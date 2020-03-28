import os
import sys
import json
import time 
import copy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tqdm
import pandas as pd 
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms

from models import get_model
from data_loader import setup_dataset
from utils.train import ObjFromDict

def load_model(run_dir, metric): 
    with open(os.path.join(run_dir,'config.json')) as json_file:
        config = json.load(json_file)
    config = ObjFromDict(config)
    model = get_model(config.model)
    checkpoint_path = os.path.join(run_dir, 'best_{}.pth'.format(metric)) 
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model, config

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


def infer(model, dataloader, output_path, device='cuda'):
    assert output_path.endswith('.csv')
    model = model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                prob_pred_top3, class_pred_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None)
                top3array = pd.DataFrame(np.squeeze(class_pred_top3.to('cpu').numpy()))
                target['predicted_1'] = top3array[0]
                target['predicted_2'] = top3array[1]
                target['predicted_3'] = top3array[2]
                results.append(pd.DataFrame(target, index=None))
    result = pd.concat(results)
    result.to_csv(output_path, index=None)
    return result

# def get_embedding(model, dataloader, output_path, output_path_label, device='cuda'):
#     assert output_path.endswith('.npy')
#     model = model.to(device)
#     model.eval()
#     embedding = torch.nn.Sequential(*list(model.children())[:-1])
#     embeddings = []
#     labels = []
#     with torch.no_grad():
#         for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
#                 data = data.to(device)
#                 output = embedding(data).cpu().numpy()
#                 embeddings.append(output)
#                 labels.append(copy.deepcopy(target['label'].cpu().numpy()))
#     labels = np.concatenate(labels)
#     embeddings = np.concatenate(embeddings)
#     embeddings = np.squeeze(embeddings)
#     np.save(output_path, embeddings)
#     np.save(output_path_label, labels)
#     return embeddings, labels

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# def get_accuracy(results, k):
#     """
#     function that computes the top k accuracy from a dataframe with a column label and the top k predicted 
#     """
#     acc = 0
#     try:
#         for i in range(1, k + 1):
#             acc += len(results.loc[(results['label'] == results['predicted_{}'.format(i)])]) / len(results)
#     except:
#         pass
#     return acc

# def create_acc_per_occ_curve(run_dir, unique_occ, res, k):
#     acc_per_occ = []
#     for n in unique_occ:
#         acc_per_occ.append(get_accuracy(res.loc[(res['occ'] == n)],k))
#     fg, ax = plt.subplots(1,1)
#     ax.set_title('top {} accuracy per number of occurences in the training set'.format(k))
#     ax.set_xlabel('number of example in the training set')
#     ax.set_ylabel('top {} accuracy'.format(k))
#     ax.scatter(unique_occ, acc_per_occ)
#     fg.savefig(os.path.join(run_dir,'top{}_accuracy_per_occ.jpg'.format(k)))
    
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# def get_tnse(embeddings, output_path_tnse, output_path_pca_plot, n_comp_pca=10):
#     assert output_path_tnse.endswith('.npy')
#     print('starting tsne embedding')
#     start = time.time()
#     pca = PCA(n_comp_pca)
#     small_embeddings = pca.fit_transform(embeddings)
#     fg, ax = plt.subplots(1,1)
#     ax.set_title('PCA explained variance before tsne')
#     ax.plot(pca.explained_variance_ratio_)
#     fg.savefig(output_path_pca_plot)
#     tnse_embedded = TSNE(n_components=2,).fit_transform(small_embeddings)
#     np.save(output_path_tnse, tnse_embedded)
#     print('time taken for the tsne embedding {:.1f}'.format(time.time() - start))
#     return tnse_embedded

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


def compute_grad_weights(model, last_conv_layer, target_class, sample, device):
    store = {}
    
    def register_convmap(self, in_map, out_map):
        store["conv_maps"] = out_map
    
    def register_grads(self, in_grads, out_grads):
        store["grads"] = out_grads
    
    handle_fw = last_conv_layer.register_forward_hook(register_convmap)
    handle_bw = last_conv_layer.register_backward_hook(register_grads)
    
    try:
        output = model(sample)
        
        one_hot = torch.FloatTensor(1, model.num_classes)
        one_hot.zero_()
        one_hot[:, target_class] = 1
        
        try:
            one_hot = torch.sum(one_hot * output)
        except:
            output = output.to(device)
            one_hot = one_hot.to(device)
            one_hot = torch.sum(one_hot * output)
        
        model.zero_grad()
        one_hot.backward()

    finally:
        handle_fw.remove()
        handle_bw.remove()
    
    conv_grads = store["grads"]# torch.autograd.grad(outputs = output[:, target_class][0],
                 #                   inputs = store["conv_maps"])[0]
    grad_weights = torch.mean(conv_grads[0], dim=[2, 3])
    
    return grad_weights, store["conv_maps"]

def compute_heatmap(conv_maps, grad_weights, model_out_size, device, interpolation_mode='bilinear', original_out_size=(400, 400)):
    with torch.no_grad():
        for i in range(conv_maps.shape[1]):
            conv_maps[:, i] *= grad_weights[:, i]
        try :
            coarse_heatmap = torch.relu(conv_maps.sum(dim=1)).detach()
        except:
            conv_maps = conv_maps.to(device)
            coarse_heatmap = torch.relu(conv_maps.sum(dim=1)).detach()
        coarse_heatmap -= coarse_heatmap.min()
        coarse_heatmap /= coarse_heatmap.max()
        
        #resnets have an output in 13x13 with input size of 400x400
        heatmap = torch.nn.functional.interpolate(coarse_heatmap.view(1, -1, model_out_size, model_out_size),
                                                  size=original_out_size,
                                                  mode=interpolation_mode)
    return coarse_heatmap, heatmap

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
          
def main(raw_args=None): 
   
    parser = argparse.ArgumentParser()    
    parser.add_argument('--run_dir', required=True, 
                        type=str,help = 'path to the run directory to perform the error analysis on')
    parser.add_argument('--recompute', action='store_true', default=False,
                        help='whether to recompute the embedding if already in the folder')
    args = parser.parse_args(raw_args)
    
    recompute = args.recompute
    run_dir = args.run_dir
    model, config = load_model(run_dir)
    
    if config.dataset.name == 'fieldguide':
        highlighted_classes = [0, 2521, 4084, 2066, 2840]
    elif config.dataset.name == 'butterfly':
        highlighted_classes = []
        while len(highlighted_classes)<3:
            selected_class = np.random.randint(1, config.model.num_classes)
            if selected_class not in highlighted_classes:
                highlighted_classes.append(selected_class)
    
    output_dir =  os.path.join(run_dir, 'error_analysis')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    config.training.batch_size = 32
    train_dataloader, val_dataloader = setup_dataset(config)
    
    if not recompute: 
        try: 
            result = pd.read_csv(os.path.join(output_dir, 'val_pred.csv'))
            print('Predictions recovered from previous inference')
        except: 
            print('Not able to recover predictions from previous inference, starting inference again ...')
            result = infer(model, val_dataloader,os.path.join(output_dir, 'val_pred.csv'))
    else:
        result = infer(model, val_dataloader,os.path.join(output_dir, 'val_pred.csv'))
        
    occ =  train_dataloader.dataset.occ
    occ = pd.DataFrame(occ, index=None).reset_index()
    occ = occ.rename({'index':'label','label':'occ'}, axis=1)
    results_with_occ = pd.merge(result,occ, on='label', how='inner')
    unique_occ = occ['occ'].unique()
    
    create_acc_per_occ_curve(output_dir, unique_occ, results_with_occ, 1)
    create_acc_per_occ_curve(output_dir, unique_occ, results_with_occ, 3)
    
    if not recompute: 
        try: 
            embeddings, labels = np.load(os.path.join(output_dir, 'embedding.npy')), np.load(os.path.join(output_dir, 'labels.npy'))
            print('Embeddings recovered from previous inference')
            try:
                tnse_embedding = np.load(os.path.join(output_dir, 'tnse_embedding.npy'))
                print('TNSE embeddings recovered from previous inference')
            except: 
                print('Not able to recover TNSE embeddings from previous inference')
                tnse_embedding = get_tnse(embeddings, 
                                          os.path.join(output_dir, 'tnse_embedding.npy'), 
                                          os.path.join(output_dir,'pca_explained_ratio.jpg'))
        except: 
            print('Not able to recover embeddings from previous inference, starting inference again ...')
            embeddings, labels = get_embedding(model, 
                                               val_dataloader, 
                                               os.path.join(output_dir, 'embedding.npy'),
                                               os.path.join(output_dir, 'labels.npy'))
            tnse_embedding = get_tnse(embeddings, 
                                      os.path.join(output_dir, 'tnse_embedding.npy'), 
                                      os.path.join(output_dir,'pca_explained_ratio.jpg'))
    else:
        embeddings, labels = get_embedding(model,
                                           val_dataloader,
                                           os.path.join(output_dir, 'embedding.npy'),
                                           os.path.join(output_dir, 'labels.npy'))
        tnse_embedding = get_tnse(embeddings,
                                  os.path.join(output_dir, 'tnse_embedding.npy'),
                                  os.path.join(output_dir,'pca_explained_ratio.jpg'))
    
    for label in highlighted_classes: 
        fg, ax = plt.subplots(1,1)
        ax.set_title('TNSE embedding class {} vs others \n n train samples : {} | top 1 acc : {:.3f}'.format(label, 
                                                                occ.loc[occ['label']==1]['occ'].values[0],
                                                                get_accuracy(results_with_occ.loc[results_with_occ['label']==label],1)))
        ax.scatter(tnse_embedding[:,0], tnse_embedding[:,1],s=0.05, c=1-(labels==label) )
        fg.savefig(os.path.join(output_dir, 'tnse_class_{}.jpg'.format(label)))
        
        
    # Add a Grad-CAM visualization to interpret our model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trunc_model = nn.Sequential(*list(model.children())[:-2])
    trunc_model.eval()
    with torch.no_grad():
        empty = torch.zeros((1,3,400,400))
        try:
            model_out_size = trunc_model(empty).shape[-1]
        except:
            empty = empty.to(device)
            model_out_size = trunc_model(empty).shape[-1]
        
    last_conv_layer = list(model.children())[-3][-1][-1].conv2
    for index in range(5):
        sample, infos = val_dataloader.dataset[index]
        sample = torch.from_numpy(sample).view(1,3,400,400)
        # 'sample' is the preprocessed image

        # Load original image preprocessed without normalization
        try:
            path = infos['file_name']
            img_path = os.path.join(os.path.join(config.dataset.root, 'train_images'), path)
        except:
            path = infos['path']
            img_path = os.path.join(config.dataset.image_dir, path)
        input_image = Image.open(img_path).convert('RGB')
        input_image = transforms.Resize(400)(input_image)

        with torch.no_grad():
            try:
                output = model(sample)
            except:
                sample = sample.to(device)
                output = model(sample)
            values, indices = torch.topk(output, 3)

        fig = plt.figure(figsize=(12, 3))
        fig.suptitle("Heatmaps for image #{}".format(index), y=1.10)
        ax = fig.add_subplot(141)
        ax.set_title("Original Image")
        ax.imshow(input_image)
        #ax.set_xticks(None)
        #ax.set_yticks(None)

        for i, target_class in enumerate(indices[0]):
            weights, conv_maps = compute_grad_weights(model, last_conv_layer, target_class, sample, device)
            coarse_heatmap, heatmap = compute_heatmap(conv_maps, weights, model_out_size, device)
            ax = fig.add_subplot(142 + i)
            title = "Class #{} predicted".format(target_class.item())
            ax.set_title(title)
            ax.imshow(input_image)
            try:
                ax.imshow(heatmap.numpy()[0, 0], cmap="jet", alpha=0.5)
            except:
                ax.imshow(heatmap.cpu().numpy()[0, 0], cmap="jet", alpha=0.5)
            #ax.set_xticks(None)
            #ax.set_yticks(None)
            
        fig.savefig(os.path.join(output_dir, 'grad_cam_{}.jpg'.format(index)))
        
        
    with open(os.path.join(output_dir, 'error_analysis.html'),'w') as file: 
        file.write('<!DOCTYPE html>\n'
                   '<html lang="fr">\n'
                   '  <head>\n'
                   '    <meta charset="utf-8">\n'
                   '    <title>Error_Analysis | {}</title>\n'
                   '  </head>\n'
                   '  <body>\n'
                   '    <div> \n'
                   '       <h1> Error_Analysis | {} </h1>\n'                   
                   '       <h2> Global Metrics : </h2>\n'
                   '        <p> Acc: {} <br /> Top3 acc : {}  <br /> '
                   '       </p>\n'
                   '    </div>\n'.format(run_dir.split('/')[-1],
                                         run_dir.split('/')[-1],
                                         get_accuracy(result,1), get_accuracy(result,3)
                                        ))
        file.write('    <div> \n'
                   '       <h2> Impact of the number of occurences : </h2>\n'
                   '     <img src="top1_accuracy_per_occ.jpg">\n'
                   '     <img src="top3_accuracy_per_occ.jpg">\n'
                   '    </div>\n')
        file.write('    <div> \n'
                   '       <h2> TNSE Embedding : </h2>\n'
                   '     <img src="pca_explained_ratio.jpg">\n')
        for i in highlighted_classes: 
            file.write('     <img src="tnse_class_{}.jpg">\n'.format(i))
        file.write('    </div>\n')
        
        file.write('    <div> \n'
                   '       <h2> Grad-CAM visualizations : </h2>\n')
        for index in range(5): 
            file.write('     <img src="grad_cam_{}.jpg">\n'.format(index))
        file.write('    </div>\n')
        
        file.write('  </body>\n'
                   '</html>\n')
    

    
if __name__ == '__main__': 
    main()   