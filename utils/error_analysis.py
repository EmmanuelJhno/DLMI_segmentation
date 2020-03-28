import os
import sys
import json
import time 
import copy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SimpleITK as sitk
import tqdm
import pandas as pd 
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from models import get_model
from data_loader.LiTS import LiTSDataset
from utils.train import ObjFromDict


def load_model(run_dir, metric='validation_dice'): 
    with open(os.path.join(run_dir,'config.json')) as json_file:
        config = json.load(json_file)
    config = ObjFromDict(config)
    model = get_model(config.model)
    checkpoint_path = os.path.join(run_dir, 'best_{}.pth'.format(metric)) 
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model, config



          
def main(raw_args=None): 
   
    parser = argparse.ArgumentParser()    
    parser.add_argument('--run_dir', required=True, 
                        type=str,help = 'path to the run directory to perform the error analysis on')
    parser.add_argument('--error_analysis', required=True, 
                        type=str,help = 'path to the run directory to perform the error analysis on')
    parser.add_argument('--test_set_path', default=None, 
                        type=str,help = 'path to the test to perform the inference on')
    parser.add_argument('--submision_path', default=None, 
                        type=str,help = 'path to the folder to store the submission')
    
    args = parser.parse_args(raw_args)
    
    run_dir = args.run_dir
    model, config = load_model(run_dir)

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