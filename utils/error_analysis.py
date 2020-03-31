import os
import sys
import json
import time 
import copy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SimpleITK as sitk
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import get_model
from data_loader.LiTS import LiTSDataset
from utils.train import ObjFromDict
from skimage.measure import label

def load_model(run_dir, metric='validation_dice'): 
    with open(os.path.join(run_dir,'config.json')) as json_file:
        config = json.load(json_file)
    config = ObjFromDict(config)
    model = get_model(config.model)
    checkpoint_path = os.path.join(run_dir, 'best_{}.pth'.format(metric)) 
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model, config


def compute_dice(gt, pred): 
    eps = 1e-5
    intersection = np.sum(gt * pred)
    return ((2*  intersection) + eps)/ (eps + np.sum(gt+pred))

          
def main(raw_args=None): 
   
    parser = argparse.ArgumentParser()    
    parser.add_argument('--run_dir', required=True, 
                        type=str,help = 'path to the run directory to perform the error analysis on')
    parser.add_argument('--post_process', action='store_true', default=False,
                        help='whether to use postprocessing or not')
    parser.add_argument('--device', default='cuda:0', 
                        type=str,help = 'device to use')

    args = parser.parse_args(raw_args)
    
    device=args.device
    run_dir = args.run_dir
    model, config = load_model(run_dir)

    output_dir =  os.path.join(run_dir, 'error_analysis')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    config.training.batch_size = 1
    
    data_path = config.dataset.root
    print('data_path ', data_path)

    # fix the seed for the split
    split_seed = 0 
    np.random.seed(split_seed)

    image_dir = os.listdir(os.path.join(data_path,'Training Batch 1')) + os.listdir(os.path.join(data_path,'Training Batch 2'))
    all_indexes = [ int(file_name[7:-4]) for file_name in image_dir if 'volume' in file_name]
    split = np.random.permutation(all_indexes)
    n_train, n_val, n_test = int(0.8 * len(split)), int(0.1 * len(split)), int(0.1 * len(split))

    train = split[: n_train]
    val = split[n_train : n_train+n_val]
    test = split[n_train + n_val :]
    
    batch_size=1
    
    
    model = model.to(device)
    model.eval()
    
    #### add test for physical reference size
    dataset = LiTSDataset(data_path, test, 
                          physical_reference_size = (512, 512, 512), 
                          spacing=2,
                          bounding_box=[0,1,0,1,0,1],
                          no_tumor=True, 
                          inference_mode=True)
    dataloader = DataLoader(dataset=dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    
    dices = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):

                data = data.to(device)
                output = model(data)
                for i in range(batch_size):
                    img = sitk.ReadImage(target['original_image_path'][i])
                    normalization_transform = dataset.get_normalization_transform(img)
                    inv_normalization_transform = normalization_transform.GetInverse()
                    out_mask = np.round(output.cpu().numpy()[i,1,:,:,:])

                    bb = target['bounding_box'].cpu().numpy()[i]

                    ref_img = sitk.Resample(img, dataset.reference_image, normalization_transform)

                    big_mask_numpy = sitk.GetArrayFromImage(ref_img)
                    big_mask_numpy[:,:,:] = 0
                    big_mask_numpy[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = out_mask

                    out_mask_image = sitk.GetImageFromArray(big_mask_numpy)
                    out_mask_image.SetOrigin(ref_img.GetOrigin())
                    out_mask_image.SetDirection(ref_img.GetDirection())
                    out_mask_image.SetSpacing(ref_img.GetSpacing())
                    out_mask_image_original_space = sitk.Resample(out_mask_image, 
                                                                  img, 
                                                                  inv_normalization_transform, 
                                                                  sitk.sitkNearestNeighbor)
                    original_mask = sitk.ReadImage(target['original_mask_path'][i])
                    original_mask_array = np.clip(sitk.GetArrayFromImage(original_mask),0,1)
                    out_mask_image_original_space_array = np.clip(sitk.GetArrayFromImage(out_mask_image_original_space),0,1)
                    if args.post_process:
                        labels, num = label(out_mask_image_original_space_array, return_num=True)
                        one_hot_labels = np.zeros(labels.shape + (1+num,))
                        labels = np.expand_dims(labels, axis=-1)
                        one_hot_labels = (labels == np.arange(1+num).reshape(1, 1 + num)).astype('int')
                        biggest_component_idx = np.argsort(np.sum(one_hot_labels, axis=(0,1,2)))[-2]
                        out_mask_image_original_space_array = one_hot_labels[:,:,:,biggest_component_idx]
                    dice = compute_dice(out_mask_image_original_space_array, original_mask_array)
                    dices.append(dice)
                    print(dice)
                    img_array = np.clip(sitk.GetArrayFromImage(img),-200,250)
                    fg, ax = plt.subplots(3,3,figsize=(10,10))
                    h,w,d = original_mask_array.shape
                    best_slice_h = np.argmax(np.sum(np.abs(out_mask_image_original_space_array - 
                                                           original_mask_array), 
                                                    axis=(1,2)))
                    best_slice_w = np.argmax(np.sum(np.abs(out_mask_image_original_space_array - 
                                                           original_mask_array), 
                                                    axis=(0,2)))
                    best_slice_d = np.argmax(np.sum(np.abs(out_mask_image_original_space_array - 
                                                           original_mask_array), 
                                                    axis=(0,1,)))
                    ax[0,0].set_title('original image')
                    ax[0,1].set_title('ground truth')
                    ax[0,2].set_title('prediction')
                    ax[0,0].imshow(np.rot90(img_array[:,:,best_slice_d],2))
                    ax[1,0].imshow(np.rot90(img_array[:,best_slice_w ,:],2))
                    ax[2,0].imshow(np.rot90(img_array[best_slice_h,:,:],2))
                    ax[0,1].imshow(np.rot90(original_mask_array[:,:,best_slice_d],2))
                    ax[1,1].imshow(np.rot90(original_mask_array[:,best_slice_w ,:],2))
                    ax[2,1].imshow(np.rot90(original_mask_array[best_slice_h,:,:],2))
                    ax[0,2].imshow(np.rot90(out_mask_image_original_space_array[:,:,best_slice_d],2))
                    ax[1,2].imshow(np.rot90(out_mask_image_original_space_array[:,best_slice_w ,:],2))
                    ax[2,2].imshow(np.rot90(out_mask_image_original_space_array[best_slice_h,:,:],2))
                    fg.suptitle('prediction {} dice : {}'.format(test[int(batch_idx*batch_size+i)], dice))
                    plt.savefig(os.path.join(output_dir,'prediction_{}.png'.format(test[int(batch_idx * batch_size + i)])))
                    
        print('mean dice : ', np.mean(dices))
        print('dices',dices)
        
    
if __name__ == '__main__': 
    main()   