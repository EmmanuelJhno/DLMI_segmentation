import sys 
import time
from os import listdir
from os.path import join
import datetime

import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import torchvision.transforms


class LiTSDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False, reference_size=(128,128,128)):
        """
        returns a dataset for the LiTS challenge.
        Args:
        root_dir: string, the directory 
                containing the training images as they can be downloaded from the
                competition website. 
        split: list of int, the id of the files in the split to be loaded (eg: 1 
                for volume-1.nii).
        """

        super(LiTSDataset, self).__init__()
        self.transform = transform
        self.image_filenames = []
        self.mask_filenames = []
        self.reference_size = np.asarray(reference_size)
        path_to_batch_1 =  join(root_dir,'Training Batch 1')
        path_to_batch_2 =  join(root_dir,'Training Batch 2')
        
        image_dir = listdir(path_to_batch_1) + listdir(path_to_batch_2)
        all_indexes = [ int(file_name[7:-4]) for file_name in image_dir if 'volume' in file_name]
        all_indexes.remove(47)

        spliter = np.random.permutation(all_indexes)
        n_train, n_val, n_test = int(0.8 * len(spliter)), int(0.1 * len(spliter)), int(0.1 * len(spliter))
        if split=='train':
            data_idx = spliter[: n_train]
        elif split=='validation':
            data_idx = spliter[n_train : n_train+n_val]
        else:
            data_idx = spliter[n_train + n_val :]

        for idx in data_idx:
            volume_filename = 'volume-{}.nii'.format(idx)
            mask_filename = 'segmentation-{}.nii'.format(idx)
            if volume_filename in listdir(path_to_batch_1):
                self.image_filenames.append(join(path_to_batch_1, volume_filename))
                self.mask_filenames.append(join(path_to_batch_1, mask_filename))
            elif volume_filename in listdir(path_to_batch_2):
                self.image_filenames.append(join(path_to_batch_2, volume_filename))
                self.mask_filenames.append(join(path_to_batch_2, mask_filename))
        assert len(self.image_filenames) == len(self.mask_filenames)

        self.set_reference_space()



    def __getitem__(self, index):
        img = sitk.ReadImage(self.image_filenames[index])
        mask = sitk.ReadImage(self.mask_filenames[index])

        transform = self.get_normalization_transform(img)

        if self.transform: 
            None 

        input = sitk.GetArrayFromImage(sitk.Resample(img, self.reference_image, transform))
        target = sitk.GetArrayFromImage(sitk.Resample(mask, self.reference_image, transform))
        input = np.expand_dims(input, axis=0)
        target = np.expand_dims(target, axis=0)
        return input.astype('float32'), target.astype('float32')


    def set_reference_space(self):
        img = sitk.ReadImage(self.image_filenames[0])
        self.dimension = img.GetDimension()
        ##########create the physical space for the dataset#####################

        # Physical image size is set arbitrary
        reference_physical_size = self.reference_size

        reference_origin = np.zeros(self.dimension)
        reference_direction = np.identity(self.dimension).flatten()
        reference_size = [128]*self.dimension 
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
        reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)
        im_size = np.array(reference_image.GetSize())
        im_center = im_size / 2
        ref = reference_image.TransformContinuousIndexToPhysicalPoint(im_center)
        self.reference_image = reference_image
        self.reference_center = np.array(ref)
        self.reference_origin = reference_origin


    def get_normalization_transform(self, img):
        ##########create the normalization transform############################
        transform = sitk.AffineTransform(self.dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - self.reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(self.dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - self.reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)
        normalization_transform = centered_transform
        return normalization_transform
        
    def __len__(self):
        return len(self.image_filenames)
