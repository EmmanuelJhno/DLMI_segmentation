import os


import SimpleITK as sitk
import numpy as np
import torch.utils.data as data
import torch

def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    # Compute quaternion: 
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv


class LiTSDataset(data.Dataset):
    default_aug = {'thetaX': [-10*np.pi/180,10*np.pi/180],
                   'thetaY': [-10*np.pi/180,10*np.pi/180],
                   'thetaZ': [-10*np.pi/180,10*np.pi/180],
                   'transX': [-10,10],
                   'transY': [-5,5],
                   'transZ': [-5,5],
                   'scale' : [0.8,1.2]
                   }
    # default bounding box for the liver (adjusted on the train set)
    default_bounding_box = np.asarray([0.35,0.85,0.2,0.7,0.1,0.6])

    def __init__(self, root_dir, split, augment=None, 
                 physical_reference_size=(512,512,512),spacing = 2, 
                 aug_parameters=default_aug,
                 bounding_box=default_bounding_box, 
                 no_tumor=True,
                 inference_mode=False):
        """
        returns a dataset for the LiTS challenge.
        Args:
        root_dir: string, the directory 
                containing the training images as they can be downloaded from the
                competition website. 
        split: list of int, the id of the files in the split to be loaded (eg: 1 
                for volume-1.nii).
        inferenece_mode: boolean in inference we return a dictionnary instead of the target 
        """

        super(LiTSDataset, self).__init__()
        self.no_tumor = no_tumor
        self.augment = augment
        self.inference_mode = inference_mode
        if self.inference_mode :
            self.transform_dict = {}
        self.aug_parameters = aug_parameters
        self.bounding_box = bounding_box
        self.image_filenames = []
        self.mask_filenames = []
        self.physical_reference_size = np.asarray(physical_reference_size) 
        self.spacing = spacing
        path_to_batch_1 =  os.path.join(root_dir,'Training Batch 1')
        path_to_batch_2 =  os.path.join(root_dir,'Training Batch 2')

        for idx in split:
            volume_filename = 'volume-{}.nii'.format(idx)
            mask_filename = 'segmentation-{}.nii'.format(idx)
            if volume_filename in  os.listdir(path_to_batch_1):
                self.image_filenames.append(os.path.join(path_to_batch_1,
                                                            volume_filename))
                self.mask_filenames.append(os.path.join(path_to_batch_1,
                                                        mask_filename))
            elif volume_filename in  os.listdir(path_to_batch_2):
                self.image_filenames.append(os.path.join(path_to_batch_2,
                                                            volume_filename))
                self.mask_filenames.append(os.path.join(path_to_batch_2,
                                                        mask_filename))
        assert len(self.image_filenames) == len(self.mask_filenames)

        self.set_reference_space()

    def __getitem__(self, index):
        img = sitk.ReadImage(self.image_filenames[index])

        # we generate the SiTK transform for normalization and data-augmentation(if needed)
        transform = self.get_normalization_transform(img)
        if self.augment: 
            aug_transform = sitk.Similarity3DTransform()
            aug_transform.SetCenter(self.reference_center)
            aug = self.aug_parameters
            aug_parameters = [(aug['thetaX'][1]-aug['thetaX'][0])*np.random.random() + aug['thetaX'][0],
                              (aug['thetaY'][1]-aug['thetaY'][0])*np.random.random() + aug['thetaY'][0],
                              (aug['thetaZ'][1]-aug['thetaZ'][0])*np.random.random() + aug['thetaZ'][0],
                              (aug['transX'][1]-aug['transX'][0])*np.random.random() + aug['transX'][0],
                              (aug['transY'][1]-aug['transY'][0])*np.random.random() + aug['transY'][0],
                              (aug['transZ'][1]-aug['transZ'][0])*np.random.random() + aug['transZ'][0],
                              (aug['scale'][1]-aug['scale'][0])*np.random.random() + aug['scale'][0]]
            aug_transform.SetParameters(aug_parameters) 
            transform.AddTransform(aug_transform)
            
        input = sitk.Resample(img, self.reference_image, transform)
        input = sitk.GetArrayFromImage(input)
        
        ax_1, ax_2, ax_3 = self.physical_reference_size
        bb = np.array([ax_1, ax_1, ax_2, ax_2, ax_3, ax_3])
        bb = self.bounding_box * bb / self.spacing
        bb = bb.astype('int')
        input = input.astype('float32')[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]
        input = np.clip(input, -200., 250.)
        input = (input + 200.) / 450.
        input = torch.from_numpy(np.expand_dims(input,axis=0)) # we add the channel dimension
        
        dic = {}
        one_hot_target = None
        if os.path.exists(self.mask_filenames[index]):
            mask = sitk.ReadImage(self.mask_filenames[index])
            target = sitk.Resample(mask, self.reference_image, transform)
            target = sitk.GetArrayFromImage(target)
            if self.no_tumor:
                target = np.clip(target, a_min=0, a_max=1)
            target = target.astype('int')[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]

            one_hot_target = np.zeros(target.shape + (target.max()+1,))
            target = np.expand_dims(target,axis=-1)
            if self.no_tumor:
                num_class = 2
            else : 
                num_class = 3   
            one_hot_target = (torch.from_numpy(target) == torch.arange(num_class).reshape(1, num_class)).int()
            one_hot_target = one_hot_target.permute(3,0,1,2)
            dic["original_mask_path"] = self.mask_filenames[index] 
            
        if self.inference_mode: 
            dic["bounding_box"] = bb
#             dic["inverse_transform"] = transform.GetInverse()
            dic["original_image_path"] = self.image_filenames[index] 
            dic["one_hot_target"] = one_hot_target
            self.transform_dict[self.image_filenames[index]] = dic
            return input, dic
#             return input, {"orginal_image_path" : self.image_filenames[index], "one_hot_target" : one_hot_target}
        
        elif not one_hot_target is None: 
            return input, one_hot_target
        
        else :
            raise ValueError('Error Loading the mask')
            
            
    def set_reference_space(self):
        img = sitk.ReadImage(self.image_filenames[0])
        self.dimension = img.GetDimension()
        ##########create the physical space for the dataset#####################

        # Physical image size is set arbitrary
        reference_physical_size = self.physical_reference_size

        self.reference_origin = np.zeros(self.dimension)
        reference_direction = np.identity(self.dimension).flatten()
        reference_size = (1/self.spacing * reference_physical_size).astype('int').tolist()
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
        reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
        reference_image.SetOrigin(self.reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)
        im_size = np.array(reference_image.GetSize())
        im_center = im_size / 2
        ref = reference_image.TransformContinuousIndexToPhysicalPoint(im_center)
        self.reference_image = reference_image
        self.reference_center = np.array(ref)


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
        return centered_transform
        
    def __len__(self):
        return len(self.image_filenames)
