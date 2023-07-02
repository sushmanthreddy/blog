+++
title =  "GSOC WEEK 4 "
tags = ["DevoLearn", "GSOC","OpenWorm" ,"INCF"]
date = "2023-06-28"

+++


# DevoNet

I have started working on the devonet model as per my gsoc proposal ,As we are using cell tracking dataset and dataset we have the 3d images of with nucleus detection and nucleus segmentation process whole data is in the format of TIF format and images are grey scale , labels and detection.

## Dataset.py 

Here is the code for the dataset.py file it reads the 3d images

```bash
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from skimage import io
from skimage import io
from skimage import transform as tr

import torch
from torch.utils.data import Dataset


def read_img(path, arr_type="npz"):
    """read image array from path
    Args:
        path (str)          : path to directory which images are stored.
        arr_type (str)      : type of reading file {'npz','jpg','png','tif'}
    Returns:
        image (np.ndarray)  : image array
    """
    if arr_type == "npz":
        image = np.load(path)["arr_0"]
    elif arr_type in ("png", "jpg"):
        image = io.imread(path, mode="L")
    elif arr_type == "tif":
        image = io.imread(path)
    else:
        raise ValueError("invalid --input_type : {}".format(arr_type))

    return image.astype(np.int32)


def crop_pair_3d(
    image1,
    image2,
    crop_size=(128, 128, 128),
    nonzero_image1_thr=0.0001,
    # nonzero_image1_thr=0.0,
    nonzero_image2_thr=0.0001,
    # nonzero_image2_thr=0.0,
    nb_crop=1,
    augmentation=True,
):
    """3d {image, label} patches are cropped from array.
    Args:
        image1 (np.ndarray)                  : Input 3d image array from 1st domain
        image2 (np.ndarray)                  : Input 3d label array from 2nd domain
        crop_size ((int, int, int))         : Crop image patch from array randomly
        nonzero_image1_thr (float)           : Crop if nonzero pixel ratio is higher than threshold
        nonzero_image2_thr (float)           : Crop if nonzero pixel ratio is higher than threshold
        nb_crop (int)                       : Number of cropping patches at once
    Returns:
        if nb_crop == 1:
            cropped_image1 (np.ndarray)  : cropped 3d image array
            cropped_image2 (np.ndarray)  : cropped 3d label array
        if nb_crop > 1:
            cropped_images1 (list)       : cropped 3d image arrays
            cropped_images2 (list)       : cropped 3d label arrays
    """
    z_len, y_len, x_len = image1.shape
    # _, x_len, y_len, z_len = image1.shape
    assert x_len >= crop_size[0]
    assert y_len >= crop_size[1]
    assert z_len >= crop_size[2]
    cropped_images1 = []
    cropped_images2 = []

    while 1:
        # get cropping position (image)
        top = random.randint(0, x_len - crop_size[0] - 1) if x_len > crop_size[0] else 0
        left = (
            random.randint(0, y_len - crop_size[1] - 1) if y_len > crop_size[1] else 0
        )
        front = (
            random.randint(0, z_len - crop_size[2] - 1) if z_len > crop_size[2] else 0
        )
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        rear = front + crop_size[2]

        # crop image
        cropped_image1 = image1[front:rear, left:right, top:bottom]
        cropped_image2 = image2[front:rear, left:right, top:bottom]
        # get nonzero ratio
        nonzero_image1_ratio = np.nonzero(cropped_image1)[0].size / float(
            cropped_image1.size
        )
        nonzero_image2_ratio = np.nonzero(cropped_image2)[0].size / float(
            cropped_image2.size
        )

        # rotate {image_A, image_B}
        if augmentation:
            aug_flag = random.randint(0, 3)
            for z in range(cropped_image1.shape[0]):
                cropped_image1[z] = np.rot90(cropped_image1[z], k=aug_flag)
                cropped_image2[z] = np.rot90(cropped_image2[z], k=aug_flag)

        # break loop
        if (nonzero_image1_ratio >= nonzero_image1_thr) and (
            nonzero_image2_ratio >= nonzero_image2_thr
        ):
            if nb_crop == 1:
                return cropped_image1, cropped_image2
            elif nb_crop > 1:
                cropped_images1.append(cropped_image1)
                cropped_images2.append(cropped_image2)
                if len(cropped_images1) == nb_crop:
                    return np.array(cropped_images1), np.array(cropped_images2)
            else:
                raise ValueError("invalid value nb_crop :", nb_crop)


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        root_path,
        split_list,
        train=True,
        model="NSN",
        arr_type="npz",
        normalization=False,
        augmentation=True,
        scaling=True,
        resolution=eval("(1.0, 1.0, 2.18)"),
        crop_size=eval("(96, 96, 96)"),
        ndim=3,
    ):
        self.root_path = root_path
        self.split_list = split_list
        self.model = model
        self.arr_type = arr_type
        self.normalization = normalization
        self.augmentation = augmentation
        self.scaling = scaling
        self.resolution = resolution
        self.crop_size = crop_size
        self.train = train
        self.ndim = ndim

        with open(split_list, "r") as f:
            self.img_path = f.read().split()

    def __len__(self):
        return len(self.img_path)

    def _get_image_3d(self, i):
        image = read_img(
            os.path.join(self.root_path, "images_raw", self.img_path[i]), self.arr_type
        )
        ip_size = (
            int(image.shape[0] * self.resolution[2]),
            int(image.shape[1] * self.resolution[1]),
            int(image.shape[2] * self.resolution[0]),
        )
        image = tr.resize(
            image.astype(np.float32), ip_size, order=1, preserve_range=True
        )
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            image = np.pad(image, pad_width=pad_size, mode="reflect")
        if self.scaling:
            image = (image - image.min()) / (image.max() - image.min())
        return image.astype(np.float32)

    def _get_label_3d(self, i):
        if self.model == "NSN" or self.model == "3DUNet":
            label = read_img(
                os.path.join(self.root_path, "images_nsn", self.img_path[i]),
                self.arr_type,
            )
        elif self.model == "NDN":
            label = read_img(
                os.path.join(self.root_path, "images_ndn", self.img_path[i]),
                self.arr_type,
            )
        else:
            print("Warning: select model")
            sys.exit()
        ip_size = (
            int(label.shape[0] * self.resolution[2]),
            int(label.shape[1] * self.resolution[1]),
            int(label.shape[2] * self.resolution[0]),
        )
        label = (tr.resize(label, ip_size, order=1, preserve_range=True) > 0) * 1
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            label = np.pad(label, pad_width=pad_size, mode="reflect")
        return label.astype(np.int32)

    def __getitem__(self, i):
        if self.ndim == 3:
            image = self._get_image_3d(i)
            label = self._get_label_3d(i)
        else:
            print("Error: ndim must be 3 dimensions")
        if self.train:
            if self.ndim == 3:
                x, t = crop_pair_3d(image, label, crop_size=self.crop_size)
            return np.expand_dims(x.astype(np.float32), axis=0), t.astype(np.int32)
        else:
            return image.astype(np.float32), label.astype(np.int32)
```

The provided code consists of utility functions and a dataset class for preprocessing and handling 3D images and labels. Let's go through the code and explain each part:

1. Importing packages:
   - The code imports necessary packages such as os, sys, random, numpy, skimage.io, skimage.transform, and torch for various functionalities related to file operations, image processing, and dataset handling.

2. read_img function:
   - This function reads an image array from a given path.
   - It takes the path to the image directory (path) and the type of the reading file (arr_type) as input.
   - Depending on the arr_type, it loads the image array using the appropriate method (np.load for npz files, io.imread for jpg, png, and io.imread for tif files).
   - The function returns the image array as a numpy array of type np.int32.

3. crop_pair_3d function:
   - This function crops 3D patches from an input image and label array.
   - It takes several arguments including the input image array (image1), input label array (image2), crop size (crop_size), nonzero pixel thresholds for the image and label (nonzero_image1_thr and nonzero_image2_thr), the number of cropping patches (nb_crop), and a flag for augmentation (augmentation).
   - The function crops random patches from the image and label arrays based on the specified crop size. If the arrays are larger than the crop size, random positions are chosen for cropping. If the arrays are smaller than the crop size, no cropping is performed.
   - The function calculates the nonzero pixel ratios for the cropped image and label arrays.
   - If augmentation is enabled, the function randomly rotates the cropped image and label arrays.
   - The function returns the cropped image and label arrays, either as individual arrays if nb_crop is 1 or as lists of arrays if nb_crop is greater than 1.

4. PreprocessedDataset class:
   - This class is a custom dataset class for handling preprocessed 3D images and labels.
   - It inherits from the torch.utils.data.Dataset class and overrides its __len__ and __getitem__ methods.
   - The class takes several arguments including the root path of the dataset (root_path), a split list file (split_list), a flag indicating if it's a training dataset (train), the model type (model), the type of input files (arr_type), flags for normalization and augmentation, scaling and resolution factors, crop size, and the number of dimensions (ndim).
   - In the __init__ method, the split list file is read and the image paths are stored in the img_path attribute.
   - The __len__ method returns the total number of images in the dataset.
   - The _get_image_3d method reads and preprocesses the input image based on the specified resolution, scaling, and pad size.
   - The _get_label_3d method reads and preprocesses the input label based on the specified resolution and pad size. The processing involves resizing the label, converting it to binary form, and applying padding.
   - The __getitem__ method is called when an item from the dataset is accessed. It retrieves the image and label corresponding to the given index and performs cropping if it's a training dataset.
   - The method returns the image and label as numpy arrays.

Overall, this code provides utility functions to read and preprocess 3D images and labels

## Loss function

As per the suggestions have heard the Diceloss will be great for the segmentation and have wrote the basic diceloss function for the model.

```bash
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def dice_loss(x, t, eps=0.0):
    return DiceLoss()(x, t, smooth=eps)

def softmax_dice_loss(x, t, eps=1e-7):
    return DiceLoss()(F.softmax(x, axis=1), t, smooth=eps)
```

The provided code contains a Dice Loss implementation and two utility functions for calculating the Dice Loss and Softmax Dice Loss for segmentation tasks. Let's go through the code and explain each part:

1. Importing packages:
   - The code imports necessary packages such as numpy and torch for numerical computations and deep learning operations.

2. DiceLoss class:
   - This class is a custom implementation of the Dice Loss for segmentation tasks.
   - It inherits from the torch.nn.Module class and overrides its forward method.
   - The class takes two optional arguments: weight and size_average. However, these arguments are not used in the implementation.
   - In the forward method, the inputs and targets are expected as arguments.
   - If the model contains a sigmoid or equivalent activation layer, the inputs are passed through the F.sigmoid function to ensure the values are between 0 and 1.
   - The inputs and targets tensors are then flattened using the view method.
   - The intersection between the flattened inputs and targets is calculated by element-wise multiplication and summation.
   - The Dice coefficient is computed using the intersection, inputs' sum, and targets' sum, with a smoothing factor added to avoid division by zero.
   - Finally, the Dice Loss is obtained by subtracting the Dice coefficient from 1 and returned.

3. dice_loss function:
   - This utility function calculates the Dice Loss given the inputs and targets.
   - It takes the inputs (x), targets (t), and an optional smoothing factor (eps) as arguments.
   - The function creates an instance of the DiceLoss class and calls it with the inputs, targets, and the smoothing factor.
   - The calculated Dice Loss is returned.

4. softmax_dice_loss function:
   - This utility function calculates the Softmax Dice Loss given the inputs and targets.
   - It takes the inputs (x), targets (t), and an optional smoothing factor (eps) as arguments.
   - The function applies the softmax function to the inputs along the second axis (channel dimension) using F.softmax function from PyTorch.
   - Then, it creates an instance of the DiceLoss class and calls it with the softmaxed inputs, targets, and the smoothing factor.
   - The calculated Softmax Dice Loss is returned.

The Dice Loss is a popular loss function for evaluating the similarity between predicted segmentation masks and ground truth labels. It measures the overlap between the predicted and target masks using the Dice coefficient. The Dice Loss is defined as 1 minus the Dice coefficient, where a higher value indicates better segmentation performance.

The utility functions provided in the code allow you to easily calculate the Dice Loss and Softmax Dice Loss for segmentation tasks using PyTorch tensors.