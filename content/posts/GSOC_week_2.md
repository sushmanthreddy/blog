+++
title =  "GSOC WEEK 2"
tags = ["DevoLearn", "GSOC","OpenWorm" ,"INCF"]
date = "2023-06-16"

+++

Here in this week most of time went on making the data for the prompt encoder as it accepts the prompt embeding , I have to convert all masks into bounding boxes.
Most of the weeek have worked on the custom dataset class .

Here First step have took all images into csv file and the images are call from the csv file .
with the corresponding label images .
```bash
ids=[]
label_filenames = [f for f in listdir(label_path) if isfile(join(label_path, f))]
feature_filenames = [f for f in listdir(output_features_path) if isfile(join(output_features_path, f))]
for i in range(len(feature_filenames)):
  ids.append(feature_filenames[i][1:])
print(len(ids))

df = pd.DataFrame(ids ,columns=["file_ids"])
df.to_csv('file_ids.csv', index=False)

#sanity check
df = pd.read_csv('file_ids.csv')
df.head()
````


Here real work started with the customdataset class.

## Custom dataset class
```bash
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

class segmentationDataset(Dataset):
    def __init__(self, csv, augmentation=None, transform_image=None, transform_label=None):
        self.df = pd.read_csv(csv)
        self.ids = self.df["file_ids"]
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.augmentation = augmentation

    def __getitem__(self, idx):
        image = np.array(Image.open("/kaggle/input/nucleus-data/nucleus_data/features/F" + self.ids[idx]))
        mask = np.array(Image.open("/kaggle/input/nucleus-data/nucleus_data/segmentation_maps/L" + self.ids[idx]))

        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = self.transform_image(image)
        mask = self.transform_label(mask)

        b_boxes = self.masks_to_boxes(mask)

        return image.float(), mask.float(), b_boxes.float()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def masks_to_boxes(mask):
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]  # Exclude background (0) label
        b_boxes = []
        for obj_id in obj_ids:
            obj_mask = mask == obj_id
            pos = torch.where(obj_mask)
            x_min = torch.min(pos[1])
            x_max = torch.max(pos[1])
            y_min = torch.min(pos[0])
            y_max = torch.max(pos[0])
            b_box = torch.tensor([x_min, y_min, x_max, y_max])
            b_boxes.append(b_box)
        if b_boxes:
            b_boxes = torch.stack(b_boxes)
        else:
            b_boxes = torch.empty(0, 4)
        return b_boxes


def collate_fn(batch):
    images = []
    masks = []
    b_boxes_list = []

    for image, mask, b_boxes in batch:
        images.append(image)
        masks.append(mask)
        b_boxes_list.append(b_boxes)

    max_num_b_boxes = max(len(b_boxes) for b_boxes in b_boxes_list)

    padded_b_boxes = []
    for b_boxes in b_boxes_list:
        padded_b_boxes.append(torch.cat([b_boxes, torch.zeros(max_num_b_boxes - len(b_boxes), 4)]))

    return torch.stack(images), torch.stack(masks), torch.stack(padded_b_boxes)
```

The given code defines a custom dataset class and a collate function for handling the dataset during training or testing. Let's break down the code step by step:

1. The __init__ method:
   - This method is the constructor of the custom dataset class.
   - It takes four parameters: csv, augmentation, transform_image, and transform_label.
   - csv is the path to a CSV file containing information about the dataset.
   - augmentation, transform_image, and transform_label are optional transformation functions to be applied to the images and masks.
   - The CSV file is read using pd.read_csv and stored as a DataFrame (self.df).
   - The file IDs are extracted from the DataFrame (self.ids) to be used later for loading the images and masks.

2. The __getitem__ method:
   - This method is used to retrieve an item from the dataset at a specific index (idx).
   - It loads the corresponding image and mask using the file IDs stored in self.ids.
   - If an augmentation function is provided, it applies the augmentation to the image and mask.
   - The image and mask are then transformed using the provided transform_image and transform_label functions, respectively.
   - The method also applies a static method called masks_to_boxes to convert the mask into bounding boxes.
   - Finally, it returns the transformed image, mask, and bounding boxes as tensors.

3. The __len__ method:
   - This method returns the length of the dataset, which is the number of items in self.ids.

4. The masks_to_boxes static method:
   - This method takes a mask tensor as input and converts it into bounding boxes.
   - It first identifies unique object IDs in the mask tensor (excluding the background label).
   - Then, for each object ID, it creates a binary mask for that object, finds the minimum and maximum positions of non-zero elements in the mask along both dimensions, and constructs a bounding box tensor.
   - The method returns a tensor containing all the bounding boxes.

5. The collate_fn function:
   - This function is used by the data loader to collate a batch of samples into a single batch.
   - It takes a list of samples, where each sample consists of an image tensor, mask tensor, and bounding box tensor.
   - It creates separate lists for images, masks, and bounding boxes from the samples.
   - It then finds the maximum number of bounding boxes among all the samples.
   - Next, it pads the bounding boxes in each sample with zeros to match the maximum number of bounding boxes.
   - Finally, it returns the stacked tensors of images, masks, and padded bounding boxes as the collated batch.

The provided code is designed to handle a dataset containing images and corresponding masks, where the masks are represented as segmentation maps with different object labels. The collate_fn function ensures that the batched tensors have consistent shapes for efficient processing during training or testing.

Most of the time went on understanding the collate function due to various number of the bounding_boxes. Dataloader havent worked ,so understand and it took almost whole week to write the custom dataset class for the sam model.

## transformations are applied for the model.
```bash
full_dataset = segmentationDataset(csv = "file_ids.csv",
                                    augmentation =  Compose([
                                                            #GridDistortion(p=0.5),
                                                            Transpose(p=0.5),
                                                            VerticalFlip(p=0.5),
                                                            HorizontalFlip(p=0.5),
                                                            RandomRotate90(p=0.5),
                                                            ShiftScaleRotate(p=0.1),
                                                            OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1),
                                                            ]),

                                    transform_image = transforms.Compose([ 
                                                                    
                                                                    transforms.ToPILImage(),
                                                                    ToTensor(),
                                                                    transforms.RandomApply([AddGaussianNoise( mean = 0.5,std= 0.05)], p=0.5)
                                                                ]),                                  
                                    transform_label = transforms.Compose([ 
                                                                                                      
                                                                    transforms.ToPILImage(),
                                                                    ToTensor(),

                                                                ]))
```

we added the gausina noise to images and some tranformations and augmentations are applied.

week 2 completely went on working with the customa dataset class and the image augmentation for the sam model


## Refernce 

Here is the link for the references , I have used for variable number of items in dataset class.
[link](https://pytorch.org/vision/master/auto_examples/plot_repurposing_annotations.html)



