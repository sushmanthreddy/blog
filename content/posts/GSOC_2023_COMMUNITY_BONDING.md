+++
title =  "GSOC_2023_community_bonding"
tags = ["DevoLearn", "GSOC","OpenWorm" ,"INCF"]
date = "2023-06-04"

+++

Finally ,after 5 months of consistent work and sleepless night the d-day has approached ,continous doubts started poping my head and was really confused that , have given my best shot.
At night around 10 clock have received mail that have been selected to google summer of code  under INCF under mentoring under bradly alicea and mayukh deb.

# Community Bonding Period.

I have started working in my community bonding period itself , because of my semester exams are starting exactly after community bonding period .
so, I have started reading some paper regarding cells and there tracking in C. elegans and instance segmentation. 

## C. elegans 

Segmenting and tracking moving cells in time-lapse video sequences is a challenging task, required for many applications in both scientific and industrial settings. Properly characterizing how cells change shapes and move as they interact with their surrounding environment is key to understanding the mechanobiology of cell migration and its multiple implications in both normal tissue development and many diseases.we objectively compare and evaluate state-of-the-art whole-cell and nucleus segmentation and tracking methods using both real (2D and 3D) time-lapse microscopy videos of cells and nuclei, along with computer-generated (2D and 3D) video sequences simulating whole cells and nuclei moving in realistic environments in DevoLearn.

Understanding the data is very important in DeepLearning, the C .elegans are very  tricky , when they are in worm stage they divide from 2 cell stage to 320 cell stage and there volume of cell decrease completly as they divide .
And name of cells will be given according to there position according to time. 
Here we took data from cell tracking dataset where the cells are stored in *.tif format

### Data capturing

```bash
Microscope: Zeiss LSM 510 Meta

Objective lens: Plan-Apochromat 63x/1.4 (oil)

Voxel size (microns): 0.09 x 0.09 x 1.0

Time step (min): 1 (1.5)

```

### Folder Format and files meaning in Dataset

man_trackT.tif - 16-bit multi-page tiff file (markers have unique positive labels propagated over time, background has zero label). It contains ground truth markers for the corresponding original image tT.tif. The man_trackT.tif file is provided for every tT.tif file.

man_segT.tif - 16-bit multi-page tiff file (segmented objects have unique positive labels that are not necessarily propagated over time, background has zero label). It contains reference segmentation for the corresponding original image tT.tif. In the case of gold segmentation truth, only selected frames are annotated (i.e., the man_segT.tif file does not have to be provided for every tT.tif file). However, in those frames, all objects are segmented. In the case of silver segmentation truth, all frames tend to be completely annotated. Nevertheless, because of being difficult to segment automatically, some objects may be missing there. If that involves all objects in a particular frame, the reference segmentation annotation is not released at all.

man_seg_T_Z.tif (gold segmentation truth only) - 16-bit multi-page tiff file (segmented objects have unique positive labels that are not necessarily propagated over time, background has zero label). It contains reference segmentation for the Z-th slice from the corresponding original image tT.tif. Not all objects have to be segmented. The man_seg_T_Z.tif file does not have to be provided for every slice of each tT.tif file. Only the slices with non-empty reference segmentation are released.

### Data Analysis

I have started analyzing the dataset, I have found out that ,flouroscene images , ground truth segmentation and ground truth markers of the segmentation  of those datsets.

I have used simpleITK library for the analyzing .*tif files and have zero experince with the tif files and understanding them took a lot of time.I have converted them to image array and then converted png/jpeg format

here are the results of the dataset.
<img src="../images/gsoc_community_background/inferno_sbs_celltrackingchallenge.png" alt="" width="700" height="">

every tif file has 34 slices of images and this is how it looks when we ploted it combinedly

<img src="../images/gsoc_community_background/inferno_sbs_celltrackingchallenge_grid.png" alt="" width="1300" height="">

and this doesnt have only ground segmentation but also have marker point indicating the cell

<img src="../images/gsoc_community_background/seg_marker.png" alt="" width="600" height="">

I have converted them to png and jpeg format and ploted the compression loss of images in png and jpeg format.

here are the compression plots of png format
<img src="../images/gsoc_community_background/newplot-2.png" alt="" width="500" height="">

here are the compression plots of jpeg format
<img src="../images/gsoc_community_background/newplot.png" alt="" width="500" height="">

and these are converted and into png and jpeg format and stored in seperate folders.

## instance segmentation

As per my reading [DevoLearn](https://github.com/DevoLearn) has semantiuc segmentation but it doesnt have instance segmentation for and this year for gsoc we are adding instance segmentation for the DevoLearn lib

As at my time the most popular model from metaAI has released its own model for instance segmentation named with segment anything model,this model has zeroshot learning features it can segment any model organism , which is not even trained the particular data.

a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks.

Here is the model architecture of segment anything.

<img src="../images/gsoc_community_background/Screenshot 2023-06-04 at 1.35.28 PM.png" alt="" width="500" height="">


[GITHUB_LINK_FOR_GSOC_WEEKLY_WORK](https://github.com/sushmanthreddy/GSOC_2023/tree/main)


## Coming weeks commitments

* Fine tune the segment anything model on model C .elegans extracted data and extract the centroids of each cell and volume of each cell.
* And start implementing DevoNet which proposed in my GSOC proposal and cross verify the results between fine tuned model,DevoLearn and DevoNet.
* final phase work would be using using DevoNet and integrate with DevoGraph.


