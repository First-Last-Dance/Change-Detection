# Change Detection for Remote Sensing Project

## Overview

This project focuses on detecting changes in remote sensing images using both classical and deep learning models. The objective is to identify areas that have undergone significant changes over time. The project explores several techniques and models to achieve high accuracy in change detection, even with an unbalanced dataset.

## Models and Methods

### Classical Methods

1. **Image Differencing**: This method involves subtracting the pixel values of one image from another taken at a different time. Significant differences indicate changes.

2. **Image Ratioing**: This technique calculates the ratio of pixel values between two images. Areas with no change will have a ratio close to one, while changed areas will deviate significantly.

3. **Change Vector Analysis (CVA)**: CVA is a technique that analyzes the difference in spectral properties between two images. The magnitude of the change vector indicates the extent of change.

### Deep Learning Models

1. **UNet Model**: A convolutional neural network (CNN) designed for semantic segmentation. It captures both spatial and contextual information, making it suitable for detecting changes in remote sensing images.

2. **UNet_Siamese**: This model combines the architecture of UNet with the concept of Siamese networks. It processes two images simultaneously and compares their feature maps to detect changes. We got this idea from the paper [Ultralightweight Spatial–Spectral Feature Cooperation Network for Change Detection in Remote Sensing Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081023).

## Dataset

The dataset used for this project consists of remote sensing images taken at different times. The dataset is unbalanced, with many images showing no changes. To address this, data augmentation techniques were applied to images with changes to help the models focus on changed areas.

## Data Augmentation

Data augmentation techniques such as rotation and reflection were applied to images with changes to balance the dataset. This helps the models learn to identify changes more effectively.

## Results

The UNet_Siamese model achieved the best performance with a Jaccard index of 85% on the entire dataset. The classical methods also provided valuable insights but were less accurate compared to the deep learning models.

## Code Structure

- #### Classical Methods: 

    The `Classical_Model.py` file contains implementations of the three classical change detection techniques: Image Differencing, Image Ratioing, and CVA.

- #### Deep Learning Models

    The `Deep_Model.py` file includes implementations of the UNet and UNet_Siamese models. It also contains code for training the models, applying data augmentation, and evaluating performance using the Jaccard index.

## How to Run

In each file, you can put the id of the dataset file to download it from the google drive or if you have the dataset you can put the path of the dataset in the code and skip the download part.
