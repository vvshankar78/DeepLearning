

<h1 align="center">Extensive Vision AI (EVA5)</h1>

<h2 align="center">Session 14 - Dataset For Monocular Depth Estimation and Segmentation </h2>

<h3 align="center"> Team Members: Prasad, Dheeraj, Rajesh, Vidhya Shankar </h3>

---
###	Objective
Create a custom dataset for monocular depth estimation and segmentation simultaneously.

As a alternative to depth camera, we use a pre-trained depth model to generate the depth maps which will be used as the ground truth for our model.

The objective of this assignment is to build depth map, surface planes and bounding boxes for custom datasets that contains Construction works with Vest, Boots, Hard hat and Mask.


#### Approach (bg)
 - Look at this model: https://github.com/intel-isl/MiDaS (Links to an external site.)
 - Look at this model: https://github.com/NVlabs/planercnn
 - Prior assignment had the custom datasets on helmet, mask, PPE, and boots for which object detection was performed using YOLO V3
 - Get the Depth Images of the above data set using Midas.
 - Get the planer images using PlanerCNN model and get planer images.
 -
 - Upload to your google drive with a shareable link to everyone, and add a GitHub repo that describes the dataset properly.


### Dataset Samples
Background:

<div align="center">
<img src= https://github.com/uday96/EVA4-TSAI/blob/master/S14-15/images/bg.png?raw=true>
</div>

### Dataset Creation

#### Background (bg)
 - "scene" images. Like the front of shops, etc.
 - 100 images of streets were downloaded from the internet.
 - Each image was resized to 224 x 224
 - Number of images: 100
 - Image dimensions: (224, 224, 3)
 - Directory size: 2.5M
 - Mean: [0.5039, 0.5001, 0.4849]
 - Std: [0.2465, 0.2463, 0.2582]
