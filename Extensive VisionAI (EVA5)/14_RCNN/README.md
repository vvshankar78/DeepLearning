

<h1 align="center">Extensive Vision AI (EVA5)</h1>

<h2 align="center">Session 14 - Dataset For Monocular Depth Estimation and Segmentation </h2>

<h3 align="center"> Team Members: Prasad, Dheeraj, Rajesh, Vidhya Shankar </h3>

---
###	Objective
Create a custom dataset for monocular depth estimation and segmentation simultaneously.

As a alternative to depth camera, we use a pre-trained depth model to generate the depth maps which will be used as the ground truth for our model.

The objective of this assignment is to build depth map, surface planes and bounding boxes for custom datasets that contains Construction works with Vest, Boots, Hard hat and Mask.


#### Approach
 - Prior assignment had the custom datasets on helmet, mask, PPE, and boots for which object detection was performed using YOLO V3
 - Get the Depth Images of the above data set using Midas. (Look at this model: https://github.com/intel-isl/MiDaS)
 - Get the planer images using PlanerCNN model and get planer images. (Look at this model: https://github.com/NVlabs/planercnn)
 - The git hup repo and the google drive sharble link is provided as part of Readme.



 #### Google Drive Links
 ##### Image Segmentation  (PlanerCNN)
 https://drive.google.com/drive/folders/1AX8oEpVVO2ixVp-WF_PSnhcpKNpMhL2J

 ##### Depth Images (MIDAS)
https://drive.google.com/drive/folders/1-Am6Naqqtfb2KuCc8_WAJaSUNw14dFds


##### Github Links
https://github.com/vvshankar78/DeepLearning/tree/master/Extensive%20VisionAI%20(EVA5)/14_RCNN




### Dataset Samples

The Data set consist of construction workers wearing hard hard, masks, vests and boots. This custom datasets was downloaded from google and a set of about 3000+ images is used for identification of objects using YOLOV3.


### Get the Depth Images using MIDAS
###  

#### Approach
 - Pass the custom dataset (3000+ images) through pre-trained monocular depth estimator model. ( https://github.com/intel-isl/MiDaS)

<div align="center">
<img src= https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/14_RCNN/Images/4f925f333e.png?raw=true>
</div>

<div align="center">
<img src= https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/14_RCNN/Images/5.png?raw=true>
</div>


### Get the Segmented Images using planercnn
###  

#### Approach
 - Environment Set up - Install Cuda8.0.
 - compile nms and roili (https://github.com/NVlabs/planercnn)
 - The script also verifies by running on the test images. So do check out test/inference to validate proper setup.
 - Perform the same action as above for generated depth perception for the cutom dataset images.
 - sample output images for the custom dataset is shown below.

<div align="center">
<img src= https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/14_RCNN/Images/0_segmentation_0_final.png?raw=true>
</div>






<div align="center">
<img src= https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/14_RCNN/Images/0_image_0.png?raw=true>
</div>


<div align="center">
<img src= https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/14_RCNN/Images/62_image_0.png?raw=true>
</div>


<div align="center">
<img src= https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/14_RCNN/Images/62_segmentation_0_final.png?raw=true>
</div>
