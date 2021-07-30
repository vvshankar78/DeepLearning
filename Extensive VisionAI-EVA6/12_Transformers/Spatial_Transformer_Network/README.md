# Spatial Transformers

### Team Members

Vidya Shankar, Mahesh, 

---



### CNN's are not invariant -

CNN (Convolution neural networks) are typically lace the invariance property to the input image. This means, they suffer from 

- *Scale / size variation in the input data*.
- *Rotation variation in the input data.*
- *Clutter in the input data.*

CNN performance are not good when there are variation though Max Pooling does help solve this issue for small variances, how ever they do not  help in making CNN's invariant when there is large variation in the data. To address this, we use Spatial Transformer Network (STN) which applies transformation to properly scale, resize and crop the image. 

### What are Spatial Transformer Networks (STNs)

STN's consist of Spatial transformer modules are neural network  (CNN and MLP) where in we apply transformations to properly scale, resize, crop an image. Since the transformation parameters come from a neural network module, they are learnable. STN's can be applied to both input images and feature maps. They can be inserted into any part of the CNN Architecture. 

**STN's acts as an attention mechanism and knows where to focus on the input image. **



![](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/images/fig-1.jpg?raw=false)

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/10_Object_Localization/TinyImagenet/outputs/LR_finder.png?raw=false" style="zoom: 105%;" />





https://colab.research.google.com/drive/1I7s6XTGJaQ8oyaOdG-9DNc6w3PF3Ened?usp=sharing





## [TinyImageNet](https://github.com/vvshankar78/DeepLearning/tree/master/Extensive%20VisionAI-EVA6/10_Object_Localization/TinyImagenet)

1. Download this [TINY IMAGENET ](http://cs231n.stanford.edu/tiny-imagenet-200.zip)dataset. 
2. Train ResNet18 on this dataset for 50 Epochs. Target 50%+ Validation Accuracy. 
3. Used LR finder and One cycle policy for learning rates. 



## [CocoDataset Anchor Box using K-means](https://github.com/vvshankar78/DeepLearning/tree/master/Extensive%20VisionAI-EVA6/10_Object_Localization/Coco_Anchor_Box)

1. Download [COCO DATASET ](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and understand the schema 
2. class distribution (along with the class names) along with a graph 
3. Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.

