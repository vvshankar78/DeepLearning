# Advanced Training Concepts - Class Activation Maps

## Team Members

Vidya Shankar, Bhaskar Gaur

Regular Submission notebook:
https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/08_Resnet/Cifar_resnet_GradCam.ipynb


### Objective:

---
Clone Resnet18 model from : https://github.com/kuangliu/pytorch-cifar

Create modular code for models, train, test, test-train plots,  plot misclassified images and GradCam. 

1. pull your Github code to google colab (don't copy-paste code)

2. our colab file must:

   1. train resnet18 for 20 epochs on the CIFAR10 dataset
   2. show loss curves for test and train datasets
   3. show a gallery of 10 misclassified images
   4. show gradcam output on 10 misclassified images.

   

   ### Parameters and Hyperparameters

   - Model : Resnet18
   - Data: CIFAR10
   - Loss Function: Cross Entropy Loss
   - Optimizer: SGD
   - Scheduler: StepLR
   - Batch Size: 64
   - Learning Rate: lr=0.01
   - Epochs: 20
   - Dropout: 0.
   - L1 decay: 0
   - L2 decay: 0

   ### Transformations :

   - HorrizontalFlip
   - ShiftScaleRotate
   - Pad
   - CourseDropout

   ### Results:

   Achieved 88.95% accuracy in 20th Epock




### <img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/08_Resnet/Images/Test_train_curves.png?raw=false" style="zoom: 100%;" />


Top-20 Misclassified Images:

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/08_Resnet/Images/20_misclassified_images.jpg?raw=false" style="zoom: 100%;" />



### Class Activation Map - Misclassified Images

Gradient-weighted Class Activation Mapping (GradCAM) uses the gradients of any target concept (say logits for 'dog' or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. 

We take the final convolutional feature map, and then we **weigh** every channel in that feature with the gradient of the class with respect to the channel. It tells us how intensely the input image activates different channels by how important each channel is with regard to the class. It does not require any re-training or change in the existing architecture. 

For this exercise, the GradCAM maps for Resnet18 is generated each layer (layer0, layer1, layer2, layer3, layer4) . The Gradcam is imported from https://github.com/jacobgil/pytorch-grad-cam

The referred code block for generating gradcam images is shown below. 

```
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
target_layer = model.layer4[-1]
input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.use_cuda)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam)
```



<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/08_Resnet/Images/cam-1.jpg?raw=false" style="zoom: 100%;" />



<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/08_Resnet/Images/cam-2.jpg?raw=false" style="zoom: 100%;" />



#### References

https://canvas.instructure.com/courses/2734471/assignments/22785148

https://arxiv.org/abs/1610.02391

