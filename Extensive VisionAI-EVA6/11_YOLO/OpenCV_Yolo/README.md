# YOLO object detection using Opencv with Python

Vidya Shankar, Mahesh, Pratima, Praveen - Group 10

---




### Objective:

---
- Run this above code provided in this this [link](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
- Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
- Run this image through the code above. 



Link for the notebook in Google Colabs. 



---

### The modular code - [Object Detection Toolkit](https://github.com/vvshankar78/Object_detection_toolkit/tree/master/OpenCV_Yolo)

**A modular module is created in the Object_detection Toolkit repository. The current code is folder OpenCV_Yolo **

[Click here for the link to modular code](https://github.com/vvshankar78/Object_detection_toolkit/tree/master/OpenCV_Yolo)

```
Object_detection_toolkit
├── OpenCV_Yolo
|   ├── main.py
|   ├── coco.names
|   ├── yolov3.cfg
```



main.py is the consolidation of following -

1. download weights from the following [link](https://pjreddie.com/media/files/yolov3.weights)
2. load the network
3. load the classes
4. show objects - read the input image and generate object detection. 



### How to run the object detection

1. Clone object detection tool kit and import main file.  

```
!git clone "https://github.com/vvshankar78/Object_detection_toolkit.git"
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
```

2. Change the current working directory, if you are running on colabs, then use the following folder structure. 

```
my_path = '/content/Object_detection_toolkit/OpenCV_Yolo'
os.chdir(my_path)
```

3. Run the main file. Downloads the weights file, initializes yolo network and loads classes. 

```
yolo = main.RunYolo(weights_url, my_path)
```

4. Now you are ready to do the inferences for your image. Load any images into inputs folder

```
im_path  = 'input/3.jpg'
output_dict = yolo.show_object(im_path)
```

The output is a dictionary that provides image file along with image name, object classes found in the image. The output is shown below. 



#### Output

Image Name :  3.jpg 

object classes in the image ['cow', 'person']

![](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/11_YOLO/OpenCV_Yolo/output/Screenshot_1.jpg?raw=False)









#### References

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

https://github.com/davidtvs/pytorch-lr-finder

https://github.com/lokeshpara/Freecodecamp/blob/master/course_project/Assignment5___course_project.ipynb

https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py





