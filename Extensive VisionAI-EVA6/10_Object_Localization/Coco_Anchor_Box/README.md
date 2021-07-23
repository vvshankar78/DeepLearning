# COCO Dataset - EDA and Anchor box using K-Means

Vidya Shankar



### Objective:

---
1.  Learn how COCO object detection dataset's schema is.
2. EDA on COCO data - data for class distribution (along with the class names) along with a graph 
3. Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.



### About [COCO](https://cocodataset.org/) :

COCO (Common Objects in Context) is large scale object detection, segmentation and captioning dataset.  COCO has several features:

- Object segmentation
- Recognition in context
- Superpixel stuff segmentation
- 330K images (>200K labeled)
- 1.5 million object instances
- 80 object categories
- 91 stuff categories
- 5 captions per image
- 250,000 people with keypoints



### Coco Dataset Schema



**Major keys of the COCO data sets consists of **

```
dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
```

'info' -  Annotater tool metadata

 'licenses' - unique internal identifier for the license

 'images' - image names, url, dimensions, date captured, source url, image id. 

'annotations' - annodation details for image id - segmentation, area, iscrowd, bounding box, category id

 'categories' - maps the category id to names 



**Annotations:**

Annotations represents the details of annotations for an image. The descriptions of the annotations with Sample data is shown below

```
  **Annotations Description**
  
  "annotations": [
    {
      "id": unique internal identifier for the annotation,
      "image_id": identifier mapping to the image through images.id,
      "category_id": identifier mapping to the category through categories.id,
      "segmentation": [
        4 pairs of (x,y) coordinates of the bbox starting from the top left corner,
        in the clockwise direction, assuming the origin is at the top left corner
        of the image
      ],
      "area": pixel area of the bbox,
      "bbox": [
        assuming the origin is at the top left corner of the image
        x: x co-ordinate of top left corner of bbox,
        y: y co-ordinate of top left corner of bbox ,
        w: width of the bbox,
        h: height of the bbox,
      ],
      "iscrowd": denotes if there is are multiple objects or single object
    },
  ],

**Sample Data**

{'area': 2765.1486500000005,
 'bbox': [199.84, 200.46, 77.71, 70.88],
 'category_id': 58,
 'id': 156,
 'image_id': 558840,
 'iscrowd': 0,
 'segmentation': [[239.97,
   260.24,
   222.04,
   270.49,
   228.87,
   271.34]]}
```



**Images:**

```
 **Description**
 "images": [
    {
      "id": unique internal identifier for the image,
      "width": width of the image,
      "height": height of the image,
      "file_name": image file name,
      "license": identifier for the correspnding license,
      "date_captured": date when captured
    },
  ],

**Sample Data**
{'coco_url': 'http://images.cocodataset.org/train2017/000000391895.jpg',
 'date_captured': '2013-11-14 11:18:45',
 'file_name': '000000391895.jpg',
 'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
 'height': 360,
 'id': 391895,
 'license': 3,
 'width': 640}
```



**Categories**

```
**Description**
  "categories": [
    {
      "id": unique internal identifier for the class,
      "name": class name,
      "supercategory": "class"
    },
  ]
}

**Sample Data**
{'id': 3, 'name': 'car', 'supercategory': 'vehicle'}
```



**Info and Licenses**

```
  "info": {
    Annotater tool metadata
  },

  "licenses": [
    {
      "id": unique internal identifier for the license,
      "name": license name,
      "url": license url
    }
  ],


**Sample**
{'id': 1,
 'name': 'Attribution-NonCommercial-ShareAlike License',
 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'}

```



### **EDA**

Annotations Data Size -  860001 

Images Data Size  - 118287

number of classes 80



**Count of bounding boxes by Categories:**



<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/10_Object_Localization/Coco_Anchor_Box/outputs/bounding%20box%20by%20categories.png?raw=false" style="zoom: 100%;" />



### **Anchor Box**

One Size does not fit all, but templates provides a better way to represent them. What am I saying!!!!. 

Each of the annotated bounding box comes with its own size and there is no way to to categorize a bounding box for object category (Baba Ramdev). So the yolo team came up with the idea of template anchor boxes that closely matches with the bounding box. The template anchor boxes are then adjusted by ratio metric to map to the actual bounding box in the yolo algorithm. 

##### HOW TO CALCULATE ANCHOR BOXES 

The optimized number of anchor boxes is figured out based on the width and height of each bounding box normalized by image dimensions. Plot (scatter plot) between the normalized height and width. 

Use K-means clustering to compute the cluster centers. 

Compute the different numbers of clusters and compute the mean of maximum IOU between bounding boxes and individual anchors. 

pick the number of centroids which givens mean IOU > 65% (yolo v2)

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/10_Object_Localization/Coco_Anchor_Box/outputs/elbow%20curve.png?raw=false" style="zoom: 150%;" />



<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/10_Object_Localization/Coco_Anchor_Box/outputs/centroid.png?raw=false" style="zoom: 150%;" />



We graph the relationship between the number of clusters and Within Cluster Sum of Squares (WCSS) then we select the number of clusters (about 4 in our case) where the change in WCSS begins to level off (elbow method).









#### References

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

https://github.com/davidtvs/pytorch-lr-finder

https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

 (https://www.youtube.com/watch?v=4b5d3muPQmA)

