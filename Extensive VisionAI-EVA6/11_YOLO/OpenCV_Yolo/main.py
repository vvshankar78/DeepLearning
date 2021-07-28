import cv2
import numpy as np
import urllib.request
from PIL import Image
from google.colab.patches import cv2_imshow


class RunYolo:
    def __init__(self, url, path):
        self.url = url
        self.path = path
        self.download_weight()
        self.net = self.load_network()
        self.classes = self.load_classes()


    def download_weight(self):
        path = self.path +  "/yolov3.weights"
        print('begining download')
        urllib.request.urlretrieve(self.url, path)
        print('download completed')

    def load_network(self):
        print('loading yolo network')
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        return net

    def load_classes(self):
        classes = []
        print('loading classes')
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print(classes)
        return classes

    def show_object(self, im_path):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # Read image
        im = Image.open(im_path)
        img = cv2.imread(im_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        # cv2_imshow(img)
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non Max Supression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        class_labels = []
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[i]
                class_labels.append(label)
                # print(label)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        exif = im.getexif()
        creation_time = exif.get(36867)
        img_name = str.split(im_path,'/')[-1]
        obj_classes =  list(set(class_labels))
        # print('time of creation of image - ',creation_time)
        # print('Object classes in the image are - ',obj_classes)
        # print('image name - ',img_name)
        output_dict = {"Image":img,
                "Image_name": img_name,
                "object_classes": obj_classes,
                "Image_save_time" : creation_time
                }
        # plt.imshow(img);
        cv2_imshow(img)
        return output_dict
















