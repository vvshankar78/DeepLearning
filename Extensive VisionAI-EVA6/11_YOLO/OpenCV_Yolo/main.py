import cv2
import numpy as np
import urllib.request


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
