import cv2
import numpy as np
import urllib.request



class RunYolo:
    def __init__(self, url, path):
        self.url = url
        self.path = path

    def download_weight(self):
        path = self.path +  "/yolov3.weights"
        urllib.request.urlretrieve(self.url, path)
