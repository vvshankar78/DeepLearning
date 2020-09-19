# EVA4-CIFAR-pytorch

This repository shows the following things. 
1. Modular approach to building a pytorch model for CIFAR10. The induvidual model consist of 

    a. Model Architecture  - vs_net.py
  
    b. Model Training  - vs_train.py
  
    c. Model Accuracy(test) - vs_accuracy.py
  
    d. Class Accuracy - vs_class_acc.py
  
  
 2. The architecture is built based on following consideration : 
 
    a. Deeper architecture (padding=1) 
  
    b. 1 layer having dilated convolution 
  
    c. 1 layer having depthwise convolution. 
  
