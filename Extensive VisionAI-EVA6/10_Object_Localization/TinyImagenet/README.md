# TinyImageNet training using ResNet18

Vidya Shankar




### Objective:

---
The objective is to train ResNet18 on tinyimagenet dataset (with 70/30 split) for 50 Epochs and achieve 50%+ Validation Accuracy.



      
    ├── data
    |   ├── data_download.py 
    |   ├── data_engine.py 
    |   ├── data_transforms.py
    ├── models  
    |   ├── Resnet_Custom.py 
    |   ├── Resnet.py
    ├── gradcam  
    |   ├── __init__.py 
    |   ├── gradcam.py
    |   ├── visualize.py
    ├── utils.py
    ├── train.py
    ├── test1.py 
    ├── config.py
    ├── main.py     
    ├── README.md  



**Details of the Training **

1. Model - ResNet18 (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)

**Data Augmentation **

RandomCrop

HorizontalFlip

Rotate

RGBShift

**Parameters and Hyperparameters**

- Model : ResNet18
- Data: TinyImageNet
- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Scheduler: One cycle policy
- Batch Size: 256
- Learning Rate: lr=0.1 (max_lr for ocp)
- Epochs: 50
- Max at Epoch = 10
- Dropout: 0.
- L1 decay: 0
- L2 decay: 0

---

### The modular code  

**A modular training module is created created to train the model. The folder structure of this module is shown below **

[Click here for the link to modular code](https://github.com/vvshankar78/Pytorch_Wrapper)

```
├── data
|   ├── data_download.py 
|   ├── data_engine.py 
|   ├── data_transforms.py
├── models  
|   ├── Resnet_Custom.py 
|   ├── Resnet.py
├── gradcam  
|   ├── __init__.py 
|   ├── gradcam.py
|   ├── visualize.py
├── utils.py
├── train.py
├── test1.py 
├── config.py
├── main.py     
├── README.md  
```



### Learning rate finder and One Cycle Policy

The learning rate range test is a test that provides valuable information about the optimal learning rate. During a pre-training run, the learning rate is increased linearly or exponentially between two boundaries. The low initial learning rate allows the network to start converging and as the learning rate is increased it will eventually be too large and the network will diverge.

The code implementation of lr finder is shown below. The output of lr finder provides the optimal learning rate. 

```
pip install torch-lr-finder

def get_lr_finder(model, train_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=1e-2)
  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
  lr_finder.plot()
  return
  
get_lr_finder(model, train_loader)
```

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/10_Object_Localization/TinyImagenet/outputs/LR_finder.png?raw=false" style="zoom: 105%;" />



**One Cycle policy**

The 1cycle learning rate policy changes the learning rate after every batch. step should be called after a batch has been used for training.

In our case, we have 50 epochs and the need to peak at 10th epoch. 



```
def get_ocp_plot(train_loader, model, max_lr=0.1):   
  EPOCHS = args.epochs
  peak = args.peak
  peak_pct = peak/EPOCHS
  optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=1e-2)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 	 steps_per_epoch=len(train_loader), epochs=EPOCHS,pct_start=peak_pct, anneal_strategy='linear')
  lrs = []

  for i in range(EPOCHS*len(train_loader)):
      optimizer.step()
      lrs.append(optimizer.param_groups[0]["lr"])
      scheduler.step()

  plt.plot(lrs)
  return
```

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/10_Object_Localization/TinyImagenet/outputs/OCP.png?raw=false" style="zoom: 100%;" />



- 



### Results:






### <img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/Images/train-test-curves.png?raw=false" style="zoom: 100%;" />





#### References

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

https://github.com/davidtvs/pytorch-lr-finder

https://github.com/lokeshpara/Freecodecamp/blob/master/course_project/Assignment5___course_project.ipynb

https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py





