# ResNets, LR Schedulers and Higher Receptive Fields

## Team Members

Vidya Shankar

Regular Submission notebook:
https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/CIFAR10_Custom_RESNET.ipynb


### Objective:

---
**Write a custom Resnet like architecture as described below**

1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
2. Layer1 -
   1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
   2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
   3. Add(X, R1)
3. Layer 2 -
   1. Conv 3x3 [256k]
   2. MaxPooling2D
   3. BN
   4. ReLU
4. Layer 3 -
   1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
   2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
   3. Add(X, R2)
5. MaxPooling with Kernel Size 4
6. FC Layer 
7. SoftMax

**Uses One Cycle Policy such that:**

1. Total Epochs = 24
2. Max at Epoch = 5
3. LRMIN = FIND
4. LRMAX = FIND
5. NO Annihilation

**Transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)**

**Batch size = 512**



---

### The modular code  

***main.py -*** all the functions needed to run the whole pipeline. 

***CIFAR10_Custom_RESNET.ipynb -*** calls individual model from main file 

***model folder*** - custom resnet model 

***Experiments folder*** - notebooks with various experimentation done. 

***support python files*** - train.py, test.py, test-train plots.py,  plot misclassified images.py



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

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/Images/lr_finder.jpg?raw=false" style="zoom: 75%;" />



**One Cycle policy**

The 1cycle learning rate policy changes the learning rate after every batch. step should be called after a batch has been used for training.

In our case, we have 24 epochs and the need to peak at 5th epoch. So to peak at 5th epoch, we use the parameter pct_peak which is calculated by  - 

pct_peak = Epoch / peak epoch = 24/5 

steps_per_epoch = len(train_loader) = 98 batches per epoch. 

max_lr = 10  X suggested_lr (from lr_finder)

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

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/Images/OCP.jpg?raw=false" style="zoom: 100%;" />



### Parameters and Hyperparameters

- Model : Custom_Resnet
- Data: CIFAR10
- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Scheduler: One cycle policy
- Batch Size: 512
- Learning Rate: lr=2.3e-2 (max_lr for ocp)
- Epochs: 24
- Dropout: 0.
- L1 decay: 0
- L2 decay: 0



### Results:

| Trial                              | Train Accuracy | Test Accuracy | Notebook                                                     |
| ---------------------------------- | -------------- | ------------- | ------------------------------------------------------------ |
| Baseline - cutout 8x8              | 99.9%          | 88.9%         | https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/experiments/CIFAR10_Custom_RESNET_cutout_8x8.ipynb |
| Cutout 16x16                       | 98.88%         | 88.94%        | https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/experiments/CIFAR10_Custom_RESNET_16_cutout.ipynb |
| change in lr peak to max_lr=3.5e-2 | 99.86%         | 89.17%        | https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/experiments/CIFAR10_Custom_RESNET_reduced_lr.ipynb |

The models seem to be over fitting. The additional set of experimentation is to look at regularization like l1, l2, drop outs etc., 




### <img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/Images/train-test-curves.png?raw=false" style="zoom: 100%;" />

**Top-20 Misclassified Images:**

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/09_Custom_Resnet/Images/misclassified_images.png?raw=false" style="zoom: 100%;" />





#### References

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

https://github.com/davidtvs/pytorch-lr-finder



