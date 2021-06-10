# Submission for Week 6 - Normalization and Regularization

## Team Members

Vidya Shankar, Bhaskar Gaur



### Objective:

---

1. You are making 3 versions of your 5th assignment's best model (or pick one from best assignments):
   1. Network with Group Normalization
   2. Network with Layer Normalization
   3. Network with L1 + BN
2. You MUST:
   1. Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include
   2. Write a single notebook file to run all the 3 models above for 20 epochs each
   3. Create these graphs:
      1. Graph 1: Test/Validation Loss for all 3 models together
      2. Graph 2: Test/Validation Accuracy for 3 models together
      3. *graphs must have proper annotation*
   4. Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 



### Approach for Normalization (Batch, Layer & Group)

---

Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization, and in some cases eliminates the need for Dropout (http://proceedings.mlr.press/v37/ioffe15.pdf)

The calculation of normalization parameters, mean and variance and how to make the shift and scale parameters learnable so that the network can understand the necessity/need for level of normalization is provided below. 

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/week6/images/Batch%20norm-equation.png?raw=false" style="zoom: 80%;" />

There are 3 types of normalization that are used. While Batch normalization is more commonly used, this exercise is to provide an intuition and understand of other normalization techniques like Layer Normalization and Group Normalization. 

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/week6/images/batch_norm_images.png?raw=false" style="zoom: 100%;" />

The intuition behind the calculation of layer norm can be a little tricky. The image shown below provides a understand of how layer normalization is calculated by with mean and variance of each image across all layers/channels.  

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/week6/images/Layer_Norm.png?raw=false" style="zoom: 80%;" />



#### Write a model that includes GN/LN/BN and takes an argument to decide which normalization to include

A function is below is used to pass on the required parameters and returns the kind of normalization that needs to be added to the layers in nn module of pytorch. 

```
    def get_bn(self, out_features, channel_size, norm_type, num_groups):
        if norm_type=="BN":
          m = nn.BatchNorm2d(out_features)
        elif norm_type == "LN":
          m = nn.LayerNorm((out_features, channel_size, channel_size))
        elif norm_type == "GN":
          if (out_features % num_groups) != 0:
            print('Error: the number of groups should be divisible by output features')
          else:
            m = nn.GroupNorm(num_groups, out_features)
        return m
```



### Regularization (L1 and L2)

---

Regularization techniques are often used to address the over fitting issues faced while we train our models. The intuition behind the regularization is to add penalizing factor to the loss function by taking the sum of mod of all coeffients(L1) or sum of square of coefficients (L2)

The code for L1 and L2 regularization goes as a part of train function is provided below



```
    if l1_decay > 0:
      l1_loss = 0
      for param in model.parameters():
        l1_loss += torch.norm(param,1)
      loss += l1_decay * l1_loss
    if l2_decay > 0:
      l2_loss = 0
      for param in model.parameters():
        l2_loss += torch.norm(param,2)
      loss += l2_decay * l2_loss
    train_loss_list.append(loss.item())
```



### Results:

---

1. ##### Network with Group Normalization

   ```
   # Input parameters for the model
   EPOCHS = 20
   l1_decay=0.0
   l2_decay=0.0
   norm_type = "GN" -----> LAYER NORMALIZATION
   num_groups=2
   input_img_size=(1, 28, 28)
   
   # Run model
   model = Net2(norm_type, input_img_size, num_groups).to(device) --> PARAMETERS FOR LAYER NORMALIZATION TO NETWORK
   optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
   scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
   
   for epoch in range(EPOCHS):
       print("EPOCH:", epoch+1)
       train_loss_list, train_accuracy_list = train(
       model, device, train_loader, optimizer, epoch,
       l1_decay, l2_decay, #------------------------------------------> l1 AND l2 REGULARIZATION PARAMETERS. 
       train_loss_list, train_accuracy_list)
       print("\nlearning rate", np.round(optimizer.param_groups[0]['lr'],4))
       scheduler.step()
       test_loss_list, test_accuracy_list, misclassified_imgs = test(model, device, test_loader, test_loss_list, test_accuracy_list, misclassified_imgs, epoch==EPOCHS-1)
   ```

   

2. ##### Network with Layer Normalization

   ```
   # Input parameters for the model
   EPOCHS = 20
   l1_decay=0.0
   l2_decay=0.0
   norm_type = "LN" -----> LAYER NORMALIZATION
   num_groups=2
   input_img_size=(1, 28, 28)
   ```

   <img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/week6/images/misclassified_imgs_LN.png?raw=false" style="zoom: 80%;" />

3. ##### Network with L1 + BN

```
# Input parameters for the model
EPOCHS = 20
l1_decay=0.0005
l2_decay=0.0
norm_type = "GN"
num_groups=2
input_img_size=(1, 28, 28)
```

<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/week6/images/misclassified_imgs_GN.png?raw=false" style="zoom: 80%;" />

### Overall Results:

---



<img src="https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/week6/images/WhatsApp%20Image%202021-06-10%20at%2010.29.03.jpeg?raw=false" style="zoom: 80%;" />



