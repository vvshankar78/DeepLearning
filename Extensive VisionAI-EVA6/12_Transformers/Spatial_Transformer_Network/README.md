# Spatial Transformers

### Team Members

Pratima, Vidya Shankar, Mahesh, Praveen 

---

### Colab File Reference:

github: https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/spatial_transformer_tutorial.ipynb

colab: https://colab.research.google.com/github/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/spatial_transformer_tutorial.ipynb

### CNN's are not invariant -

CNN (Convolution neural networks) are typically lace the invariance property to the input image. This means, they suffer from 

- *Scale / size variation in the input data*.
- *Rotation variation in the input data.*
- *Clutter in the input data.*

CNN performance are not good when there are variation though Max Pooling does help solve this issue for small variances, how ever they do not  help in making CNN's invariant when there is large variation in the data. To address this, we use Spatial Transformer Network (STN) which applies transformation to properly scale, resize and crop the image. 

### What are Spatial Transformer Networks (STNs)

STN's consist of Spatial transformer modules are neural network  (CNN and MLP) where in we apply transformations to properly scale, resize, crop an image. Since the transformation parameters come from a neural network module, they are learnable. STN's can be applied to both input images and feature maps. They can be inserted into any part of the CNN Architecture. 

**STN's acts as an attention mechanism and knows where to focus on the input image.**



![](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/images/fig-1.jpg?raw=false)

In the above figure, Column (a) shows the input image to the Spatial Transformer Network. We can see that some images are deformed and some contain clutter as well. Column (b) shows where the localization network part of the STN focuses on applying the transformations. In column (c) we can see the output after the transformations



### STN Architecture:

The STN has 3 parts  - 

- *The localization network.*
- *The parameterized sampling grid.*
- *And differentiable image sampling.*



![](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/images/fig-2-architecture.jpg?raw=false)



#### Localization Network

The localization network takes the input feature map (input image in our case) and outputs the parameters of the spatial transformations that should be applied to the feature map. The localization network is a very simple stacking of convolutional layers. The output of the localization network is parameters θ which is a of dimensions (3,2) matrix. 

```
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),  # 32x32x3 |  24x24x16
            nn.MaxPool2d(2, stride=2),       # 24x24x16 | 12x12x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5), #12x12x16 | 8x8x32
            nn.MaxPool2d(2, stride=2), # 4x4x32
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*4*4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
```



#### Parametrized Sampling grid

To get the desired output, the input feature map are parameterized sampling grid. The grid generator outputs the parameterized sampling grid.

Let G be the sampling grid. we need to transform the input feature map to get the desirable results? we apply the transformation Tθ to the grid G. That is, Tθ(G).

![](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/images/fig-3.jpg?raw=false)

The above figure shows the result of warping the regular grid with the affine transformation Tθ(G).After the sampling grid operation, we have the Differentiable Image Sampling.



```
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 128*4*4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
```



#### Differentiable Image Sampling

We have the input feature map and also the parameterized sampling grid. To perform the sampling, we give the feature map U and sampling grid Tθ(G) as input to the sampler (see **figure 2**). The sampling kernel is applied to the source coordinates using the parameters θ and we get the output V.



#### Benefits of Spatial Transformer Networks

There are mainly three benefits of Spatial Transformer Networks which makes them easy to use.

- Spatial transformer module can be inserted anywhere in an existing CNN model architecture.
- Spatial Transformer Networks are dynamic and flexible. We can easily train STNs with backpropagation algorithm
- STN works on both input image data or feature maps output from convolutions. 



### STN using Pytorch:

 Pytorch implementation of the STN is run for CIFAR10 data set and the output results are shown below. 

### 

#### Model Architecture

```
Net(
  (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=1600, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
  (localization): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=2048, out_features=32, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=32, out_features=6, bias=True)
  )
)
```

#### Training Logs

```

Test set: Average loss: 1.8181, Accuracy: 3642/10000 (36%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 2.002690
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.772292

Test set: Average loss: 1.6084, Accuracy: 4275/10000 (43%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.637450
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.698842

Test set: Average loss: 1.5512, Accuracy: 4460/10000 (45%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.606140
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.792879

Test set: Average loss: 1.4508, Accuracy: 4786/10000 (48%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 1.506480
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.667079

Test set: Average loss: 1.3979, Accuracy: 4966/10000 (50%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.555439
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.675636

Test set: Average loss: 1.3419, Accuracy: 5174/10000 (52%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.255403
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.374131

Test set: Average loss: 1.2899, Accuracy: 5408/10000 (54%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.543791
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.386232

Test set: Average loss: 1.4615, Accuracy: 4840/10000 (48%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.609339
Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.397793

Test set: Average loss: 1.2788, Accuracy: 5467/10000 (55%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 1.278730
Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.388397

Test set: Average loss: 1.2004, Accuracy: 5745/10000 (57%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 1.499066
Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.335857

Test set: Average loss: 1.1819, Accuracy: 5803/10000 (58%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 1.290704
Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.444100

Test set: Average loss: 1.2476, Accuracy: 5653/10000 (57%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 1.500814
Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.380567

Test set: Average loss: 1.1345, Accuracy: 5964/10000 (60%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 1.182164
Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.524851

Test set: Average loss: 1.1230, Accuracy: 6016/10000 (60%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 1.210536
Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.246433

Test set: Average loss: 1.1152, Accuracy: 6154/10000 (62%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 1.476574
Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.403886

Test set: Average loss: 1.1105, Accuracy: 6023/10000 (60%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 1.114907
Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.159935

Test set: Average loss: 1.1193, Accuracy: 6077/10000 (61%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 1.370826
Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.439328

Test set: Average loss: 1.0803, Accuracy: 6216/10000 (62%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 0.975361
Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.165208

Test set: Average loss: 1.0593, Accuracy: 6313/10000 (63%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 1.414292
Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.205971

Test set: Average loss: 1.0476, Accuracy: 6308/10000 (63%)
```



#### STN Results Visualization:



![](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI-EVA6/12_Transformers/Spatial_Transformer_Network/images/fig-4-results.jpg?raw=false)





## Reference

1. https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html

2. https://debuggercafe.com/spatial-transformer-network-using-pytorch/

3. https://arxiv.org/pdf/1506.02025v3.pdf

4. https://www.youtube.com/watch?v=25dO4fLhEMY

   



