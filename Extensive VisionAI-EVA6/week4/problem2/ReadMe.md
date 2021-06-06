## Problem Statement

Train the MNIST dataset under the following restrictions

1. 99.4% validation accuracy
2. Less than 20k Parameters
3. You can use anything from above you want. 
4. Less than 20 Epochs
5. Have used BN, Dropout, a Fully connected layer, **have used GAP.** 

## Solution

A [model](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/Session_4_model.ipynb) was trained with **19,678** parameters and **99.56**% test accuracy.

#### Model parameters

Number of layers: 7 Convolution Layers, 2 Transition Layers

Batch Size: 64
Optimizer: SGD
Loss Function: NLL Loss

Learning rate for training:

<img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/images/chart.png?raw=true" style="zoom: 80%;" />


#### Model Architecture

12->24->36->MP->12->12->24->MP->12->12->24->GAP(7x7)->Dense-> 10 outputs
<img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/images/model_diagram.jpeg?raw=true" style="zoom: 80%;" />

###### Highlights

- No Batch Norm and Dropout used after the final Convolution layer.
- GAP (Global Average Pooling) layer on 7x7 kernel size
- Used Batch Normalization after ReLU.
- Dropout of value 0.05 is used after every convolution layer (except the last) 
- Batch size experiments showed that a smaller batch gave better results



###### Additional Models that satisfy the model requirements

|                            Model                             | No. Parameters | Accuracy |
| :----------------------------------------------------------: | :------------: | :------: |
| [model_1](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/Experiments/Session_4_model1.ipynb)<br />12->24->MP->12->12->24->MP-><br />12->12->24->GAP(7x7)->Dense |     11,650     |  99.41   |
| [model_2](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/Experiments/Session_4_model2.ipynb)<br />10->20->30>MP->10->10->20->MP-><br />10->10->20->GAP->Dense |     13,780     |  99.50   |
| [model_3](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/Experiments/Session_4_model2.ipynb)<br />2->24->24->MP->12->12->24->MP-><br />12->12->24->GAP(7x7)->Dense |     16,906     |  99.49   |
|                                                              |                |          |



###### Experiments and Learnings

The MNIST dataset is a set of small (28 by 28) images. If you zoom in, to see the pixelated image, the handwriting stride is within a 5x5 box. Hence, a 5x5 receptive field before a Max pooling layer should be enough to retain all the information related to edges/corners of the image. Keeping this in mind, the base network that we started experimenting with as below:

<img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/images/base_network.jpg?raw=true" alt="img" style="zoom:35%;" />



Listing some of the parameters that affect the accuracy of the model:

1. Batch Normalization and Dropout

   To start with BN and Dropout were applied after each Convolution layer following the standard practice. Since the last convolution goes into a GAP layer, the BN and Dropout was removed. As a result, it was observed that the training and accuracy curves came closer

   <img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/images/exp_1.jpg?raw=true" alt="img" style="zoom:35%;" />

2. The distance of MaxPooling from Prediction

   To play with this, the second Max pooling layer was removed and a deeper model was trained. This model had 5 convolution layers after the first Max pooling layer. This removed the gap between the training and the test curves even more.
   <img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/images/exp_2.jpg?raw=true" alt="img" style="zoom:50%;" />

3. Receptive Field of 7x7 in the First Block

   This gave advantage on accuracy number. For a particular model the accuracy after 10 epochs increased from 98.1% to 98.9%.

4. Batch Size

   Experimented with batch size of 32, 64 and 128. 32 and 64 were close enough, so the batch size of 64 was chosen

5. Dropout Values

   Different values of dropout were tested. We ran these tests for only 5 epochs to get quick results. Values tested: 0.05, 0.1, 0.2, 0.5. The best value in terms of the Accuracy values and the gap between training and test was **0.05**. As an experiment, Dropouts were used only before Max pooling. However, this did not give any advantage.

6. Learning Rate

   This seemed to have a major effect on how the model converged. Since the requirement was to have a model that trained within 20 epochs, it is necessary to start with a relatively higher learning rate and then reduce it at the right moment. Just the right combination helped increase the accuracy from 99.33% to 99.56%.

7. Parameters that did not help in improving the accuracy:

   - increasing the number of channels from 24 to 32 for the last Conv layer, just before GAP.
   - Adding another Dense layer to transform 24 inputs to 48 and then 48 to 10, brought the curves closer but it was an overkill in terms of number of parameters.





###### Further Experimentation

- Use Augmentation
- Test with live data OR actual handwritten numbers

