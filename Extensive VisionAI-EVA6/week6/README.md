# Submission for Week 6 - Normalization and Regularization
[TOC]

## Objective:

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



## Approach for Normalization (Batch, Layer & Group)





[CODE](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week5/Session_5_Bhaskar_exp1.ipynb)

1. Make a light model with less than 10k parameters to train MNIST

2. Results:

   Trained three different models architectures here, for the final model

   - Parameters: **6,866**
   - Best Train Accuracy: **97.97**%
   - Best Test Accuracy: **97.59**%

3. Analysis: 

   ```
   First two layer for receptive field of 5 were reduced from 16->32 to 8->16, to reduce parameter count.
   The channel count of 32 is reduced to 20 in order to reduce parameter count below 8K. We are aiming for bonus and we dont see any major dip in the model characteristics.
   
   32 was felt to be an overkill for MNIST, especially when I had seen models with max 20 filters at any layer achieving target previously.
   Only quirk is the absence of max pool before second 1x1. Did this to gain depth in the network while keeping it simple. This also results in a larger GAP of 6x6, which doesnt seem to negatively impact the model.
   
   However the accuracy doesn't seem to be improving much.
   ```



## Code 2 - Regularization

[CODE](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week5/Session_5_Bhaskar_exp2.ipynb)

1. Targets: Allow the network to train more in order to improve accuracies. 
2. Results:
   - Parameters: **7002**
   - Best Train Accuracy: **99.39**%
   - Best Test Accuracy: **99.43**%
3. Analysis: 
   - The training and test accuracies continue to improve as we train the model. 
   - We hit the accuracy of 99.43% in 15th epoch, however the model was not able to maintain the same. Both training and test accuracies dropped after that.
   - We should be able to push the network more.



## Code 3 - Augmentation

[CODE](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week5/Session_5_Bhaskar_exp3.ipynb)

1. Targets: Make training more challenging
2. Results:
   - Parameters: 7,002
   - Best Train Accuracy: **99.19**%
   - Best Test Accuracy: **99.38**%
3. Analysis: 
   - By adding augmentation in terms of rotation and sheer, the training accuracy dropped compared to the previous experiment. However, since it is ever increasing, we see that there is a gap created that can be filled to improve the model further.
   - The accuracies increase constantly in this experiment, hence the promise to maintain the good number.
   - There accuracies have been achieved in 20 epochs. Need to work further to achieve 99.4% within 15 epochs



## Code 4 - Final Model

[CODE](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week5/Session_5_Bhaskar_exp4.ipynb)

1. Targets: Achieve the final accuracy within 15 epochs
2. Results:
   - Parameters: 7,002
   - Best Train Accuracy: **99.37**%
   - Best Test Accuracy: **99.51**% (14th epoch) , 
3. Analysis: 
   - Played with the learning rate scheduler to achieve the high accuracy faster.
   - However, we trained it further to see that accuracy > 99.4 was maintained. 

<img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week5/images/CAPACITY.jpg?raw=false" style="zoom: 60%;" />

## Team Members

Neha Mittal, Vidya Shankar, Bhaskar Gaur, Abhijit Das
