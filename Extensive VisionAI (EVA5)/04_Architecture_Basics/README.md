<h1 align="center">Extensive Vision AI (EVA5)</h1>

<h2 align="center">Assignment- Architectural Basics</h2>

<h3 align="center"> Team Members: Prasad, Dheeraj, Rajesh, Vidhya Shankar </h3>

---
**Convolutions Architecture that achieves 99.4% accuracy for Mnist dataset**
---



**APPROACH AND SOLUTION**
----
The solution to identifying the right architecture that achieves the accuracy of 99.4% with less than 20K parameters and in 20 epochs is performed iteratively. A total of 6 different architectures were built iteratively and the summary of the trails are listed below. Refer to the link for the code.

**Summary of the trials**

1. Name: Net

   Description: Base architecture as provided by EVA, removed relu at conv7

   Parameters: 6.37 million

   number of Epochs - 2

   Accuracy - 98.7%

------------

2. Name: Net2

  Description: Reduced the number of parameters by reducing channel size

  Parameters: 37k

  number of Epochs - 10

  Accuracy - 99%
  
------------

3. Name: Net 3

  Description : Add GAP to 5x5

  parameters = 19.25K

  number of epochs -20

  accuracy - 98.98 % -

--------------

4. Name: Net 4

  Description : GAP + Batch Normalization

  parameters = 19.45K

  number of epochs -20

  accuracy - 99.16 % 
  
---------------------

5. Name: Net 5

  Description : Remove Padding. Go Deeper.

  parameters = 8.5K

  number of epochs -20

  accuracy - 99.32 % -

Go Deeper - Remove Padding and increase parameters (channels)

--------------------------

6. Name: Net 6

  Description : Go deeper, Remove Padding, increase parameters

  parameters = 19.3k

  number of epochs -20

  accuracy - 99.47 % -

----------------------------------------
**Code Link**

colab - https://drive.google.com/file/d/1dsEGp_ku6r4K40RYcnT71Qm5lRmtuHjH/view?usp=sharing

github - https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/04_Architecture_Basics/Architectural_Basics.ipynb

-------------------

**ASK OF THE ASSIGNMENT**


1.  We have considered many many points in our last 4 lectures. Some of these we have covered directly and some indirectly. They are:
    1.  How many layers,
    2.  MaxPooling,
    3.  1x1 Convolutions,
    4.  3x3 Convolutions,
    5.  Receptive Field,
    6.  SoftMax,
    7.  Learning Rate,
    8.  Kernels and how do we decide the number of kernels?
    9.  Batch Normalization,
    10.  Image Normalization,
    11.  Position of MaxPooling,
    12.  Concept of Transition Layers,
    13.  Position of Transition Layer,
    14.  DropOut
    15.  When do we introduce DropOut, or when do we know we have some overfitting
    16.  The distance of MaxPooling from Prediction,
    17.  The distance of Batch Normalization from Prediction,
    18.  When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
    19.  How do we know our network is not going well, comparatively, very early
    20.  Batch Size, and effects of batch size
    21.  etc (you can add more if we missed it here)
2.  Refer to this code:  [COLABLINK (links to an external site)](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx)
    -  **WRITE IT AGAIN SUCH THAT IT ACHIEVES**  
        1.  99.4% validation accuracy
        2.  Less than 20k Parameters
        3.  You can use anything from above you want.
        4.  Less than 20 Epochs
        5.  No fully connected layer
