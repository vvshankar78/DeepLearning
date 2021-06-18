# Submission for Week 7 - Advanced Concepts

## Team Members

Bhaskar Gaur

Vidya Shankar had emergency surgery this week. He gave me guidence on the phone.


### Objective:

---
#### Met
change the code such that it uses GPU.

change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead)

total RF must be more than 44

one of the layers must use Depthwise Separable Convolution

one of the layers must use Dilated Convolution

use GAP (compulsory):- We removed the FC after GAP (optional)

parameter count of 85k

Used Transforms:
    horizontal flip (50% prob)
    scale (0.9 to 1.1)
    rotation (-5 to 5)
    translate/shift (0.1 for x/y)

#### Not Met
use albumentation library and apply:

    horizontal flip
    shiftScaleRotate
    coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k. 

upload to Github

Attempt S7-Assignment Solution. Questions in the Assignment QnA are:

    Which assignment are you submitting? (early/late)

    Please mention the name of your partners who are submitting EXACTLY the same assignment. Please note if the assignments are different, then all the names mentioned here will get the lowest score. So please check with your team if they are submitting even a slightly different assignment. 

    copy paste your model code from your model.py file (full code) [125]

    copy paste output of torchsummary [125]

    copy-paste the code where you implemented albumentation transformation for all three transformations [125]

    copy paste your training log (you must be running validation/text after each Epoch [125]

    Share the link for your README.md file. [200]


### Tasks done:
1. Switched the notebook to use GPU.
2. Added normalization parameter calculation using both test and train data, used them in the transformation to get bump from 54% to 57% for 1000 images test accuracy.
