<h1 align="center">Extensive Vision AI (EVA5)</h1>

<h2 align="center">Assignment- Coding Drill Down</h2>

<h3 align="center"> Team Members: Prasad, Dheeraj, Rajesh, Vidhya Shankar </h3>

---
**Achieves 99.4% accuracy for MNIST dataset**
---



**APPROACH AND SOLUTION**
----
The solution to identifying the right architecture and combination of convolution architectures like batch normalization, dropouts, GAP, MaxPooling, GAP, regularization and lr scheduler etc., that achieves the accuracy of 99.4% with less than 10K parameters and in 15 epochs is performed iteratively. A total of 4 different architectures were built iteratively and the summary of the trails are listed below. Refer to the link for the code.

**Summary of the trials**

 <h3> 01 Baseline architecture provided by EVA <h3>

  <h4>
     Results:

     Parameters: 194k
     Best Train Accuracy: 99.4
     Best Test Accuracy: 98.98

     Analysis:
     The model is still large and have some over fitting.
  <h4>

------------


 <h3> 02 Lighter Model - Reduce parameters + Batch Norm
 <h3>

  <h4> Results:

    Parameters: 10.79k
    Best Train Accuracy: 99.74
    Best Test Accuracy: 99.22

    Analysis: Over fitting and results are not so consistant.

  <h4>


  ------------




 <h3> 03 Lighter Model - GAP + Add Capacity + regularization(dropouts every layer
 <h3>

  <h4> Results:

    Parameters: 9.886k
    Best Train Accuracy: 99.16
    Best Test Accuracy: 99.46

    Analysis: Good Results, achieve consistant accuracy of > 99.4%



  ------------

 <h3> 04 Lighter Model - LR Scheduler + 20 epochs
 <h3>

  <h4>
    Results:

    Parameters: 9.886k
    Best Train Accuracy: 99.21
    Best Test Accuracy: 99.47


    Analysis: Over fitting and results are not so consistant.

  <h4>

----------------------------------------
**Code Link**

colab - https://drive.google.com/file/d/18VxQg-pP5CaLEpGe4sOWdz8octcr5nxw/view?usp=sharing

github - https://github.com/vvshankar78/DeepLearning/tree/master/Extensive%20VisionAI%20(EVA5)/05_Coding%20DrillDown

-------------------
