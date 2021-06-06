# How a Neural Network Works (Back Propagation)

*Neha, Vidhya Shankar, Bhaskar, Abhijit from Team 11*





[TOC]

## Neural Network 

This exercise is primarily focused on understanding the math and logic behind how a neural network works. Solved with a simple 2 layer Fully connected layer network. In this exercise, we will look at how forward propagation works, how loss is calculated for different problem objectives, how we perform back propagation to update the weights (learning process) and finally how learning rate plays a role in weights update. 

### Network Architecture

The architecture used for example problem solved is a regression problem with architecture consisting of 2 hidden layers and 2 neurons per hidden layer. 

![image-20210526205650395](C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210526205650395.png)



## Forward propagation



<img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210527074436721.png" alt="image-20210527074436721" style="zoom:50%;" />



### Steps for Forward propagation:

The following steps are also relevant if we had many more hidden layers (deep) as well as many more neurons in each of the hidden layer. This example shows only 2 hidden layer and 2 neurons per layer. 

**First hidden layer:**  the inputs i1, i2 are multiplied with weights (w1, w2, w3, s4) to form the first hidden layer (2 neurons h1 & h2) - *Refer to hidden-1 calculations below*

**Activation**: since h1, h2 are in linear form, activation function is applied to get outputs a_h1, a_h2. While the most common is relu, in this example sigmoid activation is applied.  ***(some of activation functions : relu, sigmoid, or tanh)***

**Second hidden layer**: the outputs of activations from first hidden layer (a_h1, a_h2) is then further multiplied (*sum product*) to formulate the hidden layer  -2. The outputs is represented as o1, o2. Similar to first layer, an activation is applied to o1 & o2 represented by a_o1, a_o2. *Refer to hidden-2 calculation below*

**Error**: The outputs of the neurons a_o1, a_o2 is compared with the actual inputs (i1, i2). The representation of this comparison is shown as errors (*E1, E2*) which shows how far off we are from the real data. There are multiple ways to represent the error, the the most common form are shown below and figure 

​			**Mean Square Error** for regression problem

​			**cross entropy loss**  for a classification problem

**Total Error (Total Loss)**: Now that we have errors from individual data,  they are summed up to arrive at the total loss. 



<img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210526201059293.png" alt="image-20210526201059293" style="zoom: 35%;" />







### Forward propagation Calculations: 

**Forward prop - Hidden Layer-1**
$$ {align*}
\\
\begin{align*}
h1 &= w1* i1 + w2 * i2					&- 	(1) \\
h2 &= w3 * i1 + w4 * i2					&- 	(2) \\
out_h1 &= σ (h1)						&- 	(3) \\
out_h2 &= σ (h2)						&- 	(4)\\
\end{align*}
$$
**Forward prop - Hidden Layer-2**
$$
\\
\begin{align*}
o1 &= w5 * out_h1 + w6 * out_h2					&- 	(5) \\
o1 &= w7 * out_h1 + w8 * out_h2					&- 	(6) \\
out_o1 &= σ (h1)						&- 	(7) \\
out_o2 &= σ (h2)						&- 	(8)
\end{align*}
$$
**Forward prop - Loss Calculation**
$$
\begin{align*}
E1 	&= ½ * (t1 - out_o1)2  = ½ * (t1 - σ(o1))2		&- 	(9)\\
E2 	&= ½ * (t2 - out_o2)2  = ½ * (t2 - σ(o2))2  	&	-	(10)\\
...\\


&Total Loss \\
E_Total &= E = E1 + E2
\end{align*}
$$

$$
Sigmoid function \\
σ (x) = 1/(1+e-x)
$$

### Solved Example: 

https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/Backpropagation%20sheet.xlsx

Based on the network shown above, a calculation is performed using excel tool to show , how neural networks work. 

The below figure shows, forward propagation is applied to a sample data of (input = i1, i2) with randomly initialized weights and the error is compared against the output (t1, t2). Further E_tot represents the total weights. 





![image-20210527081734097](C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210527081734097.png)





## Back Propagation:

### Intuition

Now that we have performed one step of forward propagation and calculated the Errors (loss) with respect to the outputs, the idea of training a network is to achieve a prediction that is close to my actual outputs or have the loss to be minimized. 

**What is it that I need to update:**  In order to minimize the loss, there has to be something in the neural network that needs to be tuned/updated. All we have is the data (x, y) and the weights. while the data cannot change, the only thing that can be updated is weights of the neural network. 



<img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210527084745924.png" alt="image-20210527084745924" style="zoom:50%;" />



**Ok!! Now that I have to tune my weight... How do I do that...** : While a naive way of updating the weights could random trial and error and pray to got for some magic to happen or have a direction of how to update the weights so that my loss is minimized. Thanks to **<u>Gradient descent,</u>** we now have a means or direction in which the weights can be updated so that I can keep minimizing my loss. Also note, that this update is done in small steps multiple times and not a single shot solution... 



<img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210527090758683.png" alt="image-20210527090758683" style="zoom:67%;" />

### Chain Rule :

Chain rule plays an important role in propagating backwards towards the weights with respect to the loss. The intuition of chain role is shown in the figure below. 

<img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210526211626411.png" alt="image-20210526211626411" style="zoom: 50%;" />



### Back Propagation Calculation : 

Given our network having 2 hidden layers, we would be calculating the derivative of the Error with respect to each of the weights (𝟃E/𝟃w). This provides the update each of our weights in the direction of minimizing the loss. 

The back propagation is calculated backwards, starting from loss , calculating the derivatives of outputs of last hidden layer and then calculating the weights of last hidden layer  (𝟃E/𝟃w5, 𝟃E/𝟃w6, 𝟃E/𝟃w7, 𝟃E/𝟃w8), and then further moving towards the first hidden layer, calculating the derivatives of the neurons and then weights of first hidden layer (𝟃E/𝟃w1, 𝟃E/𝟃w2, 𝟃E/𝟃w3, 𝟃E/𝟃w4). Having calculated the derivative of total error with respect to each weights, the weights are updated with the help of learning rate function.
$$
w = w - η*𝟃E/𝟃w \\
.\\
η - Learning.rate
$$
Having updated the weights, the cycle of forward propagation is performed to calculate the new errors and predictions. This cycle is repeated multiple times till the error is no longer reducing (global minima)

 

#### Derivatives

Before we could proceed with the back propagation, we would be formulating few of the partial derivatives values essential for our calculation.



***Derivates Sigmoid***
$$
\begin{align*}
σ(x) &= 1/( 1 + e^{-x})  \\
σ’(x) &= σ(x) ( 1- σ(x)) \ -(11)\\
\end{align*}
$$


***Derivates Hidden Layer-1***
$$
o1 = w5 * out_h1 + w6 * out_h2	\\ 
o2 = w7 * out_h1 + w8 * out_h2	\\ 
...\\
\begin{align*}
𝟃o1/𝟃w5 &= out_h1  \\
𝟃o1/𝟃w6 &= out_h2  &-(12) \\
...\\
𝟃o2/𝟃w7 &= out_h1  \\
𝟃o2/𝟃w8 &= out_h2  &-(13)
\end{align*}
$$


***Derivates Hidden Layer-2***
$$
Derivative-Linear\ Outputs \\

Hidden -Layer:1\\
...\\
h1 = w1 * i1 + w2 * i2 	\\ 
h2 = w3 * i1 + w4 * i2	\\ 
...\\
\begin{align*}
𝟃h1/𝟃w1 &= i1  \\
𝟃h1/𝟃w2 &= i2      \                &-(14) \\
...\\
𝟃h2/𝟃w3 &= i1  \\
𝟃h2/𝟃w4 &= i2  \ &-(15)
\end{align*}
$$


***Derivate of 𝟃E/𝟃w5***
$$
\begin{align*}
𝟃E/𝟃w5  &= 𝟃E1/𝟃w5 + 𝟃E2/𝟃w5\\
&= 𝟃E1/𝟃w5  + 0	\\
&= -1 * (t1 - out_o1) * 𝟃(σ(o1))/𝟃w5  \\
&= (out_o1 - t1) * [𝟃(σ(o1))/𝟃o1] * [𝟃o1/𝟃w5 ]   &- By \ Chain Rule \\
&= (out_o1 - t1) * [ σ(o1) * (1 - σ(o1)) ] * [out_h1]	&- ref. (11)\  and\  (13)\\
&= (out_o1 - t1) * [ out_o1 * (1 - out_o1) ] * [out_h1] &- subst\  Values\  from\  (7)
\end{align*}
$$

$$
Hence, \\
∂E/∂w5  =(out_o1 - t1) * [ out_o1 * (1 -out_o1) ] * [out_h1]
\\
$$
<img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210528055556049.png" alt="image-20210528055556049" style="zoom: 50%;" />



***Similarly, The derivatives for other weights (𝟃E/𝟃w6, 𝟃E/𝟃w7, 𝟃E/𝟃w8),in this layer can be derived -*** 
$$
∂E/∂w6  =(out_o1 - t1) * [ out_o1 * (1 -out_o1) ] * [out_h2] \\
∂E/∂w7  =(out_o2 - t2) * [ out_o2 * (1 -out_o2) ] * [out_h1] \\
∂E/∂w8  =(out_o2 - t2) * [ out_o2 * (1 -out_o2) ] * [out_h2]
$$


***Further, going backwards to calculate  (𝟃E/𝟃w1, 𝟃E/𝟃w2, 𝟃E/𝟃w3, 𝟃E/𝟃w4)***
$$
\begin{align*}
Derive \ &-\ 𝟃E/𝟃w1 \\
...\\
𝟃E/𝟃w1  &= 𝟃E1/𝟃w1 + 𝟃E2/𝟃w1 		& Substitution values from (9) and (10)\\
		&= {-1 * (t1 - out_o1) * 𝟃(out_o1)/𝟃w1} + { -1 * (t2 - out_o2) * 𝟃(out_o2)/𝟃w1 }  &- (16)
\end{align*}
$$


***Derivation of 𝟃(out_o1)/𝟃w1 and 𝟃(out_o2)/𝟃w1***


$$
\begin{align*}
𝟃(out_o1)/𝟃w1  &= 𝟃(σ(o1))/𝟃o1 * 𝟃o1/𝟃w1  				\\
 -Using (11), we\  get\\
		&= [ σ(o1) * (1 - σ(o1)) ] * 𝟃o1/𝟃w1 			&- (17)\\
 𝟃o1/𝟃w1	&= w5 * 𝟃out_h1/𝟃w1 + w6 * 𝟃out_h2/𝟃w1	\\
 
 - Using (1)\\
		&= [w5 * 𝟃[σ(h1)]/𝟃w1] + [w6 * 𝟃[σ(h2)]/𝟃w1]\\
		&= [w5 * 𝟃(σ(h1))/𝟃h1 * 𝟃h1/𝟃w1] + [w6 * 𝟃(σ(h2))/𝟃h2 * 𝟃h2/𝟃w1] - By chain rule\\
		&= [w5 * σ(h1) * (1 - σ(h1)) * i1] + [w6 * σ(h2) * (1 - σ(h2)) * 0]	\\
- Using (11)\  and\  (14)\\
		&= [w5 * σ(h1) * (1 - σ(h1)) * i1] + 0			&- (18)\\

\end{align*}
$$

$$
Similarly\\
\begin{align*}

𝟃(out_o2)/𝟃w1 &= 𝟃(σ(o2))/𝟃o2 * 𝟃o2/𝟃w1\\
		&= [ σ(o2) * (1 - σ(o2)) ] * 𝟃o2/𝟃w1 			&- (19)\\
𝟃(o2)/𝟃w1 	&= w7 * 𝟃out_h1/𝟃w1 + w8 * 𝟃out_h2/𝟃w1\\
		&=  [w7 * σ(h1) * (1 - σ(h1)) * i1]  + 0			&- (20)\\
\end{align*}
$$


***Finally here comes  𝟃E/𝟃w1, 𝟃E/𝟃w2, 𝟃E/𝟃w3, 𝟃E/𝟃w4***
$$
\begin{align*}
Substituting (17), (18), (19), (20) into (16), we \ get\\
𝟃E/𝟃w1 = \  &{ -1 * (t1 - out_o1) * [ σ(o1) * (1 - σ(o1)) ] * w5 * σ(h1) * (1 - σ(h1)) * i1}\ +\\
  	    &{ -1 * (t2 - out_o2) * [ σ(o2) * (1 - σ(o2)) ] * w7 * σ(h1) * (1 - σ(h1)) * i1 }\\

\end{align*}
$$


**Taking out the common part of substitution and applying similar calculations for other weights in this layer. **
$$
\begin{align*}
𝟃E/𝟃w1 = &out_h1 * (1 - out_h1) * i1  \ * \\
&		     { (out_o1 - t1) * out_o1* (1 - out_o1) * w5  + (out_o2 - t2) * out_o2 * (1 - out_o2) * w7 }\\
𝟃E/𝟃w2 = &out_h1 * (1 - out_h1) * i2  * \\
		  &   { (out_o1 - t1) * out_o1* (1 - out_o1) * w5  + (out_o2 - t2) * out_o2 * (1 - out_o2) * w7 }\\

𝟃E/𝟃w3 = &out_h2 * (1 - out_h2) * i1  * \\
		  &   { (out_o1 - t1) * out_o1* (1 - out_o1) * w6  + (out_o2 - t2) * out_o2 * (1 - out_o2) * w8 }\\

𝟃E/𝟃w4 = &out_h2 * (1 - out_h2) * i2  * \\
		 &    { (out_o1 - t1) * out_o1* (1 - out_o1) * w6  + (out_o2 - t2) * out_o2 * (1 - out_o2) * w8 }\\
\end{align*}
$$


### Weights Update and Learning rate: 

The individual weights are updated by subtracting the initial weights or weights from previous epoch, using the derivative of error w.r.t each weights. ***Important note:*** The weights are updated individually, based on the assumption that only one weight is updated while others are kept constant. 
$$
w = w - η*𝟃E/𝟃w \\
.\\
η - Learning.rate
$$
The learning rate is an important hyper parameter that plays a role of how fast or slow we would want the weight update step to be. In this solved example, we show the loss reduction based on different learning rates as well. 



## Results:

### Calculation (Tables)

The calculation table shows the back propagation as described above for the example problem.

 Each row (epoch) is a repeat cycle of forward propagation --> prediction -> loss calculation -> back propagation -> weights update and move to next row repeating the same. 

![image-20210528065725716](C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210528065725716.png)



### Working with Learning Rates

Having calculated for a simple neural network, we would like to play with various learning rates to see how it impacts the loss reduction . 

As seen in the below table, higher learning rate (2.0) helps to converge to low loss much faster, and beyond a certain point, there is no further improvement in loss. How ever, lower learning rates, the convergence is quite slow. 

the learning rate convergence is specific to this example, this does not imply that we have to go with higher learning rates. There are various techniques  adopted to arrive at a best learning rate depending on the complexity of the data and model that we are building. 

 <img src="C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210528070301014.png" alt="image-20210528070301014" style="zoom: 80%;" />





## Gradient Descent Algorithm

![image-20210528074303893](C:\Users\vvsha\AppData\Roaming\Typora\typora-user-images\image-20210528074303893.png)



## References:

http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf

https://www.youtube.com/watch?v=XPcmzEIdoZI&t=4129s

https://www.youtube.com/watch?v=bdBtmg5ZE94

https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/Backpropagation%20sheet.xlsx



