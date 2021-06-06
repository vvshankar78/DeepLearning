# Extensive Vision AI (EVA6)
## Group Members: Neha Mittal, Vidya Shankar, Bhaskar Gaur, Abhijit Das

**What are Channels and Kernels (according to EVA)?**

**Kernels**: 
In Convolutional neural network, the kernel is  a filter that  extract the features from the images.kernal is a computer , it is going to compute or extract something and result of then extraction  is called neuron, The kernel is a matrix that moves over the input data,convolves with the input data and gets the output as the matrix of dot products. Kernel moves on the input data by the stride value. If the stride value is 2, then kernel moves by 2 columns of pixels in the input matrix. In short, the kernel is used to extract high-level features like edges from the image.

**Channels**:
Now after the convolution operation the output we get is called the neuron , such collection of all neurons that contains information about a specific feature is known as Channel. It is the collection of same features which are extracted through kernal.

 
 **Why should we (nearly) always use 3x3 kernels?**
1. Symmetry: A kernel with odd size (3x3, 5x5, 7x7) is always preferred as it offers an axis of symmetry at the center of the kernel matrix. This helps to capture the symmetric information.
2. Resource optimization: Instead of using a larger kernel like 5x5, which uses 25 parameters/operations, we can use 3x3 twice. It will lead to 18 param/ops. This helps in reduction of number of computations. The growth of operation becomes linear rather than exponential.
3. GPU Optimized: Today’s GPU’s from nvidia are highly optimized to perform 3x3 convolutions.
4. Square Shaped: The pattern can occur both horizontally or vertically and by using square-shaped filters both patterns can be extracted without prior knowledge of pattern.

**How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)**
Number of layers is 99.
> 199x199-> 197x197-> 195x195-> 193x193-> 191x191-> 189x189-> 187x187-> 185x185-> 183x183-> 181x181-> 179x179-> 177x177-> 175x175-> 173x173-> 171x171-> 169x169-> 167x167-> 165x165-> 163x163-> 161x161-> 159x159-> 157x157-> 155x155-> 153x153-> 151x151-> 149x149-> 147x147-> 145x145-> 143x143-> 141x141-> 139x139-> 137x137-> 135x135-> 133x133-> 131x131-> 129x129-> 127x127-> 125x125-> 123x123-> 121x121-> 119x119-> 117x117-> 115x115-> 113x113-> 111x111-> 109x109-> 107x107-> 105x105-> 103x103-> 101x101-> 99x99-> 97x97-> 95x95-> 93x93-> 91x91-> 89x89-> 87x87-> 85x85-> 83x83-> 81x81-> 79x79-> 77x77-> 75x75-> 73x73-> 71x71-> 69x69-> 67x67-> 65x65-> 63x63-> 61x61-> 59x59-> 57x57-> 55x55-> 53x53-> 51x51-> 49x49-> 47x47-> 45x45-> 43x43-> 41x41-> 39x39-> 37x37-> 35x35-> 33x33-> 31x31-> 29x29-> 27x27-> 25x25-> 23x23-> 21x21-> 19x19-> 17x17-> 15x15-> 13x13-> 11x11-> 9x9-> 7x7-> 5x5-> 3x3-> 1x1

**How are kernels initialized?**
During the training, the gradients of the weights are calculated using the process called back-propagation like chain rule to calculate the gradients. Different optimizers like ADAM, SGD or RMS prop then update the weights based on the functions of the calculated gradients. The basic objective of appropriate kernel initialization is to avoid either divergence or slow training.

Factors to keep in mind while initializing kernels are:
1. Avoid initializing all weights and biases with zero or a constant value. This will lead to symmetrical evolution during training, and would make the network linear.
2. Avoid initializing too large weights as it leads to exploding gradient problem.
3. Avoid initializing too small weights as it leads to vanishing gradient problem.
4. To prevent the gradients of the network’s activations from vanishing or exploding we ensure that the mean of activations should be zero and their variance should stay same across every layer.
5. He Initialization - He et al., proposed activation aware initialization of weights used for ReLU activation function. Xavier initialization is similar to He initialization but used for tanh functions.
6. If doing transfer learning, then weights are initialized from pretrained model.

**What happens during the training of a DNN?**
A Deep neural network has a various forms depending on the type of architecture and applications, like multi layer perceptrons, Convolution neural network, RNN etc., At the core of the DNN architecture is operations like activation, drop outs, convolutions, max pooling etc., During the training process, two steps happen:
1. There is a forward pass where in the weights are multiplied by the previous activation outputs and this operations moves sequentially (also skip) in from beginning layer to last layer. Once the prediction is made, the loss is calculated and based on the loss, the gradients are calculated which is a derivative of loss with respect to weights.
2. Then there is a backward pass, where gradients are backpropagated (backprop) and weights are updated. It will receive the gradient of loss with respect to its outputs and return the gradient of loss with respect to its inputs.

Optimization algorithms exists such as Stochastic Gradient Descent with Momentum, Nesterov Accelerated Gradient, AdaGrad, RMSProp and Adam Optimizer, which make sure that we reach the optimal value with the least epochs. Further the weights are updated using the gradients for all the batches or training data resulting in 1 epoch. This iterative process is continued for multiple iterations, epochs till we find that the loss or accuracy metric does not change any further. 

**Reference:**

https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

https://www.deeplearning.ai/ai-notes/initialization/

https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94

https://pathmind.com/wiki/neural-network

https://towardsdatascience.com/training-a-convolutional-neural-network-from-scratch-2235c2a25754


