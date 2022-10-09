# Image-Classification-CNN
Classification of the CIFAR-10 Image Dataset using Convolutional Neural Networks

The CIFAR-10 is a dataset of 50,000 32x32 coloured training images and 10,000 test images, labelled with 10 
categories. These 10 classes are — Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck 
respectively. This project will focus on implementing two classification algorithms on the CIFAR-10 dataset using 
Python. 

## Basic CNN Model with Dropout

In this model, there are two convolutional layers stacked on each other with 3x3 filters followed by a MaxPooling 
layer. ReLU activation functions are used for each of the convolutional layers and a Dropout is used in each of 
the convolutional layer as well. Dropout is a regularization method that approximates training a large number of 
neural networks with different architectures in parallel. During training, some number of layer outputs are 
randomly ignored. This has the effect of making the layer look-like and be treated-like a layer with a different 
number of nodes and connectivity to the prior layer. A classifier is used along with this Neural network to predict 
the class to which the photo would belong to. The classifier built is optimized using the Adam optimizer. The
Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum. It uses 
the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using 
moving average of the gradient instead of gradient itself like SGD with momentum. Categorical cross entropy
function would be used to minimise loss and the metric to be monitored would be the classification accuracy.

## VGG Model (CNN) with Dropout and Batch Normalization

In this architecture, convolutional layers are stacked on top of each other along with 3x3 filters followed by a max 
pooling layer forming a block. These blocks can be repeated along with the number of filters. As the depth of the 
network increases the number of filters also increase. In order to ensure that the output feature maps match the 
input, padding is implemented on the convolutional layers. Each layer of the convolutional network uses the ReLU 
activation function and the He weight initialization. ReLU activation function helps prevent the exponential 
growth in the computation required to operate the neural network. He initializations are used for layers with 
ReLU activation function whereas Xavier initialization works better along with sigmoid functions.
In the project, a 3 Block VGG style architecture has been used along with a classifier to predict the class to which 
the photo would belong to. The classifier built is optimized using Stochastic Gradient Descent. A Learning rate 
of 0.001 and a categorical cross entropy function would be used to minimise loss and the metric to be monitored 
would be the classification accuracy. The feature maps output from the 3 block VGG must be flattened. The 
output layer that should be used for prediction should have 10 nodes since we have 10 classes. Since this is a
multiclass classification problem, a SoftMax activation function is used to predict a multinomial probability 
distribution. A dropout layer has been added to each of the VGG blocks to make the model more robust. Batch
normalization is used in each layer to stabilize the learning and accelerate the learning process.

# Results

The first model shows a test accuracy of approximately 76% and
the second model is the model with a better accuracy of 86% on the test dataset.
