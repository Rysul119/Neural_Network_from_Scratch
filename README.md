# Neural_Network_from_Scratch
This project facilitates training a fully connected neural network to predict outputs for given input data. 

### Till now this project has:

  - Forward propagation to get an estimated output
  - Calculating the cost using, MSE (X, \boldsymbol{\theta })= \frac{1}{2N}\sum_{i = 1}^{N}\left ( \hat{y}_{i} - y_{i} \right )^{2}
  - Backward propagation to calculate the layer errors
  - Calculate the derivatives of the cost w.r.t the trainable variables i.e. the weights and biases of the neural network
  - Update the weights and biases of the neural network using gradient descent to optimize the network.
  - Activation function: sigmoid, S(x) = \frac{1}{1+e^{-x}}; linear

### More to come:

  - Mini-batch training
  - Regularization
  - More activation functions, e.g., ReLU
  
### Functionality:
  - It should be able to build a fully connected neural network given the architecture as a dictionary i.e. dictionary should have this form {"layer_number: 3", "node_array": [3, 3, 1], "activation_array": ["sigmoid", "sigmoid", "linear"]}.
  - It will be able to train a fully connected neural network given training data with corresponding label.
  - It will also calculate respective layer errors.
  - It optimizes the weights using gradient descent