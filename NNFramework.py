import numpy as np

# defining sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1- fx)

# FC Layer class
class Layer:
    # initializing
    def __init__(self, input_shape, node, activation):
        self.node = node
        # assuming input has a shape of (n, 1)
        self.weights = np.random.normal(size=(input_shape.shape[0], self.node))
        self.bias = np.random.randn() # only one bias for the layer
        self.activation = activation

    # return layer variables
    def get_variables(self):
        return  self.weights, self.bias

    def feedforward(self, input):
        # calculating the output using matrix multiplication z = matmul(w,x)+b
        output = np.matmul(input, self.weights) + self.bias
        # returns an output of shape(node, 1) after applying activation function
        if(self.activation == "sigmoid"):
            # for now lets return the weights as well. Later i will find if we have a better alternative
            return output, sigmoid(output)


class NNModel:
    # initializing
    # takes a dictionary with (keys layer_number, node_array, activation_array) for both [hidden + output]
    def __init__(self, arc, input_shape, learning_rate):
        self.layerN = arc["layer_number"]
        self.nodeA = arc["node_array"]
        self.activationA = arc["activation_array"]
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layers = []
        self.model_weights = []
        self.model_biases = []
        self.z = []  # to store output before activation
        self.h = []  # to store output after activation
        self.layer_error = [] # to store layer errors
        self.deriv = [] # to store the derivative of cost function w.r.t weights
        for i in range(self.layerN):
            layer = Layer(self.input_shape, self.nodeA[i], self.activationA[i])
            self.layers.append(layer)
            weights, biases = layer.get_variables()
            self.model_weights.append(weights)
            self.model_biases.append(biases)

    def forward(self, input):
        inputI = input # input should be flattened
        self.h.append(input) # for calculating the deriv in line 84
        for i in range(self.layerN):
            output, output_activation = self.layers[i].feedforward(inputI)
            self.z.append(output) # has to clear out at a later stage as function forward will be used in a for loop but will need this value for a particular iteration
            self.h.append(output_activation) # has to clear out at a later stage
            inputI = output_activation
        self.h.pop() # pop the activation value for the output layer for calculating deriv in line 84, as the output is not needed
        # outputs the final result that is the output node after applying activation
        return inputI

    # backprop
    def backprop(self, y_pred, y_test):
        # get the z value for the output layer for calculating the output layer error
        layer_error = (y_pred - y_test) * deriv_sigmoid(self.z[self.layerN-1])
        layer_error = np.array([layer_error], ndmin = 2) # to make it a (1, 1) array
        self.layer_error.insert(0, layer_error)
        # to get all the layer errors
        for n in reversed(range(self.layerN - 1)): # check if it should be n-1 or n<< think it is good but should still check
            # fix function deriv_sigmoid so that it can output array << Its already good
            ilayer_error = np.multiply(np.matmul(self.layers[n+1].weights, self.layer_error[0]), deriv_sigmoid(self.z[n])) # (n_node, 1) shape array
            self.layer_error.insert(0, ilayer_error) # stores the layer errors

    # calculate the derivatives
    def calderiv(self):
        # getting the derivatives with respect to the variables using layer errors
        for i in range(self.layerN):
            # think about the first one.. with input x
            ideriv = np.matmul(self.layer_error[i], self.h[i])  # outputs the same dim deriv matrix as the weight at that layer.
            assert self.model_weights[i].shape == ideriv.shape  # checking if the derivative and weight shape are the same
            self.deriv.append(ideriv)

    # update the weights according to the gradient descent
    def updateweights(self):
        # update the weights after getting the derivatives for gradient descent
        for i in range(self.layerN):
            self.model_weights[i] = self.model_weights[i] - self.learning_rate*self.deriv[i]

        # clear out all the stored lists except the network weights and layer lists
        self.z = []  # to store output before activation
        self.h = []  # to store output after activation
        self.layer_error = [] # to store layer errors
        self.deriv = [] # to store the derivative of cost function w.r.t weights

    #fit function will call for forward(calculate the output) and backprop(calculate the error, gradient and then update the weigts and biases) once at each iteration
    def fit(self, x_test, y_test):
        #maybe add a training method to incorporate batch fit..
        y_pred = self.forward(x_test) # call forward function to get the output y_pred
        self.backprop(y_pred, y_test) #calculate the layer errors
        self.calderiv() # calculate the derivatives
        self.updateweights() # update the weights according to gradient descent