import numpy as np

#defining sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#FC Layer class
class Layer:
    #initializing
    def __init__(self, input_shape, node, activation):
        self.node = node
        #assuming input has a shape of (n, 1)
        self.weights = np.random.normal(size=(input_shape.shape[0], self.node))
        self.bias = np.random.randn() # only one bias for the layer
        self.activation = activation

    #return layer variables
    def get_variables(self):
        return  self.weights, self.bias

    def feedforward(self, input):
        #calculating the output using matrix multiplication z = matmul(w,x)+b
        output = np.matmul(input, self.weights) + self.bias
        #returns an output of shape(node, 1) after applying activation function
        if(self.activation == "sigmoid"):
            #for now lets return the weights as well. Later i will find if we have a better alternative
            return sigmoid(output)


class NNModel:
    #initializing
    #takes a dictionary with (keys layer_number, node_array[hidden + output], activation_array) layer number and corresponding hidden layer nodes
    def __init__(self, arc, input_shape):
        self.layerN = arc["layer_number"]
        self.nodeA = arc["node_array"]
        self.activationA = arc["activation_array"]
        self.input_shape = input_shape
        self.layers = []
        self.model_weights = []
        self.model_biases = []
        for i in range(self.layerN):
            layer = Layer(self.input_shape, self.nodeA[i], self.activationA[i])
            self.layers.append(layer)
            weights, biases = layer.get_variables()
            self.model_weights.append(weights)
            self.model_biases.append(biases)

    def forward(self, input):
        inputI = input
        for i in range(self.layerN):
            output = self.layers[i].feedforward(inputI)
            inputI = output
        #outputs the final result that is the output node after applying activation
        return inputI

    #backprop
