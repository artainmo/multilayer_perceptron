import numpy as np
from random import randint
import matplotlib
matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac
import matplotlib.pyplot as mpl

#Used on output layer to classify output to one category, the softmax function assumes each example can only be part of one class
#Useful when more than 2 mutually exclusive classes, otherwise sigmoid can be used
#Can converge faster by putting more emphasis through exponentials on probably correct answers
#Takes vector and transforms into probabilities that sum to one
def softmax(predicted):
    return np.array([np.exp(predicted) / np.sum(np.exp(predicted))])
    #e_exponentials = []
    #normalized = []
    #for _class in predicted:
    #    e_exponentials.append(np.exp(_class))
    #normalization_term = np.sum(e_exponentials)
    #for num in e_exponentials:
    #    normalized.append(num / normalization_term)
    #return np.array([normalized])  

#activation function, sets value between 0 and 1
def sigmoid(x):
    return  np.divide(1, np.add(1, (np.exp((np.multiply(x, -1))))))

def derivative_sigmoid(x):#used to find gradient, calculates slope of predicted value
    return x * (1.0 - x)

#activation function sets value between -1,1
def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - np.square(x)

def relu(x):
    for i in range(len(x[0])):
        if x[0][i] < 0:
            x[0][i] = 0
    return x

def derivative_relu(x):
    for i in range(len(x[0])):
        if x[0][i] <= 0:
            x[0][i] = 0
        else:
            x[0][i] = 1
    return x

def init_bias(weights):
    bias = []
    for layer in weights:
        bias.append(np.zeros([1, layer.shape[1]])) #Initialize bias to zero as default
    return bias

#Shape of each weight matrix consists of firstlayernodesXfollowinglayernodes
def init_weights(layers):
    weights = []
    first = None 
    for layer in layers:
        if first != None:
            weights.append(np.ones([first, layer.shape[0]]))
        first = layer.shape[0]
    return weights

#Here we follow pyramid structure with input layers as base, making each following layer smaller
def layer_length(input_nodes, output_nodes, deep_layers):
    _range = range(input_nodes, output_nodes -1, -1)
    base_percentile = 100 / deep_layers
    percentile = base_percentile
    yield input_nodes
    while percentile < 100:
        next_layer_nodes = np.percentile(_range, 100 - percentile).astype(np.int64)
        yield next_layer_nodes
        percentile += base_percentile
    yield output_nodes

def init_layers(deep_layers, input_nodes, output_nodes):
    layers = []
    for layer in layer_length(input_nodes, output_nodes, deep_layers):
        layers.append(np.zeros([layer, 1]),)
    return layers

def show_object(name, obj):
    print(name + ":")
    for elem in obj:
        print(elem.shape)
    print("----------")


def init_deep_gradient(copy):
    ret = []
    for cpy in copy:
        ret.append(np.zeros(cpy.shape))
    return ret

def get_mini_batch(inputs, expected, b):
    length = len(inputs)
    last = 0
    pos = 0
    while True:
        pos += b
        if pos > length:
            ret = inputs[last:length]
            pos -= length
            while pos > length: 
                pos -= length
                np.concatenate((ret, inputs))
            yield np.concatenate((ret, inputs[0:pos])), np.concatenate((expected[last:length], expected[0:pos]))
        else:
            yield inputs[last:pos], expected[last:pos]
        last = pos

class MyNeuralNetwork():
    def __init__(self, inputs, expected, deep_layers=1, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="relu", activation_function_output="softmax", feedback=True):
        if gradient_descend == "stochastic":
            self.gradient_descend = self.__stochastic
        elif gradient_descend == "batch":
            self.gradient_descend = self.__batch
        elif gradient_descend == "mini-batch":
            self.gradient_descend = self.__mini_batch
        else:
            print("Error: My_Neural_Network gradient descend, choose between stochastic, batch, mini-batch")
        if activation_function_layers == "sigmoid":
            self.layers_activation_function = sigmoid
            self.derivative_layers_activation_function = derivative_sigmoid
        elif activation_function_layers == "tanh":
            self.layers_activation_function = tanh
            self.derivative_layers_activation_function = derivative_tanh
        elif activation_function_layers == "relu":
            self.layers_activation_function = relu
            self.derivative_layers_activation_function = derivative_relu
        else:
            print("Error: My_Neural_Network activation function layers, choose between stochastic, batch, mini-batch")
        self.inputs = inputs
        self.expected = expected
        self.layers = init_layers(deep_layers + 1, inputs.shape[1], self.expected.shape[1])
        self.weights = init_weights(self.layers)
        self.bias = init_bias(self.weights)
        self.__reset_gradients()
        self.alpha = learning_rate
        self.n_cycles = n_cycles
        self.b = b #mini-batch size
        self.softmax = softmax
        self.costs = []
        self.feedback = feedback
        self.show_all()
   
    def show_all(self):
        print("---------------------------------------------------------------------------------")
        show_object("Layer", self.layers)
        show_object("Weight", self.weights)
        show_object("Bias", self.bias)
        show_object("Output gradient weight", self.output_gradient_weight)
        show_object("Output gradient bias", self.output_gradient_bias)
        show_object("Deep gradient weight", self.deep_gradient_weight)
        show_object("Deep gradient bias", self.deep_gradient_bias)
        print("---------------------------------------------------------------------------------")

    #create output based on input and weights and biases
    def forward_propagation(self, inputs):
        self.layers[0] = np.array([inputs])
        for i in range(len(self.layers) - 2):
            self.layers[i + 1] = self.layers_activation_function(np.dot(self.layers[i], self.weights[i]) + self.bias[i])
        self.layers[-1] = softmax((np.dot(self.layers[-2], self.weights[-1]) + self.bias[-1])[0])
        self.predicted = self.layers[-1]

    def cost(self, expected): #cost function calculates total error of made prediction #cost is calculated using sum of square error
        ret =  np.square(np.dot(self.predicted, -expected))
        self.costs.append(ret)
        return ret
 
    def __derivative_delta_output_layer(self, expected):#Used for convenience of separating mathematical formula derivative
        return (self.predicted - expected) * self.derivative_layers_activation_function(self.predicted)
  
    def __output_layer_partial_derivatives(self, expected):
        return np.dot(self.layers[-2].T, self.__derivative_delta_output_layer(expected))
    
    def __delta_derivative_deep_layer(self, position, expected):
        return np.dot(self.__derivative_delta_output_layer(expected), self.weights[position + 1].T) * self.derivative_layers_activation_function(self.layers[position + 1])

    def __deep_layer_partial_derivatives(self, position, expected): #More complex as has change in node has also effect on following nodes
        return np.dot(self.layers[position].T, self.__delta_derivative_deep_layer(position, expected)) 

    #Adjust weight and bias values, based on gradient descend  
    #gradient descend searches for error minima point
    #gradient = derivative = slope = rate of change
    #partial derivatives are used to verify how each weight and bias affect the error individually
    def backward_propagation(self, expected):
        self.output_gradient_weight[0] = self.output_gradient_weight[0] + self.__output_layer_partial_derivatives(expected)
        self.output_gradient_bias[0] = self.output_gradient_bias[0] + self.__derivative_delta_output_layer(expected)
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.deep_gradient_weight[i] = self.deep_gradient_weight[i] + self.__deep_layer_partial_derivatives(i, expected)
            self.deep_gradient_bias[i] = self.deep_gradient_bias[i] + self.__delta_derivative_deep_layer(i, expected)
  
    def __reset_gradients(self):
        self.output_gradient_weight = init_deep_gradient([self.weights[-1]])
        self.output_gradient_bias = init_deep_gradient([self.bias[-1]])
        self.deep_gradient_weight = init_deep_gradient(self.weights[0:-1])
        self.deep_gradient_bias = init_deep_gradient(self.bias[0:-1])

    def __update_weights(self, epoch, expected):
        self.weights[-1] = self.weights[-1] + (self.alpha * self.output_gradient_weight[0])
        self.bias[-1] = self.bias[-1] + (self.alpha * self.output_gradient_bias[0])
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.weights[i] = self.weights[i] + (self.alpha * self.deep_gradient_weight[i])
            self.bias[i] = self.bias[i] + (self.alpha * self.deep_gradient_bias[i])
        self.__reset_gradients()
        if self.feedback == True:
            print("Epoch: " + str(epoch) + "/" + str(self.n_cycles) + " -> Cost: " + str(self.cost(expected)))
 
    def __cycle(self, inputs, expected):
         self.forward_propagation(inputs)
         self.backward_propagation(expected)
    
    #slow but more computanional efficient on big datasets
    #Stable convergence but risk of local minima or premature convergence
    def __batch(self):
        for i in range(self.n_cycles):
            for inputs, expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(inputs, expected)
            self.__update_weights(i + 1, expected)

    #mini-batch sits between stochastic and batch, trying to optimize benefits of both, and is the recommended variant of gradient descend
    def __mini_batch(self):
        generator = get_mini_batch(self.inputs, self.expected, self.b)
        for i in range(self.n_cycles):
            inputs, expected = next(generator)
            for _inputs, _expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(_inputs, _expected)
            self.__update_weights(i + 1, _expected)

    #faster convergence on small datasets but slower on big datasets due to constant weight update
    #can avoid local minimas or premature convergence but has higher variance in results due to randomness
    def __stochastic(self):
        length = len(self.inputs) - 1
        for i in range(self.n_cycles):
            random = randint(0, length)
            self.__cycle(self.inputs[random], self.expected[random])
            self.__update_weights(i + 1, self.expected[random])

    def __feedback_cost_graph(self):
        input("========================\nPress Enter To See Graph\n========================")
        mpl.title("Starting Cost: " + str(round(self.costs[0][0], 5))  + "\nFinal Cost: " + str(round(self.costs[-1][0], 5)))
        mpl.plot(range(len(self.costs)), self.costs)
        mpl.show()


    def fit(self):
        self.costs.clear()
        self.gradient_descend()
        if self.feedback == True:
            self.__feedback_cost_graph()


#if __name__ == "__main__":
#    x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #4X3 -> 4 examples and 3 inputs expected
#    y = np.array([[0, 1],[1, 1],[1, 0],[0, 1]]) #4X2 -> 4 examples and 2 outputs expected
#    test = MyNeuralNetwork(x, y)
#    test.fit()
