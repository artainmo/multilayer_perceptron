import numpy as np
from random import randint
import matplotlib
matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac
import matplotlib.pyplot as mpl
from activation_functions import *
from init_neural_network import *
from cost_functions import *

def show_object(name, obj):
    print(name + ":")
    for elem in obj:
        print(elem.shape)
    print("----------")

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
    def __init__(self, inputs, expected, deep_layers=1, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="sigmoid", weight_init="xavier", cost_function="MSE", feedback=True):
        if gradient_descend == "stochastic":
            self.gradient_descend = self.__stochastic
        elif gradient_descend == "batch":
            self.gradient_descend = self.__batch
        elif gradient_descend == "mini-batch":
            self.gradient_descend = self.__mini_batch
        else:
            print("Error: My_Neural_Network gradient descend, choose between stochastic, batch, mini-batch")
            exit()
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
            print("Error: My_Neural_Network activation function layers")
            exit()
        if activation_function_output == "sigmoid":
            self.output_activation_function = sigmoid
            self.derivative_output_activation_function = derivative_sigmoid
        elif activation_function_output == "softmax":
            self.output_activation_function = softmax
            self.derivative_output_activation_function = derivative_softmax
        elif activation_function_output == "relu":
            self.output_activation_function = call_relu
            self.derivative_output_activation_function = call_derivative_relu
        else:
            print("Error: My_Neural_Network activation function output")
            exit()
        if weight_init == "xavier":
            weight_init = xavier
        elif weight_init == "he":
            weight_init = he
        elif weight_init == None:
            weight_init = normal
        else:
            print("Error: weight init type")
            exit()
        if cost_function == "MSE":
            self.cost_function = mean_square_error
            self.derivative_cost_function = derivative_mean_square_error
        elif cost_function == "CE":
            self.cost_function = call_cross_entropy
            pass #self.derivative_cost_function =
        else:
            print("Error: cost function")
            exit()
        self.inputs = inputs
        self.expected = expected
        self.layers = init_layers(deep_layers + 1, inputs.shape[1], self.expected.shape[1])
        self.weights = init_weights(self.layers, inputs.shape[1], self.expected.shape[1], weight_init)
        self.delta = copy_object_shape(self.weights) #delta is used in backpropagation to indicate the error of each layer final activation value
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
        print(self.gradient_descend)
        print(self.layers_activation_function)
        print(self.derivative_layers_activation_function)
        print(self.output_activation_function)
        print(self.derivative_output_activation_function)
        print("---------------------------------------------------------------------------------")
        input("=============================\nPress Enter To Start Training\n=============================")


    #create output based on input and weights and biases
    def forward_propagation(self, inputs):
        self.layers[0] = np.array([inputs])
        for i in range(len(self.layers) - 2):
            self.layers[i + 1] = self.layers_activation_function(np.dot(self.layers[i], self.weights[i]) + self.bias[i])
        self.layers[-1] = self.output_activation_function((np.dot(self.layers[-2], self.weights[-1]) + self.bias[-1]))
        self.predicted = self.layers[-1]

    def one_cost(self, expected): #cost function calculates total error of made prediction
        total_error = self.cost_function(self.predicted, expected)
        total_error = np.sum(total_error) #Transform vector of errors into one total error value
        self.costs.append(total_error)
        return ret

    def __output_layer_partial_derivatives(self, expected):
        Delta = self.derivative_cost_function(self.predicted, expected) * self.derivative_output_activation_function(self.predicted)
        return np.dot(self.layers[-2].T, Delta), Delta

    def __deep_layer_partial_derivatives(self, position, expected, Delta): #More complex as has change in node has also effect on following nodes
        Delta = np.dot(self.weights[position + 1].T, Delta) * self.derivative_layers_activation_function(self.layers[position + 1])
        return np.dot(self.layers[position].T, Delta)

    #Adjust weight and bias values, based on gradient descend
    #gradient descend searches for error minima point
    #gradient = derivative = slope = rate of change
    #partial derivatives are used to verify how each weight and bias affect the error individually
    def backward_propagation(self, expected):
        gradient, Delta = self.__output_layer_partial_derivatives(expected)
        self.output_gradient_weight[-1] = self.output_gradient_weight[0] - gradient
        self.output_gradient_bias[-1] = self.output_gradient_bias[0] - Delta #bias weight does not need to get multiplied by prior bias node as it is equal to one
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            gradient, Delta = self.__deep_layer_partial_derivatives(i, expected, Delta)
            self.deep_gradient_weight[i] = self.deep_gradient_weight[i] - gradient
            self.deep_gradient_bias[i] = self.deep_gradient_bias[i] - Delta

    def __reset_gradients(self):
        self.output_gradient_weight = copy_object_shape([self.weights[-1]])
        self.output_gradient_bias = copy_object_shape([self.bias[-1]])
        self.deep_gradient_weight = copy_object_shape(self.weights[0:-1])
        self.deep_gradient_bias = copy_object_shape(self.bias[0:-1])

    def __update_weights(self, epoch, expected):
        self.weights[-1] = self.weights[-1] - (self.alpha * self.output_gradient_weight[0])
        self.bias[-1] = self.bias[-1] - (self.alpha * self.output_gradient_bias[0])
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.weights[i] = self.weights[i] - (self.alpha * self.deep_gradient_weight[i])
            self.bias[i] = self.bias[i] - (self.alpha * self.deep_gradient_bias[i])
        self.__reset_gradients()
        if self.feedback == True:
            print("Epoch: " + str(epoch) + "/" + str(self.n_cycles) + " -> Cost: " + str(self.one_cost(expected)))

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
        mpl.title("Starting Cost: " + str(round(self.costs[0], 5))  + "\nFinal Cost: " + str(round(self.costs[-1], 5)))
        mpl.plot(range(len(self.costs)), self.costs)
        mpl.show()


    def fit(self):
        self.costs.clear()
        self.gradient_descend()
        if self.feedback == True:
            self.__feedback_cost_graph()


if __name__ == "__main__":
    x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #4X3 -> 4 examples and 3 inputs expected
    y = np.array([[0, 1],[1, 1],[1, 0],[0, 1]]) #4X2 -> 4 examples and 2 outputs expected
    test = MyNeuralNetwork(x, y)
    test.fit()
