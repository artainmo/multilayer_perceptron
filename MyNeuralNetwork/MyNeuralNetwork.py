import numpy as np
from random import randint
import matplotlib
matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac
import matplotlib.pyplot as mpl
from .activation_functions import *
from .init_neural_network import *
from .cost_functions import *
from .manipulate_data import *

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
    def __init__(self, inputs, expected, test_set_x=None, test_set_y=None, deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=False, validation_hold_outset="Default", feedback=True):
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
            self.layers_activation_function = call_relu
            self.derivative_layers_activation_function = call_derivative_relu
        else:
            print("Error: My_Neural_Network activation function layers, choose between sigmoid, tanh and relu")
            exit()
        if activation_function_output == "sigmoid":
            self.output_activation_function = sigmoid
            self.derivative_output_activation_function = derivative_sigmoid
            self.probabilities_to_answer = sigmoid_to_answer
        elif activation_function_output == "softmax":
            self.output_activation_function = softmax
            self.derivative_output_activation_function = derivative_softmax
            self.probabilities_to_answer = softmax_to_answer
        elif activation_function_output == "relu":
            self.output_activation_function = call_relu
            self.derivative_output_activation_function = call_derivative_relu
            self.probabilities_to_answer = relu_to_answer
        else:
            print("Error: My_Neural_Network activation function output, choose between sigmoid, softmax and relu")
            exit()
        if weight_init == "xavier":
            weight_init = xavier
        elif weight_init == "he":
            weight_init = he
        elif weight_init == None:
            weight_init = normal
        else:
            print("Error: weight init type, choose between xavier, he and None")
            exit()
        if cost_function == "MSE":
            self.cost_function = mean_square_error
            self.derivative_cost_function = derivative_mean_square_error
        elif cost_function == "CE":
            self.cost_function = cross_entropy
            self.derivative_cost_function = derivative_cross_entropy
        else:
            print("Error: cost function, choose between MSE and CE")
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
        self.costs = []
        self.costs_test_set = []
        self.feedback = feedback
        self.test_set_x = test_set_x
        self.test_set_y = test_set_y
        if early_stopping == True and self.test_set_x is not None and self.test_set_y is not None:
            self.early_stopping = True
            if validation_hold_outset == "Default":
                self.validation_hold_outset = int(self.inputs.shape[0] / 100 * 10)
            else:
                self.validation_hold_outset = validation_hold_outset
            self.cost_rising = 0
            self.lowest_cost_index = 0
            self.best_weights = copy_object_shape(self.weights)
            self.best_bias = copy_object_shape(self.bias)
        else:
            self.early_stopping = False
        if self.feedback == True:
            self.show_all()

    def show_all(self):
        print("--------------------------------------DEEP NEURAL NETWORK STRUCTURE--------------------------------------")
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
        print(self.probabilities_to_answer)
        print("---------------------------------------------------------------------------------------------------------")

    #If no lowering of costs compared to lowest cost after 50epochs, early stop and whenever stopping always keep weights and bias associated with lowest cost and cut graphs until lowest cost
    def __early_stopping(self, epoch):
        if epoch == 1:
            self.lowest_cost_index = epoch - 1
            self.cost_rising = 0
        elif self.costs_test_set[self.lowest_cost_index] > self.costs_test_set[-1]:
            self.lowest_cost_index = epoch - 1
            self.best_weights = self.weights
            self.best_bias = self.bias
            self.cost_rising = 0
        else:
            self.cost_rising += 1
        if self.cost_rising >= self.validation_hold_outset or (epoch == self.n_cycles and self.lowest_cost_index != epoch - 1):
            self.weights = self.best_weights
            self.bias = self.best_bias
            return 1
        return 0

    def cost(self, epoch=None, feedback=False): #cost function calculates total error of made prediction, mean over output nodes
        total_error = np.sum([self.cost_function(predicted, expected) for predicted, expected in zip(self.predict(self.inputs, probabilities_to_answer=False), self.expected)]) / self.inputs.shape[0]
        self.costs.append(total_error)
        if self.test_set_x is not None and self.test_set_y is not None:
            total_error_test = np.sum([self.cost_function(predicted, expected) for predicted, expected in zip(self.predict(self.test_set_x, probabilities_to_answer=False), self.test_set_y)]) / self.test_set_x.shape[0]
            self.costs_test_set.append(total_error_test)
            if feedback == True:
                print("Epoch: " + str(epoch) + "/" + str(self.n_cycles) + " -> Cost: " + str(total_error) + " --> Test set Cost: " + str(total_error_test))
        elif feedback == True:
            print("Epoch: " + str(epoch) + "/" + str(self.n_cycles) + " -> Cost: " + str(total_error))
        return total_error

    def __feedback_cost_graph(self):
        input("========================\nPress Enter To See Graph\n========================")
        mpl.title("Starting Cost: " + str(round(self.costs[0], 5))  + "\nFinal Cost: " + str(round(self.costs[-1], 5)))
        mpl.plot(range(len(self.costs)), self.costs, label="training set")
        if self.early_stopping == True:
            mpl.plot(range(len(self.costs[0:self.lowest_cost_index])), self.costs[0:self.lowest_cost_index], label="training set stop")
        if self.test_set_x is not None and self.test_set_y is not None:
            mpl.plot(range(len(self.costs_test_set)), self.costs_test_set, label="test set")
            if self.early_stopping == True:
                mpl.plot(range(len(self.costs_test_set[0:self.lowest_cost_index])), self.costs_test_set[0:self.lowest_cost_index], label="test set stop")
            mpl.legend()
        mpl.show()

    #create output based on input and weights and biases
    def forward_propagation(self, inputs):
        self.layers[0] = np.array([inputs], dtype=np.float128)
        for i in range(len(self.layers) - 2):
            self.layers[i + 1] = self.layers_activation_function(np.dot(self.layers[i], self.weights[i]) + self.bias[i])
        self.layers[-1] = self.output_activation_function((np.dot(self.layers[-2], self.weights[-1]) + self.bias[-1]))
        self.predicted = self.layers[-1]

    def __output_layer_partial_derivatives(self, expected):
        Delta = self.derivative_cost_function(self.predicted, expected) * self.derivative_output_activation_function(self.predicted)
        return np.dot(self.layers[-2].T, Delta), Delta

    def __deep_layer_partial_derivatives(self, position, Delta): #More complex as has change in node has also effect on following nodes
        Delta = (np.dot(self.weights[position + 1], Delta.T) * (self.derivative_layers_activation_function(self.layers[position + 1])).T).T
        return np.dot(self.layers[position].T, Delta), Delta

    #Adjust weight and bias values, based on gradient descend
    #gradient descend searches for error minima point
    #gradient = derivative = slope = rate of change
    #partial derivatives are used to verify how each weight and bias affect the error individually
    def backward_propagation(self, expected):
        gradient, Delta = self.__output_layer_partial_derivatives(expected)
        self.output_gradient_weight[0] = self.output_gradient_weight[0] + gradient
        self.output_gradient_bias[0] = self.output_gradient_bias[0] + Delta #bias weight does not need to get multiplied by prior bias node as it is equal to one
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 1)
            gradient, Delta = self.__deep_layer_partial_derivatives(i, Delta)
            self.deep_gradient_weight[i] = self.deep_gradient_weight[i] + gradient
            self.deep_gradient_bias[i] = self.deep_gradient_bias[i] + Delta

    def __reset_gradients(self):
        self.output_gradient_weight = copy_object_shape([self.weights[-1]])
        self.output_gradient_bias = copy_object_shape([self.bias[-1]])
        self.deep_gradient_weight = copy_object_shape(self.weights[0:-1])
        self.deep_gradient_bias = copy_object_shape(self.bias[0:-1])

    def __update_weights(self, _epoch):
        self.weights[-1] = self.weights[-1] - (self.alpha * self.output_gradient_weight[0])
        self.bias[-1] = self.bias[-1] - (self.alpha * self.output_gradient_bias[0])
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.weights[i] = self.weights[i] - (self.alpha * self.deep_gradient_weight[i])
            self.bias[i] = self.bias[i] - (self.alpha * self.deep_gradient_bias[i])
        self.__reset_gradients()
        self.cost(epoch=_epoch, feedback=self.feedback)

    def __cycle(self, inputs, expected):
         self.forward_propagation(inputs)
         if self.early_stopping == True:
             self.last_weights = self.weights
             self.last_bias = self.bias
         self.backward_propagation(expected)

    #slow but more computanional efficient on big datasets
    #Stable convergence but risk of local minima or premature convergence
    def __batch(self):
        for i in range(self.n_cycles):
            for inputs, expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(inputs, expected)
            self.__update_weights(i + 1)
            if self.early_stopping == True and self.__early_stopping(i + 1):
                print("Early Stopping was used")
                break

    #mini-batch sits between stochastic and batch, trying to optimize benefits of both, and is the recommended variant of gradient descend
    def __mini_batch(self):
        generator = get_mini_batch(self.inputs, self.expected, self.b)
        for i in range(self.n_cycles):
            inputs, expected = next(generator)
            for _inputs, _expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(_inputs, _expected)
            self.__update_weights(i + 1)
            if self.early_stopping == True and self.__early_stopping(i + 1):
                print("Early Stopping was used")
                break

    #faster convergence on small datasets but slower on big datasets due to constant weight update
    #can avoid local minimas or premature convergence but has higher variance in results due to randomness
    def __stochastic(self):
        length = len(self.inputs) - 1
        for i in range(self.n_cycles):
            random = randint(0, length)
            self.__cycle(self.inputs[random], self.expected[random])
            self.__update_weights(i + 1)
            if self.early_stopping == True and self.__early_stopping(i + 1):
                print("Early Stopping was used")
                break


    def fit(self):
        input("=============================\nPress Enter To Start Training\n=============================")
        self.costs.clear()
        self.costs_test_set.clear()
        self.gradient_descend()
        if self.feedback == True:
            self.__feedback_cost_graph()

    def predict(self, inputs, probabilities_to_answer=True):
        answers = np.zeros((inputs.shape[0], self.expected.shape[1]))
        for i in range(inputs.shape[0]):
            self.forward_propagation(inputs[i])
            answers[i] = self.predicted
        if probabilities_to_answer == True:
            return self.probabilities_to_answer(answers)
        else:
            return answers


# if __name__ == "__main__":
#     x = np.array([[0,0,1,1,0,0],[0,1,1,1,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0]]) #4X3 -> 4 examples and 3 inputs expected
#     y = np.array([[0, 1],[1, 1],[1, 0],[1, 0]]) #4X2 -> 4 examples and 2 outputs expected
#     test = MyNeuralNetwork(x, y)
#     test.fit()
