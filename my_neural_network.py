import numpy as np

#activation function, sets value between 0 and 1
def sigmoid(x):
    return  np.divide(1, np.add(1, (np.exp((np.multiply(x, -1))))))

def derivative_sigmoid(x):#used to find gradient, calculates slope of predicted value
    return x * (1.0 - x)

def derivative_delta_output_layer(predicted, expected):#Used for convenience of separating mathematical formula derivative
    return (predicted - expected) * derivative_sigmoid(predicted)

def init_bias(weights):
    bias = []
    for layer in weights:
        bias.append(np.ones([1, layer.shape[1]])) #Initialize bias to value one, to make sure node outputs are never stuck on zero
    return bias

#Shape of each weight matrix consists of firstlayernodesXfollowinglayernodes
def init_weights(layers):
    weights = []
    first = None 
    for layer in layers:
        if first != None:
            weights.append(np.zeros([first, layer.shape[0]]))
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

class My_Neural_Network():
    def __init__(self, inputs, expected, deep_layers=1, learning_rate=0.1, n_cycles=1000):
        self.inputs = inputs
        self.expected = expected
        self.predicted = np.zeros(expected.shape)
        self.layers = init_layers(deep_layers, inputs.shape[1], self.expected.shape[1])
        self.weights = init_weights(self.layers)
        self.bias = init_bias(self.weights)
        self.__reset_gradients()
        self.alpha = learning_rate
        self.n_cycles = n_cycles
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
        for i in range(len(self.layers) - 1):
            self.layers[i + 1] = sigmoid(np.dot(self.layers[i], self.weights[i]) + self.bias[i])
        self.predicted = self.layers[-1]

    def cost(self, expected): #cost function calculates total error of made prediction #cost is calculated using sum of square error
        return np.square(np.dot(self.predicted, -expected))
   
    def __output_layer_partial_derivatives(self, expected):
        return np.dot(self.layers[-2].T, derivative_delta_output_layer(self.predicted, expected))
    
    def __delta_derivative_deep_layer(self, position, expected):
        return np.dot(derivative_delta_output_layer(self.predicted, expected), self.weights[position + 1].T) * derivative_sigmoid(self.layers[position + 1])

    def __deep_layer_partial_derivatives(self, position, expected): #More complex as has change in node has also effect on following nodes
        return np.dot(self.layers[position].T, self.__delta_derivative_deep_layer(position, expected)) 

    #Adjust weight and bias values, based on gradient descend  
    #gradient descend searches for error minima point
    #gradient = derivative = slope = rate of change
    #partial derivatives are used to verify how each weight and bias affect the error individually
    def backward_propagation(self, expected):
        self.output_gradient_weight[0] = self.output_gradient_weight[0] + self.__output_layer_partial_derivatives(expected)
        self.output_gradient_bias[0] = self.output_gradient_bias[0] + derivative_delta_output_layer(self.predicted, expected)
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.deep_gradient_weight[i] = self.deep_gradient_weight[i] + self.__deep_layer_partial_derivatives(i, expected)
            self.deep_gradient_bias[i] = self.deep_gradient_bias[i] + self.__delta_derivative_deep_layer(i, expected)
  
    def __reset_gradients(self):
        self.output_gradient_weight = init_deep_gradient([self.weights[-1]])
        self.output_gradient_bias = init_deep_gradient([self.bias[-1]])
        self.deep_gradient_weight = init_deep_gradient(self.weights[0:-1])
        self.deep_gradient_bias = init_deep_gradient(self.bias[0:-1])

    def __update_weights(self):
        self.weights[-1] = self.weights[-1] - (self.alpha * self.output_gradient_weight[0])
        self.bias[-1] = self.bias[-1] - (self.alpha * self.output_gradient_bias[0])
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.weights[i] = self.weights[i] - (self.alpha * self.deep_gradient_weight[i])
            self.bias[i] = self.bias[i] - (self.alpha * self.deep_gradient_bias[i])
        self.__reset_gradients()
 
    def __cycle(self, inputs, expected):
         self.forward_propagation(inputs)
         self.backward_propagation(expected)

    def __batch_cycle(self):
        for i in range(self.n_cycles):
            for inputs, expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(inputs, expected)
            #self.show_all()
            self.__update_weights()
            #self.show_all()
            #print("Cost: " + str(self.cost(expected)))

    def __mini_batch(self):
        pass

    def __stochastic_cycle(self):
        shuffle(self.inputs)
        for i in range(self.n_cycles):
            self.__cycle(inputs, expected)     

    def fit(self):
        self.__batch_cycle()   


if __name__ == "__main__":
    x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    test = My_Neural_Network(x, y, 2)
    test.fit()
