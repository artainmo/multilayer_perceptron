import numpy as np

def softmax(predicted):
    return np.exp(predicted) / np.sum(np.exp(predicted))

#Output of total vector values equals one
def derivative_softmax(predicted, expected):
    return predicted - expected

#activation function, sets value between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):#used to find gradient, calculates slope of predicted value
    return x * (1 - x)

#activation function sets value between -1,1
def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - np.square(x)

#Value stays same unless under zero than equal to zero
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
