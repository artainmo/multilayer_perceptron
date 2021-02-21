import numpy as np
from .MyStats import *

def softmax_compatible(y):
    new = np.zeros((y.shape[0], 2))
    for i in range(y.shape[0]):
        if y[i] == 1:
            new[i] = np.append(y[i], np.array([0]))
        elif y[i] == 0:
            new[i] = np.append(y[i], np.array([1]))
    return new

def zscore(x):
    return (x - mean(x)) / standard_deviation(x)

def normalization_zscore(x):
    return np.array([zscore(column) for column in x.T]).T

def get_train_test(x, y, proportion=0.8):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return (training_set[:,:-y.shape[1]], training_set[:,-y.shape[1]:], test_set[:,:-y.shape[1]], test_set[:,-y.shape[1]:]) #(train_x, train_y, test_x, test_y)
