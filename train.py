import numpy as np
import pandas as pn
import sys
import os
from .MyNeuralNetwork import *
import pickle

def save_neural_network(NN):
    pickle.dump(NN.weights, open("saved/weights.pkl", 'wb', closefd=True))
    print("Weights saved in saved/weights.pkl")
    pickle.dump(NN.bias, open("saved/bias.pkl", 'wb', closefd=True))
    print("Bias saved in saved/bias.pkl")

def get_x_y(path, column_range_x, column_y):
    if os.path.isfile(path) == False:
        print("Error: file argument does not exist")
        exit()
    x = np.array(pn.read_csv(path, header=None, usecols=column_range_x))
    x = normalization_zscore(x)
    y = pn.read_csv(path, header=None, usecols=[column_y])
    y.replace({"M": 1, "B": 0}, inplace=True)
    y = np.array(y)
    y = softmax_compatible(y)
    return x, y

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    # visualize_data(path)
    x, y = get_x_y(path, list(range(2, 32)), 1)
    data = get_train_test(x, y)
    NN = MyNeuralNetwork(data[0], data[1], deep_layers=2, learning_rate=0.01, n_cycles=50, gradient_descend="batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", feedback=True)
    NN.fit()
    save_neural_network(NN)
