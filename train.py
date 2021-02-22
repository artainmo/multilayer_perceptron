import numpy as np
import pandas as pn
import sys
import os
from MyNeuralNetwork import *
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
    if input("Visualize the data?(y/n):") == "y":
        visualize_data(path)
    x, y = get_x_y(path, list(range(2, 32)), 1)
    data = get_train_test(x, y)
    NN = MyNeuralNetwork(data[0], data[1], test_set_x=data[2], test_set_y=data[3],deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN.fit()
    if input("Evaluate predictions for training set?(y/n):") == "y":
        evaluate(NN.predict(data[0]), data[1])
    if input("Evaluate predictions for test set?(y/n):") == "y":
        evaluate(NN.predict(data[2]), data[3])
    if input("Save this network?(y/n):") == "y":
        save_neural_network(NN)

#Best learning rate is smallest default one
#Less deep layers probably better but not accepted based on what is demanded in pdf
#gradient descend type makes no difference as long as enough n_cycles follow and early stopping is used
#tanh and relu with appropriate weight init seems superior over sigmoid, tanh most stable
#cross entropy is default for classification and demanded in subject
