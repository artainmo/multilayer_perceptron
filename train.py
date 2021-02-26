import numpy as np
import pandas as pn
import sys
import os
from MyNeuralNetwork import *

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

#returns "y" if network has been accepted and saved
def single_network_evaluation(NN, data):
    if input("Evaluate predictions for training set?(y/n):") == "y":
        evaluate(NN.predict(data[0]), data[1])
    if input("Evaluate predictions for test set?(y/n):") == "y":
        evaluate(NN.predict(data[2]), data[3])
    saved = input("Save this network?(y/n):")
    if saved == "y":
        save_neural_network(NN)
    return saved

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    if input("Visualize the data?(y/n):") == "y":
        visualize_data(path)
    x, y = get_x_y(path, list(range(2, 32)), 1)
    data = get_train_test(x, y)
    while True: #Continue looping until ideal random data split and weight init occured
        NN = MyNeuralNetwork(name="mini-batch | tanh | xavier | CE", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
        NN.fit()
        if single_network_evaluation(NN, data) == 'y':
            break
    NN2 = MyNeuralNetwork(name="batch | tanh | xavier | MSE", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="MSE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN2.fit()
    single_network_evaluation(NN2, data)
    NN3 = MyNeuralNetwork(name="stochastic | tanh | xavier | CE", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=2, learning_rate=0.01, n_cycles=10000, gradient_descend="stochastic", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN3.fit()
    single_network_evaluation(NN3, data)
    NN4 = MyNeuralNetwork(name="stochastic | tanh | xavier | CE | momentum", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="stochastic", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=True, validation_hold_outset="Default", momentum=True, feedback=True)
    NN4.fit()
    single_network_evaluation(NN4, data)
    NN5 = MyNeuralNetwork(name="mini-batch | sigmoid | radom init | CE", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="sigmoid", activation_function_output="softmax", weight_init=None, cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN5.fit()
    single_network_evaluation(NN5, data)
    NN6 = MyNeuralNetwork(name="mini-batch | relu | he | CE", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="relu", activation_function_output="softmax", weight_init="he", cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN6.fit()
    single_network_evaluation(NN6, data)
    NN7 = MyNeuralNetwork(name="four deep layer", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=4, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN7.fit()
    single_network_evaluation(NN7, data)
    NN8 = MyNeuralNetwork(name="zero deep layer", inputs=data[0], expected=data[1], test_set_x=data[2], test_set_y=data[3], deep_layers=0, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=True, validation_hold_outset="Default", feedback=True)
    NN8.fit()
    NN8.training_metric_history()
    single_network_evaluation(NN8, data)
    compare_different_neural_networks([NN, NN2, NN3, NN4, NN5, NN6, NN7, NN8])

#Best learning rate is smallest default one
#gradient descend type makes no difference as long as enough n_cycles follow and early stopping is used
#tanh and relu with appropriate weight init seems superior over sigmoid, tanh most stable
#cross entropy is default for classification and demanded in subject
