import numpy as np
import pandas as pn
import sys
import os

def get_x_y(path, column_range_x, column_y):
    if os.path.isfile(path) == False:
        print("Error: file argument does not exist")
        exit()
    x = np.array(pn.read_csv(path, header=None, usecols=column_range_x))
    y = pn.read_csv(path, header=None, usecols=[column_y])
    y.replace({"M": 1, "B": 0}, inplace=True)
    y = np.array(y)
    return x, y

def get_train_test(x, y, proportion=0.8):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return (training_set[:,:-1], training_set[:,-1:], test_set[:,:-1], test_set[:,-1:])

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    x, y = get_x_y(path, list(range(2, 32)), 1)
    data = get_train_test(x, y)
