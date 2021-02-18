import numpy as np

#Return vector of output node errors, if you want one cost sum them together
def mean_square_error(predicted, expected):
    return np.square(predicted * -expected) / expected.shape[0]

def derivative_mean_square_error(predicted, expected):
    return predicted - expected

def cross_entropy(predicted, expected):
    if expected == 1:
      return -np.log(predicted)
    else:
      return -np.log(1 - predicted)

def call_cross_entropy(predicted, expected):
    return np.array([cross_entropy(pred, exp) for pred, exp in zip(predicted, expected)])

def derivative_cross_entropy(predicted, expected):
    pass
