import numpy as np

#Return vector of output node errors, if you want one cost sum them together
def mean_square_error(predicted, expected):
    return np.square(predicted * -expected) / expected.shape[0]

def derivative_mean_square_error(predicted, expected):
    return predicted - expected

#Also called logistic cost function
def cross_entropy(predicted, expected):
    return (expected * np.log(predicted)) + ((1 - expected) * np.log(1 - predicted)) * -1

def call_cross_entropy(predicted, expected):
    return np.array([cross_entropy(pred, exp) for pred, exp in zip(predicted, expected)])

def derivative_cross_entropy(predicted, expected):
    pass

def call_derivative_cross_entropy(predicted, expected):
    pass
