from train import *


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    x, y = get_x_y(path, list(range(2, 32)), 1)
    NN = MyNeuralNetwork(x, y, deep_layers=2, learning_rate=0.01, n_cycles=50, gradient_descend="batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", feedback=False)
    NN.weights = pickle.load(open("saved/weights.pkl", 'rb'))
    NN.bias = pickle.load(open("saved/bias.pkl", 'rb'))
    print(NN.predict(x))
    print(y)
