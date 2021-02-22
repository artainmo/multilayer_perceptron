from train import *
import pandas as pn

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    x, y = get_x_y(path, list(range(2, 32)), 1)
    NN = MyNeuralNetwork(x, y, deep_layers=2, learning_rate=0.01, n_cycles=10, gradient_descend="mini-batch", b=18, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", feedback=True)
    NN.weights = pickle.load(open("saved/weights.pkl", 'rb', closefd=True))
    NN.bias = pickle.load(open("saved/bias.pkl", 'rb', closefd=True))
    predictions = NN.predict(x)
    if input("Evaluate predictions?(y/n):") == "y":
        evaluate(predictions, y)
    if input("Save predictions?(y/n):") == "y":
        try:
            pn.DataFrame(predictions, columns=["malignant", "benign"]).to_csv("saved/predictions.csv")
            print("Predictions saved in saved/predictions.csv")
        except:
            print("Error: save/predictions.csv does not exist")
