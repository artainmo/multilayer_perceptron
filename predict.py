from train import *
import pandas as pn

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    x, y = get_x_y(path, list(range(2, 32)), 1)
    NN = load_neural_network("saved/neural_network.pkl")
    predictions = NN.predict(x)
    if input("Evaluate predictions?(y/n):") == "y":
        print("Cross entropy", end=" ")
        print(NN.cost()) #demanded for correction
        evaluate(predictions, y)
    if input("Save predictions?(y/n):") == "y":
        try:
            pn.DataFrame(predictions, columns=["malignant", "benign"]).to_csv("saved/predictions.csv")
            print("Predictions saved in saved/predictions.csv")
        except:
            print("Error: save/predictions.csv does not exist")
