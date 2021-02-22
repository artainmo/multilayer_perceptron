all: train

train:
	@python3 train.py

predict:
	@python3 predict.py

clean:
	rm -rf ./__pycache__
	rm -rf ./lib/MyNeuralNetwork/MyStats/__pycache__
	rm -rf ./lib/MyNeuralNetwork/MyStats/stats/__pycache__
	rm -rf ./lib/MyNeuralNetwork/__pycache__
