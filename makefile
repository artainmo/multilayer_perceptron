all: train

train:
	@python3 train.py

predict:
	@python3 predict.py

clean:
	rm correction/data_test.csv
	rm correction/data_training.csv
	rm -rf ./__pycache__
	rm -rf ./lib/MyNeuralNetwork/MyStats/__pycache__
	rm -rf ./lib/MyNeuralNetwork/MyStats/stats/__pycache__
	rm -rf ./lib/MyNeuralNetwork/__pycache__
	rm -rf ./documentation/.DS_Store
	rm -rf ./.DS_Store
	rm -rf ./datasets/.DS_Store

env:
	pip3 install numpy
	pip3 install pandas
	pip3 install matplotlib
	pip3 install seaborn
	#sudo apt-get install python3-tk
