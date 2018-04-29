"""
Main file for News Popularity Prediction Task.

Assignment: Final Project
Class: Data Mining | CSC 440
Programmer: Gregory D. Hunkins 
"""
import os
from data_utils import read_clean_data, BinaryY
from models import (LinearRegression, LogisticRegression, 
					DecisionTree, SVM, NeuralNet)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main():
	X, Y = read_clean_data()
	Y_binary = BinaryY(Y)
	#LinearRegression(X, Y)
	LogisticRegression(X, Y_binary)
	#DecisionTree(X, Y_binary)
	#SVM(X, Y)
	#NeuralNet(X, Y)

if __name__ == "__main__":
	main()
