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
	# Get Clean Data
	X, Y = read_clean_data()
	Y_binary = BinaryY(Y)
	# Linear Regression
	try:
		LinearRegression(X, Y)
	except Exception as e:
		print e
	# Logistic Regression
	try:
		LogisticRegression(X, Y_binary)
	except Exception as e:
		print e
	# Decision Tree
	try:
		DecisionTree(X, Y_binary)
	except Exception as e:
		print e
	# Support Vector Machine
	try:
		SVM(X, Y_binary)
	except Exception as e:
		print e
	#NeuralNet(X, Y_binary)

if __name__ == "__main__":
	main()
