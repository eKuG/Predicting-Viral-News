"""
Main file for News Popularity Prediction Task.

Assignment: Final Project
Class: Data Mining | CSC 440
Programmer: Gregory D. Hunkins 
"""
import os
import argparse
from data_utils import read_clean_data, BinaryY
from models import (LinearRegression, LogisticRegression, 
					DecisionTree, SVM, NeuralNet)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(grid):
	# Get Clean Data
	X, Y = read_clean_data()
	Y_binary = BinaryY(Y)
	# Linear Regression
	try:
		pass
		#LinearRegression(X, Y, grid)
	except Exception as e:
		print e
	# Logistic Regression
	try:
		pass
		#LogisticRegression(X, Y_binary, grid)
	except Exception as e:
		print e
	# Decision Tree
	try:
		pass
		#DecisionTree(X, Y_binary, grid)
	except Exception as e:
		print e
	# Support Vector Machine
	try:
		SVM(X, Y_binary, grid)
	except Exception as e:
		print e
	#NeuralNet(X, Y_binary, grid)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", help="Enable grid search.", action="store_true")
	args = parser.parse_args()
	main(args.grid)
