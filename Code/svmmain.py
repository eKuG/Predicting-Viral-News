"""
Main file for News Popularity Prediction Task.

Assignment: Final Project
Class: Data Mining | CSC 440
Programmer: Gregory D. Hunkins 
"""
import os
import argparse
from data_utils import read_clean_data, BinaryY
from svm import SVM_I, SVM_II, SVM_III

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(svmI, svmII, svmIII):
	# Get Clean Data
	X, Y = read_clean_data()
	Y_binary = BinaryY(Y)
	if svmI:
		SVM_I(X, Y_binary, None)
	if svmII:
		SVM_II(X, Y_binary, None)
	if svmIII:
		SVM_III(X, Y_binary, None)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--svmI", help="Run SVM I.", action="store_true")
	parser.add_argument("--svmII", help="Run SVM II.", action="store_true")
	parser.add_argument("--svmIII", help="Run SVM III.", action="store_true")
	args = parser.parse_args()

	main(args.svmI, args.svmII, args.svmIII)
