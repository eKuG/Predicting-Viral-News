"""
Models file for News Popularity Prediction Task.

Models:
- Linear Regression
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Neural Network 

Assignment: Final Project
Class: Data Mining | CSC 440
Programmer: Gregory D. Hunkins 
"""
import warnings
warnings.warn = lambda *args, **kwargs: None
import numpy
import pandas
import pickle
import dill
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm, linear_model
from pactools.grid_search import GridSearchCVProgressBar
from tempfile import mkdtemp
from shutil import rmtree
from os import system

from data_utils import TrainTestSplit

RANDOM_STATE = 0
VIS_ROOT = '../Visualizations'
CSV_ROOT = '../Model_Results'
DIVIDER = '================================================='

CLASS_CUTOFF = 1400


def SVM_I(X, Y, grid):
	"""SVM classifier."""
	print(DIVIDER)
	# define model
	mSVM = svm.SVC(random_state=RANDOM_STATE)
	# create cache
	cachedir = mkdtemp()
	# set-up pipeline: normalize, reduce components, model
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mSVM)],
					memory=cachedir)
	# grid search parameters
	grid = {'scale': [None],
			'pca': [None],
			'model__kernel': ['rbf', 'linear'],
			'model__C': numpy.logspace(-4, 4, 5),
			'model__gamma': numpy.logspace(-4, 4, 5)
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=2)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/svm_I.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("SVM Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)

def SVM_II(X, Y, grid):
	"""SVM classifier."""
	print(DIVIDER)
	# define model
	mSVM = svm.SVC(random_state=RANDOM_STATE, kernel='rbf')
	# create cache
	cachedir = mkdtemp()
	# set-up pipeline: normalize, reduce components, model
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mSVM)],
					memory=cachedir)
	# grid search parameters
	grid = {'scale': [RobustScaler(), StandardScaler()],
			'pca': [None, PCA(20), PCA(40), PCA(40, whiten=True)]}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=2)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/svm_II.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("SVM Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)

def SVM_III(X, Y, grid):
	"""SVM classifier."""
	print(DIVIDER)
	# define model
	mSVM = svm.SVC(random_state=RANDOM_STATE)
	# create cache
	cachedir = mkdtemp()
	# set-up pipeline: normalize, reduce components, model
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mSVM)],
					memory=cachedir)
	# grid search parameters
	grid = {'scale': [None],
			'pca': [None],
			'model__kernel': ['poly', 'sigmoid'],
			'model__C': numpy.logspace(-4, 4, 5),
			'model__gamma': numpy.logspace(-4, 4, 5)
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=2)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/svm_III.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("SVM Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)




