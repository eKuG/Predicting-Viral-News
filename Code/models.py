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
import time
import pprint
import itertools
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

from data_utils import TrainTestSplit, NNTrainTestGraph

RANDOM_STATE = 0
VIS_ROOT = '../Visualizations'
CSV_ROOT = '../Model_Results'
DIVIDER = '================================================='
PP = pprint.PrettyPrinter(indent=4)

CLASS_CUTOFF = 1400

def LinearRegression(X, Y, grid):
	"""Linear regression classifier."""
	print(DIVIDER)
	# custom score function to turn regression into classification
	classify = lambda x: x >= CLASS_CUTOFF
	score = lambda x, y: sum([classify(_x) == classify(y[i]) for i, _x in enumerate(x)]) / float(len(x))
	linear_score = make_scorer(score, greater_is_better=True)

	# define model
	mLinear = linear_model.LinearRegression()
	# create cache
	cachedir = mkdtemp()
	# set-up pipeline: normalize, reduce components, model
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mLinear)],
					memory=cachedir)
	# grid search parameters
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), PCA(20, whiten=True),
			 PCA(40, whiten=True)],
			'model__normalize': [False, True]
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring=linear_score, n_jobs=1, verbose=2)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/linear_regression.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	# print best and loop through all grid results
	print("Linear Regression Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)

def LogisticRegression(X, Y, grid):
	"""Logistic regression classifier."""
	print(DIVIDER)
	# define model
	mLogistic = linear_model.LogisticRegression(solver='saga', n_jobs=-1)
	# create cache
	cachedir = mkdtemp()
	# set-up pipeline: normalize, reduce components, model
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mLogistic)],
					memory=cachedir)
	# grid search parameters
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), PCA(20, whiten=True),
				    PCA(40, whiten=True), RFE(mLogistic), RFE(mLogistic, 20)],
			'model__C': numpy.logspace(-4, 4, 10),
			'model__penalty': ['l1', 'l2']
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=1)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/logistic_regression.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("Logistic Regression Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)

def DecisionTree(X, Y, grid, save=True):
	"""Decision Tree classifier."""
	print(DIVIDER)
	# define model
	mDT = DecisionTreeClassifier(random_state=RANDOM_STATE)
	# create cache
	cachedir = mkdtemp()
	# set-up pipeline: normalize, reduce components, model
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mDT)],
					memory=cachedir)
	# grid search parameters
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), PCA(20, whiten=True),
				    PCA(40, whiten=True), RFE(mDT), RFE(mDT, 20)],
			'model__criterion': ['gini', 'entropy'],
			'model__max_depth': [None, 3, 5, 7, 10]
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=1)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/decision_tree.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("Decision Tree Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	if save:
		try:
			with open("tree.dot", "w") as f:
				f = export_graphviz(estimator.best_estimator_, out_file=f, feature_names=list(X),
					filled=True, rounded=True, special_characters=True)
			system('dot -Tpng tree.dot -o tree.png')
			system('mv tree.png {0}'.format(VIS_ROOT))
			system('mv tree.dot {0}'.format(VIS_ROOT))
		except Exception as e:
			print e
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)

def SVM(X, Y, grid):
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
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), PCA(20, whiten=True),
			 PCA(40, whiten=True)],
			'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
			'model__C': numpy.logspace(-4, 4, 5),
			'model__gamma': numpy.logspace(-4, 4, 5)
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=1)
	estimator.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/svm.csv'.format(CSV_ROOT))
	# prepare variables for printing
	means = 100 * estimator.cv_results_['mean_test_score']
	stds = 100 * estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("SVM Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	print(DIVIDER)
	# remove cache
	rmtree(cachedir)

def NeuralNet(X, Y, grid):
	"""
	Neural Network grid search.
	Due to Tensorflow pickling issues, the grid search is done manually.
	"""
	try:
		from keras.models import Sequential
		from keras.layers import Dense
		from keras.wrappers.scikit_learn import KerasClassifier
	except Exception:
		print('Keras import failed.')

	numpy.random.seed(RANDOM_STATE)

	def NN_dynamic(optimizer='adam', loss='binary_crossentropy',
				   num_hidden=1, hidden_layer_width=16, activation='relu'):
		# create model
		model = Sequential()
		model.add(Dense(hidden_layer_width, input_dim=58, kernel_initializer='normal', activation=activation))
		for hidden in range(num_hidden):
			model.add(Dense(hidden_layer_width, input_dim=hidden_layer_width,
						    kernel_initializer='normal', activation=activation))
		model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
		# Compile and return model
		model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
		return model

	S = StandardScaler().fit(X)
	_X = S.transform(X)

	optimizers = ['adam', 'rmsprop']
	losses = ['binary_crossentropy', 'mse']
	activation = ['relu', 'tanh']
	num_hidden = [1, 2, 3]
	hidden_widths = [16, 32, 64]

	BATCH_SIZE = 512
	EPOCHS = 25

	X_train, X_val, Y_train, Y_val = TrainTestSplit(_X, Y)

	val_len = len(X_train)

	NN_config = {}
	for index, (o, l, a, n, w) in enumerate(itertools.product(optimizers, losses, activation, num_hidden, hidden_widths)):
		print DIVIDER
		grid_config = {
			'optimizer': o,
			'loss': l,
			'activation': a,
			'num_hidden': n,
			'batch_size': BATCH_SIZE,
			'epochs': EPOCHS
		}
		PP.pprint(grid_config)
		N = NN_dynamic(optimizer=o, loss=l, num_hidden=n, hidden_layer_width=w, activation=a)
		_time = time.time()
		history = N.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
		NNTrainTestGraph(history, index)
		NN_config['fit_time'] = time.time() - _time
		val_scores = N.evaluate(X_val, Y_val)
		NN_config['val_eval_time'] = time.time() - _time - NN_config['fit_time']
		train_scores = N.evaluate(X_train, Y_train)
		NN_config['train_eval_time'] = time.time() - _time - NN_config['fit_time'] - NN_config['val_eval_time']
		print("\n%s: %.2f%%" % (N.metrics_names[1], val_scores[1]*100))
		grid_config['val_score'] = val_scores[1]*100
		grid_config['train_score'] = train_scores[1]*100
		NN_config[index] = grid_config
		print("NeuralNet Accuracy: %.2f%% (%.2f%%) with %r" % (val_scores[1]*100, val_scores[0]*100, grid_config))

	NN_df = pandas.DataFrame.from_dict(NN_config, orient='index')
	NN_df.to_csv('{0}/NN.csv'.format(CSV_ROOT))


