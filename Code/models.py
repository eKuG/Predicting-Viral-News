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

def LinearRegression(X, Y, grid):
	"""Linear regression classifier."""
	print(DIVIDER)
	# custom score function to turn regression into classification
	classify = lambda x: x >= CLASS_CUTOFF
	score = lambda x, y: sum([classify(_x) == classify(y[i]) for i, _x in enumerate(x)]) / float(len(x))
	# use pickle for parallel jobs
	pickled_score = pickle.dumps(score)
	score = pickle.loads(pickled_score)
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
				    PCA(40, whiten=True), RFE(mLinear), RFE(mLinear, 20)],
			'model__normalize': [False, True]
	}
	# define grid search and fit the values
	estimator = GridSearchCVProgressBar(pipe, grid, scoring=linear_score, n_jobs=-1, verbose=1)
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
			'pca': [None, PCA(20), PCA(40), PCA(20, whiten=True), PCA(40, whiten=True)],
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
	"""Neural Network baseline."""
	try:
		from keras.models import Sequential
		from keras.layers import Dense
		from keras.wrappers.scikit_learn import KerasClassifier
	except Exception:
		print('Keras import failed.')

	numpy.random.seed(RANDOM_STATE)

	def NN1(optimizer='adam', init='normal'):
		"""Baseline. Retain dimensionality."""
		# create model
		model = Sequential()
		model.add(Dense(58, input_dim=58, kernel_initializer=init, activation='relu'))
		model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return model

	def NN2(optimizer='adam', init='normal'):
		"""Reduction. Retain dimensionality then reduce dimensionality."""
		# create model
		model = Sequential()
		model.add(Dense(58, input_dim=58, kernel_initializer=init, activation='relu'))
		model.add(Dense(20, input_dim=58, kernel_initializer=init, activation='relu'))
		model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return model

	NN1_model = KerasClassifier(build_fn=NN1, verbose=0)
	NN2_model = KerasClassifier(build_fn=NN2, verbose=0)
	# grid search parameters
	grid = {'optimizer': ['rmsprop', 'adam'],
			'epochs': [50, 100, 150],
			'batch_size': [5, 10, 20],
			'init': ['glorot_uniform', 'normal', 'uniform']
	}
	# define grid search and fit the values
	estimator_NN1 = GridSearchCVProgressBar(NN1_model, grid, scoring='accuracy', n_jobs=-1, verbose=1)
	estimator_NN2 = GridSearchCVProgressBar(NN2_model, grid, scoring='accuracy', n_jobs=-1, verbose=1)
	estimator_NN1.fit(X.values, Y.values.ravel())
	estimator_NN2.fit(X.values, Y.values.ravel())
	# store the results of grid search in CSV
	best_df_NN1 = pandas.DataFrame.from_dict(estimator_NN1.cv_results_)
	best_df_NN2 = pandas.DataFrame.from_dict(estimator_NN2.cv_results_)
	best_df_NN1.to_csv('{0}/NN1.csv'.format(CSV_ROOT))
	best_df_NN2.to_csv('{0}/NN2.csv'.format(CSV_ROOT))
	# prepare variables for printing --> NN1
	means_NN1 = 100 * estimator_NN1.cv_results_['mean_test_score']
	stds_NN1 = 100 * estimator_NN1.cv_results_['std_test_score']
	params_NN1 = estimator_NN1.cv_results_['params']
	i_NN1 = estimator_NN1.best_index_
	print("NeuralNet Accuracy: %.2f%% (%.2f%%) with %r" % (means_NN1[i_NN1], stds_NN1[i_NN1], params_NN1[i_NN1]))
	# prepare variables for printing --> NN2
	means_NN2 = 100 * estimator_NN2.cv_results_['mean_test_score']
	stds_NN2 = 100 * estimator_NN2.cv_results_['std_test_score']
	params_NN2 = estimator_NN2.cv_results_['params']
	i_NN2 = estimator_NN2.best_index_
	print("NeuralNet Accuracy: %.2f%% (%.2f%%) with %r" % (means_NN2[i_NN2], stds_NN2[i_NN2], params_NN2[i_NN2]))
	print(DIVIDER)


