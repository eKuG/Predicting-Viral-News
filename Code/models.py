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
import numpy
import pandas
import pickle
import dill
import warnings
try:
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.wrappers.scikit_learn import KerasClassifier
except Exception as e:
	print('Keras import failed.')
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm, linear_model
from os import system

from data_utils import TrainTestSplit

warnings.warn = lambda *args, **kwargs: None

RANDOM_STATE = 0
VIS_ROOT = '../Visualizations'
CSV_ROOT = '../Model_Results'
DIVIDER = '================================================='

CLASS_CUTOFF = 1400

def LinearRegression(X, Y):
	"""Linear regression classifier."""
	print(DIVIDER)
	classify = lambda x: x >= CLASS_CUTOFF
	score = lambda x, y: sum([classify(_x) == classify(y[i]) for i, _x in enumerate(x)]) / float(len(x))
	pickled_score = pickle.dumps(score)
	score = pickle.loads(pickled_score)

	linear_score = make_scorer(score, greater_is_better=True)

	mLinear = linear_model.LinearRegression()

	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mLinear)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), RFE(mLinear), RFE(mLinear, 10)]
	}
	estimator = GridSearchCV(pipe, grid, scoring=linear_score, n_jobs=-1)
	estimator.fit(X.values, Y.values.ravel())
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/linear_regression.csv'.format(CSV_ROOT))
	means = 100*estimator.cv_results_['mean_test_score']
	stds = 100*estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("Linear Regression: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	for mean, stdev, param in zip(means, stds, params):
	    print("%.2f%% (%.2f%%) with: %r" % (mean, stdev, param))
	print(DIVIDER)

def LogisticRegression(X, Y):
	"""Logistic regression classifier."""
	print(DIVIDER)
	mLogistic = linear_model.LogisticRegression(solver='saga', n_jobs=-1)
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mLogistic)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), RFE(mLogistic), RFE(mLogistic, 10)],
			'model__C': numpy.logspace(-4, 4, 5),
			'model__penalty': ['l1', 'l2']
	}
	estimator = GridSearchCV(pipe, grid, scoring='accuracy', n_jobs=-1)
	estimator.fit(X.values, Y.values.ravel())
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/logistic_regression.csv'.format(CSV_ROOT))
	means = 100*estimator.cv_results_['mean_test_score']
	stds = 100*estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("Logistic Regression: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	for mean, stdev, param in zip(means, stds, params):
	    print("%.2f%% (%.2f%%) with: %r" % (mean, stdev, param))
	print(DIVIDER)

def DecisionTree(X, Y, save=True):
	"""Decision Tree classifier."""
	print(DIVIDER)
	mDT = DecisionTreeClassifier(random_state=RANDOM_STATE)
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mDT)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), RFE(mDT), RFE(mDT, 10)],
			'model__criterion': ['gini', 'entropy'],
			'model__max_depth': [None, 3, 5, 10]
	}
	estimator = GridSearchCV(pipe, grid, scoring='accuracy', n_jobs=-1)
	estimator.fit(X.values, Y.values.ravel())
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/decision_tree.csv'.format(CSV_ROOT))
	means = 100*estimator.cv_results_['mean_test_score']
	stds = 100*estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("Decision Tree: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	for mean, stdev, param in zip(means, stds, params):
	    print("%.2f%% (%.2f%%) with: %r" % (mean, stdev, param))
	if save:
		with open("{0}/tree.dot".format(VIS_ROOT), "w") as f:
			f = export_graphviz(estimator.best_estimator_, out_file=f, feature_names=list(X),
				filled=True, rounded=True, special_characters=True)
		system('dot -Tpng {0}/tree.dot -o {0}/tree.png'.format(VIS_ROOT))
		system('open {0}/tree.png'.format(VIS_ROOT))
	print(DIVIDER)

def SVM(X, Y):
	"""SVM classifier."""
	print(DIVIDER)
	mSVM = svm.SVC(random_state=RANDOM_STATE)
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mSVM)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), RFE(mSVM)],
			'model__kernel': ['rbf', 'entropy'],
			'model__C': numpy.logspace(-4, 4, 5),
			'model__gamma': numpy.logspace(-4, 4, 5)
	}
	estimator = GridSearchCV(pipe, grid, scoring='accuracy', n_jobs=-1)
	estimator.fit(X.values, Y.values.ravel())
	best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
	best_df.to_csv('{0}/svm.csv'.format(CSV_ROOT))
	means = 100*estimator.cv_results_['mean_test_score']
	stds = 100*estimator.cv_results_['std_test_score']
	params = estimator.cv_results_['params']
	i = estimator.best_index_
	print("SVM: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
	for mean, stdev, param in zip(means, stds, params):
	    print("%.2f%% (%.2f%%) with: %r" % (mean, stdev, param))
	print(DIVIDER)

def NeuralNet(X, Y):
	"""Neural Network baseline."""
	numpy.random.seed(RANDOM_STATE)

	def NN1():
		"""Baseline. Retain dimensionality."""
		# create model
		model = Sequential()
		model.add(Dense(58, input_dim=58, kernel_initializer='normal', activation='relu'))
		model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	def NN2():
		"""Reduction. Retain dimensionality then reduce dimensionality."""
		# create model
		model = Sequential()
		model.add(Dense(58, input_dim=58, kernel_initializer='normal', activation='relu'))
		model.add(Dense(20, input_dim=58, kernel_initializer='normal', activation='relu'))
		model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


	pipeline = KerasClassifier(build_fn=NN1, epochs=10, batch_size=5, verbose=2)
	kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
	results = cross_val_score(pipeline, X.values, Y.values, cv=kfold, scoring='accuracy', verbose=2)
	print("NeuralNet Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
