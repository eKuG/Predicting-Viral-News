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
import pprint
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm, linear_model
from os import system

from data_utils import TrainTestSplit

RANDOM_STATE = 0
SAVE_ROOT = '../Visualizations/'

CLASS_CUTOFF = 1400

def LinearRegression(X, Y):
	"""Linear regression classifier."""
	classify = lambda x: x >= CLASS_CUTOFF
	score = lambda x, y: sum([classify(_x) == classify(y[i]) for i, _x in enumerate(x)]) / float(len(x))

	linear_score = make_scorer(score, greater_is_better=True)

	mLinear = linear_model.LinearRegression()

	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mLinear)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(5), PCA(10), PCA(15), PCA(20), RFE(mLinear), RFE(mLinear, 10)]
	}
	estimator = GridSearchCV(pipe, grid, scoring=linear_score)
	estimator.fit(X.values, Y.values.ravel())
	print "Best Parameters:", estimator.best_params_
	print "Best Score:", estimator.best_score_
	print estimator.predict(X.head(20))
	print Y.head(20).values.ravel()
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
	results = cross_val_score(estimator.best_estimator_, X.values, Y.values.ravel(), scoring=linear_score, cv=kfold)
	print("Linear Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def LogisticRegression(X, Y):
	"""Logistic regression classifier."""
	mLogistic = linear_model.LogisticRegression(solver='sag', n_jobs=-1)
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mLogistic)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(30), RFE(mLogistic)],
			'model__C': numpy.logspace(-4, 4, 3),
			'model__penalty': ['l1', 'l2']
	}
	estimator = GridSearchCV(pipe, grid, scoring='accuracy')
	estimator.fit(X.values, Y.values.ravel())
	print "Best Parameters:"
	pprint.pprint(estimator.best_params_)
	kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
	results = cross_val_score(estimator.best_estimator_, X.values, Y.values.ravel(), scoring='accuracy', cv=kfold)
	print("Logistic Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def DecisionTree(X, Y, save=True):
	"""Decision Tree classifier."""
	mDT = DecisionTreeClassifier(random_state=RANDOM_STATE)
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mDT)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40)],
			'model__criterion': ['gini', 'entropy'],
			'model__max_depth': [None, 3, 5, 10]
	}
	estimator = GridSearchCV(pipe, grid, scoring='accuracy')
	estimator.fit(X.values, Y.values.ravel())
	print "Best Parameters:"
	pprint.pprint(estimator.best_params_)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
	results = cross_val_score(estimator.best_estimator_, X.values, Y.values.ravel(), scoring='accuracy', cv=kfold)
	print("Decision Tree Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	if save:
		with open("tree.dot", "w") as f:
			f = export_graphviz(estimator.best_estimator_, out_file=f, feature_names=list(X),
				filled=True, rounded=True, special_characters=True)
		system('dot -Tpng tree.dot -o tree.png')
		system('open tree.png')

def SVM(X, Y):
	"""SVM classifier."""
	mSVM = svm.SVC(random_state=RANDOM_STATE)
	pipe = Pipeline(steps=[('scale', None),
						   ('pca', None),
						   ('model', mSVM)])
	grid = {'scale': [None, StandardScaler(), RobustScaler()],
			'pca': [None, PCA(20), PCA(40), RFE()],
			'model__kernel': ['rbf', 'entropy'],
			'model__max_depth': [None, 3, 5, 10]
	}
	estimator = GridSearchCV(pipe, grid, scoring='accuracy')
	estimator.fit(X.values, Y.values.ravel())
	print "Best Parameters:"
	pprint.pprint(estimator.best_params_)
	mSVM.fit(X.values, Y.values.ravel())
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
	results = cross_val_score(mSVM, X.values, Y.values.ravel(), scoring='accuracy', cv=kfold)
	print("SVM Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

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
