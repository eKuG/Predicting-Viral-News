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
import os
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
from sklearn.ensemble import (RandomForestClassifier,
                              BaggingClassifier,
                              AdaBoostClassifier)
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

def RandomForest(X, Y, grid):
    print(DIVIDER)
    # define model
    mRF = RandomForestClassifier(random_state=RANDOM_STATE)
    # create cache
    cachedir = mkdtemp()
    # set-up pipeline: normalize, reduce components, model
    pipe = Pipeline(steps=[('scale', None),
                           ('pca', None),
                           ('model', mRF)],
                    memory=cachedir)
    # grid search parameters
    grid = {#'scale': [None, StandardScaler()],
            #'pca': [None, PCA(20), PCA(40)],
            'scale': [None, StandardScaler()],
            'pca': [None, PCA(40)],
            'model__n_estimators': [10],
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [None, 5, 10]
    }
    # define grid search and fit the values
    estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=1)
    estimator.fit(X.values, Y.values.ravel())
    # store the results of grid search in CSV
    best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
    best_df.to_csv('{0}/random_forest.csv'.format(CSV_ROOT))
    # prepare variables for printing
    means = 100 * estimator.cv_results_['mean_test_score']
    stds = 100 * estimator.cv_results_['std_test_score']
    params = estimator.cv_results_['params']
    i = estimator.best_index_
    print("Decision Tree Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
    print(DIVIDER)
    # remove cache
    rmtree(cachedir)


def Bagging(X, Y, grid):
    print(DIVIDER)
    # define model
    mBag = BaggingClassifier()
    # create cache
    cachedir = mkdtemp()
    # set-up pipeline: normalize, reduce components, model
    pipe = Pipeline(steps=[('scale', None),
                           ('pca', None),
                           ('model', mBag)],
                    memory=cachedir)
    # grid search parameters
    grid = {'scale': [None, StandardScaler()],
            'pca': [None, PCA(40)],
            'model__base_estimator': [DecisionTreeClassifier(max_depth=5)],
                                      #linear_model.LogisticRegression(penalty='l1'),
                                      #svm.SVC()],
            'model__n_estimators': [10, 20],
    }
    # define grid search and fit the values
    estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=2)
    estimator.fit(X.values, Y.values.ravel())
    # store the results of grid search in CSV
    best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
    best_df.to_csv('{0}/bagging.csv'.format(CSV_ROOT))
    # prepare variables for printing
    means = 100 * estimator.cv_results_['mean_test_score']
    stds = 100 * estimator.cv_results_['std_test_score']
    params = estimator.cv_results_['params']
    i = estimator.best_index_
    print("Bagging Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
    print(DIVIDER)
    # remove cache
    rmtree(cachedir)

def Boost(X, Y, grid):
    print(DIVIDER)
    # define model
    mBoost = AdaBoostClassifier()
    # create cache
    cachedir = mkdtemp()
    # set-up pipeline: normalize, reduce components, model
    pipe = Pipeline(steps=[('scale', None),
                           ('pca', None),
                           ('model', mBag)],
                    memory=cachedir)
    # grid search parameters
    grid = {'scale': [None, StandardScaler()],
            'pca': [None, PCA(40)],
            'model__base_estimator': [DecisionTreeClassifier(max_depth=5)],
                                      #linear_model.LogisticRegression(penalty='l1'),
                                      #svm.SVC()],
            'model__n_estimators': [10, 20],
    }
    # define grid search and fit the values
    estimator = GridSearchCVProgressBar(pipe, grid, scoring='accuracy', n_jobs=-1, verbose=2)
    estimator.fit(X.values, Y.values.ravel())
    # store the results of grid search in CSV
    best_df = pandas.DataFrame.from_dict(estimator.cv_results_)
    best_df.to_csv('{0}/bagging.csv'.format(CSV_ROOT))
    # prepare variables for printing
    means = 100 * estimator.cv_results_['mean_test_score']
    stds = 100 * estimator.cv_results_['std_test_score']
    params = estimator.cv_results_['params']
    i = estimator.best_index_
    print("Bagging Best Results: %.2f%% (%.2f%%) with %r" % (means[i], stds[i], params[i]))
    print(DIVIDER)
    # remove cache
    rmtree(cachedir)


def NeuralNet(X, Y, grid):
    """
    Neural Network grid search.
    Due to Tensorflow pickling issues, the grid search is done manually.
    """
    try:
        import tensorflow as tf
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        from keras.optimizers import Adam, SGD, RMSprop
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.callbacks import EarlyStopping
        from keras.wrappers.scikit_learn import KerasClassifier
        print('Imports successful.')
    except Exception:
        print('Keras import failed.')

    with tf.device('/gpu:0'):
        numpy.random.seed(RANDOM_STATE)

        optimizer_dropout = {
            'sgd': SGD(lr=0.1, momen tum=0.9, decay=1e-6),
            'adam': Adam(lr=0.01, decay=1e-6),
            'rmsprop': RMSprop(lr=0.01, decay=1e-6)
        }

        def NN_dynamic(optimizer='adam', loss='binary_crossentropy', dropout=False,
                       hidden=(32), activation='relu'):
            # optimizer if dropout
            if dropout:
                optimizer = optimizer_dropout[optimizer]
            # create model
            model = Sequential()
            model.add(Dense(hidden[0], input_dim=58, kernel_initializer='normal', activation=activation))
            for i, h in enumerate(hidden[1:-1]):
                model.add(Dense(hidden[i+1], input_dim=h,
                                kernel_initializer='normal', activation=activation))
                if dropout:
                    model.add(Dropout(0.2))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            # Compile and return model
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model

        S = StandardScaler().fit(X)
        _X = S.transform(X)

        optimizers = ['adam', 'rmsprop', 'sgd']
        losses = ['binary_crossentropy', 'mse']
        activation = ['relu', 'tanh']
        hidden = [[58],
                  [32],
                  [16],
                  [58, 58],
                  [58, 32],
                  [58, 16],
                  [32, 32],
                  [32, 16],
                  [16, 16],
                  [58, 58, 32],
                  [58, 58, 16],
                  [58, 32, 32],
                  [58, 32, 16],
                  [32, 32, 16],
                  [32, 16, 16],
                  [16, 16, 16]]
        batch_size = [64, 512]
        dropout = [True, False]

        BATCH_SIZE = 512
        EPOCHS = 100

        # define 10-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cvscores = []

        # make history folder
        if not os.path.isdir("./history"):
            os.system('mkdir history')

        NN_config = {}
        scores = [0]
        num_configs = len(list(itertools.product(optimizers, losses, activation,
                                                 hidden, dropout, batch_size)))
        for index, (o, l, a, h, d, b) in enumerate(itertools.product(optimizers, losses, activation,
                                                                     hidden, dropout, batch_size)):
            grid_config = {
                'optimizer': o,
                'loss': l,
                'activation': a,
                'hidden': h,
                'batch_size': b,
                'dropout': d,
                'max_epochs': EPOCHS
            }
            # print 
            print DIVIDER
            # get the dynamic neural net
            N = NN_dynamic(optimizer=o, loss=l, hidden=h, activation=a)
            # set callbacks
            callbacks = [] if dropout else [EarlyStopping(monitor='val_acc', patience=10, mode='max')]
            # initialize cv scores
            cvscores = []
            cvconfigs = []
            for i, (train, test) in enumerate(kfold.split(_X, Y)):
                # initialize
                cvconfig = {}
                _time = time.time()
                v_data = (_X[test], Y.values[test].ravel())
                # fit the data
                history = N.fit(_X[train], Y.values[train].ravel(), epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=v_data, verbose=0, callbacks=callbacks)
                hist_dict = {k: history.history[k][-1] for k in history.history}
                cvconfig = dict(cvconfig, **hist_dict)
                # save full history
                os.chdir('history')
                with open('history_grid{0}_cv{1}.pickle'.format(index, i), 'wb') as file:
                    pickle.dump(history.history, file)
                os.chdir('..')
                # save meta-history
                grid_config['val_acc_{0}'.format(i)] = cvconfig['val_acc']*100
                grid_config['train_acc_{0}'.format(i)] = cvconfig['acc']*100
                # append config
                cvconfigs.append(cvconfig)

            # get average of cvscores, cvconfigs
            N = float(len(cvconfigs))
            cvconfig = {k: sum(c[k] for c in cvconfigs)/N for k in cvconfigs[0]}
            # merge 
            grid_config = dict(grid_config, **cvconfig)
            # get acc pct
            acc_pct = grid_config['val_acc'] * 100
            tra_pcc = grid_config['acc'] * 100
            # add dictionary to overall history
            NN_config[index] = grid_config
            print("Finished running {0} out of {1} configs.".format(index, num_configs))
            print("NeuralNet Accuracy: %.2f%% (Train: %.2f%%)." % (acc_pct, tra_pcc))
            PP.pprint(grid_config)
            try:
                if acc_pct > max(scores):
                    print('NEW MAX: {0}'.format(acc_pct))
                else:
                    print ('MAX: {0}'.format(max(scores)))
            except:
                pass
            scores.append(acc_pct)

        try:
            NN_df = pandas.DataFrame.from_dict(NN_config, orient='index')
            NN_df.to_csv('{0}/NN.csv'.format(CSV_ROOT))
        except Exception:
            with open('{0}/NN.txt'.format(CSV_ROOT), 'w') as file:
                file.write(pickle.dumps(NN_config))


