"""
File I/O.

Assignment: Final Project
Class: Data Mining | CSC 440
Programmer: Gregory D. Hunkins 
"""
import pandas
import numpy
import os
import matplotlib
if 'scratch' in os.path.abspath('.'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

FILEPATH = "../OnlineNewsPopularity/OnlineNewsPopularity.csv"

NON_PREDICTIVE_COLS = ['url', 'timedelta', 'shares']
TARGET_COL = ['shares']
XY_IGNORE_COLS = ['url', 'timedelta']

def read_clean_data():
    """Load and return the Online News Popularity Dataset."""
    full_data = clean_cols(pandas.read_csv(FILEPATH))
    X = full_data[[x for x in list(full_data) if x not in NON_PREDICTIVE_COLS]]
    Y = full_data[TARGET_COL]
    return X, Y

def readXYcombined():
    """Read in X, Y combined in a single matrix."""
    full_data = clean_cols(pandas.read_csv(FILEPATH))
    XY = full_data[[x for x in list(full_data) if x not in XY_IGNORE_COLS]]
    return XY

def clean_cols(data):
    """Clean the column names by stripping and lowercase."""
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)

def BinaryY(Y):
    """Encode the Y vector for ML work."""
    Y['shares'] = Y['shares'].map(lambda x: 1 if x >= 1400 else 0)
    return Y

def TrainTestSplit(X, Y, R=0, test_size=0.2):
    """Easy Train Test Split call."""
    return train_test_split(X, Y, test_size=test_size, random_state=R)

def PCAVisual(X, Y):
    """Visualize the dataset with PCA in 2-D and 3-D."""
    Y = BinaryY(Y)


def CorrelationMatrix(X=None):
    """Plotting the correlation matrix.
    Reference: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    """
    if X is None:
        X = readXYcombined()
    correlations = X.corr()
    names = list(X)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    font = {'family' : 'normal',
            'size'   : 6}
    matplotlib.rc('font', **font)
    plt.xticks(rotation=90)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


def NNTrainTestGraph(history, index):
    """
    Plot.
    Reference: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    """
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('NN_{0}_acc.png'.format(index))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('NN_{0}_loss.png'.format(index))





