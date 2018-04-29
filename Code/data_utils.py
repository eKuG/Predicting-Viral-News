"""
File I/O.

Assignment: Final Project
Class: Data Mining | CSC 440
Programmer: Gregory D. Hunkins 
"""
import pandas
from sklearn.model_selection import train_test_split

FILEPATH = "../OnlineNewsPopularity/OnlineNewsPopularity.csv"

NON_PREDICTIVE_COLS = ['url', 'timedelta', 'shares']
TARGET_COL = ['shares']

def read_clean_data():
    """Load and return the Online News Popularity Dataset."""
    full_data = clean_cols(pandas.read_csv(FILEPATH))
    X = full_data[[x for x in list(full_data) if x not in NON_PREDICTIVE_COLS]]
    Y = full_data[TARGET_COL]
    return X, Y

def clean_cols(data):
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)

def BinaryY(Y):
    """Encode the Y vector for ML work."""
    Y['shares'] = Y['shares'].map(lambda x: 1 if x >= 1400 else 0)
    return Y

def TrainTestSplit(X, Y, R=0, test_size=0.2):
    return train_test_split(X, Y, test_size=test_size, random_state=R)




