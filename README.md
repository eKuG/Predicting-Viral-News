# :fire: Predicting-Viral-News :fire:

Investigation in `Python` of predicting the virality of a news article based on its meta-data. Uses the [UCI dataset](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) from the paper below.

`K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.`

## Running the Code

#### Installing Requirements

`Keras`
`Numpy`
`Pandas`
`Scikit Learn`

You can install the requirements via:

`pip install -r requirements.txt`

For faster training of the Neural Net, I recommend using a GPU. Create a new `Anaconda` environment and install all the requirements _except_ for Keras. Then run: 

`conda install -c anaconda keras-gpu`

If you need help getting started with `Anaconda`: [Getting Started With Anaconda](https://conda.io/docs/user-guide/getting-started.html)

#### Run :running:

To run the best parameters simply run:

`python Code/main.py`

To re-run the grid search, run:

`python Code/main.py --grid`

## Models

### Linear Regression

`Cross-Validation Accuracy:` **X.XX%**

`Best Parameters:`

### Logistic Regression

`Cross-Validation Accuracy:` **X.XX%**

`Best Parameters:`

### Decision Tree

`Cross-Validation Accuracy:` **X.XX%**

`Best Parameters:`

### Support Vector Machine

`Cross-Validation Accuracy:` **X.XX%**

`Best Parameters:`

### Neural Network

`Cross-Validation Accuracy:` **X.XX%**

`Best Parameters:`



