# :fire: Predicting-Viral-News :fire:

Investigation in `Python` of predicting the virality of a news article based on its meta-data. Uses the [UCI dataset](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) from the paper below. A top accuracy of **67%** is achieved via a **Neural Network** which is a competitive result (original paper baseline: 67%, current state-of-the-art: 69%).

`K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.`

The full report can be viewed [**here**](https://github.com/ghunkins/Predicting-Viral-News/blob/master/Online_News_Popularity_Hunkins.pdf).

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

Navigate into the `Code` directory.

To run the full grid search, run:

`python main.py`

## Correlation Matrix

Correlation matrix computed via Pearson's coefficient is shown below. 

<img src="https://raw.githubusercontent.com/ghunkins/Predicting-Viral-News/master/Visualizations/correlation.png" alt="baseline">

## Results

Below is a table summary of the results. **Bolded** values represented best-in-class results for the experiments provided. 

<img src="https://raw.githubusercontent.com/ghunkins/Predicting-Viral-News/master/Visualizations/results.png" alt="baseline">

## Model Grid Search Summary

Below is a table summary of the results. **Bolded** values represented best-in-class results for the experiments provided. 


<img src="https://raw.githubusercontent.com/ghunkins/Predicting-Viral-News/master/Visualizations/crossvalid.png" alt="baseline">







