import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import sem

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline(steps=[
    ("scalar", StandardScaler()),
    ("classifier", LinearSVC())
])

parameter_grid = [
    {
        "classifier": [LinearSVC()],
        "classifier__loss": ['hinge', 'squared_hinge'],
        "classifier__C": [0.1, 1, 10]
    },
    {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [3, 5, 7],
        "classifier__weights": ['uniform', 'distance'],
        "classifier__metric": ['minkowski', 'euclidean', 'manhattan']
    },
    {
        "classifier": [SVC()],
        "classifier__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ['scale', 'auto']
    },
    {
        "classifier": [BaggingClassifier()],
        "classifier__bootstrap": [True, False],
        "classifier__bootstrap_features": [True, False],    
        "classifier__n_estimators": [5, 10, 15],
        "classifier__max_samples" : [0.6, 0.8, 1.0]
    },
    {
        "classifier": [DecisionTreeClassifier()],
        "classifier__criterion": ['entropy', 'gini'],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    },
    {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    },
    {
        "classifier": [ExtraTreesClassifier()],
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }
]

def train(training_data):
    """
    1. Reads in a labelled dataset. \n 
    2. Trains the models with that  dataset. \n
    3. Measure and return the accuracy of the model against that dataset.
    """

    print("Seperating/processing the dataset...")

    # Shuffle the dataset to prevent bias in the training.
    training_data = shuffle(training_data, random_state=42)

    # Seperate the input 'features' from the expect output labels.
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values

    print("Tunning, training and comparing models...")

    # Tune, train and compare models with varying parameters.
    grid_search = GridSearchCV(pipe, parameter_grid, n_jobs=-1, verbose=3, scoring='accuracy')
    grid_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Get the best trained model with the best parameters.
    return grid_search.best_estimator_

def test(model, testing_data):
    """
    1. Executes a model on the dataset to predict labels for each trace. \n
    2. Appends the labels to last column on the dataset. \n
    3. Returns the modifed dataset.
    """

    # Get the feature values.
    X = testing_data.values

    # Predict the labels using the model.
    y = model.predict(X)

    # Append the predicted labels to the last column on the dataset.
    testing_data['labels'] = y

    return testing_data

def main():

    # Read the training dataset from the training data file.
    training_data = pd.read_csv(sys.argv[1], header=None)

    # Read the testing dataset from the testing data file.
    testing_data = pd.read_csv(sys.argv[2], header=None)

    # Get a model that is optimised against the training dataset.
    model = train(training_data)

    # Predict and append labels for the testing dataset.
    test(model, testing_data)

    # Write the dataset with the predicted labels to the results data file.
    testing_data.to_csv(sys.argv[3], header=False, index=False)

main()