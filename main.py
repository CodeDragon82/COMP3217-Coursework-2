import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import sem

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

scaler = StandardScaler()

# Dictionary of selectable models where each model has a selection of tunable parameters.
models = {
    "linear": [ LinearSVC(), { 'loss': ['hinge', 'squared_hinge'],
                               'C': [0.1, 1, 10] }],

    "kneighbours": [KNeighborsClassifier(), { 'n_neighbors': [3, 5, 7],
                                              'weights': ['uniform', 'distance'],
                                              'metric': ['minkowski', 'euclidean', 'manhattan'] }],

    "svc": [SVC(), { 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'C': [0.1, 1, 10],
                     'gamma': ['scale', 'auto'] }],

    "bagging": [BaggingClassifier(), { 'bootstrap': [True, False],
                                       'bootstrap_features': [True, False],    
                                       'n_estimators': [5, 10, 15],
                                    'max_samples' : [0.6, 0.8, 1.0] }],

    "decisiontree": [DecisionTreeClassifier(), { 'criterion': ['entropy', 'gini'],
                                                 'max_depth': [None, 5, 10],
                                                 'min_samples_split': [2, 5, 10],
                                                 'min_samples_leaf': [1, 2, 4] }],

    "forest": [RandomForestClassifier(), { 'n_estimators': [100, 200, 500],
                                           'max_depth': [None, 5, 10],
                                           'min_samples_split': [2, 5, 10],
                                           'min_samples_leaf': [1, 2, 4] }],

    "eforest": [ExtraTreesClassifier(), { 'n_estimators': [100, 200, 500],
                                          'max_depth': [None, 5, 10],
                                          'min_samples_split': [2, 5, 10],
                                          'min_samples_leaf': [1, 2, 4] }]
}

def experiment(training_data):
    """
    Trains each available model on a training dataset and compares their accuracies on a plotted graph.
    """

    model_names = []
    trained_accuracies = []
    untrained_accuracies = []

    for [model, parameter_grid] in models.values():

        # Train and tune the model against the training dataset.
        (trained_accuracy, untrained_accuracy) = train(model, parameter_grid, training_data)

        model_names.append(model.__class__.__name__)
        trained_accuracies.append(trained_accuracy)
        untrained_accuracies.append(untrained_accuracy)


    plt.plot(model_names, trained_accuracies, marker='o', color='black', label="Accuracy on Training Data")
    plt.plot(model_names, untrained_accuracies, marker='s', color='black', label="Accuracy on Validation Data")

    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def train(model, parameter_grid, training_data):
    """
    1. Reads in a labelled dataset. \n 
    2. Trains the models with that  dataset. \n
    3. Measure and return the accuracy of the model against that dataset.
    """

    print("Seperating/processing the dataset...")

    # Seperate the input 'features' from the expect output labels.
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values

    # Normalise and transform data.
    X = scaler.fit_transform(X)
    X, y = shuffle(X, y, random_state=42)

    print("Model selected: ", model.__class__.__name__)

    print("Tunning and training the model...")

    # Tune, train and compare model with varying parameters.
    grid_search = GridSearchCV(model, parameter_grid, n_jobs=-1, verbose=3, scoring='accuracy')
    grid_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Get the best trained model with the best parameters.
    model = grid_search.best_estimator_

    # print("Calulating accuracy...")

    # # # Use the first part of the training data to validate the accuracy of the model on data it HAS seen.
    # trained_accuracy = model.score(X_1, y_1)

    # # # Use the second part of the training data to validate the accuracy of the model on data it HASN'T seen.
    # untrained_accuracy = model.score(X_2, y_2)

    return (trained_accuracy, untrained_accuracy)

def test(model, testing_data):
    """
    1. Executes a model on the dataset to predict labels for each trace. \n
    2. Appends the labels to last column on the dataset. \n
    3. Returns the modifed dataset.
    """

    # Get the feature values.
    X = testing_data.values

    # Normalise and transform the data.
    X = scaler.fit_transform(X)

    # Predict the labels using the model.
    y = model.predict(X)

    # Append the predicted labels to the last column on the dataset.
    testing_data['labels'] = y

    return testing_data

def main():
    mode = int(sys.argv[1])

    if (mode == 1):

        # Read the dataset from the training data file.
        training_data = pd.read_csv(sys.argv[3], header=None)

        # Read the dataset from the testing data file.
        testing_data = pd.read_csv(sys.argv[4], header=None)

        # Instantiate model.
        model = models[sys.argv[2]][0]
        parameter_grid = models[sys.argv[2]][1]

        # Train model with training data and calculate the model accuracy.
        training_accuracy, validation_accuracy = train(model, parameter_grid, training_data)

        print("Training Accuracy: ", training_accuracy)
        print("Validation Accuracy: ", validation_accuracy)

        # Test model with validating data.
        test(model, testing_data)

        # Write the dataset with the predicted labels to the results data file.
        testing_data.to_csv(sys.argv[5], header=False, index=False)

    elif (mode == 2):

        # Read the dataset from the training data file.
        training_data = pd.read_csv(sys.argv[2], header=None)

        # Compare accuracy of different models on training dataset and plot results.
        experiment(training_data)

    else:
        print("ERROR: Mode doesn't exist!")

main()