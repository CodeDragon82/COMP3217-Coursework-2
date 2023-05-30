import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import sem

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

scaler = StandardScaler()

# Dictionary of selectable models.
models = {
    "linear": LinearSVC(),
    "kneighbours": KNeighborsClassifier(),
    "svclinear":  SVC(kernel='linear'),
    "svcpoly": SVC(kernel='poly'),
    "svcrbf": SVC(kernel='rbf'),
    "svcsigmoid": SVC(kernel='sigmoid'),
    "bagging":  BaggingClassifier(),
    "decisiontree": DecisionTreeClassifier(), 
    "forest": RandomForestClassifier(),
    "eforest": ExtraTreesClassifier()
}

def experiment(training_data):
    model_names = []
    training_accuracy_means = []
    training_accuracy_errors = []
    validation_accuracy_means = []
    validation_accuracy_errors = []

    for name, model in models.items():
        training_accuracies = []
        validation_accuracies = []

        for i in range(0, 10):
            (trained_accuracy, untrained_accuracy) = train(model, training_data)
            training_accuracies.append(trained_accuracy)
            validation_accuracies.append(untrained_accuracy)

        model_names.append(name)
        training_accuracy_means.append(mean(training_accuracies))
        training_accuracy_errors.append(sem(training_accuracies))
        validation_accuracy_means.append(mean(validation_accuracies))
        validation_accuracy_errors.append(sem(validation_accuracies))

    plt.errorbar(model_names, training_accuracy_means, training_accuracy_errors, fmt='o', color='black', label='Training Accuracy')
    plt.errorbar(model_names, validation_accuracy_means, validation_accuracy_errors, fmt='s', color='black', label='Validation Accuracy')

    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def train(model, training_data):
    """
    1. Reads in a labelled dataset. \n 
    2. Trains the models with that  dataset. \n
    3. Measure and return the accuracy of the model against that dataset.
    """

    # Seperate the input 'features' from the expect output labels.
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values

    # Normalise and transform data.
    X = scaler.fit_transform(X)

    # Seperate the training data into 2 parts.
    X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=0.9)

    # Use the first part of the training data to train the model.
    model.fit(X_1, y_1)

    # Use the first part of the training data to validate the accuracy of the model on data it HAS seen.
    y_predicted = model.predict(X_1)
    trained_accuracy = accuracy_score(y_1, y_predicted)

    # Use the second part of the training data to validate the accuracy of the model on data it HASN'T seen.
    y_predicted = model.predict(X_2)
    untrained_accuracy = accuracy_score(y_2, y_predicted)

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
        model = models[sys.argv[2]]

        # Train model with training data.
        train(model, training_data)

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