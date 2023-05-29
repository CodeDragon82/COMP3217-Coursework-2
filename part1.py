import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()

# Dictionary of selectable models.
models = {
    "linear": LinearSVC(),
    "kneighbours": KNeighborsClassifier(n_neighbors=3),
    "svclinear":  SVC(kernel='linear'),
    "svcpoly": SVC(kernel='poly'),
    "svcrbf": SVC(kernel='rbf'),
    "svcsigmoid": SVC(kernel='sigmoid'),
    "forest": RandomForestClassifier(n_estimators=100)
}


def train(model, training_data_file):
    """
    1. Reads in a labelling dataset. \n 
    2. and then measures the accuracy of the model against that dataset.\n
    """

    data = pd.read_csv(training_data_file, header=None)

    # Seperate the input 'features' from the expect output labels.
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 
    X = scaler.fit_transform(X)

    # Seperate the training data into 2 parts.
    X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=0.9)

    # Use the first part of the training data to train the model.
    model.fit(X_1, y_1)

    # Use the first part of the training data to validate the accuracy of the model on data it HAS seen.
    y_predicted = model.predict(X_1)
    print('Accuracy on trained data: ', accuracy_score(y_1, y_predicted))

    # Use the second part of the training data to validate the accuracy of the model on data it HASN'T seen.
    y_predicted = model.predict(X_2)
    print('Accuracy on untrained data: ', accuracy_score(y_2, y_predicted))

    # print('Precision: ', precision_score(y_2, y_predicted))
    # print('Recall: ', recall_score(y_2, y_predicted))

def test(model, testing_data_file, results_data_file):
    """
    1. Reads in an unlabelled dataset. \n
    2. Executes a model on that dataset to predict labels for each trace. \n
    3. Writes the dataset with the predicted labels to a new file.
    """

    data = pd.read_csv(testing_data_file, header=None)

    X = data.values
    X = scaler.fit_transform(X)

    y = model.predict(X)

    data['labels'] = y

    data.to_csv(results_data_file, header=False, index=False)

def main():
    # Instantiate model.
    model = models[sys.argv[1]]

    # Train model with training data.
    train(model, sys.argv[2])

    # Test model with validating data.
    test(model, sys.argv[3], sys.argv[4])

main()