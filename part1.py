import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

TRAINING_DATA_FILE = 'TrainingDataBinary.csv'
TESTING_DATA_FILE = 'TestingDataBinary.csv'
RESULTS_DATA_FILE = 'TestingResultsBinary.csv'

scaler = StandardScaler()

def train(model):
    data = pd.read_csv(TRAINING_DATA_FILE, header=None)

    # Seperate the input 'features' from the expect output labels.
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X = scaler.fit_transform(X)

    # Seperate the training data into 2 parts.
    X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=0.7)

    # Use the first part of the training data to train the model.
    model.fit(X_1, y_1)

    # Use the second part of the training data to validate the accuracy of the model.
    y_predicted = model.predict(X_2)
    print('Accuracy: ', accuracy_score(y_2, y_predicted))
    print('Precision: ', precision_score(y_2, y_predicted))
    print('Recall: ', recall_score(y_2, y_predicted))

def test(model):
    data = pd.read_csv(TESTING_DATA_FILE, header=None)

    X = data.values
    X = scaler.fit_transform(X)

    y = model.predict(X)

    data['labels'] = y

    data.to_csv(RESULTS_DATA_FILE, header=False, index=False)

# Instantiate model.
model = LinearSVC()

# Train model with training data.
train(model)

# Test model with validating data.
test(model)