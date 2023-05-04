import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

scaler = StandardScaler()

def train(model, data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X = scaler.fit_transform(X)

    model.fit(X, y)

def test(model, data):
    X = data.iloc[:, :-1].values
    y_expected = data.iloc[:, -1].values

    X = scaler.fit_transform(X)

    y_actual = model.predict(X)

    accuracy = accuracy_score(y_expected, y_actual)
    print('Accuracy: ', accuracy)

# Load in dataset.
data = pd.read_csv('TrainingDataBinary.csv')

# Seperate dataset into training and testing data.
training_data, testing_data = train_test_split(data, train_size=0.7)

# Instantiate linear SVC model.
model = LinearSVC()

# Train model with training data.
train(model, training_data)

# Test model with testing data.
test(model, testing_data)