# COMP3217 Coursework 2
 Utilising machine learning to detect attacks against a power grid.

## Executing Program
Execute the following command to train a model against a labelled dataset and generate labels for a unlabelled dataset:

`python3 main.py <training data file> <testing data file> <results data file>`

## Manually Selecting a Model
The sklearn models that are trained and evaluated by the `GridSearchCV` are defined in `parameter_grid`. This can be modified to only focus on a specific model by commenting out unwanted models.