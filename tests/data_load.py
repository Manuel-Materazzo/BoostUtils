import os
import re

import pandas as pd


def load_regression_data():
    # Load the data
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/../resources/regression_unit_test.csv'
    data = pd.read_csv(file_path)
    # standardize column names
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    # Remove rows with missing target, separate target from predictors
    pruned_data = data.dropna(axis=0, subset=['SalePrice'])
    y = pruned_data['SalePrice']
    X = pruned_data.drop(['SalePrice'], axis=1)
    return X, y

def load_classification_data():
    # Load the data
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/../resources/classification_unit_test.csv'
    data = pd.read_csv(file_path)
    # standardize column names
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    # Remove rows with missing target, separate target from predictors
    pruned_data = data.dropna(axis=0, subset=['SalePrice'])
    y = pruned_data['SalePrice']
    X = pruned_data.drop(['SalePrice'], axis=1)
    return X, y
