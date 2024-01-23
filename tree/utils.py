import pandas as pd
import numpy as np
from scipy.special import xlogy


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    total_entropy = entropy(Y)
    weighted_entropy = 0
    for value in attr.unique():
        subset = Y[attr == value]
        subset_entropy = entropy(subset)
        subset_probability = subset.shape[0] / Y.shape[0]
        weighted_entropy += subset_probability * subset_entropy
    return total_entropy - weighted_entropy


def variance(data: np.array) -> float:
    # this function is used to calculate impurity in numerical dtype column
    if len(data[:, -1]) == 0:  # leaf node no further splitting possible.
        std = 0
    else:
        std = np.mean((data[:, -1] - np.mean(data[:, -1])) ** 2)
    return std


def entropy(data: np.array) -> float:
    probabilities = np.unique(data[:, -1], return_counts=True)[1] / np.unique(data[:, -1], return_counts=True)[1].sum()
    entropy_ = sum(xlogy(probabilities, probabilities))
    return entropy_


def gini_index(data: np.array) -> float:
    probabilities = np.unique(data[:, -1], return_counts=True)[1] / np.unique(data[:, -1], return_counts=True)[1].sum()
    gini = sum(probabilities * (1 - probabilities))  # formula
    return gini


def impurity(left_node, right_node, criterion) -> float:
    # this function finds the impurity of left and right nodes
    total_length = (len(left_node) + len(right_node))
    weight_left_node = len(left_node) / total_length
    weight_right_node = len(right_node) / total_length
    weighted_impurity = (weight_left_node * criterion(left_node) + weight_right_node * criterion(right_node))
    # here criterion can be entropy or gini_index
    return weighted_impurity


def find_feature_type(df):
    # this function finds the type of feature, i.e., discrete or numerical/real
    feature_types = []
    features = df.columns
    for i in range(len(features) - 1):
        one_value = df.iloc[:, i].unique()[0]
        # checks whether one_value is string or the number of unique values of that column are less than 5 (assumed)
        if (isinstance(one_value, str)) or (len(df.iloc[:, i].unique()) <= 5):
            # then termed as discrete
            feature_types.append("discrete")
        else:
            feature_types.append("numerical")
    return feature_types
