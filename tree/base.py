from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pprint as pprint
from tree.utils import entropy, variance, information_gain, gini_index, impurity, find_feature_type

np.random.seed(42)


@dataclass
class DecisionTree:
    output_type: Literal["discrete", "numerical"]
    criterion: Literal["entropy", "gini_index"]  # criterion won't be used for regression
    node: dict()
    max_depth: int = 10  # The maximum depth the tree can grow to

    def __init__(self, output_type: Literal["discrete", "numerical"], criterion: Literal["entropy", "gini_index"],
                 max_depth: int):
        self.output_type = output_type
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X) -> None:
        self.node = self.make_tree(train_data=X, output_type=self.output_type, criterion=self.criterion,
                                   max_depth=self.max_depth)
        return None

    def predict(self, X) -> pd.Series:
        n = len(X.columns)
        predictions = []
        for i in range(len(X)):
            predictions.append(self.Output(X.iloc[i, :n], self.node))
        answers = pd.Series(predictions)
        return answers

    def leaf_node_value(self, data, output_type) -> float:
        output_column = data[:, -1]
        if output_type == "numerical":
            # regression (takes mean of datapoints)
            leaf = np.mean(output_column)
        else:
            # classification (does mode of datapoints)
            leaf = np.unique(output_column, return_counts=True)[0][
                np.unique(output_column, return_counts=True)[1].argmax()]
        return leaf

    def find_all_splits(self, data) -> dict:
        # finds all the unique values of each column in the dataset which are taken as splits
        all_splits = {}
        for column in range(data.shape[1] - 1):
            all_splits[column] = np.unique(data[:, column])
        return all_splits

    def split_node_values(self, data, split_feature, split_point):
        # this function splits the given column into right and left node values based on numerical and discrete
        split_feature_column = data[:, split_feature]
        feature_type = Feature_Type[split_feature]  # first to find the feature type of splitting column

        if feature_type == "numerical":
            left_node = data[split_feature_column <= split_point]
            right_node = data[split_feature_column > split_point]
        else:  # feature is discrete
            left_node = data[split_feature_column == split_point]
            right_node = data[split_feature_column != split_point]
        return left_node, right_node

    def max_gain_split(self, data, all_splits, output_type, criterion):
        # this function is used to determine the best split at particular node: for each value in each column it
        # splits the data and calculates impurity for left and right split data.
        global weighted_impurity, min_weighted_impurity, max_gain_column, max_gain_value
        non_empty_data = 1  # handling empty data
        for column in all_splits:  # iterating over all possible features.
            for value in all_splits[column]:  # iterating over all possible splits
                left_data, right_data = self.split_node_values(data, split_feature=column, split_point=value)

                if output_type == "numerical":
                    weighted_impurity = impurity(left_data, right_data, criterion=variance)
                else:  # discrete
                    if criterion == "entropy":
                        weighted_impurity = impurity(left_data, right_data, criterion=entropy)
                    elif criterion == "gini_index":
                        weighted_impurity = impurity(left_data, right_data, criterion=gini_index)

                # this part of code reduces impurity, and stores column number and splitting value in global variables
                if non_empty_data == 1 or weighted_impurity <= min_weighted_impurity:
                    non_empty_data = 0
                    min_weighted_impurity = weighted_impurity
                    max_gain_column = column
                    max_gain_value = value
        return max_gain_column, max_gain_value

    def make_tree(self, train_data, output_type, criterion=None, level=0, max_depth=4):
        if level == 0:
            global Attribute, Feature_Type
            Attribute = train_data.columns
            Feature_Type = find_feature_type(train_data)
            data = train_data.values
        else:
            data = train_data
        # base cases
        # i.e when the node has just one category, or when it has reached the maximum depth
        if (len(np.unique(data[:, -1])) == 1) or (level == max_depth):
            # find leaf node value and return the tree
            leaf = self.leaf_node_value(data, output_type)
            return leaf
        else:
            level += 1
            # find all possible splits
            all_splits = self.find_all_splits(data)
            # find split with the highest gain
            split_feature, split_point = self.max_gain_split(data, all_splits, output_type, criterion)
            # divide the dataset on basis of split feature and split point
            left_node, right_node = self.split_node_values(data, split_feature, split_point)
            # if data is empty i.e we have reached the leaf, then simply find the value of leaf node and return that.
            if len(left_node) == 0 or len(right_node) == 0:
                leaf = self.leaf_node_value(data, output_type)
                return leaf
            # make subtrees based on the split_feature and value
            feature_name = Attribute[split_feature]
            feature_type = Feature_Type[split_feature]
            # numerical feature
            if feature_type == "numerical":
                comparision = "{} <= {}".format(feature_name, split_point)
            # discrete feature
            else:
                comparision = "{} = {}".format(feature_name, split_point)
            # make a subtree
            sub_tree = {comparision: []}

            # create splits, by recursion.
            left_sub_tree = self.make_tree(left_node, output_type, criterion, level, max_depth)
            right_sub_tree = self.make_tree(right_node, output_type, criterion, level, max_depth)

            # if there is no decrease in entropy on splitting the data, then there is no need to split the data.
            if left_sub_tree == right_sub_tree:
                sub_tree = right_sub_tree
            else:
                sub_tree[comparision].append(left_sub_tree)
                sub_tree[comparision].append(right_sub_tree)
            return sub_tree

    def Output(self, input, node):
        # base case: if a node, doesn't have any child, return node
        if not isinstance(node, dict):
            return node
        comparison = list(node.keys())[0]
        attribute, comparison_operator, point = comparison.split(" ")
        if comparison_operator == "<=":  # feature is numerical
            if input[attribute] <= float(point):
                answer = node[comparison][0]
            else:
                answer = node[comparison][1]
        else: # feature is categorical
            if str(input[attribute]) == point:
                answer = node[comparison][0]
            else:
                answer = node[comparison][1]
        return self.Output(input, answer)