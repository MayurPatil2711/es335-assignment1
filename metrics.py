from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:

    assert y_hat.size == y.size  # Ensure that sizes of y_hat and y are equal.

    # Calculate accuracy
    correct_predictions = (y_hat == y).sum()
    total_samples = y.size
    accuracy = correct_predictions / total_samples

    return accuracy

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    # Ensure that sizes of y_hat and y are equal.
    assert y_hat.size == y.size

    # Calculate precision for the specified class
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_positive = ((y_hat == cls) & (y != cls)).sum()

    # Handling the special case where there are no positives are predicted, to avoid division by zero
    if (true_positive + false_positive) == 0:
        return 0.0

    precision_value = true_positive / (true_positive + false_positive)

    return precision_value

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    # Ensure that sizes of y_hat and y are equal.
    assert y_hat.size == y.size

    # Calculate recall for the specified class
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_negative = ((y_hat != cls) & (y == cls)).sum()

    # Handle the case where there are no actual positives to avoid division by zero
    if (true_positive + false_negative) == 0:
        return 0.0

    recall_value = true_positive / (true_positive + false_negative)

    return recall_value

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    # Ensure that sizes of y_hat and y are equal.
    assert y_hat.size == y.size
    assert y_hat.dtype in ['float','int']
    assert y.dtype in ['float','int']

    # Calculate squared differences between predicted and true values
    squared_errors = (y_hat - y) ** 2

    # Calculate mean squared error
    mse = squared_errors.mean()

    # Calculate root mean squared error
    rmse_value = np.sqrt(mse)

    return rmse_value

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    # Ensure that sizes of y_hat and y are equal.
    assert y_hat.size == y.size
    assert y_hat.dtype in ['float','int']
    assert y.dtype in ['float','int']

    # Calculate absolute differences between predicted and true values
    absolute_errors = np.abs(y_hat - y)

    # Calculate mean absolute error
    mae_value = absolute_errors.mean()

    return mae_value


