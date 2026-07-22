"""
Prediction utilities.
"""

import pandas as pd

from .model_utils import load_model


def predict(model, X: pd.DataFrame):
    """
    Return class predictions.
    """

    return model.predict(X)


def predict_probability(model, X: pd.DataFrame):
    """
    Return fraud probabilities.
    """

    return model.predict_proba(X)[:, 1]


def load_and_predict(
    model_path,
    X: pd.DataFrame,
):
    """
    Load a saved model and perform prediction.
    """

    model = load_model(model_path)

    return predict(model, X)