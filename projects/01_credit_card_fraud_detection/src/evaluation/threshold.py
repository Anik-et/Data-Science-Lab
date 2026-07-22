"""
threshold.py
============

Utilities for evaluating binary classifiers at different probability thresholds.
"""

import numpy as np
import pandas as pd

import sklearn.metrics

def predict_with_threshold(
    model,
    X,
    threshold=0.5,
):
    """
    Predict classes using a custom threshold.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions


def evaluate_threshold(
    model,
    X_test,
    y_test,
    threshold=0.5,
):
    """
    Evaluate model performance at a specific threshold.
    """

    y_pred = predict_with_threshold(
        model,
        X_test,
        threshold,
    )

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    return {
        "Threshold": threshold,
        "Precision": sklearn.metrics.precision_score(y_test, y_pred),
        "Recall": sklearn.metrics.recall_score(y_test, y_pred),
        "F1": sklearn.metrics.f1_score(y_test, y_pred),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def threshold_report(
    model,
    X_test,
    y_test,
    thresholds=np.arange(0.1, 1.0, 0.1),
):
    """
    Evaluate multiple thresholds.
    """

    results = []

    for threshold in thresholds:

        results.append(
            evaluate_threshold(
                model,
                X_test,
                y_test,
                threshold,
            )
        )

    return pd.DataFrame(results)


def find_best_threshold(
    model,
    X_test,
    y_test,
    metric: str = "F1",
    thresholds=None,
):
    """
    Find the threshold that maximizes a chosen metric.

    Parameters
    ----------
    model : trained classifier

    X_test : pd.DataFrame

    y_test : pd.Series

    metric : str
        One of:
        - "Precision"
        - "Recall"
        - "F1"

    thresholds : iterable, optional
        Threshold values to evaluate.
        Defaults to np.arange(0.01, 1.00, 0.01)

    Returns
    -------
    best_threshold : float

    best_score : float

    results_df : pd.DataFrame
    """

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            model,
            X_test,
            y_test,
            threshold,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. "
            f"Choose one of: Precision, Recall, F1."
        )

    best_row = results_df.loc[results_df[metric].idxmax()]

    return (
        best_row["Threshold"],
        best_row[metric],
        results_df,
    )