"""
metrics.py
==========

Utility functions for evaluating classification models.
"""

from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test) -> Dict:
    """
    Evaluate a classification model.

    Parameters
    ----------
    model : trained estimator
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
        ),
    }

    return metrics


def metrics_to_dataframe(metrics: Dict) -> pd.DataFrame:
    """
    Convert metrics dictionary into a DataFrame.
    """

    return pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC AUC",
            ],
            "Value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
            ],
        }
    )