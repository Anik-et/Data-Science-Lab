"""
comparison.py
=============

Utilities for comparing multiple classification models.
"""

import pandas as pd

from .metrics import evaluate_model


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compare multiple trained models.

    Parameters
    ----------
    models : dict
        Dictionary of {"Model Name": trained_model}

    X_test : pd.DataFrame

    y_test : pd.Series

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """

    results = []

    for model_name, model in models.items():

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )

    return pd.DataFrame(results)