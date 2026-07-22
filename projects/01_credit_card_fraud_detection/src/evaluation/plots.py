"""
plots.py
========

Visualization utilities.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix.
    """

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC Curve.
    """

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("ROC Curve")
    plt.show()


def plot_feature_importance(feature_importance_df, top_n=15):
    """
    Plot feature importance.
    """

    df = (
        feature_importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        df["Feature"],
        df["Importance"],
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")

    plt.title(f"Top {top_n} Features")

    plt.show()