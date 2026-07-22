"""
shap_analysis.py
================

Utility functions for SHAP explainability.
"""

import shap
import matplotlib.pyplot as plt


def create_explainer(model):
    """
    Create a SHAP TreeExplainer for a tree-based model.

    Parameters
    ----------
    model : trained model
        Trained tree-based model (Random Forest, XGBoost, etc.)

    Returns
    -------
    shap.TreeExplainer
    """

    return shap.TreeExplainer(model)


def summary_plot(explainer, X):
    """
    Display SHAP summary plot.

    Parameters
    ----------
    explainer : shap.TreeExplainer

    X : pd.DataFrame
    """

    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        show=True,
    )


def bar_plot(explainer, X):
    """
    Display SHAP feature importance bar plot.

    Parameters
    ----------
    explainer : shap.TreeExplainer

    X : pd.DataFrame
    """

    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=True,
    )


def waterfall_plot(explainer, X, row_index=0):
    """
    Display SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    explainer : shap.TreeExplainer

    X : pd.DataFrame

    row_index : int
        Row to explain.
    """

    explanation = explainer(X.iloc[[row_index]])

    shap.plots.waterfall(
        explanation[0],
        show=True,
    )