"""
Training utilities for XGBoost.
"""

from xgboost import XGBClassifier



def train_xgboost(
    X_train,
    y_train,
    **kwargs,
):
    """
    Train an XGBoost classifier.
    """

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        **kwargs,
    )

    model.fit(X_train, y_train)

    return model