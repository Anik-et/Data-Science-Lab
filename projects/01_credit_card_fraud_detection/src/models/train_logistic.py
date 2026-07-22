"""
Training utilities for Logistic Regression.
"""

from sklearn.linear_model import LogisticRegression


def train_logistic(
    X_train,
    y_train,
    class_weight=None,
    max_iter=50000,
    random_state=42,
):
    """
    Train a Logistic Regression model.
    """

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model