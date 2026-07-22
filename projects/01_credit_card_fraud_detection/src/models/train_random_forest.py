"""
Training utilities for Random Forest.
"""

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
    **kwargs,
):
    """
    Train a Random Forest model.

    Extra keyword arguments are passed directly to
    RandomForestClassifier.
    """

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **kwargs,
    )

    model.fit(X_train, y_train)

    return model