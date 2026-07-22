"""
preprocess.py
=============

Data preprocessing utilities.

Responsibilities
----------------
- Train/Test Split
- Feature Scaling
- SMOTE Oversampling

This module should NOT:
- Train models
- Evaluate models
- Save models
"""

from typing import Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


def split_data(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split dataframe into train and test sets.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    X = df.drop(columns=[target_column])

    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_amount(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Standardize the Amount column.

    Returns
    -------
    X_train
    X_test
    scaler
    """

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["Amount"] = scaler.fit_transform(
        X_train[["Amount"]]
    )

    X_test["Amount"] = scaler.transform(
        X_test[["Amount"]]
    )

    return X_train, X_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
):
    """
    Apply SMOTE only on training data.

    Returns
    -------
    Resampled X_train and y_train.
    """

    smote = SMOTE(
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train,
    )

    return X_resampled, y_resampled