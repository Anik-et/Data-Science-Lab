"""
load_data.py
============

Utility functions for loading datasets used in the Credit Card Fraud Detection
project.

Responsibilities
----------------
- Read datasets from disk
- Validate file existence
- Return pandas DataFrames

This module should NOT perform:
- Data cleaning
- Feature engineering
- Train-test split
- Scaling
"""

from pathlib import Path
import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the csv file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(file_path)


def load_processed_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load processed dataset.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    pd.DataFrame
    """

    return load_csv(file_path)