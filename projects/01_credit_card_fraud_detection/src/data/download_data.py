"""
Module: download_data.py

Purpose:
    Download the project dataset from Kaggle if it does not
    already exist locally.

Author: Aniket Mali
Project: Credit Card Fraud Detection
"""

from pathlib import Path
import subprocess

from src.utils.config import load_config


def _create_data_directory(path: Path) -> None:
    """
    Create the data directory if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def _dataset_exists(dataset_path: Path) -> bool:
    """
    Check whether the dataset already exists.

    Parameters
    ----------
    dataset_path : Path
        Full path of dataset.

    Returns
    -------
    bool
        True if dataset exists, else False.
    """
    return dataset_path.exists()


def download_dataset() -> Path:
    """
    Download the dataset from Kaggle if required.

    Returns
    -------
    Path
        Path to the downloaded dataset.
    """

    # -----------------------------
    # Load configuration
    # -----------------------------
    config = load_config()

    dataset = (
        f"{config['dataset']['owner']}/"
        f"{config['dataset']['name']}"
    )

    raw_path = Path(config["paths"]["raw_data"])

    filename = config["dataset"]["filename"]

    force_download = config["download"]["force_download"]

    dataset_path = raw_path / filename

    # -----------------------------
    # Create directory
    # -----------------------------
    _create_data_directory(raw_path)

    # -----------------------------
    # Check existing dataset
    # -----------------------------
    if _dataset_exists(dataset_path) and not force_download:
        print("Dataset already exists.")
        return dataset_path

    print("Downloading dataset from Kaggle...")

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(raw_path),
        "--unzip",
    ]

    subprocess.run(command, check=True)

    # -----------------------------
    # Validate download
    # -----------------------------
    if not _dataset_exists(dataset_path):
        raise FileNotFoundError(
            "Dataset download failed. CSV file not found."
        )

    print("Dataset downloaded successfully.")

    return dataset_path