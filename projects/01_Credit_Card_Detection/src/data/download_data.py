from pathlib import Path
import subprocess


DATASET = "mlg-ulb/creditcardfraud"

RAW_PATH = Path("data/raw")

def download_dataset():

    """
    Download dataset if it does not already exist.

    Returns
    -------
    Path
        Location of downloaded dataset.
    """

    pass

def download_dataset():

    RAW_PATH.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(RAW_PATH),
        "--unzip"
    ])

    print("Dataset Downloaded")