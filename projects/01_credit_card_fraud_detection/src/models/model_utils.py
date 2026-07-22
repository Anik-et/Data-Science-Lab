import joblib
import json
from pathlib import Path


def save_model(model, filepath):
    """
    Save a trained model to disk.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a saved model.
    """
    return joblib.load(filepath)


def save_metadata(metadata, filepath):
    """
    Save metadata (dictionary) as JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)