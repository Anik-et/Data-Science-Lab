"""
Configuration Loader

Purpose:
    Load project configuration from config/config.yaml
"""

from pathlib import Path
import yaml


def load_config():
    """
    Load the project configuration.

    Returns
    -------
    dict
        Parsed configuration from config.yaml.
    """

    project_root = Path(__file__).resolve().parents[2]

    config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config