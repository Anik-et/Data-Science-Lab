from pathlib import Path
import yaml


def load_config():

    project_root = Path(__file__).resolve().parents[2]

    config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config