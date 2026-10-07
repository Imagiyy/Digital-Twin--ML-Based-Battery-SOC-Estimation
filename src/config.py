"""Configuration loader and schema definitions for the SOC Digital Twin."""

from pathlib import Path
from typing import Any, Dict, List
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Global singleton configuration
CONFIG = load_config()


def get_path(path_key: str) -> Path:
    """Resolve a relative path from the config to an absolute path."""
    rel_path = CONFIG["paths"].get(path_key, "")
    return ROOT_DIR / rel_path
