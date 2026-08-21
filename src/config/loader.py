from pathlib import Path
from typing import Any

import yaml

from .validator import ConfigError


def load_yaml(path: str | Path) -> Any:
    """Read and parse a UTF-8 YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"could not read configuration file {config_path}: {error}") from error

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        detail = str(error).splitlines()[0]
        raise ConfigError(f"invalid YAML in {config_path}: {detail}") from error
