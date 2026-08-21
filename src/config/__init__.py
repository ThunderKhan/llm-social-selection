from pathlib import Path

from .canonical import canonicalize, hash_config
from .loader import load_yaml
from .schema import AppConfig
from .validator import ConfigError, validate_config


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a configuration file."""
    return validate_config(load_yaml(path))


__all__ = [
    "AppConfig",
    "ConfigError",
    "canonicalize",
    "hash_config",
    "load_config",
    "load_yaml",
    "validate_config",
]
