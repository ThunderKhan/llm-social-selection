import hashlib
import json
from dataclasses import asdict

from .schema import AppConfig


def canonicalize(config: AppConfig) -> str:
    """Serialize a validated configuration to deterministic JSON."""
    return json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def hash_config(config: AppConfig) -> str:
    """Return the full SHA-256 hash of a validated configuration."""
    return hashlib.sha256(canonicalize(config).encode("utf-8")).hexdigest()
