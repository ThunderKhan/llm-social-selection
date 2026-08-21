from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .models import Provenance


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_code_commit(repository: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_provenance(
    *,
    provider_name: str,
    model_name: str,
    code_commit: str | None = None,
    repository: str | Path | None = None,
    created_at: str | None = None,
) -> Provenance:
    return Provenance(
        code_commit=(
            code_commit
            if code_commit is not None
            else discover_code_commit(repository)
        ),
        python_version=platform.python_version(),
        platform=platform.platform(),
        provider_name=provider_name,
        model_name=model_name,
        created_at=created_at or utc_now(),
    )
