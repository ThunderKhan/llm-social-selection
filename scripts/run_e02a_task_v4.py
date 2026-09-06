from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_e02a_task_profile_revision import main  # noqa: E402


if __name__ == "__main__":
    user_args = sys.argv[1:]
    sys.argv = [
        sys.argv[0],
        "--candidates",
        str(ROOT / "tasks" / "e02a_candidates_v4.json"),
        "--validated-output",
        str(ROOT / "tasks" / "e02a_validated_v4.json"),
        "--output-dir",
        str(ROOT / "experiments" / "e02a_task_v4"),
        *user_args,
    ]
    raise SystemExit(main())
