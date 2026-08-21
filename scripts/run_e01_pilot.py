from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.e01 import (  # noqa: E402
    E01_DEFAULT_MASTER_SEED,
    E01_DEFAULT_MODEL,
    E01Error,
    build_e01_plan,
    build_ollama_provider,
    run_e01,
)
from src.models import ModelProviderError  # noqa: E402
from src.storage import StorageError  # noqa: E402
from src.tasks.calibration import TaskSetValidationError  # noqa: E402
from src.tournament import TrialError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable E01 real-model pilot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the 18-round smoke plan")
    mode.add_argument("--full", action="store_true", help="Run the 300-round pilot plan")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--model", default=E01_DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--master-seed", type=int, default=E01_DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--stop-after-commits",
        type=int,
        help="Stop cleanly after N new commits; intended for resume verification",
    )
    args = parser.parse_args()
    selected_mode = "full" if args.full else "smoke"
    if selected_mode == "full" and not args.confirm_full_run:
        parser.error("--full requires --confirm-full-run")
    if args.confirm_full_run and selected_mode != "full":
        parser.error("--confirm-full-run is valid only with --full")
    database_path = args.db or (
        ROOT / "experiments" / "e01" / f"e01_{selected_mode}.sqlite"
    )
    task_path = ROOT / "tasks" / "e01_validated_v1.json"

    try:
        plan = build_e01_plan(
            mode=selected_mode,
            database_path=database_path,
            task_path=task_path,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            master_seed=args.master_seed,
        )
        provider = build_ollama_provider(plan)

        def check_provider() -> str:
            version = provider.check_health()
            provider.ensure_model_available()
            return version

        outcome = run_e01(
            plan,
            provider,
            repository=ROOT,
            check_provider=check_provider,
            stop_after_commits=args.stop_after_commits,
        )
    except KeyboardInterrupt:
        print(
            "Resume with the same command; only complete committed rounds will be reused.",
            file=sys.stderr,
        )
        return 130
    except (E01Error, ModelProviderError, StorageError, TaskSetValidationError, TrialError) as error:
        print(f"E01 runner failed: {error}", file=sys.stderr)
        return 1

    print(f"Summary: {plan.summary_path}")
    if outcome.controlled_stop:
        print("Resume by rerunning the command without --stop-after-commits.")
        return 75
    print(
        f"E01 {selected_mode} complete: {outcome.committed_total}/"
        f"{outcome.total_rounds} rounds; new commits={outcome.committed_this_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
