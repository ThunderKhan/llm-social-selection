from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.e01 import analyze_e01_database  # noqa: E402
from src.analysis.plotting import generate_figures  # noqa: E402
from src.analysis.reporting import write_analysis_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze persisted E01 pilot evidence without model generation."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "experiments" / "e01" / "e01_full.sqlite",
    )
    parser.add_argument(
        "--task-artifact",
        type=Path,
        default=ROOT / "tasks" / "e01_validated_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "e01" / "analysis",
    )
    args = parser.parse_args()
    staging_dir = args.output_dir.with_name(f".{args.output_dir.name}.tmp")
    try:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        report, tables = analyze_e01_database(
            args.db,
            task_artifact_path=args.task_artifact,
            repository=ROOT,
        )
        figure_names = []
        if report["integrity"]["status"] == "PASS":
            figure_names = generate_figures(report, staging_dir / "figures")
        artifacts = write_analysis_bundle(
            staging_dir, report, tables, figure_names
        )
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        staging_dir.replace(args.output_dir)
    except (OSError, ValueError, sqlite3.Error, KeyError, TypeError, StopIteration) as error:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        print(f"E01 analysis failed: {error}", file=sys.stderr)
        return 1

    print(f"DATASET INTEGRITY: {report['integrity']['status']}")
    print(
        f"Verified: trials={report['integrity']['counts']['trials']}, "
        f"rounds={report['integrity']['counts']['rounds']}, "
        f"responses={report['integrity']['counts']['responses']}, "
        f"scores={report['integrity']['counts']['scores']}, "
        f"ballots={report['integrity']['counts']['ballots']}"
    )
    print(f"Decision: {report['recommendation']['decision']}")
    print(f"Report: {args.output_dir / 'e01_analysis.md'}")
    print(f"Artifacts: {len(artifacts) + len(figure_names)}")
    return 0 if report["integrity"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
