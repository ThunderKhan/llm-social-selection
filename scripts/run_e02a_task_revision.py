from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.e02a import (
    EvidenceCheckpoint,
    choose_task_shortlist,
    protocol_hash,
    sha256_file,
    summarize_holdout,
    write_csv,
)
from src.experiments.e02a_revision import (
    E02A_TASK_REVISION_PROTOCOL,
    evaluate_task_revision_readiness,
    neutral_task_prompt,
)
from src.models import ModelProviderError, OllamaProvider, OllamaTimeoutError
from src.scoring import normalize_answer
from src.seeding import derive_seed
from src.tasks.calibration import load_task_set, output_format_valid, task_set_json

EXPECTED_E01_SHA256 = "2a3847218d47820c13647e3622773fa3e57ff081608e4842707f1bbb479d424a"
BASE_SEED = 20260906


def run_inference(
    *,
    checkpoint: EvidenceCheckpoint,
    provider: OllamaProvider,
    evaluation_id: str,
    phase: str,
    task,
    participant_index: int,
    prompt: str,
    seed: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    existing = checkpoint.get(evaluation_id)
    if existing is not None:
        return existing
    agent, _, _ = neutral_task_prompt(task, participant_index)
    record: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "phase": phase,
        "request_id": f"e02a-v2-request-{evaluation_id}",
        "agent_id": agent.agent_id,
        "task_id": task.task_id,
        "seed": seed,
        **metadata,
    }
    try:
        output = provider.generate(
            agent=agent,
            task=task,
            prompt=prompt,
            request_id=record["request_id"],
            seed=seed,
        )
    except OllamaTimeoutError as error:
        record.update(status="error", error_type="timeout", error_message=str(error))
    except ModelProviderError as error:
        record.update(
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    else:
        record.update(
            status="ok",
            raw_output=output.content,
            provider_name=output.provider_name,
            model_name=output.model_name,
            effective_seed=output.seed,
            finish_reason=output.finish_reason,
            latency_ms=output.latency_ms,
            token_count=output.token_count,
        )
    return checkpoint.append(record)


def derive_fields(rows: list[dict[str, Any]], tasks_by_id: dict[str, Any]) -> None:
    for row in rows:
        if row.get("status") != "ok":
            continue
        task = tasks_by_id[row["task_id"]]
        row["correct"] = normalize_answer(row["raw_output"]) == normalize_answer(
            task.expected_answer or ""
        )
        row["format_valid"] = output_format_valid(task, row["raw_output"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only the E02A objective-task calibration with the same versioned "
            "profile/task prompt renderer used by experimental trials."
        )
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "tasks" / "e02a_candidates_v2.json",
    )
    parser.add_argument(
        "--validated-output",
        type=Path,
        default=ROOT / "tasks" / "e02a_validated_v2r1.json",
    )
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=ROOT / "experiments" / "e02a" / "e02a_report.json",
    )
    parser.add_argument(
        "--e01-database",
        type=Path,
        default=ROOT / "experiments" / "e01" / "e01_full.sqlite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "e02a_task_v2",
    )
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    artifact = load_task_set(args.candidates)
    if artifact.model_used_for_validation != args.model:
        parser.error("candidate model_used_for_validation does not match --model")
    if not args.prior_report.exists():
        parser.error(
            "prior E02A report not found; run scripts/run_e02a.py once before this revision"
        )
    prior_report = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if prior_report.get("scope", {}).get("apparatus_validation_only") is not True:
        parser.error("prior report is not an E02A apparatus-validation report")

    e01_before = sha256_file(args.e01_database)
    if e01_before != EXPECTED_E01_SHA256:
        parser.error(
            f"E01 SHA-256 mismatch: expected {EXPECTED_E01_SHA256}, found {e01_before}"
        )

    settings = {
        "protocol_version": E02A_TASK_REVISION_PROTOCOL,
        "candidate_sha256": sha256_file(args.candidates),
        "prior_report_protocol_hash": prior_report["protocol"]["protocol_hash"],
        "e01_sha256": e01_before,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "num_predict": 32,
        "calibration_attempts_per_task": 16,
        "holdout_attempts_per_task": 8,
        "prompt_path": "src.agents.render_prompt with empty neutral profile parameters",
        "seed": BASE_SEED,
    }
    run_protocol_hash = protocol_hash(settings)
    checkpoint = EvidenceCheckpoint(
        args.output_dir / "e02a_task_v2_raw_evidence.jsonl", run_protocol_hash
    )
    provider = OllamaProvider(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=0,
        num_predict=32,
    )
    try:
        ollama_version = provider.check_health()
        provider.ensure_model_available()
    except ModelProviderError as error:
        print(f"E02A task revision failed before inference: {error}", file=sys.stderr)
        return 1

    started = perf_counter()
    print(
        f"E02A task revision {run_protocol_hash[:12]} using Ollama {ollama_version}"
    )
    print("Phase 1/2: experiment-aligned candidate calibration")

    tasks_by_id = {task.task_id: task for task in artifact.tasks}
    for task_index, task in enumerate(artifact.tasks):
        for attempt in range(16):
            _, _, prompt = neutral_task_prompt(task, attempt)
            evaluation_id = f"task-calibration-{task.task_id}-a{attempt:02d}"
            run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=evaluation_id,
                phase="task_calibration",
                task=task,
                participant_index=attempt,
                prompt=prompt,
                seed=derive_seed(
                    BASE_SEED,
                    task_index,
                    "e02a_task_calibration_v2",
                    task.task_id,
                    str(attempt),
                ),
                metadata={"attempt": attempt, "family": task.family},
            )

    calibration_rows = checkpoint.phase("task_calibration")
    derive_fields(calibration_rows, tasks_by_id)
    shortlist, candidate_diagnostics, task_rejections = choose_task_shortlist(
        artifact.tasks, calibration_rows
    )

    print(f"Phase 2/2: fresh holdout for {len(shortlist)} shortlisted tasks")
    for shortlist_index, task in enumerate(shortlist):
        for participant_index in range(8):
            _, _, prompt = neutral_task_prompt(task, 100 + participant_index)
            evaluation_id = f"task-holdout-{task.task_id}-a{participant_index}"
            run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=evaluation_id,
                phase="task_holdout",
                task=task,
                participant_index=100 + participant_index,
                prompt=prompt,
                seed=derive_seed(
                    BASE_SEED,
                    shortlist_index,
                    "e02a_fresh_holdout_v2",
                    task.task_id,
                    str(participant_index),
                ),
                metadata={"agent_index": participant_index, "family": task.family},
            )

    holdout_rows = checkpoint.phase("task_holdout")
    derive_fields(holdout_rows, tasks_by_id)
    final_tasks, holdout_summary, holdout_details = summarize_holdout(
        shortlist, holdout_rows
    )

    e01_after = sha256_file(args.e01_database)
    all_rows = list(checkpoint.records.values())
    provider_requests = len(all_rows)
    provider_failures = sum(row.get("status") != "ok" for row in all_rows)
    failure_reasons = Counter(
        row.get("error_type", "unknown")
        for row in all_rows
        if row.get("status") != "ok"
    )

    readiness = evaluate_task_revision_readiness(
        final_task_count=len(final_tasks),
        holdout_summary=holdout_summary,
        task_provider_requests=provider_requests,
        task_provider_failures=provider_failures,
        e01_hash_unchanged=e01_before == e01_after == EXPECTED_E01_SHA256,
        prior_report=prior_report,
    )

    task_gate = (
        len(final_tasks) >= 16
        and holdout_summary["mixed_score_round_rate"] >= 0.75
        and holdout_summary["degenerate_round_rate"] < 0.25
    )
    if task_gate:
        args.validated_output.parent.mkdir(parents=True, exist_ok=True)
        args.validated_output.write_text(
            task_set_json(
                artifact,
                final_tasks,
                task_set_version="e02a-validated-v2r1",
                status="validated",
            ),
            encoding="utf-8",
        )

    family_diagnostics = []
    for family in sorted({row["family"] for row in candidate_diagnostics}):
        rows = [row for row in candidate_diagnostics if row["family"] == family]
        attempts = sum(row["successful_attempts"] for row in rows)
        family_diagnostics.append(
            {
                "family": family,
                "tasks": len(rows),
                "all_incorrect_tasks": sum(row["exact_match_rate"] == 0 for row in rows),
                "all_correct_tasks": sum(row["exact_match_rate"] == 1 for row in rows),
                "mixed_tasks": sum(0 < row["exact_match_rate"] < 1 for row in rows),
                "exact_match_rate": (
                    sum(row["correct"] for row in rows) / attempts if attempts else 0.0
                ),
            }
        )

    report = {
        "protocol": {
            **settings,
            "protocol_hash": run_protocol_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": perf_counter() - started,
            "ollama_version": ollama_version,
            "checkpoint_path": str(checkpoint.path),
        },
        "revision_rationale": (
            "The first E02A run calibrated objective tasks with bare task prompts. "
            "This revision uses the same versioned profile/task/output-contract renderer "
            "used by experimental response generation. Prior position, peer, and "
            "repeatability evidence is reused rather than regenerated."
        ),
        "task_calibration": {
            "candidate_task_count": len(artifact.tasks),
            "candidate_diagnostics": candidate_diagnostics,
            "family_diagnostics": family_diagnostics,
            "shortlist_task_count": len(shortlist),
            "shortlist_task_ids": [task.task_id for task in shortlist],
            "rejections": task_rejections,
            "holdout_summary": holdout_summary,
            "holdout_details": holdout_details,
            "final_task_count": len(final_tasks),
            "final_task_ids": [task.task_id for task in final_tasks],
            "validated_artifact": str(args.validated_output) if task_gate else None,
            "scorer_version": "exact-match-v1",
            "scorer_changed": False,
        },
        "reused_e02a_v1": {
            "prior_report": str(args.prior_report),
            "position_diagnostic": prior_report["position_diagnostic"],
            "peer_diagnostic": prior_report["peer_diagnostic"],
            "repeatability": prior_report["repeatability"],
            "note": (
                "The max/min anonymous-label ratio is retained descriptively but is not "
                "a hard gate because it was not part of the frozen E02A acceptance criteria."
            ),
        },
        "provider": {
            "requests": provider_requests,
            "failures": provider_failures,
            "failure_rate": provider_failures / provider_requests if provider_requests else 1.0,
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "integrity": {
            "e01_sha256_before": e01_before,
            "e01_sha256_after": e01_after,
            "e01_hash_unchanged": e01_before == e01_after == EXPECTED_E01_SHA256,
        },
        "readiness": readiness,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "e02a_task_v2_report.json"
    markdown_path = args.output_dir / "e02a_task_v2_report.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            [
                "# E02A Task Calibration Revision",
                "",
                f"**{readiness['decision']}**",
                "",
                f"- Candidates: {len(artifact.tasks)}",
                f"- Shortlisted: {len(shortlist)}",
                f"- Final holdout-mixed tasks: {len(final_tasks)}",
                f"- Holdout mixed-score rate: {holdout_summary['mixed_score_round_rate']:.3f}",
                f"- Holdout degenerate rate: {holdout_summary['degenerate_round_rate']:.3f}",
                f"- Provider failures: {provider_failures}/{provider_requests}",
                f"- E01 unchanged: {e01_before == e01_after == EXPECTED_E01_SHA256}",
                "- Scorer: `exact-match-v1` (unchanged)",
                "- Prior position/peer/repeatability evidence reused from E02A v1.",
                "- Label max/min ratio retained as diagnostic only; frozen max-share gates remain authoritative.",
                "",
                f"Gate failures: `{readiness['failures']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "task_candidates.csv", candidate_diagnostics)
    write_csv(args.output_dir / "task_families.csv", family_diagnostics)
    write_csv(args.output_dir / "task_holdout.csv", holdout_details)

    print(
        f"Final tasks: {len(final_tasks)}; holdout mixed={holdout_summary['mixed_score_round_rate']:.3f}"
    )
    print(f"Provider failures: {provider_failures}/{provider_requests}")
    print(f"E01 unchanged: {e01_before == e01_after == EXPECTED_E01_SHA256}")
    print(f"Report: {report_path}")
    print(readiness["decision"])
    if readiness["failures"]:
        print(f"Gate failures: {readiness['failures']}")
    return 0 if not readiness["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
