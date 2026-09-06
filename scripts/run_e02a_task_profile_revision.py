from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from src.experiments.e02a_profile_calibration import (
    E02A_PROFILE_TASK_PROTOCOL,
    calibration_profiles,
    profile_manifest,
    profile_task_prompt,
)
from src.experiments.e02a_revision import evaluate_task_revision_readiness
from src.models import ModelProviderError, OllamaProvider, OllamaTimeoutError
from src.scoring import normalize_answer
from src.seeding import derive_seed
from src.tasks.calibration import load_task_set, output_format_valid, task_set_json

EXPECTED_E01_SHA256 = "2a3847218d47820c13647e3622773fa3e57ff081608e4842707f1bbb479d424a"
BASE_SEED = 20260906
CALIBRATION_REPEATS = 2


def run_inference(
    *,
    checkpoint: EvidenceCheckpoint,
    provider: OllamaProvider,
    evaluation_id: str,
    phase: str,
    task,
    agent,
    prompt: str,
    seed: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    existing = checkpoint.get(evaluation_id)
    if existing is not None:
        return existing
    record: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "phase": phase,
        "request_id": f"e02a-v3-request-{evaluation_id}",
        "agent_id": agent.agent_id,
        "profile_id": agent.profile_id,
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


def family_summary(candidate_diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_diagnostics:
        by_family[str(row["family"])].append(row)
    result = []
    for family, rows in sorted(by_family.items()):
        attempts = sum(int(row["successful_attempts"]) for row in rows)
        correct = sum(int(row["correct"]) for row in rows)
        result.append(
            {
                "family": family,
                "tasks": len(rows),
                "attempts": attempts,
                "correct": correct,
                "exact_match_rate": correct / attempts if attempts else 0.0,
                "floor_tasks": sum(float(row["exact_match_rate"]) == 0.0 for row in rows),
                "ceiling_tasks": sum(float(row["exact_match_rate"]) == 1.0 for row in rows),
                "mixed_tasks": sum(0.0 < float(row["exact_match_rate"]) < 1.0 for row in rows),
                "in_band_tasks": sum(
                    0.15 <= float(row["exact_match_rate"]) <= 0.85 for row in rows
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate E02A objective tasks across the frozen eight E01 profiles. "
            "This tests round-relevant task discrimination without changing the scorer."
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
        default=ROOT / "tasks" / "e02a_validated_v3.json",
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
        default=ROOT / "experiments" / "e02a_task_v3",
    )
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    artifact = load_task_set(args.candidates)
    if artifact.model_used_for_validation != args.model:
        parser.error("candidate model_used_for_validation does not match --model")
    if not args.prior_report.exists():
        parser.error("prior E02A report not found; run scripts/run_e02a.py first")
    prior_report = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if prior_report.get("scope", {}).get("apparatus_validation_only") is not True:
        parser.error("prior report is not an E02A apparatus-validation report")

    e01_before = sha256_file(args.e01_database)
    if e01_before != EXPECTED_E01_SHA256:
        parser.error(
            f"E01 SHA-256 mismatch: expected {EXPECTED_E01_SHA256}, found {e01_before}"
        )

    profiles = calibration_profiles()
    if len(profiles) != 8:
        parser.error(f"expected exactly eight frozen E01 profiles, found {len(profiles)}")
    manifest = profile_manifest(profiles)
    settings = {
        "protocol_version": E02A_PROFILE_TASK_PROTOCOL,
        "candidate_sha256": sha256_file(args.candidates),
        "prior_report_protocol_hash": prior_report["protocol"]["protocol_hash"],
        "e01_sha256": e01_before,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "num_predict": 32,
        "profiles": manifest,
        "calibration_repeats_per_profile": CALIBRATION_REPEATS,
        "calibration_attempts_per_task": len(profiles) * CALIBRATION_REPEATS,
        "holdout_attempts_per_task": len(profiles),
        "prompt_path": "src.agents.render_prompt with frozen E01 profile parameters",
        "profile_dependency": (
            "If E02B changes the profile set or profile prompts, this task calibration "
            "must be rerun before E03."
        ),
        "seed": BASE_SEED,
    }
    run_protocol_hash = protocol_hash(settings)
    checkpoint = EvidenceCheckpoint(
        args.output_dir / "e02a_task_v3_raw_evidence.jsonl", run_protocol_hash
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
        print(f"E02A profile task revision failed before inference: {error}", file=sys.stderr)
        return 1

    started = perf_counter()
    print(
        f"E02A profile task revision {run_protocol_hash[:12]} using Ollama {ollama_version}"
    )
    print("Phase 1/2: calibration across 8 frozen profiles x 2 repeats")

    tasks_by_id = {task.task_id: task for task in artifact.tasks}
    profile_index = {profile.profile_id: index for index, profile in enumerate(profiles)}
    for task_index, task in enumerate(artifact.tasks):
        for repeat_index in range(CALIBRATION_REPEATS):
            for profile in profiles:
                agent, prompt = profile_task_prompt(
                    task,
                    profile,
                    phase="calibration",
                    repeat_index=repeat_index,
                )
                evaluation_id = (
                    f"task-calibration-{task.task_id}-{profile.profile_id}-"
                    f"r{repeat_index:02d}"
                )
                run_inference(
                    checkpoint=checkpoint,
                    provider=provider,
                    evaluation_id=evaluation_id,
                    phase="task_calibration",
                    task=task,
                    agent=agent,
                    prompt=prompt,
                    seed=derive_seed(
                        BASE_SEED,
                        task_index,
                        "e02a_task_profile_calibration_v3",
                        task.task_id,
                        profile.profile_id,
                        str(repeat_index),
                    ),
                    metadata={
                        "family": task.family,
                        "repeat_index": repeat_index,
                        "profile_index": profile_index[profile.profile_id],
                        "profile_parameters": dict(profile.parameters),
                    },
                )

    calibration_rows = checkpoint.phase("task_calibration")
    derive_fields(calibration_rows, tasks_by_id)
    shortlist, candidate_diagnostics, task_rejections = choose_task_shortlist(
        artifact.tasks,
        calibration_rows,
        attempts_per_task=len(profiles) * CALIBRATION_REPEATS,
        maximum_tasks=24,
    )

    print(f"Phase 2/2: fresh holdout across 8 profiles for {len(shortlist)} tasks")
    for task_index, task in enumerate(shortlist):
        for profile in profiles:
            agent, prompt = profile_task_prompt(
                task,
                profile,
                phase="holdout",
                repeat_index=0,
            )
            evaluation_id = f"task-holdout-{task.task_id}-{profile.profile_id}"
            run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=evaluation_id,
                phase="task_holdout",
                task=task,
                agent=agent,
                prompt=prompt,
                seed=derive_seed(
                    BASE_SEED,
                    task_index,
                    "e02a_task_profile_holdout_v3",
                    task.task_id,
                    profile.profile_id,
                ),
                metadata={
                    "family": task.family,
                    "profile_index": profile_index[profile.profile_id],
                    "profile_parameters": dict(profile.parameters),
                },
            )

    holdout_rows = checkpoint.phase("task_holdout")
    derive_fields(holdout_rows, tasks_by_id)
    final_tasks, holdout_summary, holdout_details = summarize_holdout(
        shortlist, holdout_rows
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
                task_set_version="e02a-validated-v3",
                status="validated",
            ),
            encoding="utf-8",
        )

    e01_after = sha256_file(args.e01_database)
    all_rows = list(checkpoint.records.values())
    provider_requests = len(all_rows)
    provider_failures = sum(row.get("status") != "ok" for row in all_rows)
    failure_reasons = Counter(
        str(row.get("error_type", "unknown"))
        for row in all_rows
        if row.get("status") != "ok"
    )
    families = family_summary(candidate_diagnostics)

    readiness = evaluate_task_revision_readiness(
        final_task_count=len(final_tasks),
        holdout_summary=holdout_summary,
        task_provider_requests=provider_requests,
        task_provider_failures=provider_failures,
        e01_hash_unchanged=(
            e01_before == e01_after == EXPECTED_E01_SHA256
        ),
        prior_report=prior_report,
    )

    report: dict[str, Any] = {
        "protocol": {
            **settings,
            "protocol_hash": run_protocol_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": perf_counter() - started,
            "ollama_version": ollama_version,
            "checkpoint_path": str(checkpoint.path),
        },
        "scope": {
            "apparatus_validation_only": True,
            "population_trials_run": False,
            "profile_manipulation_run": False,
            "scorer_changed": False,
            "candidate_bank_changed_from_v2": False,
        },
        "task_calibration": {
            "candidate_task_count": len(artifact.tasks),
            "candidate_diagnostics": candidate_diagnostics,
            "family_diagnostics": families,
            "shortlist_task_count": len(shortlist),
            "shortlist_task_ids": [task.task_id for task in shortlist],
            "rejections": task_rejections,
            "holdout_summary": holdout_summary,
            "holdout_details": holdout_details,
            "final_task_count": len(final_tasks),
            "final_task_ids": [task.task_id for task in final_tasks],
            "validated_artifact": str(args.validated_output) if task_gate else None,
            "selection_rule": (
                "Across eight frozen E01 profiles and two seeded repeats each, shortlist "
                "only complete tasks with aggregate exact-match rate in [0.15,0.85]; "
                "final set keeps only complete mixed-score fresh holdout rounds."
            ),
        },
        "provider": {
            "requests": provider_requests,
            "failures": provider_failures,
            "failure_rate": provider_failures / provider_requests if provider_requests else 1.0,
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "integrity": {
            "e01_expected_sha256": EXPECTED_E01_SHA256,
            "e01_sha256_before": e01_before,
            "e01_sha256_after": e01_after,
            "e01_hash_unchanged": e01_before == e01_after == EXPECTED_E01_SHA256,
        },
        "readiness": readiness,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "e02a_task_v3_report.json"
    markdown_path = args.output_dir / "e02a_task_v3_report.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            (
                "# E02A Profile-Aware Task Calibration",
                "",
                "Apparatus validation only. No population-effect claims are made.",
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
                "- Profile dependency: if E02B changes profiles, rerun this calibration.",
                f"- Gate failures: `{readiness['failures']}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "task_candidates.csv", candidate_diagnostics)
    write_csv(args.output_dir / "task_families.csv", families)
    write_csv(args.output_dir / "task_holdout.csv", holdout_details)

    print(
        f"Shortlisted: {len(shortlist)}; final tasks: {len(final_tasks)}; "
        f"holdout mixed={holdout_summary['mixed_score_round_rate']:.3f}"
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
