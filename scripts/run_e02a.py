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

from src.agents import AgentIdentity
from src.experiments.e02a import (
    E02A_PROTOCOL_VERSION,
    PEER_TASK_PROMPT,
    POSITION_TASK_PROMPT,
    EvidenceCheckpoint,
    choice_schema,
    choose_task_shortlist,
    deterministic_random_baseline,
    e01_peer_baseline,
    evaluate_readiness,
    generate_figures,
    matched_condition_policy,
    pairwise_repeatability,
    parse_choice,
    parse_rich_choices,
    peer_plan,
    position_plan,
    protocol_hash,
    render_choice_prompt,
    render_markdown,
    render_rich_peer_prompt,
    rich_peer_schema,
    sha256_file,
    summarize_holdout,
    summarize_position,
    summarize_rich_peer,
    write_csv,
)
from src.models import (
    ModelProviderError,
    OllamaProvider,
    OllamaTimeoutError,
)
from src.scoring import normalize_answer
from src.seeding import derive_seed
from src.tasks import Task
from src.tasks.calibration import (
    load_task_set,
    output_format_valid,
    task_set_json,
)

EXPECTED_E01_SHA256 = "2a3847218d47820c13647e3622773fa3e57ff081608e4842707f1bbb479d424a"
BASE_SEED = 20260823


def diagnostic_agent(agent_id: str) -> AgentIdentity:
    return AgentIdentity(
        agent_id=agent_id,
        profile_id=f"profile-{agent_id}",
        display_label="E02A Diagnostic Participant",
        generation=0,
    )


def run_inference(
    *,
    checkpoint: EvidenceCheckpoint,
    provider: OllamaProvider,
    evaluation_id: str,
    phase: str,
    agent: AgentIdentity,
    task: Task,
    prompt: str,
    seed: int,
    response_schema: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = checkpoint.get(evaluation_id)
    if existing is not None:
        return existing
    record: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "phase": phase,
        "request_id": f"e02a-request-{evaluation_id}",
        "agent_id": agent.agent_id,
        "task_id": task.task_id,
        "seed": seed,
        **(metadata or {}),
    }
    try:
        output = provider.generate(
            agent=agent,
            task=task,
            prompt=prompt,
            request_id=record["request_id"],
            seed=seed,
            response_schema=response_schema,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run focused E02A apparatus diagnostics against local Ollama."
    )
    parser.add_argument(
        "--candidates", type=Path, default=ROOT / "tasks" / "e02a_candidates_v2.json"
    )
    parser.add_argument(
        "--validated-output",
        type=Path,
        default=ROOT / "tasks" / "e02a_validated_v2.json",
    )
    parser.add_argument(
        "--e01-database",
        type=Path,
        default=ROOT / "experiments" / "e01" / "e01_full.sqlite",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "e02a"
    )
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    artifact = load_task_set(args.candidates)
    if artifact.model_used_for_validation != args.model:
        parser.error("candidate model_used_for_validation does not match --model")
    e01_before = sha256_file(args.e01_database)
    if e01_before != EXPECTED_E01_SHA256:
        parser.error(
            f"E01 SHA-256 mismatch: expected {EXPECTED_E01_SHA256}, found {e01_before}"
        )
    settings = {
        "protocol_version": E02A_PROTOCOL_VERSION,
        "candidate_sha256": sha256_file(args.candidates),
        "e01_sha256": e01_before,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "num_predict": 32,
        "calibration_attempts_per_task": 16,
        "holdout_attempts_per_task": 8,
        "position_evaluations": 196,
        "peer_evaluations": 256,
        "repeatability_prompts_per_type": 20,
        "repeatability_attempts_per_prompt": 5,
        "seed": BASE_SEED,
    }
    run_protocol_hash = protocol_hash(settings)
    checkpoint = EvidenceCheckpoint(
        args.output_dir / "e02a_raw_evidence.jsonl", run_protocol_hash
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
        print(f"E02A failed before inference: {error}", file=sys.stderr)
        return 1

    started = perf_counter()
    print(f"E02A protocol {run_protocol_hash[:12]} using Ollama {ollama_version}")
    print("Phase 1/5: objective-task calibration")
    task_index = {task.task_id: index for index, task in enumerate(artifact.tasks)}
    for task in artifact.tasks:
        for attempt in range(16):
            evaluation_id = f"task-calibration-{task.task_id}-a{attempt:02d}"
            row = run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=evaluation_id,
                phase="task_calibration",
                agent=diagnostic_agent(f"calibration-agent-{attempt:02d}"),
                task=task,
                prompt=task.prompt,
                seed=derive_seed(
                    BASE_SEED,
                    task_index[task.task_id],
                    "e02a_task_calibration",
                    task.task_id,
                    str(attempt),
                ),
                metadata={"attempt": attempt, "family": task.family},
            )
            if row["status"] == "ok" and "correct" not in row:
                row["correct"] = normalize_answer(
                    row["raw_output"]
                ) == normalize_answer(task.expected_answer or "")
                row["format_valid"] = output_format_valid(task, row["raw_output"])
                # Derived fields are reconstructed below for resumed immutable rows.
    calibration_rows = checkpoint.phase("task_calibration")
    tasks_by_id = {task.task_id: task for task in artifact.tasks}
    for row in calibration_rows:
        if row["status"] == "ok":
            task = tasks_by_id[row["task_id"]]
            row["correct"] = normalize_answer(row["raw_output"]) == normalize_answer(
                task.expected_answer or ""
            )
            row["format_valid"] = output_format_valid(task, row["raw_output"])
    shortlist, candidate_diagnostics, task_rejections = choose_task_shortlist(
        artifact.tasks, calibration_rows
    )

    print(f"Phase 2/5: fresh holdout for {len(shortlist)} shortlisted tasks")
    for shortlist_index, task in enumerate(shortlist):
        for agent_index in range(8):
            evaluation_id = f"task-holdout-{task.task_id}-a{agent_index}"
            row = run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=evaluation_id,
                phase="task_holdout",
                agent=diagnostic_agent(f"holdout-agent-{agent_index}"),
                task=task,
                prompt=task.prompt,
                seed=derive_seed(
                    BASE_SEED,
                    shortlist_index,
                    "e02a_fresh_holdout",
                    task.task_id,
                    str(agent_index),
                ),
                metadata={"agent_index": agent_index, "family": task.family},
            )
            if row["status"] == "ok":
                row["correct"] = normalize_answer(
                    row["raw_output"]
                ) == normalize_answer(task.expected_answer or "")
    holdout_rows = checkpoint.phase("task_holdout")
    for row in holdout_rows:
        if row["status"] == "ok":
            task = tasks_by_id[row["task_id"]]
            row["correct"] = normalize_answer(row["raw_output"]) == normalize_answer(
                task.expected_answer or ""
            )
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
                task_set_version="e02a-validated-v2",
                status="validated",
            ),
            encoding="utf-8",
        )

    position_task = Task(
        task_id="e02a-position-task",
        family="diagnostic",
        prompt=POSITION_TASK_PROMPT,
        expected_answer="4",
        scorer_version="exact-match-v1",
    )
    print("Phase 3/5: crossed position/label/content diagnostic")
    plans_by_id = {plan["evaluation_id"]: plan for plan in position_plan()}
    for plan in plans_by_id.values():
        labels = [candidate["anonymous_label"] for candidate in plan["candidates"]]
        row = run_inference(
            checkpoint=checkpoint,
            provider=provider,
            evaluation_id=plan["evaluation_id"],
            phase="position",
            agent=diagnostic_agent(f"position-voter-{plan['voter']}"),
            task=position_task,
            prompt=render_choice_prompt(POSITION_TASK_PROMPT, plan["candidates"]),
            seed=derive_seed(
                BASE_SEED,
                plan["cycle"],
                "e02a_position",
                plan["scheme"],
                str(plan["repeat"]),
                str(plan["voter"]),
            ),
            response_schema=choice_schema(labels),
            metadata={
                "scheme": plan["scheme"],
                "repeat": plan["repeat"],
                "cycle": plan["cycle"],
                "voter": plan["voter"],
                "candidates": plan["candidates"],
            },
        )
        if row["status"] == "ok":
            choice, reason = parse_choice(row["raw_output"], labels)
            row.update(valid=choice is not None, invalid_reason=reason)
            if choice is not None:
                selected = next(
                    candidate
                    for candidate in plan["candidates"]
                    if candidate["anonymous_label"] == choice
                )
                row.update(
                    selected_display_index=selected["display_index"],
                    selected_anonymous_label=choice,
                    selected_candidate_id=selected["underlying_candidate_id"],
                )
    position_rows = checkpoint.phase("position")
    for row in position_rows:
        plan = plans_by_id[row["evaluation_id"]]
        if row["status"] != "ok":
            row.update(
                valid=False, invalid_reason=row.get("error_type", "provider_error")
            )
            continue
        labels = [candidate["anonymous_label"] for candidate in plan["candidates"]]
        choice, reason = parse_choice(row["raw_output"], labels)
        row.update(valid=choice is not None, invalid_reason=reason)
        if choice is not None:
            selected = next(
                candidate
                for candidate in plan["candidates"]
                if candidate["anonymous_label"] == choice
            )
            row.update(
                selected_display_index=selected["display_index"],
                selected_anonymous_label=choice,
                selected_candidate_id=selected["underlying_candidate_id"],
            )
    position_summary = summarize_position(position_rows)

    peer_task = Task(
        task_id="e02a-peer-task",
        family="diagnostic",
        prompt=PEER_TASK_PROMPT,
        expected_answer="25",
        scorer_version="exact-match-v1",
    )
    print("Phase 4/5: ranked and approval peer-mechanism probe")
    peer_plans_by_id = {plan["evaluation_id"]: plan for plan in peer_plan()}
    for plan in peer_plans_by_id.values():
        labels = [candidate["anonymous_label"] for candidate in plan["candidates"]]
        row = run_inference(
            checkpoint=checkpoint,
            provider=provider,
            evaluation_id=plan["evaluation_id"],
            phase="peer_rich",
            agent=diagnostic_agent(plan["voter_id"]),
            task=peer_task,
            prompt=render_rich_peer_prompt(plan),
            seed=derive_seed(
                BASE_SEED,
                plan["panel"],
                f"e02a_peer_{plan['mechanism']}",
                plan["voter_id"],
            ),
            response_schema=rich_peer_schema(labels, plan["mechanism"]),
            metadata={
                "panel": plan["panel"],
                "voter_id": plan["voter_id"],
                "mechanism": plan["mechanism"],
                "candidates": plan["candidates"],
            },
        )
        if row["status"] == "ok":
            choices, reason = parse_rich_choices(
                row["raw_output"], labels, plan["mechanism"]
            )
            row.update(valid=choices is not None, invalid_reason=reason)
            if choices is not None:
                by_label = {
                    candidate["anonymous_label"]: candidate["underlying_candidate_id"]
                    for candidate in plan["candidates"]
                }
                row["selected_candidate_ids"] = [by_label[choice] for choice in choices]
    peer_rows = checkpoint.phase("peer_rich")
    for row in peer_rows:
        plan = peer_plans_by_id[row["evaluation_id"]]
        if row["status"] != "ok":
            row.update(
                valid=False, invalid_reason=row.get("error_type", "provider_error")
            )
            continue
        labels = [candidate["anonymous_label"] for candidate in plan["candidates"]]
        choices, reason = parse_rich_choices(
            row["raw_output"], labels, plan["mechanism"]
        )
        row.update(valid=choices is not None, invalid_reason=reason)
        if choices is not None:
            by_label = {
                candidate["anonymous_label"]: candidate["underlying_candidate_id"]
                for candidate in plan["candidates"]
            }
            row["selected_candidate_ids"] = [by_label[choice] for choice in choices]
    peer_summary = summarize_rich_peer(peer_rows)
    peer_summary["e01_one_vote"] = e01_peer_baseline(args.e01_database)
    peer_summary["uniform_random"] = deterministic_random_baseline()
    ranked = peer_summary["ranked_top3"]
    peer_summary["elimination_risk_interpretation"] = (
        f"Ranked top-3 produced a unique minimum in "
        f"{ranked['unique_minimum_rate']:.1%} of panels versus 3.0% under E01's "
        "one-vote rule; unlike uniform random elimination, it uses response-quality "
        "signals and therefore creates meaningful variation in elimination risk."
    )

    print("Phase 5/5: fixed-prompt provider repeatability")
    response_probe_tasks = artifact.tasks[:20]
    for prompt_index, task in enumerate(response_probe_tasks):
        fixed_seed = derive_seed(
            BASE_SEED, prompt_index, "e02a_repeat_response", task.task_id
        )
        for repetition in range(5):
            row = run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=f"repeat-response-p{prompt_index:02d}-r{repetition}",
                phase="repeat_response",
                agent=diagnostic_agent(f"repeat-response-agent-{prompt_index:02d}"),
                task=task,
                prompt=task.prompt,
                seed=fixed_seed,
                metadata={
                    "prompt_id": f"response-{prompt_index:02d}",
                    "repetition": repetition,
                },
            )
            if row["status"] == "ok":
                row["semantic_result"] = normalize_answer(
                    row["raw_output"]
                ) == normalize_answer(task.expected_answer or "")
    repeat_response_rows = checkpoint.phase("repeat_response")
    repeat_task_by_prompt = {
        f"response-{index:02d}": task for index, task in enumerate(response_probe_tasks)
    }
    for row in repeat_response_rows:
        if row["status"] == "ok":
            task = repeat_task_by_prompt[row["prompt_id"]]
            row["semantic_result"] = normalize_answer(
                row["raw_output"]
            ) == normalize_answer(task.expected_answer or "")

    repeat_position_plans = tuple(position_plan())[:20]
    for prompt_index, plan in enumerate(repeat_position_plans):
        labels = [candidate["anonymous_label"] for candidate in plan["candidates"]]
        fixed_seed = derive_seed(BASE_SEED, prompt_index, "e02a_repeat_ballot")
        for repetition in range(5):
            row = run_inference(
                checkpoint=checkpoint,
                provider=provider,
                evaluation_id=f"repeat-ballot-p{prompt_index:02d}-r{repetition}",
                phase="repeat_ballot",
                agent=diagnostic_agent(f"repeat-ballot-agent-{prompt_index:02d}"),
                task=position_task,
                prompt=render_choice_prompt(POSITION_TASK_PROMPT, plan["candidates"]),
                seed=fixed_seed,
                response_schema=choice_schema(labels),
                metadata={
                    "prompt_id": f"ballot-{prompt_index:02d}",
                    "repetition": repetition,
                },
            )
            if row["status"] == "ok":
                choice, _ = parse_choice(row["raw_output"], labels)
                by_label = {
                    candidate["anonymous_label"]: candidate["underlying_candidate_id"]
                    for candidate in plan["candidates"]
                }
                row["semantic_result"] = by_label.get(choice)
    repeat_ballot_rows = checkpoint.phase("repeat_ballot")
    repeat_plan_by_prompt = {
        f"ballot-{index:02d}": plan for index, plan in enumerate(repeat_position_plans)
    }
    for row in repeat_ballot_rows:
        if row["status"] == "ok":
            plan = repeat_plan_by_prompt[row["prompt_id"]]
            labels = [candidate["anonymous_label"] for candidate in plan["candidates"]]
            choice, _ = parse_choice(row["raw_output"], labels)
            by_label = {
                candidate["anonymous_label"]: candidate["underlying_candidate_id"]
                for candidate in plan["candidates"]
            }
            row["semantic_result"] = by_label.get(choice)
    response_repeatability = pairwise_repeatability(
        repeat_response_rows, "semantic_result"
    )
    ballot_repeatability = pairwise_repeatability(repeat_ballot_rows, "semantic_result")

    e01_after = sha256_file(args.e01_database)
    all_rows = list(checkpoint.records.values())
    provider_requests = len(all_rows)
    provider_failures = sum(row.get("status") != "ok" for row in all_rows)
    failure_reasons = Counter(
        row.get("error_type", "unknown")
        for row in all_rows
        if row.get("status") != "ok"
    )
    finish_reasons = Counter(
        str(row.get("finish_reason")) for row in all_rows if row.get("status") == "ok"
    )
    family_diagnostics = []
    for family in sorted({row["family"] for row in candidate_diagnostics}):
        rows = [row for row in candidate_diagnostics if row["family"] == family]
        attempts = sum(row["successful_attempts"] for row in rows)
        family_diagnostics.append(
            {
                "family": family,
                "tasks": len(rows),
                "all_incorrect_tasks": sum(
                    row["exact_match_rate"] == 0 for row in rows
                ),
                "all_correct_tasks": sum(row["exact_match_rate"] == 1 for row in rows),
                "mixed_tasks": sum(0 < row["exact_match_rate"] < 1 for row in rows),
                "exact_match_rate": sum(row["correct"] for row in rows) / attempts,
                "format_compliance_rate": sum(
                    row["format_compliance_rate"] * row["successful_attempts"]
                    for row in rows
                )
                / attempts,
            }
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
        },
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
            "selection_rule": (
                "shortlist only complete 16-attempt tasks in [0.15,0.85], balanced "
                "by family; final set only complete mixed-score fresh holdout rounds; "
                "no ceiling fallback"
            ),
        },
        "position_diagnostic": position_summary,
        "peer_diagnostic": peer_summary,
        "repeatability": {
            "response": response_repeatability,
            "ballot": ballot_repeatability,
            "matched_condition_policy": matched_condition_policy(
                response_repeatability, ballot_repeatability
            ),
            "retry_until_match_used": False,
            "output_replay_or_cache_used": False,
        },
        "provider": {
            "requests": provider_requests,
            "failures": provider_failures,
            "failure_rate": provider_failures / provider_requests,
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "truncation_finish_count": finish_reasons.get("length", 0),
        },
        "integrity": {
            "e01_expected_sha256": EXPECTED_E01_SHA256,
            "e01_sha256_before": e01_before,
            "e01_sha256_after": e01_after,
            "e01_hash_unchanged": e01_before == e01_after == EXPECTED_E01_SHA256,
            "e01_opened_read_only_for_analysis": True,
        },
    }
    report["readiness"] = evaluate_readiness(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "e02a_report.json"
    markdown_path = args.output_dir / "e02a_report.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_csv(args.output_dir / "task_candidates.csv", candidate_diagnostics)
    write_csv(args.output_dir / "task_families.csv", family_diagnostics)
    write_csv(args.output_dir / "task_holdout.csv", holdout_details)
    write_csv(args.output_dir / "position_evidence.csv", position_rows)
    write_csv(args.output_dir / "peer_evidence.csv", peer_rows)
    write_csv(
        args.output_dir / "repeatability_prompts.csv",
        response_repeatability["prompt_details"]
        + ballot_repeatability["prompt_details"],
    )
    figures = generate_figures(report, args.output_dir / "figures")
    report["artifacts"] = {
        "json_report": str(report_path),
        "markdown_report": str(markdown_path),
        "raw_evidence": str(checkpoint.path),
        "csv_tables": [
            "task_candidates.csv",
            "task_families.csv",
            "task_holdout.csv",
            "position_evidence.csv",
            "peer_evidence.csv",
            "repeatability_prompts.csv",
        ],
        "figures": figures,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Final tasks: {len(final_tasks)}; holdout mixed={holdout_summary['mixed_score_round_rate']:.3f}"
    )
    print(
        f"Position max display={position_summary['max_display_share']:.3f}; "
        f"max label={position_summary['max_label_share']:.3f}"
    )
    print(f"Peer recommendation: {peer_summary['recommendation']}")
    print(f"Provider failures: {provider_failures}/{provider_requests}")
    print(f"E01 unchanged: {e01_before == e01_after == EXPECTED_E01_SHA256}")
    print(f"Report: {report_path}")
    print(report["readiness"]["decision"])
    if report["readiness"]["failures"]:
        print(f"Gate failures: {report['readiness']['failures']}")
    return 0 if not report["readiness"]["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
