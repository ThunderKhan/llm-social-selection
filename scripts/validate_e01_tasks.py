from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import AgentIdentity, PromptProfile, generate_agent_response  # noqa: E402
from src.models import ModelProviderError, OllamaProvider, OllamaTimeoutError  # noqa: E402
from src.scoring import score_response  # noqa: E402
from src.seeding import derive_seed  # noqa: E402
from src.tasks.calibration import (  # noqa: E402
    TaskSetValidationError,
    analyze_objective_rounds,
    analyze_task_attempts,
    evaluate_task_set_readiness,
    load_task_set,
    render_validation_markdown,
    select_validated_tasks,
    task_set_json,
    validate_task,
)


def validation_agents() -> tuple[tuple[AgentIdentity, PromptProfile], ...]:
    return tuple(
        (
            AgentIdentity(
                agent_id=f"e01-validation-agent-{index:03d}",
                profile_id=f"e01-validation-profile-{index:03d}",
                display_label=f"Validation Participant {index}",
                generation=0,
            ),
            PromptProfile(
                profile_id=f"e01-validation-profile-{index:03d}",
                parameters={},
                template_version="e01-task-validation-v1",
            ),
        )
        for index in range(1, 9)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate deterministic E01 tasks against local qwen3:0.6b."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "tasks" / "e01_candidates_v1.json",
    )
    parser.add_argument(
        "--validated-output",
        type=Path,
        default=ROOT / "tasks" / "e01_validated_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "task_validation",
    )
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--maximum-tasks", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.maximum_tasks < 12:
        parser.error("--maximum-tasks must be at least 12")

    try:
        artifact = load_task_set(args.candidates)
    except TaskSetValidationError as error:
        print(f"Task validation failed before inference: {error}", file=sys.stderr)
        return 1
    if artifact.model_used_for_validation != args.model:
        print(
            "Task validation failed before inference: candidate artifact model does not "
            f"match requested model {args.model}",
            file=sys.stderr,
        )
        return 1

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
        print(f"Task validation failed before inference: {error}", file=sys.stderr)
        return 1

    started = perf_counter()
    agents = validation_agents()
    provider_requests = 0
    provider_failures = 0
    failure_reasons: Counter[str] = Counter()
    attempt_evidence: list[dict[str, object]] = []
    diagnostics = []
    diagnostic_by_id: dict[str, dict[str, object]] = {}
    for task_index, task in enumerate(artifact.tasks):
        responses = []
        for repeat_index in range(args.repeats):
            for agent_index, (agent, profile) in enumerate(agents):
                request_id = (
                    f"e01-task-validation-{task.task_id}-"
                    f"r{repeat_index + 1:02d}-a{agent_index + 1:02d}"
                )
                seed = derive_seed(
                    args.seed,
                    task_index,
                    "e01_task_validation",
                    str(repeat_index),
                    agent.agent_id,
                )
                provider_requests += 1
                try:
                    response = generate_agent_response(
                        response_id=f"response-{request_id}",
                        trial_id="e01-task-validation",
                        round_index=task_index,
                        agent=agent,
                        profile=profile,
                        task=task,
                        provider=provider,
                        request_id=request_id,
                        seed=seed,
                    )
                except OllamaTimeoutError:
                    provider_failures += 1
                    failure_reasons["timeout"] += 1
                    continue
                except ModelProviderError:
                    provider_failures += 1
                    failure_reasons["provider_error"] += 1
                    continue
                responses.append(response)
                score = score_response(response, task)
                attempt_evidence.append(
                    {
                        "phase": "candidate_validation",
                        "task_id": task.task_id,
                        "agent_id": agent.agent_id,
                        "repeat_index": repeat_index,
                        "request_id": response.request_id,
                        "seed": response.seed,
                        "raw_output": response.content,
                        "score": score.value,
                        "provider_name": response.provider_name,
                        "model_name": response.model_name,
                        "latency_ms": response.latency_ms,
                        "token_count": response.token_count,
                    }
                )
        if responses:
            diagnostic = analyze_task_attempts(task, responses)
        else:
            diagnostic = {
                "task_id": task.task_id,
                "family": task.family,
                "attempts": 0,
                "correct": 0,
                "incorrect": 0,
                "exact_match_rate": 0.0,
                "format_compliant": 0,
                "format_compliance_rate": 0.0,
                "difficulty_class": "too_hard",
                "unique_raw_outputs": 0,
                "mean_latency_ms": None,
                "mean_token_count": None,
                "format_error_patterns": {},
                "most_common_raw_outputs": [],
            }
        diagnostics.append(diagnostic)
        diagnostic_by_id[task.task_id] = diagnostic

    selected_tasks, rejected = select_validated_tasks(
        artifact.tasks,
        diagnostic_by_id,
        minimum_attempts=len(agents) * args.repeats,
        maximum_tasks=args.maximum_tasks,
    )

    simulated_rounds: list[list[float]] = []
    simulated_round_details: list[dict[str, object]] = []
    incomplete_simulations: list[str] = []
    for task_index, task in enumerate(selected_tasks):
        scores: list[float] = []
        raw_outputs: list[str] = []
        for agent_index, (agent, profile) in enumerate(agents):
            request_id = f"e01-objective-simulation-{task.task_id}-a{agent_index + 1:02d}"
            seed = derive_seed(
                args.seed,
                task_index,
                "e01_objective_round_simulation",
                agent.agent_id,
            )
            provider_requests += 1
            try:
                response = generate_agent_response(
                    response_id=f"response-{request_id}",
                    trial_id="e01-objective-round-simulation",
                    round_index=task_index,
                    agent=agent,
                    profile=profile,
                    task=task,
                    provider=provider,
                    request_id=request_id,
                    seed=seed,
                )
            except OllamaTimeoutError:
                provider_failures += 1
                failure_reasons["timeout"] += 1
                continue
            except ModelProviderError:
                provider_failures += 1
                failure_reasons["provider_error"] += 1
                continue
            score = score_response(response, task)
            scores.append(score.value)
            raw_outputs.append(response.content)
            attempt_evidence.append(
                {
                    "phase": "objective_round_simulation",
                    "task_id": task.task_id,
                    "agent_id": agent.agent_id,
                    "request_id": response.request_id,
                    "seed": response.seed,
                    "raw_output": response.content,
                    "score": score.value,
                    "provider_name": response.provider_name,
                    "model_name": response.model_name,
                    "latency_ms": response.latency_ms,
                    "token_count": response.token_count,
                }
            )
        if len(scores) != 8:
            incomplete_simulations.append(task.task_id)
            continue
        simulated_rounds.append(scores)
        simulated_round_details.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "number_correct": int(sum(scores)),
                "number_incorrect": 8 - int(sum(scores)),
                "scores": scores,
                "raw_outputs": raw_outputs,
            }
        )

    if simulated_rounds:
        objective_summary = analyze_objective_rounds(simulated_rounds)
    else:
        objective_summary = analyze_objective_rounds(([0.0] * 8,))

    integrity_issues = [
        f"{task.task_id}: {issue}"
        for task in artifact.tasks
        for issue in validate_task(task)
    ]
    integrity_issues.extend(
        f"simulation incomplete for {task_id}" for task_id in incomplete_simulations
    )
    readiness = evaluate_task_set_readiness(
        integrity_issues=integrity_issues,
        provider_failures=provider_failures,
        provider_requests=provider_requests,
        selected_tasks=selected_tasks,
        objective_rounds=objective_summary,
    )

    family_totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"tasks": 0, "attempts": 0, "correct": 0}
    )
    for item in diagnostics:
        totals = family_totals[str(item["family"])]
        totals["tasks"] += 1
        totals["attempts"] += int(item["attempts"])
        totals["correct"] += int(item["correct"])
    for totals in family_totals.values():
        attempts = int(totals["attempts"])
        totals["exact_match_rate"] = int(totals["correct"]) / attempts if attempts else 0.0
    selected_ids = {task.task_id for task in selected_tasks}
    selected_diagnostics = [
        item for item in diagnostics if item["task_id"] in selected_ids
    ]
    selected_rates = {
        str(item["task_id"]): float(item["exact_match_rate"])
        for item in selected_diagnostics
    }
    format_patterns: Counter[str] = Counter()
    for item in diagnostics:
        format_patterns.update(item["format_error_patterns"])
    format_compliant = sum(int(item["format_compliant"]) for item in diagnostics)
    validation_attempts = sum(int(item["attempts"]) for item in diagnostics)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"e01_task_validation_{timestamp}.json"
    markdown_path = args.output_dir / f"e01_task_validation_{timestamp}.md"
    report = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": perf_counter() - started,
            "candidate_path": str(args.candidates),
            "validated_path": str(args.validated_output),
            "task_set_version": artifact.task_set_version,
            "ollama_version": ollama_version,
            "model": provider.model_name,
            "temperature": provider.temperature,
            "num_predict": provider.num_predict,
            "agents": len(agents),
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "candidate_task_count": len(artifact.tasks),
        "selected_task_count": len(selected_tasks),
        "rejected_task_count": len(rejected),
        "selected_task_ids": [task.task_id for task in selected_tasks],
        "selected_families": sorted({task.family for task in selected_tasks}),
        "rejections": rejected,
        "rejection_reason_counts": dict(sorted(Counter(rejected.values()).items())),
        "validation_attempt_count": validation_attempts,
        "selection_criteria": {
            "strong_band": [0.20, 0.80],
            "acceptable_band": [0.10, 0.90],
            "maximum_tasks": args.maximum_tasks,
            "minimum_tasks": 12,
            "fallback": "ceiling tasks only; missing families then stable task ID",
        },
        "family_diagnostics": dict(sorted(family_totals.items())),
        "task_diagnostics": diagnostics,
        "selected_task_diagnostics": selected_diagnostics,
        "selected_exact_match_rates": selected_rates,
        "format_compliance": {
            "compliant_outputs": format_compliant,
            "rate": format_compliant / validation_attempts if validation_attempts else 0.0,
            "format_violation_count": validation_attempts - format_compliant,
            "patterns": dict(sorted(format_patterns.items())),
        },
        "objective_round_simulation": objective_summary,
        "objective_round_details": simulated_round_details,
        "provider": {
            "requests": provider_requests,
            "failures": provider_failures,
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "integrity": {
            "passed": not integrity_issues,
            "issues": integrity_issues,
        },
        "attempt_evidence": attempt_evidence,
        "readiness": readiness,
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_validation_markdown(report), encoding="utf-8")

    if readiness["decision"] == "OBJECTIVE TASK SET READY":
        args.validated_output.parent.mkdir(parents=True, exist_ok=True)
        args.validated_output.write_text(
            task_set_json(
                artifact,
                selected_tasks,
                task_set_version="e01-validated-v1",
                status="validated",
            ),
            encoding="utf-8",
        )

    print("E01 Objective Task Validation")
    print(f"Model: {provider.model_name} (Ollama {ollama_version})")
    print(
        f"Candidates: {len(artifact.tasks)}; validation attempts: "
        f"{report['validation_attempt_count']}"
    )
    print(
        f"Selected: {len(selected_tasks)}; rejected: {len(rejected)}; "
        f"families: {report['selected_families']}"
    )
    print(f"Family exact-match: {dict(sorted(family_totals.items()))}")
    print(
        "Objective rounds: "
        f"mixed={objective_summary['mixed_score_round_rate']}, "
        f"degenerate={objective_summary['degenerate_objective_round_rate']}, "
        f"tie={objective_summary['objective_tie_rate']}"
    )
    print(
        f"Provider failures: {provider_failures}/{provider_requests}; "
        f"integrity passed: {not integrity_issues}"
    )
    print(f"Report: {json_path}")
    print(readiness["decision"])
    if readiness["failures"]:
        print(f"Failures: {readiness['failures']}")
    if readiness["warnings"]:
        print(f"Warnings: {readiness['warnings']}")
    return 0 if readiness["decision"] == "OBJECTIVE TASK SET READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
