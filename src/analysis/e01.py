from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from .integrity import (
    audit_e01_integrity,
    load_analysis_tables,
    open_read_only,
    sha256_file,
)
from .metrics import analyze_all_metrics


E01_ANALYSIS_VERSION = "e01-analysis-v1"
ANALYSIS_SOURCE_FILES = (
    "src/analysis/e01.py",
    "src/analysis/integrity.py",
    "src/analysis/metrics.py",
    "src/analysis/plotting.py",
    "src/analysis/reporting.py",
    "src/analysis/statistics.py",
    "src/seeding.py",
    "scripts/analyze_e01.py",
)


def _git_state(repository: str | Path | None) -> tuple[str | None, bool | None]:
    if repository is None:
        return None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, dirty


def _analysis_source_hashes(repository: str | Path | None) -> dict[str, str]:
    if repository is None:
        return {}
    root = Path(repository)
    result = {}
    for relative in ANALYSIS_SOURCE_FILES:
        source = root / relative
        normalized = source.read_text(encoding="utf-8").replace("\r\n", "\n")
        result[relative] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return result


def _profile_behavior_range(metrics: dict[str, Any]) -> float:
    values = [
        row["mean_response_score"]
        for row in metrics["profiles"]["profile_rows"]
        if row["mean_response_score"] is not None
    ]
    return 0.0 if not values else max(values) - min(values)


def evaluate_e03_readiness(
    integrity: dict[str, Any], metrics: dict[str, Any] | None
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, value: object, threshold: str) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": value,
                "gate": threshold,
            }
        )

    gate(
        "hard_dataset_integrity",
        integrity["status"] == "PASS",
        integrity["status"],
        "all critical integrity checks pass",
    )
    if metrics is None:
        return {
            "decision": "REVISE APPARATUS BEFORE E03",
            "gates": gates,
            "reasons": ["Critical dataset-integrity checks failed."],
            "recommendations": [
                "Resolve and document the integrity failure without altering the raw E01 database."
            ],
        }

    invalid_ballots = integrity["ballots"]["invalid"]
    objective_mixed_accuracy = metrics["selection"]["objective_mechanics"][
        "selected_incorrect_rate_when_mixed"
    ]
    peer_tie_rate = metrics["selection"]["peer_mechanics"]["tie_rate"]
    peer_mean_tie = metrics["selection"]["peer_mechanics"]["mean_minimum_tie_size"]
    objective_degenerate = 1 - metrics["selection"]["objective_mechanics"][
        "mixed_score_round_rate"
    ]
    profile_range = _profile_behavior_range(metrics)
    position_max_share = metrics["ballots"]["position_max_share"]
    extreme_tasks = sum(
        row["always_correct"] or row["always_incorrect"]
        for row in metrics["tasks"]["task_rows"]
    )
    extreme_task_rate = extreme_tasks / len(metrics["tasks"]["task_rows"])
    round_zero = metrics["cross_condition_comparability"]
    final_round = max(row["round_index"] for row in metrics["divergence"]["by_round"])
    final_divergence = [
        row["mean_shared_agent_fraction"]
        for row in metrics["divergence"]["by_round"]
        if row["round_index"] == final_round
    ]
    gate("provider_schema_reliability", invalid_ballots == 0, invalid_ballots, "0 invalid ballots")
    gate(
        "objective_selection_correctness",
        objective_mixed_accuracy == 1.0,
        objective_mixed_accuracy,
        "selected incorrect responder in every mixed-score round",
    )
    gate(
        "peer_selection_differentiation",
        not (peer_tie_rate >= 0.90 and peer_mean_tie >= 4.0),
        {"tie_rate": peer_tie_rate, "mean_tie_size": peer_mean_tie},
        "not both >=90% ties and mean minimum-support tie size >=4",
    )
    gate(
        "objective_score_discrimination",
        objective_degenerate <= 0.25,
        objective_degenerate,
        "<=25% score-degenerate objective rounds",
    )
    gate(
        "condition_divergence",
        any(value < 0.95 for value in final_divergence),
        final_divergence,
        "at least one final-round pair shares <95% of active identities",
    )
    gate(
        "profile_behavioral_separation",
        profile_range >= 0.05,
        profile_range,
        ">=5 percentage-point observed range in profile response accuracy",
    )
    gate(
        "anonymous_position_health",
        position_max_share <= 0.25,
        position_max_share,
        "no anonymous label receives >25% of ballots",
    )
    gate(
        "task_difficulty_coverage",
        extreme_task_rate <= 0.25,
        {"extreme_tasks": extreme_tasks, "rate": extreme_task_rate},
        "<=25% of tasks are always correct or always incorrect",
    )
    gate(
        "matched_inference_reproducibility",
        round_zero["all_round_zero_responses_identical"]
        and round_zero["all_round_zero_ballots_identical"],
        {
            "response_seed_match_rate": round_zero["response_seed_match_rate"],
            "response_content_match_rate": round_zero["response_content_match_rate"],
            "ballot_seed_match_rate": round_zero["ballot_seed_match_rate"],
            "candidate_order_match_rate": round_zero["candidate_order_match_rate"],
            "ballot_choice_match_rate": round_zero["ballot_choice_match_rate"],
        },
        "identical matched round-0 states, seeds, responses, candidate orders, and ballots",
    )

    failed = [row["name"] for row in gates if not row["passed"]]
    reasons = []
    warnings = []
    if failed:
        reasons.append(f"Readiness gates not met: {', '.join(failed)}.")
    reasons.append(
        "E01 does not expose persistent identity or history, so it cannot address reciprocity, reputation, or alliance questions."
    )
    if peer_tie_rate >= 0.90:
        warnings.append(
            "Peer selection had a minimum-support tie in at least 90% of rounds; the observed mean tied set should be treated as substantial tie-break dependence."
        )
    recommendations = [
        "Freeze all E03 outcomes, estimands, exclusions, and multiplicity rules before examining E03 data.",
        "Use fresh held-out task seeds and expand the validated task pool to reduce repeated-task dependence.",
        "Run a dedicated profile manipulation check and revise profile prompts if behavioral separation is weak.",
        "Treat identity/history exposure as a separate randomized manipulation if reciprocity or reputation is an E03 target.",
        "Power E03 from trial-level pilot variance and paired effect uncertainty rather than agent-round counts.",
    ]
    if "peer_selection_differentiation" in failed:
        recommendations.append(
            "Revise or pre-specify peer tie handling so selection is not dominated by large zero-support tie sets."
        )
    if "objective_score_discrimination" in failed:
        recommendations.append(
            "Rebalance task difficulty or use a more graded frozen scorer to increase within-round objective discrimination."
        )
    if "anonymous_position_health" in failed:
        recommendations.append(
            "Investigate and mitigate anonymous-label position anchoring before reusing LLM ballots as a selection signal."
        )
    if "task_difficulty_coverage" in failed:
        recommendations.append(
            "Replace always-correct/always-incorrect tasks and recalibrate the expanded task pool on the frozen model settings."
        )
    if "matched_inference_reproducibility" in failed:
        recommendations.append(
            "Quantify provider nondeterminism and pre-specify whether matched trials require cached common outputs or independent stochastic realizations."
        )
    if warnings:
        recommendations.append(
            "Pre-specify and stress-test peer minimum-support tie behavior even if the current tie-size gate is retained."
        )
    decision = (
        "READY TO DESIGN E03"
        if not failed
        else "REVISE APPARATUS BEFORE E03"
    )
    return {
        "decision": decision,
        "gates": gates,
        "warnings": warnings,
        "reasons": reasons,
        "recommendations": recommendations,
    }


def analyze_e01_database(
    database_path: str | Path,
    *,
    task_artifact_path: str | Path | None,
    repository: str | Path | None = None,
) -> tuple[dict[str, Any], object]:
    database = Path(database_path).resolve()
    with open_read_only(database) as connection:
        connection.execute("BEGIN")
        connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        database_hash = sha256_file(database)
        tables = load_analysis_tables(connection)
        integrity = audit_e01_integrity(
            connection, tables, task_artifact_path=task_artifact_path
        )
    metrics = analyze_all_metrics(tables) if integrity["status"] == "PASS" else None
    recommendation = evaluate_e03_readiness(integrity, metrics)
    config = tables.config
    analysis_commit, analysis_dirty = _git_state(repository)
    report: dict[str, Any] = {
        "dataset": {
            "input_database": str(database),
            "sqlite_sha256": database_hash,
            "experiment_id": tables.experiment["experiment_id"],
            "protocol_version": config.get("apparatus", {}).get("protocol_version"),
            "config_hash": tables.experiment["config_hash"],
            "task_hash": config.get("task_set", {}).get("hash"),
            "profile_hash": config.get("profiles", {}).get("hash"),
            "provider": tables.experiment["provider_name"],
            "model": tables.experiment["model_name"],
            "experiment_code_commit": tables.experiment["code_commit"],
            "trials_per_condition": config.get("trials_per_condition"),
            "rounds_per_trial": config.get("rounds_per_trial"),
            "population_size": config.get("apparatus", {}).get("population_size"),
        },
        "analysis": {
            "version": E01_ANALYSIS_VERSION,
            "git_commit": analysis_commit,
            "git_worktree_dirty": analysis_dirty,
            "source_hashes": _analysis_source_hashes(repository),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "exploratory pilot analysis; not confirmatory",
            "database_access": "SQLite URI mode=ro with PRAGMA query_only=ON",
        },
        "integrity": integrity,
        "recommendation": recommendation,
    }
    if metrics is not None:
        report.update(metrics)
        report["effects"] = metrics["performance"]["paired_differences"]
    return report, tables
