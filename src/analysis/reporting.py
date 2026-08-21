from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .integrity import AnalysisTables, Row


REQUIRED_MARKDOWN_SECTIONS = (
    "Status",
    "Dataset",
    "Integrity Audit",
    "Experimental Design",
    "Objective Performance",
    "Selection Quality",
    "Ballot Behavior",
    "Peer Selection Mechanics",
    "Objective Selection Mechanics",
    "Random Condition Sanity",
    "Profile Dynamics",
    "Replacement Dynamics",
    "Condition Divergence",
    "Task Diagnostics",
    "Exploratory Effect Estimates",
    "Apparatus Problems Found",
    "Interpretation",
    "Limitations",
    "Recommendations Before E03",
    "Go / Revise Decision",
)


def _fmt(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value * 100:.1f}%" if percent else f"{value:.4f}"
    return str(value)


def _validate_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("analysis output contains a non-finite float")
    if isinstance(value, Mapping):
        for nested in value.values():
            _validate_json(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json(nested)


def deterministic_json(report: Mapping[str, Any]) -> str:
    _validate_json(report)
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _condition_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Condition | Correct | Responses | Mean score | Trial SD | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("peer_vote", "objective", "random"):
        row = report["performance"]["conditions"][condition]
        summary = row["trial_mean_score"]
        ci = summary["bootstrap_95_ci"]
        lines.append(
            f"| `{condition}` | {row['total_correct']} | {row['total_responses']} | "
            f"{_fmt(row['mean_response_score'], percent=True)} | "
            f"{_fmt(summary['standard_deviation'])} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] |"
        )
    return lines


def render_markdown(report: dict[str, Any], figure_names: Sequence[str]) -> str:
    integrity = report["integrity"]
    recommendation = report["recommendation"]
    lines = [
        "# E01 Pilot Analysis",
        "",
        "## Status",
        "",
        f"**DATASET INTEGRITY: {integrity['status']}**",
        "",
        f"**{recommendation['decision']}**",
        "",
        "This is an exploratory pilot analysis, not confirmatory evidence.",
        "",
        "## Dataset",
        "",
        f"- Input database: `{report['dataset']['input_database']}`",
        f"- SQLite SHA-256: `{report['dataset']['sqlite_sha256']}`",
        f"- Experiment: `{report['dataset']['experiment_id']}`",
        f"- Protocol: `{report['dataset']['protocol_version']}`",
        f"- Config hash: `{report['dataset']['config_hash']}`",
        f"- Task hash: `{report['dataset']['task_hash']}`",
        f"- Profile hash: `{report['dataset']['profile_hash']}`",
        f"- Experiment commit: `{report['dataset']['experiment_code_commit']}`",
        f"- Analysis version: `{report['analysis']['version']}`",
        f"- Analysis commit: `{report['analysis']['git_commit']}`",
        f"- Analysis worktree dirty: `{report['analysis']['git_worktree_dirty']}`",
        f"- Analysis source hashes: `{len(report['analysis']['source_hashes'])}` recorded files",
        f"- Analysis timestamp: `{report['analysis']['timestamp_utc']}`",
        "",
        "## Integrity Audit",
        "",
        f"The read-only audit found {len(integrity['failures'])} critical failures.",
        "",
        "| Check | Result | Details |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{row['name']}` | {'PASS' if row['passed'] else 'FAIL'} | {row['details']} |"
        for row in integrity["checks"]
    )
    if "performance" not in report:
        for section in REQUIRED_MARKDOWN_SECTIONS[3:]:
            lines.extend(
                [
                    "",
                    f"## {section}",
                    "",
                    "Substantive analysis withheld because the dataset integrity gate failed.",
                ]
            )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "## Experimental Design",
            "",
            f"E01 contains {report['dataset']['trials_per_condition']} matched replicates in each of `peer_vote`, `objective`, and `random`, with {report['dataset']['rounds_per_trial']} rounds and {report['dataset']['population_size']} active agents per round. Ballots are anonymous and persistent identity/history are not exposed. Therefore E01 can characterize within-round support and selection, but cannot test reciprocity, reputation, alliances, or social memory.",
            "",
            "Round-0 response equality across matched conditions: "
            f"`{report['cross_condition_comparability']['all_round_zero_responses_identical']}`. "
            "Round-0 ballot equality: "
            f"`{report['cross_condition_comparability']['all_round_zero_ballots_identical']}`.",
            "",
            f"Matched round-0 response seeds agreed for {_fmt(report['cross_condition_comparability']['response_seed_match_rate'], percent=True)} of agent groups, but response content agreed for {_fmt(report['cross_condition_comparability']['response_content_match_rate'], percent=True)}. Ballot seeds and candidate orders agreed at {_fmt(report['cross_condition_comparability']['ballot_seed_match_rate'], percent=True)} and {_fmt(report['cross_condition_comparability']['candidate_order_match_rate'], percent=True)}, while ballot choices agreed at {_fmt(report['cross_condition_comparability']['ballot_choice_match_rate'], percent=True)}.",
            "",
            "## Objective Performance",
            "",
            "**OBSERVATION**",
            "",
        ]
    )
    lines.extend(_condition_table(report))
    lines.extend(
        [
            "",
            "Performance over rounds is reported in `tables/performance_by_round.csv` and `figures/objective_performance_over_round.png`.",
            "",
            *[
                f"- `{condition}`: round 0 mean {_fmt(next(row['mean'] for row in report['performance']['round_rows'] if row['condition'] == condition and row['round_index'] == 0), percent=True)}; final-round mean {_fmt(next(row['mean'] for row in report['performance']['round_rows'] if row['condition'] == condition and row['round_index'] == report['dataset']['rounds_per_trial'] - 1), percent=True)}."
                for condition in ("peer_vote", "objective", "random")
            ],
            "",
            "**INTERPRETATION**",
            "",
            "Condition differences are pilot estimates from matched trial replicates. They may reflect selection-induced population composition and repeated task schedules; they are not confirmatory causal estimates for a future protocol.",
            "The three round trajectories moved together and all peaked around round 4 before declining. Because matched conditions received the same task at each round, this observed temporal pattern is strongly entangled with task schedule and is not evidence of progressive learning or selection improvement.",
            "",
            "## Selection Quality",
            "",
            "| Condition | Selected incorrect, all rounds | Selected incorrect, mixed rounds | Mean actual survivor score |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in ("peer_vote", "objective", "random"):
        row = report["selection"]["conditions"][condition]
        lines.append(
            f"| `{condition}` | {_fmt(row['all_rounds']['selected_incorrect_rate'], percent=True)} | "
            f"{_fmt(row['mixed_score_rounds']['selected_incorrect_rate'], percent=True)} | "
            f"{_fmt(row['all_rounds']['mean_actual_survivor_score'])} |"
        )
    lines.extend(
        [
            "",
            "Mixed-score and degenerate-round counts are retained separately so binary-score ties do not obscure selection quality.",
            "",
            "## Ballot Behavior",
            "",
            "| Condition | Supports correct | Candidate-aware baseline | Best-response agreement | Tie-aware best baseline |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in ("peer_vote", "objective", "random"):
        row = report["ballots"]["conditions"][condition]
        lines.append(
            f"| `{condition}` | {_fmt(row['support_correct_rate'], percent=True)} | "
            f"{_fmt(row['mean_support_correct_chance_baseline'], percent=True)} | "
            f"{_fmt(row['best_response_agreement_rate'], percent=True)} | "
            f"{_fmt(row['mean_best_tie_aware_chance_baseline'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "Support entropy and maximum support share describe within-round concentration only. They are not evidence of coalitions.",
            "",
            f"Across all conditions, the most-used anonymous display label accounted for {_fmt(report['ballots']['position_max_share'], percent=True)} of ballots. Full A-G frequencies are in the JSON report.",
            "",
            "| Condition | Mean max support share | Mean normalized entropy | Mean supported agents |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in ("peer_vote", "objective", "random"):
        row = report["ballots"]["concentration"][condition]
        lines.append(
            f"| `{condition}` | {_fmt(row['mean_max_support_share'], percent=True)} | "
            f"{_fmt(row['mean_normalized_support_entropy'])} | "
            f"{_fmt(row['mean_agents_with_support'])} |"
        )
    lines.extend(
        [
            "",
            "## Peer Selection Mechanics",
            "",
            f"- Minimum-support tie rate: {_fmt(report['selection']['peer_mechanics']['tie_rate'], percent=True)}",
            f"- Mean minimum-support tie size: {_fmt(report['selection']['peer_mechanics']['mean_minimum_tie_size'])}",
            f"- Selected agent received zero support: {_fmt(report['selection']['peer_mechanics']['selected_zero_support_rate'], percent=True)}",
            "",
            "## Objective Selection Mechanics",
            "",
            f"- Lowest-score tie rate: {_fmt(report['selection']['objective_mechanics']['tie_rate'], percent=True)}",
            f"- Mean tied-lowest set size: {_fmt(report['selection']['objective_mechanics']['mean_lowest_tie_size'])}",
            f"- Mixed-score round rate: {_fmt(report['selection']['objective_mechanics']['mixed_score_round_rate'], percent=True)}",
            f"- Selected incorrect when mixed: {_fmt(report['selection']['objective_mechanics']['selected_incorrect_rate_when_mixed'], percent=True)}",
            "",
            "## Random Condition Sanity",
            "",
            f"Selected-agent correctness was {_fmt(report['random_sanity']['selected_correct_rate'], percent=True)} versus {_fmt(report['random_sanity']['population_correct_rate'], percent=True)} among all random-condition agent-rounds. Mean selected support was {_fmt(report['random_sanity']['mean_selected_support'])} versus {_fmt(report['random_sanity']['mean_population_support'])} overall. These are descriptive checks over {report['selection']['conditions']['random']['all_rounds']['rounds']} selections.",
            "",
            "## Profile Dynamics",
            "",
        ]
    )
    profile_rows = report["profiles"]["profile_rows"]
    for condition in ("peer_vote", "objective", "random"):
        rows = [row for row in profile_rows if row["condition"] == condition]
        highest = max(rows, key=lambda row: row["mean_response_score"])
        lowest = min(rows, key=lambda row: row["mean_response_score"])
        longest = max(rows, key=lambda row: row["mean_observed_active_rounds"])
        lines.append(
            f"- `{condition}`: highest observed profile accuracy `{highest['profile_id']}` "
            f"({_fmt(highest['mean_response_score'], percent=True)}), lowest `{lowest['profile_id']}` "
            f"({_fmt(lowest['mean_response_score'], percent=True)}), greatest mean observed "
            f"active time `{longest['profile_id']}` "
            f"({_fmt(longest['mean_observed_active_rounds'])} rounds; right-censoring retained)."
        )
    lines.extend(
        [
            "",
            "Profile IDs and their persisted approach parameters are listed in `tables/profile_metrics.csv`. Accuracy, support, selections, lifetime, censoring, and round-wise population share are descriptive because profile exposure is repeated within matched trials.",
            "",
            "## Replacement Dynamics",
            "",
            "The integrity audit verifies that all matched conditions receive the same incoming profile and agent queue at every queue index. Different outgoing selections can therefore create endogenous composition differences despite matched replacement supply.",
            "",
            "## Condition Divergence",
            "",
        ]
    )
    final_round = max(row["round_index"] for row in report["divergence"]["by_round"])
    final_divergence = [
        row
        for row in report["divergence"]["by_round"]
        if row["round_index"] == final_round
    ]
    for row in final_divergence:
        lines.append(
            f"- `{row['comparison']}` at round {final_round}: mean shared-agent fraction "
            f"{_fmt(row['mean_shared_agent_fraction'], percent=True)}; mean normalized profile L1 "
            f"distance {_fmt(row['mean_profile_count_normalized_l1'])}."
        )
    always_incorrect = [
        row["task_id"] for row in report["tasks"]["task_rows"] if row["always_incorrect"]
    ]
    always_correct = [
        row["task_id"] for row in report["tasks"]["task_rows"] if row["always_correct"]
    ]
    lines.extend(
        [
            "",
            "Shared-agent fractions and profile-count L1 distances are reported by matched replicate and round. Round 0 is an intentionally matched state; later divergence is expected after condition-specific selections.",
            "",
            "## Task Diagnostics",
            "",
            f"Always incorrect tasks: {', '.join(f'`{value}`' for value in always_incorrect) or 'none'}.",
            "",
            f"Always correct tasks: {', '.join(f'`{value}`' for value in always_correct) or 'none'}.",
            "",
            "Per-task and task-family exact-match rates appear in `tables/task_metrics.csv`. Condition skew is descriptive and may reflect endogenous population composition rather than a task-specific condition effect.",
            "",
            "## Exploratory Effect Estimates",
            "",
            "| Contrast | Mean paired difference | Median | 95% bootstrap CI | Standardized paired difference | Exploratory sign-flip p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["effects"].items():
        ci = row["bootstrap_95_ci"]
        lines.append(
            f"| `{name}` | {_fmt(row['mean_paired_difference'])} | "
            f"{_fmt(row['median_paired_difference'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"{_fmt(row['standardized_paired_difference'])} | "
            f"{_fmt(row['exploratory_exact_sign_flip_p'])} |"
        )
    failed_gates = [row for row in recommendation["gates"] if not row["passed"]]
    lines.extend(
        [
            "",
            "All intervals and tests above are exploratory pilot analyses and are not confirmatory evidence.",
            "",
            "## Apparatus Problems Found",
            "",
        ]
    )
    if failed_gates:
        lines.extend(
            f"- `{row['name']}` did not meet its explicit gate; observed={row['observed']}."
            for row in failed_gates
        )
    else:
        lines.append("- No predeclared readiness gate failed in this analysis.")
    lines.extend(f"- Warning: {value}" for value in recommendation.get("warnings", []))
    lines.extend(
        [
            "- E01's anonymous, history-free ballots cannot identify reciprocity, reputation, identity-based alliances, betrayal, or strategic cooperation.",
            "- The binary exact-match scorer creates substantial tied selection sets even when it correctly distinguishes incorrect from correct responses.",
            "",
            "## Interpretation",
            "",
            "**OBSERVATION:** The report records objective performance, selected-agent quality, anonymous support alignment, concentration, profile trajectories, and matched-condition divergence directly from persisted evidence.",
            "",
            "**INTERPRETATION:** These observed patterns provide preliminary evidence about whether the apparatus creates measurable selection pressure. They do not establish intentional behavior or generalize beyond this model, task set, and pilot protocol.",
            "",
            "## Limitations",
            "",
            "- Ten matched trial replicates per condition provide limited independent replication.",
            "- The same small validated task pool is reused across rounds and replicates.",
            "- One small local model and deterministic inference setting were studied.",
            "- Profiles are prompt configurations, not psychological traits; semantic effects require a separate manipulation check.",
            "- Later-round cross-condition ballot comparisons occur in endogenously different populations.",
            "- Final-round selection has no replacement, so terminal selection and actual removal are distinct.",
            "",
            "## Recommendations Before E03",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in recommendation["recommendations"])
    lines.extend(
        [
            "",
            "## Go / Revise Decision",
            "",
            f"**{recommendation['decision']}**",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in recommendation["reasons"])
    lines.extend(
        [
            "",
            "## Research-Question Mapping",
            "",
            "| Question | E01 can answer? | Pilot result | Next step |",
            "|---|---|---|---|",
            "| Does peer selection change objective performance? | Exploratorily | Matched pilot estimate reported above | Freeze estimand and power E03 at trial level |",
            "| Does peer support align with correctness? | Yes, descriptively | Candidate-aware alignment reported above | Replicate with held-out tasks |",
            "| Do conditions create different population trajectories? | Yes, descriptively | Shared-agent and profile distances reported | Preserve matched replacement supply |",
            "| Do profile types have different observed active time? | Exploratorily | Censored observed active rounds and composition reported | Run profile manipulation check before survival modeling |",
            "| Does anonymous peer voting show position bias? | Yes, descriptively | A-G frequencies and maximum share reported | Retain candidate-order audit |",
            "| Does peer selection produce persistent reciprocity? | No | Identity/history are not exposed in E01 | Test only in a separately randomized identity/history protocol |",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `figures/{name}`" for name in figure_names)
    return "\n".join(lines) + "\n"


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _summary_rows(report: dict[str, Any], section: str) -> list[Row]:
    if section == "condition_performance":
        return [
            {"condition": condition, **values}
            for condition, values in report["performance"]["conditions"].items()
        ]
    if section == "paired_differences":
        return [
            {"comparison": comparison, **values}
            for comparison, values in report["effects"].items()
        ]
    if section == "selection_quality":
        return [
            {"condition": condition, **values}
            for condition, values in report["selection"]["conditions"].items()
        ]
    if section == "ballot_quality":
        return [
            {"condition": condition, **values}
            for condition, values in report["ballots"]["conditions"].items()
        ]
    raise ValueError(f"unknown summary section: {section}")


def write_analysis_bundle(
    output_dir: str | Path,
    report: dict[str, Any],
    tables: AnalysisTables,
    figure_names: Sequence[str],
) -> list[str]:
    target = Path(output_dir)
    table_dir = target / "tables"
    target.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    json_path = target / "e01_analysis.json"
    markdown_path = target / "e01_analysis.md"
    json_temp = json_path.with_suffix(".json.tmp")
    markdown_temp = markdown_path.with_suffix(".md.tmp")
    json_temp.write_text(deterministic_json(report), encoding="utf-8")
    markdown_temp.write_text(render_markdown(report, figure_names), encoding="utf-8")
    os.replace(json_temp, json_path)
    os.replace(markdown_temp, markdown_path)
    artifacts.extend([str(json_path), str(markdown_path)])

    table_sets: dict[str, Sequence[Row]] = {
        "pilot_integrity.csv": report["integrity"]["checks"]
    }
    if "performance" in report:
        table_sets.update(
            {
                "round_table.csv": tables.round_table,
                "agent_round_table.csv": tables.agent_round_table,
                "ballot_table.csv": tables.ballot_table,
                "replacement_table.csv": tables.replacement_table,
                "condition_performance.csv": _summary_rows(
                    report, "condition_performance"
                ),
                "performance_by_round.csv": report["performance"]["round_rows"],
                "paired_differences.csv": _summary_rows(report, "paired_differences"),
                "selection_quality.csv": _summary_rows(report, "selection_quality"),
                "ballot_quality.csv": _summary_rows(report, "ballot_quality"),
                "profile_metrics.csv": report["profiles"]["profile_rows"],
                "profile_definitions.csv": report["profiles"]["metadata"],
                "profile_composition.csv": report["profiles"]["composition_rows"],
                "profile_lifetimes.csv": report["profiles"]["lifetime_rows"],
                "condition_divergence.csv": report["divergence"]["rows"],
                "task_metrics.csv": report["tasks"]["task_rows"],
                "task_family_metrics.csv": report["tasks"]["family_rows"],
                "position_distribution.csv": [
                    {
                        "anonymous_label": label,
                        "count": report["ballots"]["position_counts"].get(label, 0),
                        "rate": rate,
                    }
                    for label, rate in report["ballots"]["position_rates"].items()
                ],
                "selection_condition_differences.csv": [
                    {"comparison": comparison, **values}
                    for comparison, values in report["selection"][
                        "condition_differences"
                    ].items()
                ],
            }
        )
    for name, rows in table_sets.items():
        path = table_dir / name
        write_csv(path, rows)
        artifacts.append(str(path))
    return artifacts
