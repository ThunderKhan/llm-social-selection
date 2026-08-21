from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


CONDITION_ORDER = ("peer_vote", "objective", "random")


def _finish(figure: Any, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _axes(title: str, xlabel: str, ylabel: str) -> tuple[Any, Any]:
    figure, axes = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(axis="y", alpha=0.25)
    return figure, axes


def generate_figures(report: dict[str, Any], output_dir: str | Path) -> list[str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    conditions = report["performance"]["conditions"]
    values = [conditions[c]["mean_response_score"] for c in CONDITION_ORDER]
    intervals = [conditions[c]["trial_mean_score"]["bootstrap_95_ci"] for c in CONDITION_ORDER]
    lower = [value - interval[0] for value, interval in zip(values, intervals, strict=True)]
    upper = [interval[1] - value for value, interval in zip(values, intervals, strict=True)]
    figure, axes = _axes("Objective Performance by Condition", "Condition", "Mean score")
    axes.bar(CONDITION_ORDER, values, yerr=[lower, upper], capsize=4)
    axes.set_ylim(0, 1)
    path = target / "objective_performance_by_condition.png"
    _finish(figure, path)
    generated.append(path.name)

    figure, axes = _axes("Objective Performance over Rounds", "Round index", "Mean score")
    for condition in CONDITION_ORDER:
        rows = [
            row
            for row in report["performance"]["round_rows"]
            if row["condition"] == condition
        ]
        x = [row["round_index"] for row in rows]
        y = [row["mean"] for row in rows]
        axes.plot(x, y, marker="o", label=condition)
        axes.fill_between(
            x,
            [row["bootstrap_95_ci"][0] for row in rows],
            [row["bootstrap_95_ci"][1] for row in rows],
            alpha=0.15,
        )
    axes.set_ylim(0, 1)
    axes.legend()
    path = target / "objective_performance_over_round.png"
    _finish(figure, path)
    generated.append(path.name)

    selection = report["selection"]["conditions"]
    figure, axes = _axes(
        "Selected-Agent Correctness by Condition", "Condition", "Selected agent score = 1 rate"
    )
    axes.bar(
        CONDITION_ORDER,
        [selection[c]["all_rounds"]["selected_correct_rate"] for c in CONDITION_ORDER],
    )
    axes.set_ylim(0, 1)
    path = target / "selected_agent_correctness.png"
    _finish(figure, path)
    generated.append(path.name)

    ballots = report["ballots"]["conditions"]
    x = range(len(CONDITION_ORDER))
    width = 0.36
    figure, axes = _axes(
        "Support for Correct Responses", "Condition", "Ballot rate"
    )
    axes.bar(
        [index - width / 2 for index in x],
        [ballots[c]["support_correct_rate"] for c in CONDITION_ORDER],
        width,
        label="Observed",
    )
    axes.bar(
        [index + width / 2 for index in x],
        [ballots[c]["mean_support_correct_chance_baseline"] for c in CONDITION_ORDER],
        width,
        label="Available-candidate baseline",
    )
    axes.set_xticks(list(x), CONDITION_ORDER)
    axes.set_ylim(0, 1)
    axes.legend()
    path = target / "ballot_support_correctness.png"
    _finish(figure, path)
    generated.append(path.name)

    peer_rows = [
        row
        for row in report["ballots"]["concentration_rows"]
        if row["condition"] == "peer_vote"
    ]
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in peer_rows:
        by_round[row["round_index"]].append(row)
    figure, axes = _axes(
        "Peer-Condition Support Concentration", "Round index", "Mean proportion"
    )
    rounds = sorted(by_round)
    axes.plot(
        rounds,
        [
            sum(row["normalized_support_entropy"] for row in by_round[index])
            / len(by_round[index])
            for index in rounds
        ],
        marker="o",
        label="Normalized entropy",
    )
    axes.plot(
        rounds,
        [
            sum(row["max_support_share"] for row in by_round[index])
            / len(by_round[index])
            for index in rounds
        ],
        marker="o",
        label="Maximum support share",
    )
    axes.set_ylim(0, 1)
    axes.legend()
    path = target / "peer_support_concentration_over_round.png"
    _finish(figure, path)
    generated.append(path.name)

    composition = report["profiles"]["composition_rows"]
    profiles = sorted({row["profile_id"] for row in composition})
    for condition in CONDITION_ORDER:
        figure, axes = _axes(
            f"Profile Composition over Rounds: {condition}",
            "Round index",
            "Mean population share",
        )
        for profile in profiles:
            rows = [
                row
                for row in composition
                if row["condition"] == condition and row["profile_id"] == profile
            ]
            axes.plot(
                [row["round_index"] for row in rows],
                [row["mean_population_share"] for row in rows],
                marker="o",
                label=profile.replace("e01-profile-", "P"),
            )
        axes.set_ylim(0, 0.3)
        axes.legend(ncol=2, fontsize=8)
        path = target / f"profile_composition_{condition}.png"
        _finish(figure, path)
        generated.append(path.name)

    profile_rows = report["profiles"]["profile_rows"]
    figure, axes = _axes(
        "Observed Profile Follow-up by Condition", "Profile", "Mean observed active rounds"
    )
    for condition in CONDITION_ORDER:
        rows = [row for row in profile_rows if row["condition"] == condition]
        axes.plot(
            range(len(rows)),
            [row["mean_observed_active_rounds"] for row in rows],
            marker="o",
            label=condition,
        )
    axes.set_xticks(range(len(profiles)), [p.replace("e01-profile-", "P") for p in profiles])
    axes.legend()
    path = target / "profile_lifetime.png"
    _finish(figure, path)
    generated.append(path.name)

    figure, axes = _axes("Round Score Composition", "Condition", "Round rate")
    categories = ("mixed_score_round_rate", "all_correct_round_count", "all_incorrect_round_count")
    labels = ("Mixed", "All correct", "All incorrect")
    width = 0.24
    for offset, (field, label) in enumerate(zip(categories, labels, strict=True)):
        values = []
        for condition in CONDITION_ORDER:
            value = selection[condition][field]
            rounds_in_condition = selection[condition]["all_rounds"]["rounds"]
            values.append(
                value if field.endswith("rate") else value / rounds_in_condition
            )
        axes.bar(
            [index + (offset - 1) * width for index in x], values, width, label=label
        )
    axes.set_xticks(list(x), CONDITION_ORDER)
    axes.set_ylim(0, 1)
    axes.legend()
    path = target / "round_score_composition.png"
    _finish(figure, path)
    generated.append(path.name)

    figure, axes = _axes(
        "Condition Population Divergence", "Round index", "Mean shared-agent fraction"
    )
    for comparison in sorted(
        {row["comparison"] for row in report["divergence"]["by_round"]}
    ):
        rows = [
            row
            for row in report["divergence"]["by_round"]
            if row["comparison"] == comparison
        ]
        axes.plot(
            [row["round_index"] for row in rows],
            [row["mean_shared_agent_fraction"] for row in rows],
            marker="o",
            label=comparison,
        )
    axes.set_ylim(0, 1)
    axes.legend(fontsize=8)
    path = target / "condition_divergence.png"
    _finish(figure, path)
    generated.append(path.name)

    distribution = report["selection"]["peer_mechanics"]["selected_support_distribution"]
    figure, axes = _axes(
        "Peer-Selected Agent Support Counts", "Support received", "Selected agents"
    )
    keys = sorted(int(key) for key in distribution)
    axes.bar(keys, [distribution.get(key, distribution.get(str(key), 0)) for key in keys])
    axes.set_xticks(range(8))
    path = target / "peer_selected_support_distribution.png"
    _finish(figure, path)
    generated.append(path.name)

    task_rows = sorted(
        report["tasks"]["task_rows"], key=lambda row: row["exact_match_rate"]
    )
    figure, axes = _axes("Task Difficulty", "Exact-match rate", "Task")
    axes.barh(
        [row["task_id"] for row in task_rows],
        [row["exact_match_rate"] for row in task_rows],
    )
    axes.set_xlim(0, 1)
    path = target / "task_difficulty.png"
    _finish(figure, path)
    generated.append(path.name)
    return generated
