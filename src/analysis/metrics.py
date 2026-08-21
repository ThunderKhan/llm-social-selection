from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable, Sequence

from .integrity import AnalysisTables, CONDITIONS, Row
from .statistics import describe, entropy, l1_distance, paired_difference, safe_rate


COMPARISONS = (
    ("objective", "random"),
    ("peer_vote", "random"),
    ("peer_vote", "objective"),
)


def objective_mean(rows: Sequence[Row]) -> float | None:
    values = [float(row["score"]) for row in rows if row.get("score") is not None]
    return None if not values else mean(values)


def is_mixed_score(scores: Sequence[float]) -> bool:
    return 0.0 in scores and 1.0 in scores


def support_correctness(ballot: Row) -> bool | None:
    score = ballot.get("supported_agent_score")
    return None if score is None else float(score) == 1.0


def selected_agent_correctness(round_row: Row) -> bool | None:
    score = round_row.get("selected_agent_score")
    return None if score is None else float(score) == 1.0


def profile_lifetime(row: Row) -> int:
    return int(row["lifetime_rounds"])


def profile_composition(rows: Sequence[Row]) -> dict[str, int]:
    return dict(sorted(Counter(row["profile_id"] for row in rows).items()))


def _group(rows: Iterable[Row], *keys: str) -> dict[tuple[Any, ...], list[Row]]:
    grouped: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _profile_metadata(config: Row) -> list[Row]:
    result = []
    for value in config.get("profiles", {}).get("values", []):
        if isinstance(value, dict):
            result.append(
                {
                    "profile_id": value["profile_id"],
                    "parameters": value.get("parameters", {}),
                    "template_version": value.get("template_version"),
                }
            )
        else:
            result.append(
                {
                    "profile_id": value[0],
                    "parameters": {"approach": value[1]},
                    "template_version": value[2],
                }
            )
    return result


def analyze_performance(tables: AnalysisTables) -> dict[str, Any]:
    trial_groups = _group(tables.agent_round_table, "condition", "replicate")
    trial_rows = []
    for (condition, replicate), rows in sorted(trial_groups.items()):
        scores = [float(row["score"]) for row in rows]
        trial_rows.append(
            {
                "condition": condition,
                "replicate": replicate,
                "trial_id": rows[0]["trial_id"],
                "correct": sum(value == 1 for value in scores),
                "responses": len(scores),
                "mean_score": mean(scores),
            }
        )
    conditions = {}
    for condition in CONDITIONS:
        rows = [row for row in trial_rows if row["condition"] == condition]
        values = [row["mean_score"] for row in rows]
        conditions[condition] = {
            "total_correct": sum(row["correct"] for row in rows),
            "total_responses": sum(row["responses"] for row in rows),
            "mean_response_score": safe_rate(
                sum(row["correct"] for row in rows),
                sum(row["responses"] for row in rows),
            ),
            "trial_mean_score": describe(values, label=f"performance:{condition}"),
        }
    round_rows = []
    round_groups = _group(tables.round_table, "condition", "round_index")
    for (condition, round_index), rows in sorted(round_groups.items()):
        values = [float(row["mean_score"]) for row in rows]
        summary = describe(values, label=f"performance:{condition}:round:{round_index}")
        round_rows.append(
            {
                "condition": condition,
                "round_index": round_index,
                **summary,
            }
        )
    trial_lookup = {
        (row["condition"], row["replicate"]): row["mean_score"] for row in trial_rows
    }
    effects = {}
    replicates = sorted({row["replicate"] for row in trial_rows})
    for left, right in COMPARISONS:
        effects[f"{left}_minus_{right}"] = paired_difference(
            [trial_lookup[(left, replicate)] for replicate in replicates],
            [trial_lookup[(right, replicate)] for replicate in replicates],
            label=f"paired:{left}:{right}",
        )
    return {
        "conditions": conditions,
        "trial_rows": trial_rows,
        "round_rows": round_rows,
        "paired_differences": effects,
    }


def _round_support(rows: Sequence[Row]) -> list[int]:
    return [int(row["support_received"]) for row in rows]


def analyze_selection(tables: AnalysisTables) -> dict[str, Any]:
    agent_groups = _group(tables.agent_round_table, "trial_id", "round_index")
    details = []
    for round_row in tables.round_table:
        key = (round_row["trial_id"], round_row["round_index"])
        agents = agent_groups[key]
        scores = [float(row["score"]) for row in agents]
        selected = next(row for row in agents if row["selected_this_round"])
        survivor_scores = [
            float(row["score"]) for row in agents if not row["selected_this_round"]
        ]
        support_counts = _round_support(agents)
        minimum_support = min(support_counts)
        lowest_score = min(scores)
        details.append(
            {
                "condition": round_row["condition"],
                "replicate": round_row["replicate"],
                "trial_id": round_row["trial_id"],
                "round_index": round_row["round_index"],
                "selected_agent_score": float(selected["score"]),
                "selected_support": int(selected["support_received"]),
                "nonselected_mean_score": mean(survivor_scores),
                "terminal_round": bool(selected["terminal_round"]),
                "mixed_score_round": is_mixed_score(scores),
                "all_correct_round": all(value == 1 for value in scores),
                "all_incorrect_round": all(value == 0 for value in scores),
                "lowest_score_tie_size": sum(value == lowest_score for value in scores),
                "minimum_support": minimum_support,
                "minimum_support_tie_size": sum(
                    value == minimum_support for value in support_counts
                ),
            }
        )
    conditions = {}
    for condition in CONDITIONS:
        rows = [row for row in details if row["condition"] == condition]
        mixed = [row for row in rows if row["mixed_score_round"]]

        def subset_summary(subset: Sequence[Row]) -> dict[str, Any]:
            return {
                "rounds": len(subset),
                "selected_incorrect_count": sum(
                    row["selected_agent_score"] == 0 for row in subset
                ),
                "selected_incorrect_rate": safe_rate(
                    sum(row["selected_agent_score"] == 0 for row in subset), len(subset)
                ),
                "selected_correct_rate": safe_rate(
                    sum(row["selected_agent_score"] == 1 for row in subset), len(subset)
                ),
                "mean_selected_score": (
                    None
                    if not subset
                    else mean(row["selected_agent_score"] for row in subset)
                ),
                "mean_nonselected_score": (
                    None
                    if not subset
                    else mean(row["nonselected_mean_score"] for row in subset)
                ),
                "mean_actual_survivor_score": (
                    None
                    if not any(not row["terminal_round"] for row in subset)
                    else mean(
                        row["nonselected_mean_score"]
                        for row in subset
                        if not row["terminal_round"]
                    )
                ),
            }

        conditions[condition] = {
            "all_rounds": subset_summary(rows),
            "mixed_score_rounds": subset_summary(mixed),
            "mixed_score_round_count": len(mixed),
            "mixed_score_round_rate": safe_rate(len(mixed), len(rows)),
            "all_correct_round_count": sum(row["all_correct_round"] for row in rows),
            "all_incorrect_round_count": sum(row["all_incorrect_round"] for row in rows),
        }
    selection_effects = {}
    for left, right in COMPARISONS:
        left_all = conditions[left]["all_rounds"]["selected_incorrect_rate"]
        right_all = conditions[right]["all_rounds"]["selected_incorrect_rate"]
        left_mixed = conditions[left]["mixed_score_rounds"]["selected_incorrect_rate"]
        right_mixed = conditions[right]["mixed_score_rounds"]["selected_incorrect_rate"]
        selection_effects[f"{left}_minus_{right}"] = {
            "selected_incorrect_rate_difference_all_rounds": left_all - right_all,
            "selected_incorrect_rate_difference_mixed_rounds": (
                None
                if left_mixed is None or right_mixed is None
                else left_mixed - right_mixed
            ),
            "unit": "absolute proportion difference",
        }
    peer_rows = [row for row in details if row["condition"] == "peer_vote"]
    objective_rows = [row for row in details if row["condition"] == "objective"]
    peer = {
        "rounds": len(peer_rows),
        "tie_rate": safe_rate(
            sum(row["minimum_support_tie_size"] > 1 for row in peer_rows), len(peer_rows)
        ),
        "mean_minimum_tie_size": mean(
            row["minimum_support_tie_size"] for row in peer_rows
        ),
        "selected_zero_support_rate": safe_rate(
            sum(row["selected_support"] == 0 for row in peer_rows), len(peer_rows)
        ),
        "selected_support_distribution": dict(
            sorted(Counter(row["selected_support"] for row in peer_rows).items())
        ),
    }
    objective_mixed = [row for row in objective_rows if row["mixed_score_round"]]
    objective = {
        "rounds": len(objective_rows),
        "tie_rate": safe_rate(
            sum(row["lowest_score_tie_size"] > 1 for row in objective_rows),
            len(objective_rows),
        ),
        "mean_lowest_tie_size": mean(
            row["lowest_score_tie_size"] for row in objective_rows
        ),
        "mixed_score_round_rate": safe_rate(len(objective_mixed), len(objective_rows)),
        "selected_incorrect_rate_when_mixed": safe_rate(
            sum(row["selected_agent_score"] == 0 for row in objective_mixed),
            len(objective_mixed),
        ),
    }
    return {
        "conditions": conditions,
        "condition_differences": selection_effects,
        "round_details": details,
        "peer_mechanics": peer,
        "objective_mechanics": objective,
    }


def analyze_ballots(tables: AnalysisTables) -> dict[str, Any]:
    candidates_by_ballot = _group(tables.candidate_rows, "ballot_id")
    details = []
    for ballot in tables.ballot_table:
        candidates = candidates_by_ballot[(ballot.get("ballot_id"),)] if ballot.get("ballot_id") else []
        if not candidates:
            candidates = [
                row
                for row in tables.candidate_rows
                if row["trial_id"] == ballot["trial_id"]
                and row["round_index"] == ballot["round_index"]
                and row["voter_agent_id"] == ballot["voter_agent_id"]
            ]
        scores = [float(row["candidate_score"]) for row in candidates]
        supported_score = ballot["supported_agent_score"]
        best = max(scores) if scores else None
        details.append(
            {
                **ballot,
                "supports_correct": (
                    None if supported_score is None else float(supported_score) == 1
                ),
                "support_correct_chance_baseline": (
                    None
                    if not scores
                    else sum(value == 1 for value in scores) / len(scores)
                ),
                "supports_best_available": (
                    None
                    if best is None or supported_score is None
                    else float(supported_score) == best
                ),
                "best_tie_aware_chance_baseline": (
                    None
                    if best is None
                    else sum(value == best for value in scores) / len(scores)
                ),
            }
        )
    conditions = {}
    for condition in CONDITIONS:
        rows = [row for row in details if row["condition"] == condition]
        conditions[condition] = {
            "ballots": len(rows),
            "valid": sum(row["valid"] for row in rows),
            "support_correct_rate": safe_rate(
                sum(row["supports_correct"] is True for row in rows), len(rows)
            ),
            "mean_support_correct_chance_baseline": mean(
                row["support_correct_chance_baseline"] for row in rows
            ),
            "best_response_agreement_rate": safe_rate(
                sum(row["supports_best_available"] is True for row in rows), len(rows)
            ),
            "mean_best_tie_aware_chance_baseline": mean(
                row["best_tie_aware_chance_baseline"] for row in rows
            ),
        }
    agent_groups = _group(tables.agent_round_table, "trial_id", "round_index")
    concentration_rows = []
    for (trial_id, round_index), rows in sorted(agent_groups.items()):
        counts = [int(row["support_received"]) for row in rows]
        concentration_rows.append(
            {
                "condition": rows[0]["condition"],
                "replicate": rows[0]["replicate"],
                "trial_id": trial_id,
                "round_index": round_index,
                "max_support_share": max(counts) / sum(counts),
                "support_entropy_bits": entropy(counts),
                "normalized_support_entropy": entropy(counts, normalized=True),
                "agents_with_support": sum(value >= 1 for value in counts),
            }
        )
    concentration = {}
    for condition in CONDITIONS:
        rows = [row for row in concentration_rows if row["condition"] == condition]
        concentration[condition] = {
            "mean_max_support_share": mean(row["max_support_share"] for row in rows),
            "mean_support_entropy_bits": mean(
                row["support_entropy_bits"] for row in rows
            ),
            "mean_normalized_support_entropy": mean(
                row["normalized_support_entropy"] for row in rows
            ),
            "mean_agents_with_support": mean(row["agents_with_support"] for row in rows),
        }
    position_counts = Counter(
        row["anonymous_label"] for row in details if row["anonymous_label"] is not None
    )
    position_rates = {
        label: safe_rate(position_counts[label], sum(position_counts.values()))
        for label in "ABCDEFG"
    }
    return {
        "conditions": conditions,
        "concentration": concentration,
        "concentration_rows": concentration_rows,
        "position_counts": dict(position_counts),
        "position_rates": position_rates,
        "position_max_share": max(position_rates.values()),
    }


def analyze_profiles(tables: AnalysisTables) -> dict[str, Any]:
    initial_agents = [row for row in tables.agents if row["introduced_round"] is None]
    profile_rows = []
    for condition in CONDITIONS:
        profiles = sorted({row["profile_id"] for row in tables.agents})
        for profile_id in profiles:
            agent_rounds = [
                row
                for row in tables.agent_round_table
                if row["condition"] == condition and row["profile_id"] == profile_id
            ]
            lifetimes = [
                row
                for row in tables.lifetime_table
                if row["condition"] == condition and row["profile_id"] == profile_id
            ]
            profile_rows.append(
                {
                    "condition": condition,
                    "profile_id": profile_id,
                    "initial_count": sum(
                        row["condition"] == condition and row["profile_id"] == profile_id
                        for row in initial_agents
                    ),
                    "agent_rounds": len(agent_rounds),
                    "selections": sum(row["selected_this_round"] for row in agent_rounds),
                    "actual_eliminations": sum(
                        row["selected_and_replaced"] for row in lifetimes
                    ),
                    "mean_observed_active_rounds": mean(
                        profile_lifetime(row) for row in lifetimes
                    ),
                    "right_censored_instances": sum(
                        row["right_censored"] for row in lifetimes
                    ),
                    "mean_response_score": objective_mean(agent_rounds),
                    "mean_support_received": mean(
                        row["support_received"] for row in agent_rounds
                    ),
                }
            )
    composition_rows = []
    for (condition, round_index), rows in sorted(
        _group(tables.agent_round_table, "condition", "round_index").items()
    ):
        counts = Counter(row["profile_id"] for row in rows)
        trials = len({row["trial_id"] for row in rows})
        for profile_id in sorted({row["profile_id"] for row in tables.agents}):
            composition_rows.append(
                {
                    "condition": condition,
                    "round_index": round_index,
                    "profile_id": profile_id,
                    "population_count": counts[profile_id],
                    "mean_count_per_trial": counts[profile_id] / trials,
                    "mean_population_share": counts[profile_id] / len(rows),
                }
            )
    return {
        "metadata": _profile_metadata(tables.config),
        "profile_rows": profile_rows,
        "composition_rows": composition_rows,
        "lifetime_rows": list(tables.lifetime_table),
    }


def analyze_divergence(tables: AnalysisTables) -> dict[str, Any]:
    population_size = int(tables.config["apparatus"]["population_size"])
    populations = {
        (condition, replicate, round_index): rows
        for (condition, replicate, round_index), rows in _group(
            tables.agent_round_table, "condition", "replicate", "round_index"
        ).items()
    }
    selected = {
        (row["condition"], row["replicate"], row["round_index"]): row[
            "selected_agent_id"
        ]
        for row in tables.round_table
    }
    rows = []
    replicates = sorted({row["replicate"] for row in tables.trials})
    rounds = sorted({row["round_index"] for row in tables.rounds})
    for left, right in COMPARISONS:
        comparison = f"{left}_vs_{right}"
        for replicate in replicates:
            for round_index in rounds:
                left_rows = populations[(left, replicate, round_index)]
                right_rows = populations[(right, replicate, round_index)]
                left_ids = {row["agent_id"] for row in left_rows}
                right_ids = {row["agent_id"] for row in right_rows}
                left_profiles = Counter(row["profile_id"] for row in left_rows)
                right_profiles = Counter(row["profile_id"] for row in right_rows)
                rows.append(
                    {
                        "comparison": comparison,
                        "replicate": replicate,
                        "round_index": round_index,
                        "shared_agent_identities": len(left_ids & right_ids),
                        "shared_agent_fraction": len(left_ids & right_ids) / population_size,
                        "profile_count_l1_distance": l1_distance(
                            dict(left_profiles), dict(right_profiles)
                        ),
                        "profile_count_normalized_l1": l1_distance(
                            dict(left_profiles), dict(right_profiles)
                        )
                        / (2 * population_size),
                        "selected_agent_overlap": selected[
                            (left, replicate, round_index)
                        ]
                        == selected[(right, replicate, round_index)],
                    }
                )
    by_round = []
    for (comparison, round_index), grouped in sorted(
        _group(rows, "comparison", "round_index").items()
    ):
        by_round.append(
            {
                "comparison": comparison,
                "round_index": round_index,
                "mean_shared_agent_fraction": mean(
                    row["shared_agent_fraction"] for row in grouped
                ),
                "mean_profile_count_normalized_l1": mean(
                    row["profile_count_normalized_l1"] for row in grouped
                ),
                "selected_agent_overlap_rate": safe_rate(
                    sum(row["selected_agent_overlap"] for row in grouped), len(grouped)
                ),
            }
        )
    return {"rows": rows, "by_round": by_round}


def analyze_task_diagnostics(tables: AnalysisTables) -> dict[str, Any]:
    task_rows = []
    for task in tables.tasks:
        rows = [
            row for row in tables.agent_round_table if row["task_id"] == task["task_id"]
        ]
        condition_rates = {}
        for condition in CONDITIONS:
            condition_rows = [row for row in rows if row["condition"] == condition]
            condition_rates[condition] = objective_mean(condition_rows)
        rates = list(condition_rates.values())
        overall = objective_mean(rows)
        task_rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "attempts": len(rows),
                "exact_match_rate": overall,
                "condition_exact_match_rates": condition_rates,
                "always_correct": overall == 1,
                "always_incorrect": overall == 0,
                "condition_rate_range": max(rates) - min(rates),
                "strong_condition_skew": max(rates) - min(rates) >= 0.25,
            }
        )
    family_rows = []
    families = sorted({row["task_family"] for row in tables.agent_round_table})
    for family in families:
        for condition in CONDITIONS:
            rows = [
                row
                for row in tables.agent_round_table
                if row["task_family"] == family and row["condition"] == condition
            ]
            family_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "attempts": len(rows),
                    "exact_match_rate": objective_mean(rows),
                }
            )
    return {"task_rows": task_rows, "family_rows": family_rows}


def analyze_random_sanity(tables: AnalysisTables) -> dict[str, Any]:
    rows = [row for row in tables.agent_round_table if row["condition"] == "random"]
    selected = [row for row in rows if row["selected_this_round"]]
    profile_rows = []
    for profile_id in sorted({row["profile_id"] for row in rows}):
        exposed = [row for row in rows if row["profile_id"] == profile_id]
        profile_rows.append(
            {
                "profile_id": profile_id,
                "agent_round_exposure": len(exposed),
                "selection_count": sum(row["selected_this_round"] for row in exposed),
                "selection_rate": safe_rate(
                    sum(row["selected_this_round"] for row in exposed), len(exposed)
                ),
            }
        )
    return {
        "selected_correct_rate": safe_rate(
            sum(float(row["score"]) == 1 for row in selected), len(selected)
        ),
        "population_correct_rate": safe_rate(
            sum(float(row["score"]) == 1 for row in rows), len(rows)
        ),
        "mean_selected_support": mean(row["support_received"] for row in selected),
        "mean_population_support": mean(row["support_received"] for row in rows),
        "profile_selection": profile_rows,
        "interpretation": "Descriptive sanity check only; 100 selections need not be perfectly balanced.",
    }


def analyze_operational(tables: AnalysisTables) -> dict[str, Any]:
    def summarize(rows: Sequence[Row]) -> dict[str, Any]:
        latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
        tokens = [int(row["token_count"]) for row in rows if row.get("token_count") is not None]
        return {
            "requests": len(rows),
            "latency_recorded": len(latencies),
            "total_latency_seconds": sum(latencies) / 1000,
            "mean_latency_ms": None if not latencies else mean(latencies),
            "median_latency_ms": None if not latencies else median(latencies),
            "token_count_recorded": len(tokens),
            "total_tokens": sum(tokens),
            "mean_tokens": None if not tokens else mean(tokens),
        }

    return {
        "responses": summarize(list(tables.responses)),
        "ballots": summarize(list(tables.ballot_evidence)),
        "memory": "not persisted; unavailable for retrospective analysis",
    }


def analyze_round_zero_comparability(tables: AnalysisTables) -> dict[str, Any]:
    response_groups = _group(
        [row for row in tables.agent_round_table if row["round_index"] == 0],
        "replicate",
        "condition",
    )
    ballot_groups = _group(
        [row for row in tables.ballot_table if row["round_index"] == 0],
        "replicate",
        "condition",
    )
    candidates = _group(
        [row for row in tables.candidate_rows if row["round_index"] == 0],
        "replicate",
        "condition",
        "voter_agent_id",
    )
    rows = []
    response_identity_groups = 0
    response_content_matches = 0
    response_seed_matches = 0
    ballot_voter_groups = 0
    ballot_choice_matches = 0
    ballot_raw_output_matches = 0
    ballot_seed_matches = 0
    candidate_order_matches = 0
    for replicate in sorted({row["replicate"] for row in tables.trials}):
        responses_by_agent = {
            condition: {row["agent_id"]: row for row in response_groups[(replicate, condition)]}
            for condition in CONDITIONS
        }
        ballots_by_voter = {
            condition: {
                row["voter_agent_id"]: row for row in ballot_groups[(replicate, condition)]
            }
            for condition in CONDITIONS
        }
        replicate_response_matches = []
        replicate_ballot_matches = []
        for agent_id in sorted(responses_by_agent[CONDITIONS[0]]):
            condition_rows = [responses_by_agent[c][agent_id] for c in CONDITIONS]
            response_identity_groups += 1
            seed_match = len({row["response_seed"] for row in condition_rows}) == 1
            content_match = len({row["response"] for row in condition_rows}) == 1
            response_seed_matches += seed_match
            response_content_matches += content_match
            replicate_response_matches.append(seed_match and content_match)
        for voter_id in sorted(ballots_by_voter[CONDITIONS[0]]):
            condition_rows = [ballots_by_voter[c][voter_id] for c in CONDITIONS]
            ballot_voter_groups += 1
            seed_match = len({row["ballot_seed"] for row in condition_rows}) == 1
            choice_match = len(
                {
                    (row["supported_agent_id"], row["anonymous_label"])
                    for row in condition_rows
                }
            ) == 1
            raw_output_match = len({row["raw_output"] for row in condition_rows}) == 1
            orders = []
            for condition in CONDITIONS:
                orders.append(
                    tuple(
                        row["candidate_agent_id"]
                        for row in sorted(
                            candidates[(replicate, condition, voter_id)],
                            key=lambda value: value["position"],
                        )
                    )
                )
            order_match = len(set(orders)) == 1
            ballot_seed_matches += seed_match
            ballot_choice_matches += choice_match
            ballot_raw_output_matches += raw_output_match
            candidate_order_matches += order_match
            replicate_ballot_matches.append(seed_match and order_match and choice_match)
        rows.append(
            {
                "replicate": replicate,
                "responses_identical": all(replicate_response_matches),
                "ballots_identical": all(replicate_ballot_matches),
            }
        )
    return {
        "replicates": rows,
        "all_round_zero_responses_identical": all(
            row["responses_identical"] for row in rows
        ),
        "all_round_zero_ballots_identical": all(row["ballots_identical"] for row in rows),
        "response_identity_groups": response_identity_groups,
        "response_seed_match_rate": safe_rate(
            response_seed_matches, response_identity_groups
        ),
        "response_content_match_rate": safe_rate(
            response_content_matches, response_identity_groups
        ),
        "ballot_voter_groups": ballot_voter_groups,
        "ballot_seed_match_rate": safe_rate(ballot_seed_matches, ballot_voter_groups),
        "candidate_order_match_rate": safe_rate(
            candidate_order_matches, ballot_voter_groups
        ),
        "ballot_choice_match_rate": safe_rate(ballot_choice_matches, ballot_voter_groups),
        "ballot_raw_output_match_rate": safe_rate(
            ballot_raw_output_matches, ballot_voter_groups
        ),
        "seed_interpretation": (
            "Matched conditions use condition-neutral agent identity namespaces. "
            "Response and ballot seeds depend on trial seed, round, namespace, and agent ID, "
            "not condition. Persisted round-0 seeds and candidate orders can therefore be checked "
            "separately from output equality. Output mismatches under equal seeds are consistent "
            "with provider/runtime nondeterminism and weaken exact matched-output control. Later "
            "states may also diverge endogenously after selection."
        ),
    }


def analyze_all_metrics(tables: AnalysisTables) -> dict[str, Any]:
    return {
        "performance": analyze_performance(tables),
        "selection": analyze_selection(tables),
        "ballots": analyze_ballots(tables),
        "profiles": analyze_profiles(tables),
        "divergence": analyze_divergence(tables),
        "tasks": analyze_task_diagnostics(tables),
        "random_sanity": analyze_random_sanity(tables),
        "operational": analyze_operational(tables),
        "cross_condition_comparability": analyze_round_zero_comparability(tables),
    }
