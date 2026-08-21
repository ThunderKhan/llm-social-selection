from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..seeding import derive_seed


CONDITIONS = ("peer_vote", "objective", "random")
TRIAL_PATTERN = re.compile(
    r"^e01-(?:smoke|full)-(peer_vote|objective|random)-replicate-(\d{3})$"
)
REQUIRED_TABLES = (
    "experiments",
    "trials",
    "agent_instances",
    "tasks",
    "rounds",
    "responses",
    "scores",
    "ballots",
    "ballot_evidence",
    "selection_events",
    "replacement_events",
)


Row = dict[str, Any]


@dataclass(frozen=True)
class AnalysisTables:
    experiment: Row
    config: Row
    trials: tuple[Row, ...]
    tasks: tuple[Row, ...]
    agents: tuple[Row, ...]
    rounds: tuple[Row, ...]
    responses: tuple[Row, ...]
    scores: tuple[Row, ...]
    ballots: tuple[Row, ...]
    ballot_evidence: tuple[Row, ...]
    selections: tuple[Row, ...]
    replacements: tuple[Row, ...]
    candidate_rows: tuple[Row, ...]
    round_table: tuple[Row, ...]
    agent_round_table: tuple[Row, ...]
    ballot_table: tuple[Row, ...]
    replacement_table: tuple[Row, ...]
    lifetime_table: tuple[Row, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def open_read_only(path: str | Path) -> Iterator[sqlite3.Connection]:
    source = Path(path).resolve()
    connection = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        yield connection
    finally:
        connection.close()


def _rows(connection: sqlite3.Connection, query: str) -> tuple[Row, ...]:
    return tuple(dict(row) for row in connection.execute(query).fetchall())


def inspect_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    table_names = tuple(
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    )
    tables = {}
    for table in table_names:
        columns = [
            {
                "name": row[1],
                "type": row[2],
                "not_null": bool(row[3]),
                "primary_key_position": row[5],
            }
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        foreign_keys = [
            {
                "from": row[3],
                "to_table": row[2],
                "to": row[4],
            }
            for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
        ]
        tables[table] = {"columns": columns, "foreign_keys": foreign_keys}
    return {"table_names": list(table_names), "tables": tables}


def _replicate(trial_id: str) -> int:
    match = TRIAL_PATTERN.fullmatch(trial_id)
    if match is None:
        raise ValueError(f"unrecognized E01 trial ID: {trial_id}")
    return int(match.group(2))


def _with_trial_fields(rows: Sequence[Row], trial_by_id: Mapping[str, Row]) -> tuple[Row, ...]:
    enriched = []
    for original in rows:
        row = dict(original)
        trial = trial_by_id[row["trial_id"]]
        if "condition" in row:
            row["persisted_condition"] = row["condition"]
        row["condition"] = trial["condition"]
        row["replicate"] = trial["replicate"]
        enriched.append(row)
    return tuple(enriched)


def load_analysis_tables(connection: sqlite3.Connection) -> AnalysisTables:
    experiment_rows = _rows(connection, "SELECT * FROM experiments ORDER BY experiment_id")
    if len(experiment_rows) != 1:
        raise ValueError(f"expected one experiment row, found {len(experiment_rows)}")
    experiment = experiment_rows[0]
    config = json.loads(experiment["config_json"])
    trials = list(_rows(connection, "SELECT * FROM trials ORDER BY trial_id"))
    for trial in trials:
        trial["replicate"] = _replicate(trial["trial_id"])
        trial["trial_seed"] = int(trial["trial_seed"])
    trial_by_id = {row["trial_id"]: row for row in trials}
    tasks = _rows(connection, "SELECT * FROM tasks ORDER BY task_id")
    task_by_id = {row["task_id"]: row for row in tasks}
    agents = _with_trial_fields(
        _rows(connection, "SELECT * FROM agent_instances ORDER BY trial_id, ordinal"),
        trial_by_id,
    )
    rounds = _with_trial_fields(
        _rows(connection, "SELECT * FROM rounds ORDER BY trial_id, round_index"),
        trial_by_id,
    )
    responses = _with_trial_fields(
        _rows(connection, "SELECT * FROM responses ORDER BY trial_id, round_index, ordinal"),
        trial_by_id,
    )
    scores = _with_trial_fields(
        _rows(connection, "SELECT * FROM scores ORDER BY trial_id, round_index, ordinal"),
        trial_by_id,
    )
    ballots = _with_trial_fields(
        _rows(connection, "SELECT * FROM ballots ORDER BY trial_id, round_index, ordinal"),
        trial_by_id,
    )
    evidence = _with_trial_fields(
        _rows(
            connection,
            "SELECT * FROM ballot_evidence ORDER BY trial_id, round_index, ordinal",
        ),
        trial_by_id,
    )
    selections = _with_trial_fields(
        _rows(connection, "SELECT * FROM selection_events ORDER BY trial_id, round_index"),
        trial_by_id,
    )
    replacements = _with_trial_fields(
        _rows(connection, "SELECT * FROM replacement_events ORDER BY trial_id, queue_index"),
        trial_by_id,
    )

    agent_by_key = {(row["trial_id"], row["agent_id"]): row for row in agents}
    response_by_id = {row["response_id"]: row for row in responses}
    score_by_key = {
        (row["trial_id"], row["round_index"], row["agent_id"]): row
        for row in scores
    }
    evidence_by_ballot = {row["ballot_id"]: row for row in evidence}
    selection_by_round = {
        (row["trial_id"], row["round_index"]): row for row in selections
    }
    replacement_by_round = {
        (row["trial_id"], row["round_index"]): row for row in replacements
    }
    responses_by_round: dict[tuple[str, int], list[Row]] = defaultdict(list)
    ballots_by_round: dict[tuple[str, int], list[Row]] = defaultdict(list)
    support_by_round_agent: Counter[tuple[str, int, str]] = Counter()
    for row in responses:
        responses_by_round[(row["trial_id"], row["round_index"])].append(row)
    for row in ballots:
        key = (row["trial_id"], row["round_index"])
        ballots_by_round[key].append(row)
        if row["supported_agent_id"] is not None:
            support_by_round_agent[(*key, row["supported_agent_id"])] += 1

    candidate_rows: list[Row] = []
    chosen_position: dict[str, int | None] = {}
    for row in evidence:
        try:
            candidates = json.loads(row["candidate_order_json"])
        except (TypeError, json.JSONDecodeError):
            candidates = []
        chosen_position[row["ballot_id"]] = None
        for position, candidate in enumerate(candidates):
            candidate_response = response_by_id.get(candidate.get("response_id"))
            candidate_score = score_by_key.get(
                (row["trial_id"], row["round_index"], candidate.get("agent_id"))
            )
            chosen = candidate.get("label") == row["parsed_choice"]
            if chosen:
                chosen_position[row["ballot_id"]] = position
            candidate_rows.append(
                {
                    "ballot_id": row["ballot_id"],
                    "trial_id": row["trial_id"],
                    "replicate": row["replicate"],
                    "condition": row["condition"],
                    "round_index": row["round_index"],
                    "voter_agent_id": row["voter_agent_id"],
                    "position": position,
                    "label": candidate.get("label"),
                    "candidate_agent_id": candidate.get("agent_id"),
                    "candidate_response_id": candidate.get("response_id"),
                    "candidate_score": (
                        None if candidate_score is None else candidate_score["value"]
                    ),
                    "candidate_response": (
                        None if candidate_response is None else candidate_response["content"]
                    ),
                    "chosen": chosen,
                }
            )

    agent_round_table: list[Row] = []
    for response in responses:
        key = (response["trial_id"], response["round_index"])
        agent = agent_by_key[(response["trial_id"], response["agent_id"])]
        score = score_by_key.get((*key, response["agent_id"]))
        selection = selection_by_round.get(key)
        replacement = replacement_by_round.get(key)
        terminal = response["round_index"] == trial_by_id[response["trial_id"]]["total_rounds"] - 1
        agent_round_table.append(
            {
                "condition": response["condition"],
                "replicate": response["replicate"],
                "trial_id": response["trial_id"],
                "round_index": response["round_index"],
                "task_id": response["task_id"],
                "task_family": task_by_id[response["task_id"]]["family"],
                "agent_id": response["agent_id"],
                "profile_id": agent["profile_id"],
                "generation": agent["generation"],
                "introduced_round": agent["introduced_round"],
                "response": response["content"],
                "score": None if score is None else score["value"],
                "support_received": support_by_round_agent[
                    (*key, response["agent_id"])
                ],
                "selected_this_round": bool(
                    selection is not None
                    and selection["selected_agent_id"] == response["agent_id"]
                ),
                "survived_to_next_round": (
                    None
                    if terminal
                    else not (
                        replacement is not None
                        and replacement["removed_agent_id"] == response["agent_id"]
                    )
                ),
                "terminal_round": terminal,
                "response_id": response["response_id"],
                "response_seed": (
                    None if response["seed"] is None else int(response["seed"])
                ),
                "latency_ms": response["latency_ms"],
                "token_count": response["token_count"],
            }
        )

    ballot_table: list[Row] = []
    for ballot in ballots:
        evidence_row = evidence_by_ballot.get(ballot["ballot_id"])
        supported_key = (
            ballot["trial_id"],
            ballot["round_index"],
            ballot["supported_agent_id"],
        )
        supported_score = (
            score_by_key.get(supported_key)
            if ballot["supported_agent_id"] is not None
            else None
        )
        voter = agent_by_key[(ballot["trial_id"], ballot["voter_agent_id"])]
        supported = (
            agent_by_key.get((ballot["trial_id"], ballot["supported_agent_id"]))
            if ballot["supported_agent_id"] is not None
            else None
        )
        ballot_table.append(
            {
                "ballot_id": ballot["ballot_id"],
                "condition": ballot["condition"],
                "replicate": ballot["replicate"],
                "trial_id": ballot["trial_id"],
                "round_index": ballot["round_index"],
                "voter_agent_id": ballot["voter_agent_id"],
                "supported_agent_id": ballot["supported_agent_id"],
                "voter_profile": voter["profile_id"],
                "supported_profile": (
                    None if supported is None else supported["profile_id"]
                ),
                "supported_agent_score": (
                    None if supported_score is None else supported_score["value"]
                ),
                "anonymous_display_position": (
                    None
                    if evidence_row is None
                    else chosen_position[ballot["ballot_id"]]
                ),
                "anonymous_label": (
                    None if evidence_row is None else evidence_row["parsed_choice"]
                ),
                "valid": bool(evidence_row and evidence_row["valid"]),
                "invalid_reason": (
                    None if evidence_row is None else evidence_row["invalid_reason"]
                ),
                "ballot_seed": (
                    None
                    if evidence_row is None or evidence_row["seed"] is None
                    else int(evidence_row["seed"])
                ),
                "raw_output": (
                    None if evidence_row is None else evidence_row["raw_output"]
                ),
            }
        )

    round_table: list[Row] = []
    for round_row in rounds:
        key = (round_row["trial_id"], round_row["round_index"])
        participant_rows = responses_by_round[key]
        round_scores = [
            score_by_key[(*key, response["agent_id"])]["value"]
            for response in participant_rows
            if (*key, response["agent_id"]) in score_by_key
        ]
        selection = selection_by_round.get(key)
        selected_score = None
        if selection is not None:
            selected = score_by_key.get((*key, selection["selected_agent_id"]))
            selected_score = None if selected is None else selected["value"]
        replacement = replacement_by_round.get(key)
        valid_ballots = sum(
            bool(evidence_by_ballot.get(ballot["ballot_id"], {}).get("valid"))
            for ballot in ballots_by_round[key]
        )
        round_table.append(
            {
                "condition": round_row["condition"],
                "replicate": round_row["replicate"],
                "trial_id": round_row["trial_id"],
                "round_index": round_row["round_index"],
                "task_id": round_row["task_id"],
                "task_family": task_by_id[round_row["task_id"]]["family"],
                "population_ids": [row["agent_id"] for row in participant_rows],
                "number_correct": sum(value == 1 for value in round_scores),
                "number_incorrect": sum(value == 0 for value in round_scores),
                "mean_score": (
                    None if not round_scores else sum(round_scores) / len(round_scores)
                ),
                "selected_agent_id": (
                    None if selection is None else selection["selected_agent_id"]
                ),
                "selected_agent_score": selected_score,
                "selection_mechanism": (
                    None if selection is None else selection["mechanism"]
                ),
                "selection_reason": None if selection is None else selection["reason"],
                "replacement_agent_id": (
                    None if replacement is None else replacement["added_agent_id"]
                ),
                "replacement_profile": (
                    None if replacement is None else replacement["profile_id"]
                ),
                "ballot_validity_count": valid_ballots,
                "ballot_attempt_count": len(ballots_by_round[key]),
                "round_seed": int(round_row["round_seed"]),
            }
        )

    replacement_table = tuple(
        {
            "condition": row["condition"],
            "replicate": row["replicate"],
            "trial_id": row["trial_id"],
            "round_index": row["round_index"],
            "outgoing_agent": row["removed_agent_id"],
            "outgoing_profile": agent_by_key[
                (row["trial_id"], row["removed_agent_id"])
            ]["profile_id"],
            "incoming_agent": row["added_agent_id"],
            "incoming_profile": row["profile_id"],
            "queue_index": row["queue_index"],
        }
        for row in replacements
    )

    removal_by_agent = {
        (row["trial_id"], row["removed_agent_id"]): row for row in replacements
    }
    lifetime_table = []
    for agent in agents:
        removal = removal_by_agent.get((agent["trial_id"], agent["agent_id"]))
        first_round = 0 if agent["introduced_round"] is None else agent["introduced_round"] + 1
        last_round = (
            trial_by_id[agent["trial_id"]]["total_rounds"] - 1
            if removal is None
            else removal["round_index"]
        )
        lifetime_table.append(
            {
                "condition": agent["condition"],
                "replicate": agent["replicate"],
                "trial_id": agent["trial_id"],
                "agent_id": agent["agent_id"],
                "profile_id": agent["profile_id"],
                "introduction_round": agent["introduced_round"],
                "first_active_round": first_round,
                "last_active_round": last_round,
                "lifetime_rounds": max(0, last_round - first_round + 1),
                "observed_active_rounds": max(0, last_round - first_round + 1),
                "selected_and_replaced": removal is not None,
                "right_censored": removal is None,
            }
        )

    return AnalysisTables(
        experiment=experiment,
        config=config,
        trials=tuple(trials),
        tasks=tasks,
        agents=agents,
        rounds=rounds,
        responses=responses,
        scores=scores,
        ballots=ballots,
        ballot_evidence=evidence,
        selections=selections,
        replacements=replacements,
        candidate_rows=tuple(candidate_rows),
        round_table=tuple(round_table),
        agent_round_table=tuple(agent_round_table),
        ballot_table=tuple(ballot_table),
        replacement_table=replacement_table,
        lifetime_table=tuple(lifetime_table),
    )


def _check(
    checks: list[Row],
    name: str,
    passed: bool,
    details: str,
    *,
    critical: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "critical": critical,
            "details": details,
        }
    )


def audit_e01_integrity(
    connection: sqlite3.Connection,
    tables: AnalysisTables,
    *,
    task_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    checks: list[Row] = []
    schema = inspect_schema(connection)
    table_names = set(schema["table_names"])
    _check(
        checks,
        "required_schema_tables",
        set(REQUIRED_TABLES).issubset(table_names),
        f"found {len(table_names)} tables; required={len(REQUIRED_TABLES)}",
    )
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    _check(checks, "sqlite_quick_check", quick_check == "ok", str(quick_check))
    foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    _check(
        checks,
        "foreign_key_check",
        not foreign_key_rows,
        f"violations={len(foreign_key_rows)}",
    )

    expected_trials_per_condition = int(tables.config.get("trials_per_condition", 0))
    expected_rounds_per_trial = int(tables.config.get("rounds_per_trial", 0))
    population_size = int(tables.config.get("apparatus", {}).get("population_size", 0))
    expected_trials = expected_trials_per_condition * len(CONDITIONS)
    expected_rounds = expected_trials * expected_rounds_per_trial
    expected_agent_rows = expected_rounds * population_size
    expected_replacements = expected_trials * max(expected_rounds_per_trial - 1, 0)
    raw_counts = {
        table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in REQUIRED_TABLES
    }
    expected_counts = {
        "trials": expected_trials,
        "rounds": expected_rounds,
        "responses": expected_agent_rows,
        "scores": expected_agent_rows,
        "ballots": expected_agent_rows,
        "ballot_evidence": expected_agent_rows,
        "selection_events": expected_rounds,
        "replacement_events": expected_replacements,
    }
    for table, expected in expected_counts.items():
        actual = raw_counts[table]
        _check(
            checks,
            f"count_{table}",
            actual == expected,
            f"actual={actual}; expected={expected}",
        )

    condition_trial_counts = Counter(row["condition"] for row in tables.trials)
    condition_round_counts = Counter(row["condition"] for row in tables.rounds)
    _check(
        checks,
        "condition_trial_counts",
        all(condition_trial_counts[c] == expected_trials_per_condition for c in CONDITIONS),
        json.dumps(dict(sorted(condition_trial_counts.items())), sort_keys=True),
    )
    _check(
        checks,
        "condition_round_counts",
        all(
            condition_round_counts[c]
            == expected_trials_per_condition * expected_rounds_per_trial
            for c in CONDITIONS
        ),
        json.dumps(dict(sorted(condition_round_counts.items())), sort_keys=True),
    )
    trial_id_condition_violations = []
    for trial in tables.trials:
        match = TRIAL_PATTERN.fullmatch(trial["trial_id"])
        if match is None or match.group(1) != trial["condition"]:
            trial_id_condition_violations.append(trial["trial_id"])
    _check(
        checks,
        "trial_id_condition_alignment",
        not trial_id_condition_violations,
        f"violations={len(trial_id_condition_violations)}",
    )
    round_condition_violations = [
        (row["trial_id"], row["round_index"])
        for row in tables.rounds
        if row.get("persisted_condition") != row["condition"] or row["status"] != "complete"
    ]
    _check(
        checks,
        "round_condition_and_status",
        not round_condition_violations,
        f"violations={len(round_condition_violations)}",
    )
    trial_length_violations = [
        row["trial_id"]
        for row in tables.trials
        if row["total_rounds"] != expected_rounds_per_trial
    ]
    _check(
        checks,
        "trial_length_protocol",
        not trial_length_violations,
        f"violations={len(trial_length_violations)}",
    )
    incomplete = [row["trial_id"] for row in tables.trials if row["status"] != "completed"]
    _check(
        checks,
        "trial_completion",
        not incomplete,
        f"incomplete_trials={incomplete}",
    )

    rounds_by_trial: dict[str, list[int]] = defaultdict(list)
    for row in tables.rounds:
        rounds_by_trial[row["trial_id"]].append(row["round_index"])
    discontinuous = {
        trial["trial_id"]: sorted(rounds_by_trial[trial["trial_id"]])
        for trial in tables.trials
        if sorted(rounds_by_trial[trial["trial_id"]])
        != list(range(expected_rounds_per_trial))
    }
    _check(
        checks,
        "round_continuity",
        not discontinuous,
        f"violating_trials={sorted(discontinuous)}",
    )

    per_round_counts: dict[str, Counter[tuple[str, int]]] = {}
    for name, rows in (
        ("responses", tables.responses),
        ("scores", tables.scores),
        ("ballots", tables.ballots),
        ("ballot_evidence", tables.ballot_evidence),
        ("selection_events", tables.selections),
    ):
        per_round_counts[name] = Counter(
            (row["trial_id"], row["round_index"]) for row in rows
        )
    bad_round_evidence = []
    for round_row in tables.rounds:
        key = (round_row["trial_id"], round_row["round_index"])
        expected = {
            "responses": population_size,
            "scores": population_size,
            "ballots": population_size,
            "ballot_evidence": population_size,
            "selection_events": 1,
        }
        actual = {name: counts[key] for name, counts in per_round_counts.items()}
        if actual != expected:
            bad_round_evidence.append({"trial_id": key[0], "round": key[1], "counts": actual})
    _check(
        checks,
        "per_round_evidence_counts",
        not bad_round_evidence,
        f"violating_rounds={len(bad_round_evidence)}",
    )

    trials_by_replicate: dict[int, list[Row]] = defaultdict(list)
    for trial in tables.trials:
        trials_by_replicate[trial["replicate"]].append(trial)
    seed_violations = []
    for replicate, rows in sorted(trials_by_replicate.items()):
        if {row["condition"] for row in rows} != set(CONDITIONS) or len(
            {row["trial_seed"] for row in rows}
        ) != 1:
            seed_violations.append(replicate)
    _check(
        checks,
        "matched_replicate_seeds",
        not seed_violations,
        f"violating_replicates={seed_violations}",
    )

    task_map: dict[tuple[int, int], dict[str, str]] = defaultdict(dict)
    for row in tables.rounds:
        task_map[(row["replicate"], row["round_index"])][row["condition"]] = row[
            "task_id"
        ]
    task_violations = [
        {"replicate": key[0], "round": key[1], "tasks": values}
        for key, values in sorted(task_map.items())
        if set(values) != set(CONDITIONS) or len(set(values.values())) != 1
    ]
    _check(
        checks,
        "matched_task_schedule",
        not task_violations,
        f"violating_replicate_rounds={len(task_violations)}",
    )

    initial_map: dict[int, dict[str, tuple[tuple[Any, ...], ...]]] = defaultdict(dict)
    queue_map: dict[int, dict[str, tuple[tuple[Any, ...], ...]]] = defaultdict(dict)
    for trial in tables.trials:
        initial_map[trial["replicate"]][trial["condition"]] = tuple(
            (row["agent_id"], row["profile_id"], row["display_label"])
            for row in tables.agents
            if row["trial_id"] == trial["trial_id"] and row["introduced_round"] is None
        )
        queue_map[trial["replicate"]][trial["condition"]] = tuple(
            (row["queue_index"], row["added_agent_id"], row["profile_id"])
            for row in tables.replacements
            if row["trial_id"] == trial["trial_id"]
        )
    initial_violations = [
        replicate
        for replicate, values in initial_map.items()
        if set(values) != set(CONDITIONS) or len(set(values.values())) != 1
    ]
    queue_violations = [
        replicate
        for replicate, values in queue_map.items()
        if set(values) != set(CONDITIONS) or len(set(values.values())) != 1
    ]
    _check(
        checks,
        "matched_initial_profiles",
        not initial_violations,
        f"violating_replicates={initial_violations}",
    )
    _check(
        checks,
        "matched_replacement_queue",
        not queue_violations,
        f"violating_replicates={queue_violations}",
    )

    agents_by_trial: dict[str, list[Row]] = defaultdict(list)
    replacements_by_trial: dict[str, list[Row]] = defaultdict(list)
    for row in tables.agents:
        agents_by_trial[row["trial_id"]].append(row)
    for row in tables.replacements:
        replacements_by_trial[row["trial_id"]].append(row)
    active_by_round: dict[tuple[str, int], set[str]] = {}
    for round_row in tables.rounds:
        trial_id = round_row["trial_id"]
        round_index = round_row["round_index"]
        removed_before = {
            row["removed_agent_id"]
            for row in replacements_by_trial[trial_id]
            if row["round_index"] < round_index
        }
        active_by_round[(trial_id, round_index)] = {
            row["agent_id"]
            for row in agents_by_trial[trial_id]
            if (row["introduced_round"] is None or row["introduced_round"] < round_index)
            and row["agent_id"] not in removed_before
        }
    population_violations = [
        key for key, active in active_by_round.items() if len(active) != population_size
    ]
    response_population_violations = []
    for key, active in active_by_round.items():
        observed = {
            row["agent_id"]
            for row in tables.responses
            if (row["trial_id"], row["round_index"]) == key
        }
        if observed != active:
            response_population_violations.append(key)
    _check(
        checks,
        "active_population_reconstruction",
        not population_violations and not response_population_violations,
        f"size_violations={len(population_violations)}; participant_violations={len(response_population_violations)}",
    )

    self_ballots = [
        row["ballot_id"]
        for row in tables.ballots
        if row["voter_agent_id"] == row["supported_agent_id"]
    ]
    ballot_eligibility = [
        row["ballot_id"]
        for row in tables.ballots
        if row["voter_agent_id"]
        not in active_by_round[(row["trial_id"], row["round_index"])]
        or (
            row["supported_agent_id"] is not None
            and row["supported_agent_id"]
            not in active_by_round[(row["trial_id"], row["round_index"])]
        )
    ]
    _check(checks, "self_ballot_prohibition", not self_ballots, f"violations={len(self_ballots)}")
    _check(
        checks,
        "ballot_eligibility",
        not ballot_eligibility,
        f"violations={len(ballot_eligibility)}",
    )

    response_by_id = {row["response_id"]: row for row in tables.responses}
    ballot_by_id = {row["ballot_id"]: row for row in tables.ballots}
    candidate_violations = []
    raw_identity_leaks = []
    profile_ids = {row["profile_id"] for row in tables.agents}
    for evidence in tables.ballot_evidence:
        ballot = ballot_by_id.get(evidence["ballot_id"])
        try:
            candidates = json.loads(evidence["candidate_order_json"])
        except (TypeError, json.JSONDecodeError):
            candidate_violations.append(evidence["ballot_id"])
            continue
        labels = [candidate.get("label") for candidate in candidates]
        ids = [candidate.get("agent_id") for candidate in candidates]
        candidate_count = population_size - 1
        expected_labels = [chr(ord("A") + index) for index in range(candidate_count)]
        mapped = [
            candidate
            for candidate in candidates
            if candidate.get("label") == evidence["parsed_choice"]
        ]
        structural_valid = (
            ballot is not None
            and len(candidates) == candidate_count
            and labels == expected_labels
            and len(set(ids)) == 7
            and evidence["voter_agent_id"] not in ids
        )
        if evidence["valid"]:
            mapping_valid = (
                len(mapped) == 1
                and mapped[0].get("agent_id") == ballot["supported_agent_id"]
                and ballot["supported_agent_id"] is not None
            )
        else:
            mapping_valid = (
                evidence["parsed_choice"] is None
                and evidence["invalid_reason"] is not None
                and ballot is not None
                and ballot["supported_agent_id"] is None
            )
        valid = structural_valid and mapping_valid
        for candidate in candidates:
            response = response_by_id.get(candidate.get("response_id"))
            valid = valid and response is not None and (
                response["trial_id"], response["round_index"], response["agent_id"]
            ) == (
                evidence["trial_id"],
                evidence["round_index"],
                candidate.get("agent_id"),
            )
        if not valid:
            candidate_violations.append(evidence["ballot_id"])
        raw = evidence["raw_output"]
        active_ids = active_by_round[(evidence["trial_id"], evidence["round_index"])]
        if any(value in raw for value in active_ids | profile_ids):
            raw_identity_leaks.append(evidence["ballot_id"])
    _check(
        checks,
        "candidate_map_integrity",
        not candidate_violations,
        f"violations={len(candidate_violations)}",
    )
    _check(
        checks,
        "ballot_raw_output_identity_leakage",
        not raw_identity_leaks,
        f"violations={len(raw_identity_leaks)}; rendered prompts are not persisted",
    )

    valid_count = sum(bool(row["valid"]) for row in tables.ballot_evidence)
    invalid_count = len(tables.ballot_evidence) - valid_count
    _check(
        checks,
        "ballot_attempt_accounting",
        valid_count + invalid_count == expected_agent_rows,
        f"valid={valid_count}; invalid={invalid_count}; expected={expected_agent_rows}",
    )
    _check(
        checks,
        "zero_invalid_ballots",
        invalid_count == 0,
        f"valid={valid_count}; invalid={invalid_count}",
        critical=False,
    )

    score_keys = {
        (row["trial_id"], row["round_index"], row["task_id"], row["agent_id"])
        for row in tables.scores
    }
    response_keys = {
        (row["trial_id"], row["round_index"], row["task_id"], row["agent_id"])
        for row in tables.responses
    }
    _check(
        checks,
        "score_response_alignment",
        score_keys == response_keys,
        f"score_only={len(score_keys - response_keys)}; response_only={len(response_keys - score_keys)}",
    )
    expected_scorer = tables.config.get("task_set", {}).get("scorer_version")
    score_domain_violations = [
        row["score_id"]
        for row in tables.scores
        if row["value"] not in (0.0, 1.0)
        or (expected_scorer is not None and row["scorer_version"] != expected_scorer)
    ]
    _check(
        checks,
        "score_domain_and_version",
        not score_domain_violations,
        f"violations={len(score_domain_violations)}; expected_scorer={expected_scorer}",
    )
    expected_provider = tables.experiment["provider_name"]
    expected_model = tables.experiment["model_name"]
    provider_violations = [
        row.get("response_id", row.get("ballot_id"))
        for row in (*tables.responses, *tables.ballot_evidence)
        if row["provider_name"] != expected_provider or row["model_name"] != expected_model
    ]
    _check(
        checks,
        "provider_model_provenance",
        not provider_violations,
        f"violations={len(provider_violations)}; expected={expected_provider}/{expected_model}",
    )
    invalid_selections = [
        row["selection_id"]
        for row in tables.selections
        if row["selected_agent_id"]
        not in active_by_round[(row["trial_id"], row["round_index"])]
        or row["mechanism"] != row["condition"]
    ]
    _check(
        checks,
        "selection_eligibility",
        not invalid_selections,
        f"violations={len(invalid_selections)}",
    )
    score_by_round_agent = {
        (row["trial_id"], row["round_index"], row["agent_id"]): row["value"]
        for row in tables.scores
    }
    support_counts = Counter(
        (row["trial_id"], row["round_index"], row["supported_agent_id"])
        for row in tables.ballots
        if row["supported_agent_id"] is not None
    )
    trial_by_id = {row["trial_id"]: row for row in tables.trials}
    selection_rule_violations = []
    from random import Random

    for selection in tables.selections:
        key = (selection["trial_id"], selection["round_index"])
        active = active_by_round[key]
        mechanism = selection["mechanism"]
        if mechanism == "objective":
            values = {
                agent_id: score_by_round_agent[(*key, agent_id)] for agent_id in active
            }
            minimum = min(values.values())
            eligible = sorted(
                agent_id for agent_id, value in values.items() if value == minimum
            )
            namespace = "objective_tiebreak"
            expected_reason = f"lowest_objective_score;value={minimum:g}"
        elif mechanism == "peer_vote":
            values = {
                agent_id: support_counts[(*key, agent_id)] for agent_id in active
            }
            minimum = min(values.values())
            eligible = sorted(
                agent_id for agent_id, value in values.items() if value == minimum
            )
            namespace = "peer_tiebreak"
            expected_reason = f"fewest_support_votes;count={minimum}"
        else:
            eligible = sorted(active)
            namespace = "random_selection"
            expected_reason = "seeded_uniform_random"
        seed = derive_seed(
            trial_by_id[selection["trial_id"]]["trial_seed"],
            selection["round_index"],
            namespace,
        )
        expected_selected = Random(seed).choice(eligible)
        if (
            selection["selected_agent_id"] != expected_selected
            or selection["reason"] != expected_reason
        ):
            selection_rule_violations.append(selection["selection_id"])
    _check(
        checks,
        "selection_rule_reconstruction",
        not selection_rule_violations,
        f"violations={len(selection_rule_violations)}",
    )
    selection_by_round = {
        (row["trial_id"], row["round_index"]): row for row in tables.selections
    }
    bad_replacements = [
        row["replacement_id"]
        for row in tables.replacements
        if selection_by_round.get((row["trial_id"], row["round_index"]), {}).get(
            "selected_agent_id"
        )
        != row["removed_agent_id"]
        or row["queue_index"] != row["round_index"]
    ]
    agent_by_key = {
        (row["trial_id"], row["agent_id"]): row for row in tables.agents
    }
    replacement_agent_violations = [
        row["replacement_id"]
        for row in tables.replacements
        if (row["trial_id"], row["added_agent_id"]) not in agent_by_key
        or agent_by_key.get((row["trial_id"], row["added_agent_id"]), {}).get(
            "profile_id"
        )
        != row["profile_id"]
        or agent_by_key.get((row["trial_id"], row["added_agent_id"]), {}).get(
            "introduced_round"
        )
        != row["round_index"]
    ]
    terminal_replacements = [
        row["replacement_id"]
        for row in tables.replacements
        if row["round_index"] == expected_rounds_per_trial - 1
    ]
    _check(
        checks,
        "replacement_reconstruction",
        not bad_replacements and not terminal_replacements and not replacement_agent_violations,
        f"event_mismatches={len(bad_replacements)}; terminal_replacements={len(terminal_replacements)}; agent_mismatches={len(replacement_agent_violations)}",
    )

    calculated_config_hash = hashlib.sha256(
        tables.experiment["config_json"].encode("utf-8")
    ).hexdigest()
    protocol_hashes_match = (
        calculated_config_hash == tables.experiment["config_hash"]
        and all(
            row["config_hash"] == tables.experiment["config_hash"]
            for row in tables.trials
        )
    )
    _check(
        checks,
        "config_hash",
        protocol_hashes_match,
        f"calculated={calculated_config_hash}; persisted={tables.experiment['config_hash']}",
    )
    expected_profile_hash = tables.config.get("profiles", {}).get("hash")
    expected_replacement_version = tables.config.get("apparatus", {}).get(
        "replacement_version"
    )
    _check(
        checks,
        "profile_protocol_hash",
        all(row["profile_pool_hash"] == expected_profile_hash for row in tables.trials),
        f"persisted_config={expected_profile_hash}",
    )
    _check(
        checks,
        "replacement_protocol_version",
        all(
            row["replacement_version"] == expected_replacement_version
            for row in tables.trials
        ),
        f"persisted_config={expected_replacement_version}",
    )
    if task_artifact_path is not None:
        artifact_hash = sha256_file(task_artifact_path)
        expected_task_hash = tables.config.get("task_set", {}).get("hash")
        _check(
            checks,
            "task_artifact_hash",
            artifact_hash == expected_task_hash,
            f"calculated={artifact_hash}; persisted={expected_task_hash}",
        )
        try:
            artifact = json.loads(Path(task_artifact_path).read_text(encoding="utf-8"))
            artifact_tasks = {
                row["task_id"]: {
                    "family": row["family"],
                    "prompt": row["prompt"],
                    "expected_answer": row["expected_answer"],
                    "scorer_version": row["scorer_version"],
                }
                for row in artifact["tasks"]
            }
            database_tasks = {
                row["task_id"]: {
                    "family": row["family"],
                    "prompt": row["prompt"],
                    "expected_answer": row["expected_answer"],
                    "scorer_version": row["scorer_version"],
                }
                for row in tables.tasks
            }
            task_content_match = database_tasks == artifact_tasks
            task_content_details = (
                f"database_tasks={len(database_tasks)}; artifact_tasks={len(artifact_tasks)}"
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            task_content_match = False
            task_content_details = f"could not compare task content: {error}"
        _check(
            checks,
            "task_artifact_content",
            task_content_match,
            task_content_details,
        )

    critical_failures = [
        row["name"] for row in checks if row["critical"] and not row["passed"]
    ]
    return {
        "status": "PASS" if not critical_failures else "FAIL",
        "checks": checks,
        "failures": critical_failures,
        "counts": raw_counts,
        "condition_trial_counts": dict(sorted(condition_trial_counts.items())),
        "condition_round_counts": dict(sorted(condition_round_counts.items())),
        "ballots": {"valid": valid_count, "invalid": invalid_count},
        "schema": schema,
        "fields_used": {
            "experiments": [
                "experiment_id",
                "config_hash",
                "config_json",
                "code_commit",
                "provider_name",
                "model_name",
            ],
            "trials": [
                "trial_id",
                "trial_seed",
                "status",
                "condition",
                "total_rounds",
                "profile_pool_hash",
            ],
            "agent_instances": [
                "agent_id",
                "profile_id",
                "ordinal",
                "introduced_round",
            ],
            "rounds": ["round_index", "task_id", "condition", "round_seed", "status"],
            "responses": ["response_id", "agent_id", "content", "seed"],
            "scores": ["agent_id", "value", "scorer_version"],
            "ballots": ["voter_agent_id", "supported_agent_id"],
            "ballot_evidence": [
                "parsed_choice",
                "valid",
                "candidate_order_json",
                "raw_output",
                "seed",
            ],
            "selection_events": ["mechanism", "selected_agent_id", "reason"],
            "replacement_events": [
                "removed_agent_id",
                "added_agent_id",
                "profile_id",
                "queue_index",
            ],
        },
    }
