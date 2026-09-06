from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from ..seeding import derive_seed
from ..tasks import Task

E02A_PROTOCOL_VERSION = "e02a-apparatus-v1"
LETTER_LABELS = ("A", "B", "C", "D", "E", "F", "G")
OPAQUE_LABELS = ("Q7", "M2", "R9", "K4", "T6", "V3", "X8")
POSITION_CONTENTS = (
    ("content-0", "4"),
    ("content-1", "The answer is 4."),
    ("content-2", "5"),
    ("content-3", "3"),
    ("content-4", "22"),
    ("content-5", "I cannot determine it."),
    ("content-6", "2 + 2 = 5"),
)
POSITION_TASK_PROMPT = "Which integer equals two plus two? Return only the integer."
PEER_TASK_PROMPT = "Compute 17 plus 8. Return only the integer."
PEER_CONTENTS = (
    "25",
    "The answer is 25.",
    "24",
    "26",
    "17 + 8 = 25",
    "I cannot determine it.",
    "33",
    "17 + 8 = 24",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class EvidenceCheckpoint:
    """Append-only inference evidence keyed by stable evaluation identity."""

    def __init__(self, path: str | Path, expected_protocol_hash: str) -> None:
        self.path = Path(path)
        self.expected_protocol_hash = expected_protocol_hash
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid checkpoint JSON at line {line_number}: {error}"
                        ) from error
                    if record.get("protocol_hash") != expected_protocol_hash:
                        raise ValueError(
                            "checkpoint protocol hash does not match this E02A run"
                        )
                    evaluation_id = record.get("evaluation_id")
                    if not isinstance(evaluation_id, str) or not evaluation_id:
                        raise ValueError(
                            f"checkpoint line {line_number} has no evaluation_id"
                        )
                    if evaluation_id in self.records:
                        raise ValueError(
                            f"duplicate checkpoint evaluation_id: {evaluation_id}"
                        )
                    self.records[evaluation_id] = record

    def get(self, evaluation_id: str) -> dict[str, Any] | None:
        return self.records.get(evaluation_id)

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(record)
        evaluation_id = row.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("evidence record must have an evaluation_id")
        if evaluation_id in self.records:
            raise ValueError(f"duplicate evaluation_id: {evaluation_id}")
        row["protocol_hash"] = self.expected_protocol_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        self.records[evaluation_id] = row
        return row

    def phase(self, phase: str) -> list[dict[str, Any]]:
        return sorted(
            (row for row in self.records.values() if row.get("phase") == phase),
            key=lambda row: row["evaluation_id"],
        )


def choose_task_shortlist(
    tasks: Sequence[Task],
    evidence: Sequence[Mapping[str, Any]],
    *,
    attempts_per_task: int = 16,
    maximum_tasks: int = 24,
) -> tuple[tuple[Task, ...], list[dict[str, Any]], dict[str, str]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence:
        by_task[str(row["task_id"])].append(row)
    diagnostics = []
    eligible: dict[str, list[tuple[Task, float]]] = defaultdict(list)
    rejected: dict[str, str] = {}
    for task in tasks:
        rows = by_task.get(task.task_id, [])
        successful = [row for row in rows if row.get("status") == "ok"]
        correct = sum(bool(row.get("correct")) for row in successful)
        rate = correct / len(successful) if successful else 0.0
        format_count = sum(bool(row.get("format_valid")) for row in successful)
        diagnostic = {
            "task_id": task.task_id,
            "family": task.family,
            "planned_attempts": attempts_per_task,
            "successful_attempts": len(successful),
            "provider_failures": len(rows) - len(successful),
            "correct": correct,
            "exact_match_rate": rate,
            "format_compliance_rate": (
                format_count / len(successful) if successful else 0.0
            ),
            "unique_raw_outputs": len({str(row["raw_output"]) for row in successful}),
        }
        diagnostics.append(diagnostic)
        if len(rows) != attempts_per_task or len(successful) != attempts_per_task:
            rejected[task.task_id] = "incomplete_calibration"
        elif not 0.15 <= rate <= 0.85:
            rejected[task.task_id] = "outside_0.15_to_0.85_band"
        else:
            eligible[task.family].append((task, rate))
    for rows in eligible.values():
        rows.sort(key=lambda item: (abs(item[1] - 0.5), item[0].task_id))
    selected: list[Task] = []
    families = sorted(eligible)
    while len(selected) < maximum_tasks:
        added = False
        for family in families:
            if eligible[family] and len(selected) < maximum_tasks:
                selected.append(eligible[family].pop(0)[0])
                added = True
        if not added:
            break
    for rows in eligible.values():
        for task, _ in rows:
            rejected[task.task_id] = "shortlist_capacity"
    return tuple(selected), diagnostics, dict(sorted(rejected.items()))


def summarize_holdout(
    shortlisted_tasks: Sequence[Task], evidence: Sequence[Mapping[str, Any]]
) -> tuple[tuple[Task, ...], dict[str, Any], list[dict[str, Any]]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence:
        by_task[str(row["task_id"])].append(row)
    final_tasks = []
    details = []
    complete_rounds = []
    for task in shortlisted_tasks:
        rows = by_task.get(task.task_id, [])
        successful = [row for row in rows if row.get("status") == "ok"]
        scores = [int(bool(row.get("correct"))) for row in successful]
        complete = len(rows) == 8 and len(successful) == 8
        correct = sum(scores)
        mixed = complete and 0 < correct < 8
        if mixed:
            final_tasks.append(task)
        if complete:
            complete_rounds.append(scores)
        details.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "complete": complete,
                "correct": correct,
                "incorrect": len(scores) - correct,
                "mixed": mixed,
                "raw_outputs": [row.get("raw_output") for row in successful],
            }
        )
    rounds = len(shortlisted_tasks)
    complete = len(complete_rounds)
    mixed_count = sum(0 < sum(scores) < 8 for scores in complete_rounds)
    degenerate = sum(sum(scores) in (0, 8) for scores in complete_rounds)
    summary = {
        "planned_rounds": rounds,
        "complete_rounds": complete,
        "incomplete_rounds": rounds - complete,
        "mixed_score_rounds": mixed_count,
        "mixed_score_round_rate": mixed_count / complete if complete else 0.0,
        "degenerate_rounds": degenerate,
        "degenerate_round_rate": degenerate / complete if complete else 1.0,
        "all_correct_rounds": sum(sum(scores) == 8 for scores in complete_rounds),
        "all_incorrect_rounds": sum(sum(scores) == 0 for scores in complete_rounds),
    }
    return tuple(final_tasks), summary, details


def position_plan() -> tuple[dict[str, Any], ...]:
    plans = []
    for scheme, labels in (("letters", LETTER_LABELS), ("opaque", OPAQUE_LABELS)):
        for repeat in range(2):
            for cycle in range(7):
                for voter in range(7):
                    candidates = []
                    for display_index in range(7):
                        content_id, content = POSITION_CONTENTS[
                            (display_index + voter) % 7
                        ]
                        candidates.append(
                            {
                                "display_index": display_index,
                                "anonymous_label": labels[(display_index + cycle) % 7],
                                "underlying_candidate_id": content_id,
                                "content": content,
                            }
                        )
                    plans.append(
                        {
                            "evaluation_id": (
                                f"position-{scheme}-r{repeat}-c{cycle}-v{voter}"
                            ),
                            "scheme": scheme,
                            "repeat": repeat,
                            "cycle": cycle,
                            "voter": voter,
                            "candidates": candidates,
                        }
                    )
    return tuple(plans)


def render_choice_prompt(
    task_prompt: str, candidates: Sequence[Mapping[str, Any]]
) -> str:
    lines = [
        "ANONYMOUS RESPONSE EVALUATION",
        "",
        "TASK:",
        task_prompt,
        "",
        "CANDIDATE RESPONSES:",
    ]
    for candidate in candidates:
        lines.extend(
            ("", f"Response {candidate['anonymous_label']}:", str(candidate["content"]))
        )
    lines.extend(
        (
            "",
            "Support the ONE response that best answers the task.",
            "Evaluate only the response content. Do not infer authorship or identity.",
            "Return ONLY a JSON object with exactly one key named choice.",
            "The choice value must be one of the displayed response labels.",
            "Do not use Markdown, code fences, explanation, or any other text.",
        )
    )
    return "\n".join(lines)


def choice_schema(labels: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": list(labels)}},
        "required": ["choice"],
        "additionalProperties": False,
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_choice(
    raw_output: str, labels: Sequence[str]
) -> tuple[str | None, str | None]:
    try:
        parsed = json.loads(raw_output, object_pairs_hook=_reject_duplicate_json_keys)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(parsed, dict) or set(parsed) != {"choice"}:
        return None, "invalid_shape"
    choice = parsed["choice"]
    if not isinstance(choice, str) or choice not in labels:
        return None, "invalid_choice"
    return choice, None


def summarize_position(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in evidence if row.get("valid")]
    display = Counter(int(row["selected_display_index"]) for row in valid)
    labels = Counter(str(row["selected_anonymous_label"]) for row in valid)
    contents = Counter(str(row["selected_candidate_id"]) for row in valid)

    def shares(counts: Counter[Any]) -> dict[str, float]:
        return {
            str(key): value / len(valid) if valid else 0.0
            for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
        }

    display_shares = shares(display)
    label_shares = shares(labels)
    max_display = max(display_shares.values(), default=1.0)
    max_label = max(label_shares.values(), default=1.0)
    positive_labels = [value for value in labels.values() if value]
    label_ratio = (
        max(positive_labels) / min(positive_labels)
        if len(positive_labels) == 14
        else float("inf")
    )
    validity = len(valid) / len(evidence) if evidence else 0.0
    required = validity >= 0.95 and max_display <= 0.35 and max_label <= 0.35
    preferred = required and max_display <= 0.25 and max_label <= 0.25
    return {
        "evaluations": len(evidence),
        "valid_ballots": len(valid),
        "strict_validity_rate": validity,
        "display_index_counts": dict(sorted(display.items())),
        "display_index_shares": display_shares,
        "anonymous_label_counts": dict(sorted(labels.items())),
        "anonymous_label_shares": label_shares,
        "candidate_content_counts": dict(sorted(contents.items())),
        "candidate_content_shares": shares(contents),
        "max_display_share": max_display,
        "max_label_share": max_label,
        "max_to_min_label_count_ratio": label_ratio,
        "exposure_balance": {
            "candidate_exposures_per_display_index_within_scheme": 14,
            "candidate_exposures_per_anonymous_label_within_scheme": 14,
            "display_label_exposures_per_pair_within_scheme": 14,
            "label_content_exposures_per_pair_within_scheme": 14,
        },
        "ratio_interpretation": (
            "The crossed design balances exposure and content, so the observed label "
            "ratio has no exposure or content-allocation explanation."
        ),
        "required_gate_passed": required,
        "preferred_gate_passed": preferred,
        "ratio_gate_passed": label_ratio < 2.5,
    }


def peer_plan() -> tuple[dict[str, Any], ...]:
    plans = []
    agent_ids = tuple(f"peer-agent-{index}" for index in range(8))
    for panel in range(16):
        contents = PEER_CONTENTS[panel % 8 :] + PEER_CONTENTS[: panel % 8]
        content_by_agent = dict(zip(agent_ids, contents, strict=True))
        labels = LETTER_LABELS if panel % 2 == 0 else OPAQUE_LABELS
        for voter_index, voter_id in enumerate(agent_ids):
            eligible = [agent_id for agent_id in agent_ids if agent_id != voter_id]
            candidates = []
            for display_index, agent_id in enumerate(eligible):
                candidates.append(
                    {
                        "display_index": display_index,
                        "anonymous_label": labels[
                            (display_index + panel + voter_index) % 7
                        ],
                        "underlying_candidate_id": agent_id,
                        "content": content_by_agent[agent_id],
                    }
                )
            for mechanism in ("ranked_top3", "approval"):
                plans.append(
                    {
                        "evaluation_id": f"peer-{mechanism}-p{panel:02d}-v{voter_index}",
                        "panel": panel,
                        "voter_id": voter_id,
                        "mechanism": mechanism,
                        "candidates": candidates,
                    }
                )
    return tuple(plans)


def render_rich_peer_prompt(plan: Mapping[str, Any]) -> str:
    mechanism = plan["mechanism"]
    lines = [
        "ANONYMOUS RESPONSE EVALUATION",
        "",
        "TASK:",
        PEER_TASK_PROMPT,
        "",
        "CANDIDATE RESPONSES:",
    ]
    for candidate in plan["candidates"]:
        lines.extend(
            ("", f"Response {candidate['anonymous_label']}:", candidate["content"])
        )
    lines.append("")
    if mechanism == "ranked_top3":
        lines.extend(
            (
                "Rank the three responses that best answer the task, best first.",
                "Return ONLY a JSON object with exactly one key named choices.",
                "The choices value must contain three distinct displayed labels.",
            )
        )
    else:
        lines.extend(
            (
                "Approve every response that adequately answers the task.",
                "Return ONLY a JSON object with exactly one key named choices.",
                "The choices value must contain one or more distinct displayed labels.",
            )
        )
    lines.append("Do not use Markdown, code fences, explanation, or any other text.")
    return "\n".join(lines)


def rich_peer_schema(labels: Sequence[str], mechanism: str) -> dict[str, Any]:
    count = 3 if mechanism == "ranked_top3" else len(labels)
    return {
        "type": "object",
        "properties": {
            "choices": {
                "type": "array",
                "items": {"type": "string", "enum": list(labels)},
                "minItems": 3 if mechanism == "ranked_top3" else 1,
                "maxItems": count,
                "uniqueItems": True,
            }
        },
        "required": ["choices"],
        "additionalProperties": False,
    }


def parse_rich_choices(
    raw_output: str, labels: Sequence[str], mechanism: str
) -> tuple[list[str] | None, str | None]:
    try:
        parsed = json.loads(raw_output, object_pairs_hook=_reject_duplicate_json_keys)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(parsed, dict) or set(parsed) != {"choices"}:
        return None, "invalid_shape"
    choices = parsed["choices"]
    if not isinstance(choices, list) or any(
        not isinstance(item, str) for item in choices
    ):
        return None, "invalid_choices"
    required = 3 if mechanism == "ranked_top3" else None
    if required is not None and len(choices) != required:
        return None, "invalid_count"
    if mechanism == "approval" and not 1 <= len(choices) <= len(labels):
        return None, "invalid_count"
    if len(set(choices)) != len(choices) or any(item not in labels for item in choices):
        return None, "invalid_choice"
    return choices, None


def summarize_rich_peer(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mechanism in ("ranked_top3", "approval"):
        rows = [row for row in evidence if row.get("mechanism") == mechanism]
        valid = [row for row in rows if row.get("valid")]
        panels = []
        for panel in range(16):
            panel_rows = [row for row in rows if row.get("panel") == panel]
            if len(panel_rows) != 8:
                continue
            scores = Counter({f"peer-agent-{index}": 0 for index in range(8)})
            valid_panel_rows = [row for row in panel_rows if row.get("valid")]
            for row in valid_panel_rows:
                choices = row["selected_candidate_ids"]
                if mechanism == "ranked_top3":
                    for points, agent_id in zip((3, 2, 1), choices, strict=True):
                        scores[agent_id] += points
                else:
                    for agent_id in choices:
                        scores[agent_id] += 1
            minimum = min(scores.values())
            tied = sorted(
                agent_id for agent_id, value in scores.items() if value == minimum
            )
            selected = Random(derive_seed(20260823, panel, mechanism)).choice(tied)
            panels.append(
                {
                    "panel": panel,
                    "scores": dict(sorted(scores.items())),
                    "minimum_score": minimum,
                    "minimum_tie_size": len(tied),
                    "selected_agent_id": selected,
                    "valid_ballots": len(valid_panel_rows),
                    "abstentions": len(panel_rows) - len(valid_panel_rows),
                }
            )
        tie_rate = (
            sum(row["minimum_tie_size"] > 1 for row in panels) / len(panels)
            if panels
            else 1.0
        )
        result[mechanism] = {
            "requests": len(rows),
            "valid_ballots": len(valid),
            "validity_rate": len(valid) / len(rows) if rows else 0.0,
            "complete_panels": len(panels),
            "tie_rate": tie_rate,
            "unique_minimum_rate": 1 - tie_rate,
            "mean_minimum_tie_size": (
                mean(row["minimum_tie_size"] for row in panels) if panels else 8.0
            ),
            "panel_details": panels,
        }
    viable = [
        mechanism
        for mechanism, summary in result.items()
        if summary["validity_rate"] >= 0.95
        and summary["complete_panels"] == 16
        and summary["tie_rate"] <= 0.75
        and summary["mean_minimum_tie_size"] <= 3.0
    ]
    result["recommendation"] = (
        "ADOPT REVISED PEER BALLOT FOR E03"
        if viable
        else "PEER MECHANISM STILL UNRESOLVED"
    )
    result["recommended_mechanism"] = viable[0] if viable else None
    return result


def e01_peer_baseline(database: str | Path) -> dict[str, Any]:
    uri = f"file:{Path(database).resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        ballot_rows = connection.execute(
            """
            SELECT b.trial_id, b.round_index, b.voter_agent_id, b.supported_agent_id
            FROM ballots AS b
            JOIN trials AS t ON t.trial_id = b.trial_id
            WHERE t.condition = 'peer_vote'
            ORDER BY b.trial_id, b.round_index, b.ordinal
            """
        ).fetchall()
        selection_rows = connection.execute(
            """
            SELECT s.trial_id, s.round_index, s.selected_agent_id
            FROM selection_events AS s
            JOIN trials AS t ON t.trial_id = s.trial_id
            WHERE t.condition = 'peer_vote'
            """
        ).fetchall()
    finally:
        connection.close()
    selections = {
        (row["trial_id"], row["round_index"]): row["selected_agent_id"]
        for row in selection_rows
    }
    grouped: dict[tuple[str, int], list[sqlite3.Row]] = defaultdict(list)
    for row in ballot_rows:
        grouped[(row["trial_id"], row["round_index"])].append(row)
    details = []
    for key, rows in sorted(grouped.items()):
        eligible = {row["voter_agent_id"] for row in rows}
        counts = Counter({agent_id: 0 for agent_id in eligible})
        counts.update(
            row["supported_agent_id"] for row in rows if row["supported_agent_id"]
        )
        minimum = min(counts.values())
        tied = sorted(
            agent_id for agent_id, count in counts.items() if count == minimum
        )
        selected = selections[key]
        details.append(
            {
                "trial_id": key[0],
                "round_index": key[1],
                "minimum_support": minimum,
                "minimum_tie_size": len(tied),
                "selected_support": counts[selected],
            }
        )
    return {
        "mechanism": "one-positive-support-vote-v1",
        "rounds": len(details),
        "tie_rate": sum(row["minimum_tie_size"] > 1 for row in details) / len(details),
        "mean_minimum_tie_size": mean(row["minimum_tie_size"] for row in details),
        "selected_zero_support_rate": sum(
            row["selected_support"] == 0 for row in details
        )
        / len(details),
        "details": details,
    }


def deterministic_random_baseline(rounds: int = 1000) -> dict[str, Any]:
    counts = Counter(
        Random(derive_seed(20260823, index, "e02a_random_baseline")).randrange(8)
        for index in range(rounds)
    )
    return {
        "mechanism": "uniform-random-v1",
        "rounds": rounds,
        "selection_counts": {str(index): counts[index] for index in range(8)},
        "selection_shares": {str(index): counts[index] / rounds for index in range(8)},
        "max_selection_share": max(counts.values()) / rounds,
        "elimination_probability_per_agent": 0.125,
        "support_signal_used": False,
    }


def pairwise_repeatability(
    evidence: Sequence[Mapping[str, Any]], semantic_field: str
) -> dict[str, Any]:
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence:
        by_prompt[str(row["prompt_id"])].append(row)
    prompt_rows = []
    exact_matches = exact_pairs = semantic_matches = semantic_pairs = 0
    for prompt_id, rows in sorted(by_prompt.items()):
        successful = [row for row in rows if row.get("status") == "ok"]
        raw_values = [row["raw_output"] for row in successful]
        semantic_values = [row.get(semantic_field) for row in successful]
        exact = list(itertools.combinations(raw_values, 2))
        semantic = list(itertools.combinations(semantic_values, 2))
        exact_matches += sum(left == right for left, right in exact)
        exact_pairs += len(exact)
        semantic_matches += sum(left == right for left, right in semantic)
        semantic_pairs += len(semantic)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "attempts": len(rows),
                "successful_attempts": len(successful),
                "unique_raw_outputs": len(set(raw_values)),
                "unique_semantic_outputs": len(set(semantic_values)),
                "all_raw_identical": len(set(raw_values)) == 1
                and len(raw_values) == len(rows),
                "all_semantic_identical": (
                    len(set(semantic_values)) == 1 and len(semantic_values) == len(rows)
                ),
            }
        )
    return {
        "prompts": len(by_prompt),
        "attempts": len(evidence),
        "failures": sum(row.get("status") != "ok" for row in evidence),
        "exact_pairwise_match_rate": exact_matches / exact_pairs
        if exact_pairs
        else 0.0,
        "semantic_pairwise_match_rate": (
            semantic_matches / semantic_pairs if semantic_pairs else 0.0
        ),
        "all_prompts_exactly_repeatable": all(
            row["all_raw_identical"] for row in prompt_rows
        ),
        "all_prompts_semantically_repeatable": all(
            row["all_semantic_identical"] for row in prompt_rows
        ),
        "prompt_details": prompt_rows,
    }


def matched_condition_policy(
    response_summary: Mapping[str, Any], ballot_summary: Mapping[str, Any]
) -> str:
    if (
        response_summary["all_prompts_exactly_repeatable"]
        and ballot_summary["all_prompts_exactly_repeatable"]
    ):
        return (
            "Matched conditions must share prompts, schemas, task order, and seeds. "
            "Treat seeds as tested repeatability controls, retain every raw output, "
            "monitor divergence, and do not replay or cache provider outputs."
        )
    return (
        "Fresh provider generations are independent observations even under matched "
        "seeds. Match prompts, schemas, task order, and seeds; pair analyses by "
        "replicate; retain and report raw divergence; do not replay, retry-until-match, "
        "or cache provider outputs."
    )


def evaluate_readiness(report: Mapping[str, Any]) -> dict[str, Any]:
    tasks = report["task_calibration"]
    position = report["position_diagnostic"]
    peer = report["peer_diagnostic"]
    repeatability = report["repeatability"]
    failures = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            failures.append(reason)

    require(report["integrity"]["e01_hash_unchanged"], "E01 database hash changed")
    require(tasks["final_task_count"] >= 16, "fewer than 16 holdout-mixed tasks")
    require(
        tasks["holdout_summary"]["mixed_score_round_rate"] >= 0.75,
        "holdout mixed-score rate below 75%",
    )
    require(
        tasks["holdout_summary"]["degenerate_round_rate"] < 0.25,
        "holdout degenerate-round rate is not below 25%",
    )
    require(position["required_gate_passed"], "position/label required gate failed")
    require(position["ratio_gate_passed"], "anonymous-label ratio gate failed")
    require(
        peer["recommendation"] != "PEER MECHANISM STILL UNRESOLVED",
        "peer mechanism remains unresolved",
    )
    total_requests = report["provider"]["requests"]
    total_failures = report["provider"]["failures"]
    require(
        total_requests > 0 and total_failures / total_requests <= 0.05,
        "provider failure/timeout rate exceeds 5%",
    )
    require(
        repeatability["response"]["prompts"] >= 20
        and repeatability["ballot"]["prompts"] >= 20,
        "repeatability probe has fewer than 20 prompts per type",
    )
    return {
        "decision": (
            "E02A READY — PROCEED TO PROFILE MANIPULATION CHECK"
            if not failures
            else "REVISE E02A"
        ),
        "failures": failures,
        "preferred_task_gate_passed": tasks["final_task_count"] >= 20
        and tasks["holdout_summary"]["mixed_score_round_rate"] >= 0.85,
        "position_preferred_gate_passed": position["preferred_gate_passed"],
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def render_markdown(report: Mapping[str, Any]) -> str:
    task = report["task_calibration"]
    holdout = task["holdout_summary"]
    position = report["position_diagnostic"]
    peer = report["peer_diagnostic"]
    repeatability = report["repeatability"]
    readiness = report["readiness"]
    return "\n".join(
        (
            "# E02A Apparatus Revision",
            "",
            "Engineering apparatus validation only. No population or profile-effect claims are made.",
            "",
            f"**{readiness['decision']}**",
            "",
            "## Objective Tasks",
            "",
            f"- Candidates: {task['candidate_task_count']}",
            f"- Shortlisted after 16 attempts: {task['shortlist_task_count']}",
            f"- Final holdout-mixed tasks: {task['final_task_count']}",
            f"- Mixed holdout rate: {holdout['mixed_score_round_rate']:.3f}",
            f"- Degenerate holdout rate: {holdout['degenerate_round_rate']:.3f}",
            "- Scorer: `exact-match-v1` (unchanged)",
            "",
            "## Position And Label Diagnostic",
            "",
            f"- Evaluations: {position['evaluations']}",
            f"- Strict ballot validity: {position['strict_validity_rate']:.3f}",
            f"- Maximum display-position share: {position['max_display_share']:.3f}",
            f"- Maximum anonymous-label share: {position['max_label_share']:.3f}",
            f"- Maximum/minimum label count ratio: {position['max_to_min_label_count_ratio']:.3f}",
            "",
            "## Peer Mechanism",
            "",
            f"- E01 one-vote tie rate: {peer['e01_one_vote']['tie_rate']:.3f}",
            f"- Ranked top-3 tie rate: {peer['ranked_top3']['tie_rate']:.3f}",
            f"- Approval tie rate: {peer['approval']['tie_rate']:.3f}",
            f"- Recommendation: **{peer['recommendation']}**",
            f"- Recommended schema: `{peer['recommended_mechanism']}`",
            f"- Elimination-risk interpretation: {peer['elimination_risk_interpretation']}",
            "",
            "## Provider Repeatability",
            "",
            f"- Response exact pairwise match: {repeatability['response']['exact_pairwise_match_rate']:.3f}",
            f"- Response semantic pairwise match: {repeatability['response']['semantic_pairwise_match_rate']:.3f}",
            f"- Ballot exact pairwise match: {repeatability['ballot']['exact_pairwise_match_rate']:.3f}",
            f"- Ballot semantic pairwise match: {repeatability['ballot']['semantic_pairwise_match_rate']:.3f}",
            f"- Matched-condition policy: {repeatability['matched_condition_policy']}",
            "",
            "## Reliability And Integrity",
            "",
            f"- Provider failures: {report['provider']['failures']}/{report['provider']['requests']}",
            f"- E01 SHA-256 before: `{report['integrity']['e01_sha256_before']}`",
            f"- E01 SHA-256 after: `{report['integrity']['e01_sha256_after']}`",
            f"- E01 unchanged: {report['integrity']['e01_hash_unchanged']}",
            f"- Gate failures: `{readiness['failures']}`",
            "",
        )
    )


def generate_figures(report: Mapping[str, Any], output_dir: str | Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    created = []

    diagnostics = report["task_calibration"]["candidate_diagnostics"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        range(len(diagnostics)),
        [row["exact_match_rate"] for row in diagnostics],
        color="#3b6f8f",
    )
    axis.axhspan(0.15, 0.85, color="#d7e8d4", alpha=0.6)
    axis.set(
        title="E02A Candidate Difficulty",
        xlabel="Candidate task",
        ylabel="Exact-match rate",
    )
    axis.set_ylim(0, 1)
    figure.tight_layout()
    path = target / "task_candidate_difficulty.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    holdout = report["task_calibration"]["holdout_details"]
    if holdout:
        figure, axis = plt.subplots(figsize=(10, 5))
        colors = ["#3c8c5a" if row["mixed"] else "#aa4b4b" for row in holdout]
        axis.bar(range(len(holdout)), [row["correct"] for row in holdout], color=colors)
        axis.set(
            title="Fresh Holdout Rounds",
            xlabel="Shortlisted task",
            ylabel="Correct responses",
        )
        axis.set_ylim(0, 8)
        figure.tight_layout()
        path = target / "task_holdout_rounds.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path.name)
    else:
        (target / "task_holdout_rounds.png").unlink(missing_ok=True)

    position = report["position_diagnostic"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    display = position["display_index_shares"]
    labels = position["anonymous_label_shares"]
    axes[0].bar(display.keys(), display.values(), color="#7c5c9e")
    axes[0].axhline(0.35, color="#aa4b4b", linestyle="--")
    axes[0].set(title="Display Position", ylabel="Selection share", ylim=(0, 0.4))
    axes[1].bar(labels.keys(), labels.values(), color="#be7b45")
    axes[1].axhline(0.35, color="#aa4b4b", linestyle="--")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set(title="Anonymous Label", ylim=(0, 0.4))
    figure.tight_layout()
    path = target / "position_label_shares.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    peer = report["peer_diagnostic"]
    mechanisms = ("one vote", "ranked top-3", "approval")
    tie_rates = (
        peer["e01_one_vote"]["tie_rate"],
        peer["ranked_top3"]["tie_rate"],
        peer["approval"]["tie_rate"],
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(mechanisms, tie_rates, color=("#aa4b4b", "#3b6f8f", "#3c8c5a"))
    axis.set(title="Minimum-Signal Tie Rate", ylabel="Tie rate", ylim=(0, 1))
    figure.tight_layout()
    path = target / "peer_tie_rates.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    repeatability = report["repeatability"]
    figure, axis = plt.subplots(figsize=(7, 4))
    values = (
        repeatability["response"]["exact_pairwise_match_rate"],
        repeatability["response"]["semantic_pairwise_match_rate"],
        repeatability["ballot"]["exact_pairwise_match_rate"],
        repeatability["ballot"]["semantic_pairwise_match_rate"],
    )
    axis.bar(
        ("response raw", "response semantic", "ballot raw", "ballot semantic"), values
    )
    axis.tick_params(axis="x", rotation=25)
    axis.set(
        title="Fixed-Prompt Repeatability", ylabel="Pairwise match rate", ylim=(0, 1)
    )
    figure.tight_layout()
    path = target / "provider_repeatability.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)
    return created
