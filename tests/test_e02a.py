from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from src.experiments.e02a import (
    LETTER_LABELS,
    OPAQUE_LABELS,
    EvidenceCheckpoint,
    choice_schema,
    choose_task_shortlist,
    deterministic_random_baseline,
    e01_peer_baseline,
    evaluate_readiness,
    pairwise_repeatability,
    parse_choice,
    parse_rich_choices,
    peer_plan,
    position_plan,
    render_choice_prompt,
    rich_peer_schema,
    summarize_holdout,
    summarize_position,
    summarize_rich_peer,
)
from src.tasks import Task
from src.tasks.calibration import load_task_set

ROOT = Path(__file__).resolve().parents[1]


def task(task_id: str, family: str = "arithmetic") -> Task:
    return Task(
        task_id=task_id,
        family=family,
        prompt="Return only the integer.",
        expected_answer="1",
        scorer_version="exact-match-v1",
    )


def test_candidate_bank_is_new_diverse_and_contract_valid() -> None:
    artifact = load_task_set(ROOT / "tasks" / "e02a_candidates_v2.json")

    assert len(artifact.tasks) == 42
    assert len({item.family for item in artifact.tasks}) == 7
    assert not {
        "arith-002",
        "count-003",
        "sequence-002",
        "sequence-003",
        "string-005",
        "arith-004",
    } & {item.task_id for item in artifact.tasks}


def test_shortlist_has_no_ceiling_or_floor_fallback() -> None:
    tasks = (task("arith-101"), task("arith-102"), task("arith-103"))
    rates = {"arith-101": 8, "arith-102": 16, "arith-103": 0}
    evidence = []
    for item in tasks:
        for attempt in range(16):
            evidence.append(
                {
                    "task_id": item.task_id,
                    "status": "ok",
                    "correct": attempt < rates[item.task_id],
                    "format_valid": True,
                    "raw_output": str(attempt < rates[item.task_id]),
                }
            )

    selected, diagnostics, rejected = choose_task_shortlist(tasks, evidence)

    assert [item.task_id for item in selected] == ["arith-101"]
    assert len(diagnostics) == 3
    assert rejected["arith-102"] == "outside_0.15_to_0.85_band"
    assert rejected["arith-103"] == "outside_0.15_to_0.85_band"


def test_holdout_keeps_only_complete_mixed_rounds() -> None:
    tasks = (task("arith-101"), task("arith-102"), task("arith-103"))
    evidence = []
    for item, correct in zip(tasks, (4, 8, 0), strict=True):
        for attempt in range(8):
            evidence.append(
                {
                    "task_id": item.task_id,
                    "status": "ok",
                    "correct": attempt < correct,
                    "raw_output": "1",
                }
            )

    selected, summary, details = summarize_holdout(tasks, evidence)

    assert [item.task_id for item in selected] == ["arith-101"]
    assert summary["mixed_score_round_rate"] == pytest.approx(1 / 3)
    assert summary["degenerate_round_rate"] == pytest.approx(2 / 3)
    assert [row["mixed"] for row in details] == [True, False, False]


def test_position_plan_is_fully_crossed_and_separates_dimensions() -> None:
    plans = position_plan()

    assert len(plans) == 196
    assert len({plan["evaluation_id"] for plan in plans}) == 196
    for scheme, expected_labels in (
        ("letters", LETTER_LABELS),
        ("opaque", OPAQUE_LABELS),
    ):
        candidates = [
            candidate
            for plan in plans
            if plan["scheme"] == scheme
            for candidate in plan["candidates"]
        ]
        display_label = Counter(
            (row["display_index"], row["anonymous_label"]) for row in candidates
        )
        display_content = Counter(
            (row["display_index"], row["underlying_candidate_id"]) for row in candidates
        )
        label_content = Counter(
            (row["anonymous_label"], row["underlying_candidate_id"])
            for row in candidates
        )
        assert {row["anonymous_label"] for row in candidates} == set(expected_labels)
        assert set(display_label.values()) == {14}
        assert set(display_content.values()) == {14}
        assert set(label_content.values()) == {14}


def test_ballot_prompt_has_no_literal_valid_choice_example() -> None:
    plan = position_plan()[0]
    labels = [row["anonymous_label"] for row in plan["candidates"]]
    prompt = render_choice_prompt("Choose the correct response.", plan["candidates"])

    assert '{"choice":' not in prompt
    assert choice_schema(labels)["properties"]["choice"]["enum"] == labels


@pytest.mark.parametrize(
    ("raw", "valid", "reason"),
    [
        ('{"choice":"A"}', "A", None),
        ('{"choice":"A","choice":"B"}', None, "invalid_json"),
        ('{"choice":"Z"}', None, "invalid_choice"),
        ('{"choice":"A","extra":1}', None, "invalid_shape"),
        ("A", None, "invalid_json"),
    ],
)
def test_choice_parser_is_strict(
    raw: str, valid: str | None, reason: str | None
) -> None:
    assert parse_choice(raw, LETTER_LABELS) == (valid, reason)


def test_rich_peer_plan_and_schemas_are_bounded() -> None:
    plans = peer_plan()

    assert len(plans) == 256
    assert len({plan["evaluation_id"] for plan in plans}) == 256
    for plan in plans:
        assert len(plan["candidates"]) == 7
        assert plan["voter_id"] not in {
            candidate["underlying_candidate_id"] for candidate in plan["candidates"]
        }
    ranked = rich_peer_schema(LETTER_LABELS, "ranked_top3")
    approval = rich_peer_schema(OPAQUE_LABELS, "approval")
    assert ranked["properties"]["choices"]["minItems"] == 3
    assert ranked["properties"]["choices"]["maxItems"] == 3
    assert approval["properties"]["choices"]["minItems"] == 1
    assert approval["properties"]["choices"]["maxItems"] == 7


def test_rich_choice_parser_rejects_duplicates_and_wrong_counts() -> None:
    assert parse_rich_choices(
        '{"choices":["A","B","C"]}', LETTER_LABELS, "ranked_top3"
    ) == (["A", "B", "C"], None)
    assert (
        parse_rich_choices('{"choices":["A","A","B"]}', LETTER_LABELS, "ranked_top3")[1]
        == "invalid_choice"
    )
    assert (
        parse_rich_choices('{"choices":["A","B"]}', LETTER_LABELS, "ranked_top3")[1]
        == "invalid_count"
    )
    assert (
        parse_rich_choices(
            '{"choices":["A"],"choices":["B"]}', LETTER_LABELS, "approval"
        )[1]
        == "invalid_json"
    )


def test_rich_peer_invalid_ballots_are_abstentions_not_dropped_panels() -> None:
    evidence = []
    for plan in peer_plan():
        candidate_ids = [
            candidate["underlying_candidate_id"] for candidate in plan["candidates"]
        ]
        valid = not (plan["panel"] == 0 and plan["voter_id"] == "peer-agent-0")
        evidence.append(
            {
                "mechanism": plan["mechanism"],
                "panel": plan["panel"],
                "valid": valid,
                "selected_candidate_ids": (
                    candidate_ids[:3]
                    if plan["mechanism"] == "ranked_top3"
                    else candidate_ids[:1]
                ),
            }
        )

    summary = summarize_rich_peer(evidence)

    assert summary["ranked_top3"]["complete_panels"] == 16
    assert summary["approval"]["complete_panels"] == 16
    assert summary["ranked_top3"]["panel_details"][0]["abstentions"] == 1
    assert summary["approval"]["panel_details"][0]["abstentions"] == 1


def test_position_summary_applies_required_and_preferred_gates() -> None:
    evidence = []
    labels = LETTER_LABELS + OPAQUE_LABELS
    for index in range(196):
        evidence.append(
            {
                "valid": True,
                "selected_display_index": index % 7,
                "selected_anonymous_label": labels[index % 14],
                "selected_candidate_id": f"content-{index % 7}",
            }
        )

    summary = summarize_position(evidence)

    assert summary["strict_validity_rate"] == 1
    assert summary["required_gate_passed"]
    assert summary["preferred_gate_passed"]
    assert summary["ratio_gate_passed"]


def test_repeatability_reports_raw_and_semantic_matches_separately() -> None:
    evidence = [
        {
            "prompt_id": "p0",
            "status": "ok",
            "raw_output": raw,
            "semantic": "A",
        }
        for raw in ("A", "A", "a", "A", "a")
    ]

    summary = pairwise_repeatability(evidence, "semantic")

    assert summary["exact_pairwise_match_rate"] == pytest.approx(0.4)
    assert summary["semantic_pairwise_match_rate"] == 1
    assert not summary["all_prompts_exactly_repeatable"]
    assert summary["all_prompts_semantically_repeatable"]


def test_checkpoint_is_append_only_and_protocol_bound(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    checkpoint = EvidenceCheckpoint(path, "abc")
    checkpoint.append({"evaluation_id": "one", "phase": "test"})

    resumed = EvidenceCheckpoint(path, "abc")
    assert resumed.get("one") is not None
    with pytest.raises(ValueError, match="duplicate"):
        resumed.append({"evaluation_id": "one", "phase": "test"})
    with pytest.raises(ValueError, match="protocol hash"):
        EvidenceCheckpoint(path, "different")


def test_e01_peer_baseline_reads_database_without_writing(tmp_path: Path) -> None:
    database = tmp_path / "e01.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE trials (trial_id TEXT PRIMARY KEY, condition TEXT NOT NULL);
        CREATE TABLE ballots (
            trial_id TEXT, round_index INTEGER, voter_agent_id TEXT,
            supported_agent_id TEXT, ordinal INTEGER
        );
        CREATE TABLE selection_events (
            trial_id TEXT, round_index INTEGER, selected_agent_id TEXT
        );
        INSERT INTO trials VALUES ('peer-1', 'peer_vote');
        """
    )
    for voter in range(8):
        connection.execute(
            "INSERT INTO ballots VALUES (?, ?, ?, ?, ?)",
            ("peer-1", 0, f"a{voter}", f"a{(voter + 1) % 8}", voter),
        )
    connection.execute("INSERT INTO selection_events VALUES ('peer-1', 0, 'a0')")
    connection.commit()
    connection.close()
    before = database.read_bytes()

    summary = e01_peer_baseline(database)

    assert summary["rounds"] == 1
    assert summary["tie_rate"] == 1
    assert summary["mean_minimum_tie_size"] == 8
    assert database.read_bytes() == before


def test_random_baseline_is_deterministic_and_near_uniform() -> None:
    first = deterministic_random_baseline(8000)
    second = deterministic_random_baseline(8000)

    assert first == second
    assert first["support_signal_used"] is False
    assert first["max_selection_share"] < 0.14


def test_readiness_requires_every_apparatus_gate() -> None:
    report = {
        "integrity": {"e01_hash_unchanged": True},
        "task_calibration": {
            "final_task_count": 20,
            "holdout_summary": {
                "mixed_score_round_rate": 0.9,
                "degenerate_round_rate": 0.1,
            },
        },
        "position_diagnostic": {
            "required_gate_passed": True,
            "preferred_gate_passed": True,
            "ratio_gate_passed": True,
        },
        "peer_diagnostic": {"recommendation": "ADOPT REVISED PEER BALLOT FOR E03"},
        "repeatability": {"response": {"prompts": 20}, "ballot": {"prompts": 20}},
        "provider": {"requests": 100, "failures": 0},
    }

    readiness = evaluate_readiness(report)

    assert readiness["decision"].startswith("E02A READY")
    report["position_diagnostic"]["required_gate_passed"] = False
    assert evaluate_readiness(report)["decision"] == "REVISE E02A"
