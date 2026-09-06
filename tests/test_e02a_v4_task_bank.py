from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.tasks.calibration import load_task_set


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "tasks" / "e02a_candidates_v4.json"


def test_v4_candidate_bank_is_balanced_and_loadable() -> None:
    raw = json.loads(BANK.read_text(encoding="utf-8"))
    artifact = load_task_set(BANK)

    assert raw["task_set_version"] == "e02a-candidates-v4-adaptive"
    assert raw["scorer_version"] == "exact-match-v1"
    assert raw["status"] == "candidate"
    assert len(artifact.tasks) == 60

    family_counts = Counter(task.family for task in artifact.tasks)
    assert family_counts == {
        "arithmetic": 10,
        "counting": 10,
        "indexing": 10,
        "mapping": 10,
        "sequence": 10,
        "string": 10,
    }
    assert all(task.scorer_version == "exact-match-v1" for task in artifact.tasks)


def test_v4_bank_avoids_known_v3_default_token_traps() -> None:
    raw = json.loads(BANK.read_text(encoding="utf-8"))
    tasks = raw["tasks"]

    assert all(task["family"] not in {"logic", "multiple-choice"} for task in tasks)
    assert all("\nA." not in task["prompt"] for task in tasks)
    assert all("true or false" not in task["prompt"].casefold() for task in tasks)

    integer_families = {"arithmetic", "counting", "mapping", "sequence"}
    assert all(
        task["expected_answer"] != "2"
        for task in tasks
        if task["family"] in integer_families
    )


def test_v4_provenance_records_adaptive_authorship_and_fresh_holdout() -> None:
    raw = json.loads(BANK.read_text(encoding="utf-8"))
    provenance = raw["provenance"]

    assert "adaptively authored" in provenance["source"]
    assert "E02A-v3" in provenance["adaptation_basis"]
    assert "fresh 8-profile holdout" in provenance["validation_protocol"]
    assert provenance["scoring_policy"].startswith("exact-match-v1 unchanged")
    assert "rerun before E03" in provenance["dependency"]
