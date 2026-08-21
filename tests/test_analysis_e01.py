from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import pytest

from src.agents import AgentIdentity, PromptProfile
from src.analysis.e01 import analyze_e01_database
from src.analysis.integrity import (
    audit_e01_integrity,
    load_analysis_tables,
    open_read_only,
    sha256_file,
)
from src.analysis.metrics import (
    is_mixed_score,
    objective_mean,
    profile_composition,
    profile_lifetime,
    selected_agent_correctness,
    support_correctness,
)
from src.analysis.reporting import REQUIRED_MARKDOWN_SECTIONS, deterministic_json, render_markdown
from src.analysis.statistics import entropy, l1_distance, paired_difference
from src.ballots import LLMBallotProvider
from src.models import ModelOutput, ModelProvider
from src.replacement import FIXED_QUEUE_VERSION, profile_pool_hash
from src.storage import Provenance, SQLiteEventStore
from src.tasks import Task
from src.tournament import FixedTaskSource, RoundEngine, TrialRunner


class AnalysisFixtureProvider(ModelProvider):
    @property
    def provider_name(self) -> str:
        return "fixture"

    @property
    def model_name(self) -> str:
        return "fixture-v1"

    def generate(
        self,
        *,
        agent: AgentIdentity,
        task: Task,
        prompt: str,
        request_id: str,
        seed: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
    ) -> ModelOutput:
        del agent, prompt
        content = '{"choice":"A"}' if response_schema is not None else task.expected_answer
        assert content is not None
        return ModelOutput(
            content=content,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=request_id,
            seed=seed,
        )


def fixture_profiles() -> dict[str, PromptProfile]:
    return {
        f"profile-{index:03d}": PromptProfile(
            f"profile-{index:03d}", {"approach": f"style-{index}"}, "fixture-v1"
        )
        for index in range(1, 9)
    }


def create_analysis_database(path: Path) -> Path:
    profiles = fixture_profiles()
    tasks = (
        Task("task-001", "arithmetic", "Return 4.", "4", "exact-match-v1"),
        Task("task-002", "string", "Return X.", "X", "exact-match-v1"),
    )
    config = {
        "apparatus": {
            "conditions": ["peer_vote", "objective", "random"],
            "population_size": 8,
            "protocol_version": "fixture-v1",
            "replacement_version": FIXED_QUEUE_VERSION,
        },
        "profiles": {
            "hash": profile_pool_hash(profiles),
            "values": [
                {
                    "profile_id": profile.profile_id,
                    "parameters": dict(profile.parameters),
                    "template_version": profile.template_version,
                }
                for profile in profiles.values()
            ],
        },
        "rounds_per_trial": 2,
        "task_set": {"hash": "fixture-task-hash"},
        "trials_per_condition": 1,
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    provider = AnalysisFixtureProvider()
    with SQLiteEventStore(path) as store:
        store.initialize()
        store.create_experiment(
            experiment_id="fixture-experiment",
            name="Analysis fixture",
            config_schema_version=1,
            config_hash=config_hash,
            config_json=config_json,
            provenance=Provenance(
                code_commit="fixture",
                python_version="3.12",
                platform="test",
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                created_at="2026-08-21T00:00:00+00:00",
            ),
        )
        for condition in ("peer_vote", "objective", "random"):
            TrialRunner(
                experiment_id="fixture-experiment",
                trial_id=f"e01-full-{condition}-replicate-001",
                trial_seed=42,
                condition=condition,  # type: ignore[arg-type]
                total_rounds=2,
                config_hash=config_hash,
                profiles=profiles,
                task_source=FixedTaskSource(tasks),
                provider=provider,
                event_store=store,
                round_engine=RoundEngine(ballot_provider=LLMBallotProvider()),
                agent_id_namespace="fixture-replicate-001",
            ).run()
    return path


def audit(path: Path) -> dict[str, Any]:
    with open_read_only(path) as connection:
        tables = load_analysis_tables(connection)
        return audit_e01_integrity(connection, tables)


def mutate(path: Path, operation: Callable[[sqlite3.Connection], None]) -> None:
    connection = sqlite3.connect(path)
    try:
        operation(connection)
        connection.commit()
    finally:
        connection.close()


def failed_checks(result: dict[str, Any]) -> set[str]:
    return {row["name"] for row in result["checks"] if not row["passed"]}


def test_valid_fixture_passes_integrity(tmp_path: Path) -> None:
    result = audit(create_analysis_database(tmp_path / "valid.sqlite"))

    assert result["status"] == "PASS"
    assert result["counts"]["trials"] == 3
    assert result["counts"]["rounds"] == 6


def test_missing_round_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "missing-round.sqlite")

    def remove_round(connection: sqlite3.Connection) -> None:
        key = ("e01-full-random-replicate-001", 1)
        for table in (
            "ballot_evidence",
            "ballots",
            "scores",
            "responses",
            "selection_events",
            "rounds",
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE trial_id=? AND round_index=?", key
            )

    mutate(path, remove_round)
    result = audit(path)

    assert "count_rounds" in failed_checks(result)
    assert "round_continuity" in failed_checks(result)


def test_wrong_response_count_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "wrong-responses.sqlite")
    mutate(
        path,
        lambda connection: connection.execute(
            "DELETE FROM responses WHERE response_id=(SELECT response_id FROM responses LIMIT 1)"
        ),
    )

    result = audit(path)

    assert "count_responses" in failed_checks(result)
    assert "per_round_evidence_counts" in failed_checks(result)


def test_unmatched_task_schedule_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "unmatched-task.sqlite")

    def change_task(connection: sqlite3.Connection) -> None:
        trial_id = "e01-full-random-replicate-001"
        current = connection.execute(
            "SELECT task_id FROM rounds WHERE trial_id=? AND round_index=0", (trial_id,)
        ).fetchone()[0]
        replacement = "task-002" if current == "task-001" else "task-001"
        for table in ("responses", "scores", "ballot_evidence", "rounds"):
            connection.execute(
                f"UPDATE {table} SET task_id=? WHERE trial_id=? AND round_index=0",
                (replacement, trial_id),
            )

    mutate(path, change_task)

    assert "matched_task_schedule" in failed_checks(audit(path))


def test_mismatched_replicate_seed_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "seed.sqlite")
    mutate(
        path,
        lambda connection: connection.execute(
            "UPDATE trials SET trial_seed='43' WHERE trial_id='e01-full-random-replicate-001'"
        ),
    )

    assert "matched_replicate_seeds" in failed_checks(audit(path))


def test_self_ballot_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "self-ballot.sqlite")

    def create_self_ballot(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            "UPDATE ballots SET supported_agent_id=voter_agent_id "
            "WHERE ballot_id=(SELECT ballot_id FROM ballots LIMIT 1)"
        )

    mutate(path, create_self_ballot)

    assert "self_ballot_prohibition" in failed_checks(audit(path))


def test_invalid_selection_target_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "selection.sqlite")

    def change_selection(connection: sqlite3.Connection) -> None:
        trial_id = "e01-full-peer_vote-replicate-001"
        inactive = connection.execute(
            "SELECT agent_id FROM agent_instances WHERE trial_id=? AND introduced_round=0",
            (trial_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE selection_events SET selected_agent_id=? "
            "WHERE trial_id=? AND round_index=0",
            (inactive, trial_id),
        )

    mutate(path, change_selection)

    assert "selection_eligibility" in failed_checks(audit(path))


def test_wrong_active_selection_is_reconstructed(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "selection-rule.sqlite")

    def change_scores(connection: sqlite3.Connection) -> None:
        trial_id = "e01-full-objective-replicate-001"
        selected = connection.execute(
            "SELECT selected_agent_id FROM selection_events "
            "WHERE trial_id=? AND round_index=0",
            (trial_id,),
        ).fetchone()[0]
        other = connection.execute(
            "SELECT agent_id FROM scores WHERE trial_id=? AND round_index=0 "
            "AND agent_id<>? LIMIT 1",
            (trial_id, selected),
        ).fetchone()[0]
        connection.execute(
            "UPDATE scores SET value=0 WHERE trial_id=? AND round_index=0 AND agent_id=?",
            (trial_id, other),
        )

    mutate(path, change_scores)

    assert "selection_rule_reconstruction" in failed_checks(audit(path))


def test_invalid_score_domain_is_detected(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "score-domain.sqlite")
    mutate(
        path,
        lambda connection: connection.execute(
            "UPDATE scores SET value=0.5 WHERE score_id=(SELECT score_id FROM scores LIMIT 1)"
        ),
    )

    assert "score_domain_and_version" in failed_checks(audit(path))


def test_recorded_invalid_ballot_is_reliability_outcome_not_corruption(
    tmp_path: Path,
) -> None:
    path = create_analysis_database(tmp_path / "invalid-ballot.sqlite")

    def invalidate_ballot(connection: sqlite3.Connection) -> None:
        ballot_id = connection.execute(
            "SELECT ballot_id FROM ballots "
            "WHERE trial_id='e01-full-objective-replicate-001' LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE ballots SET supported_agent_id=NULL WHERE ballot_id=?", (ballot_id,)
        )
        connection.execute(
            "UPDATE ballot_evidence SET valid=0, parsed_choice=NULL, "
            "invalid_reason='fixture_invalid' WHERE ballot_id=?",
            (ballot_id,),
        )

    mutate(path, invalidate_ballot)
    result = audit(path)

    assert result["status"] == "PASS"
    zero_invalid = next(
        row for row in result["checks"] if row["name"] == "zero_invalid_ballots"
    )
    assert not zero_invalid["passed"]
    assert not zero_invalid["critical"]


def test_read_only_connection_cannot_mutate_source(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "readonly.sqlite")
    before = sha256_file(path)

    with open_read_only(path) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM rounds")

    assert sha256_file(path) == before


def test_analysis_import_does_not_import_ollama() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.analysis.e01; "
            "assert 'src.models.ollama' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_metric_primitives() -> None:
    rows = [{"score": 0.0}, {"score": 1.0}, {"score": 1.0}]

    assert objective_mean(rows) == pytest.approx(2 / 3)
    assert is_mixed_score([0.0, 1.0])
    assert not is_mixed_score([1.0, 1.0])
    assert support_correctness({"supported_agent_score": 1.0}) is True
    assert selected_agent_correctness({"selected_agent_score": 0.0}) is False
    assert entropy([4, 4, 0, 0]) == pytest.approx(1.0)
    assert entropy([1, 1, 1, 1], normalized=True) == pytest.approx(1.0)


def test_profile_and_paired_metrics() -> None:
    rows = [
        {"profile_id": "p1"},
        {"profile_id": "p1"},
        {"profile_id": "p2"},
    ]
    paired = paired_difference([0.4, 0.6, 0.5], [0.2, 0.5, 0.5], label="test")

    assert profile_composition(rows) == {"p1": 2, "p2": 1}
    assert profile_lifetime({"lifetime_rounds": 4}) == 4
    assert paired["mean_paired_difference"] == pytest.approx(0.1)
    assert paired["n_pairs"] == 3
    assert l1_distance({"p1": 3, "p2": 1}, {"p1": 1, "p2": 3}) == 4


def test_report_serialization_and_sections(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "report.sqlite")
    report, _ = analyze_e01_database(path, task_artifact_path=None, repository=None)
    first = deterministic_json(report)
    second = deterministic_json(report)
    markdown = render_markdown(report, ())

    assert first == second
    assert first.endswith("\n")
    for section in REQUIRED_MARKDOWN_SECTIONS:
        assert f"## {section}" in markdown
    assert "not confirmatory" in markdown
    assert "persistent reciprocity? | No" in markdown


def test_integrity_failure_report_withholds_substantive_analysis(tmp_path: Path) -> None:
    path = create_analysis_database(tmp_path / "failed-report.sqlite")
    mutate(
        path,
        lambda connection: connection.execute(
            "DELETE FROM responses WHERE response_id=(SELECT response_id FROM responses LIMIT 1)"
        ),
    )

    report, _ = analyze_e01_database(path, task_artifact_path=None, repository=None)
    markdown = render_markdown(report, ())

    assert report["integrity"]["status"] == "FAIL"
    assert "performance" not in report
    assert "Substantive analysis withheld" in markdown
