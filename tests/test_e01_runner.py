from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.agents import AgentIdentity
from src.experiments.e01 import E01Error, build_e01_plan, run_e01
from src.models import ModelOutput, ModelProvider
from src.storage import SQLiteEventStore
from src.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "tasks" / "e01_validated_v1.json"


class FakeOllamaProvider(ModelProvider):
    def __init__(self, *, interrupt_at: int | None = None) -> None:
        self.calls = 0
        self.interrupt_at = interrupt_at

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "qwen3:0.6b"

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
        del agent
        self.calls += 1
        if self.interrupt_at is not None and self.calls == self.interrupt_at:
            raise KeyboardInterrupt
        content = '{"choice":"A"}' if response_schema is not None else task.expected_answer
        assert content is not None
        return ModelOutput(
            content=content,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=request_id,
            seed=seed,
        )


def plan_for(path: Path):
    return build_e01_plan(
        mode="smoke",
        database_path=path,
        task_path=TASK_PATH,
    )


def test_smoke_plan_is_matched_and_frozen(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "e01.sqlite")

    assert len(plan.trials) == 6
    assert plan.rounds_per_trial == 3
    assert plan.total_rounds == 18
    for offset in (0, 3):
        matched = plan.trials[offset : offset + 3]
        assert tuple(trial.condition for trial in matched) == (
            "peer_vote",
            "objective",
            "random",
        )
        assert len({trial.seed for trial in matched}) == 1
        assert len({trial.identity_namespace for trial in matched}) == 1
        assert len({trial.trial_id for trial in matched}) == 3
    config = json.loads(plan.config_json)
    assert config["implementation_hashes"]["src/experiments/e01.py"]
    assert config["task_set"]["hash"] == plan.task_hash
    assert config["model"]["think"] is False
    assert config["model"]["temperature"] == 0
    assert config["model"]["num_predict"] == 32


def test_metadata_and_all_trials_exist_before_provider_check(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "metadata.sqlite")
    provider = FakeOllamaProvider()

    def unavailable() -> str:
        raise E01Error("provider unavailable")

    with pytest.raises(E01Error, match="provider unavailable"):
        run_e01(
            plan,
            provider,
            repository=ROOT,
            check_provider=unavailable,
            output=lambda _: None,
        )

    with SQLiteEventStore(plan.database_path) as store:
        assert store.get_experiment(plan.experiment_id).config_hash == plan.config_hash
        assert all(store.trial_exists(trial.trial_id) for trial in plan.trials)
        assert all(
            store.last_committed_round(trial.trial_id) is None
            for trial in plan.trials
        )


def test_matched_conditions_share_first_round_randomization(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "matched.sqlite")
    outcome = run_e01(
        plan,
        FakeOllamaProvider(),
        repository=ROOT,
        stop_after_commits=3,
        output=lambda _: None,
    )

    assert outcome.controlled_stop
    with SQLiteEventStore(plan.database_path) as store:
        matched = tuple(store.load_round(trial.trial_id, 0) for trial in plan.trials[:3])
        assert len({result.task.task_id for result in matched}) == 1
        assert [response.agent_id for response in matched[0].responses] == [
            response.agent_id for response in matched[1].responses
        ] == [response.agent_id for response in matched[2].responses]
        assert [response.seed for response in matched[0].responses] == [
            response.seed for response in matched[1].responses
        ] == [response.seed for response in matched[2].responses]
        assert [evidence.seed for evidence in matched[0].ballot_evidence] == [
            evidence.seed for evidence in matched[1].ballot_evidence
        ] == [evidence.seed for evidence in matched[2].ballot_evidence]
        assert [
            tuple(candidate.agent_id for candidate in evidence.candidate_order)
            for evidence in matched[0].ballot_evidence
        ] == [
            tuple(candidate.agent_id for candidate in evidence.candidate_order)
            for evidence in matched[1].ballot_evidence
        ] == [
            tuple(candidate.agent_id for candidate in evidence.candidate_order)
            for evidence in matched[2].ballot_evidence
        ]


def test_interrupt_resume_and_completed_rerun_are_incremental(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "resume.sqlite")
    interrupted = FakeOllamaProvider(interrupt_at=51)

    with pytest.raises(KeyboardInterrupt):
        run_e01(plan, interrupted, repository=ROOT, output=lambda _: None)

    with SQLiteEventStore(plan.database_path) as store:
        committed_before = 0
        for trial in plan.trials:
            last_round = store.last_committed_round(trial.trial_id)
            committed_before += 0 if last_round is None else last_round + 1
    assert committed_before == 3

    resumed = FakeOllamaProvider()
    outcome = run_e01(plan, resumed, repository=ROOT, output=lambda _: None)
    assert outcome.completed
    assert outcome.committed_this_run == 15
    assert resumed.calls == 15 * 16

    completed = FakeOllamaProvider()
    rerun = run_e01(plan, completed, repository=ROOT, output=lambda _: None)
    assert rerun.completed
    assert rerun.committed_this_run == 0
    assert completed.calls == 0
    summary = json.loads(plan.summary_path.read_text(encoding="utf-8"))
    assert summary["apparatus"]["committed_rounds"] == 18
    assert summary["apparatus"]["committed_responses"] == 144
    assert summary["apparatus"]["ballot_attempts"] == 144
    assert summary["apparatus"]["ballot_valid"] == 144
    assert summary["apparatus"]["committed_model_generations"] == 288


def test_resume_rejects_protocol_change(tmp_path: Path) -> None:
    database = tmp_path / "incompatible.sqlite"
    original = plan_for(database)
    run_e01(
        original,
        FakeOllamaProvider(),
        repository=ROOT,
        stop_after_commits=1,
        output=lambda _: None,
    )
    changed = build_e01_plan(
        mode="smoke",
        database_path=database,
        task_path=TASK_PATH,
        master_seed=original.master_seed + 1,
    )

    with pytest.raises(E01Error, match="does not match persisted"):
        run_e01(changed, FakeOllamaProvider(), repository=ROOT, output=lambda _: None)


def test_stop_on_final_commit_reports_completion(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "final-stop.sqlite")

    outcome = run_e01(
        plan,
        FakeOllamaProvider(),
        repository=ROOT,
        stop_after_commits=plan.total_rounds,
        output=lambda _: None,
    )

    assert outcome.completed
    assert not outcome.controlled_stop


def test_stale_lock_is_recovered(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "stale-lock.sqlite")
    lock_path = Path(f"{plan.database_path}.lock")
    lock_path.write_text("pid=999999999\n", encoding="ascii")

    outcome = run_e01(
        plan,
        FakeOllamaProvider(),
        repository=ROOT,
        stop_after_commits=1,
        output=lambda _: None,
    )

    assert outcome.committed_this_run == 1
    assert not lock_path.exists()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_controlled_stop_is_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises(E01Error, match="positive integer"):
        run_e01(
            plan_for(tmp_path / "invalid-stop.sqlite"),
            FakeOllamaProvider(),
            repository=ROOT,
            stop_after_commits=value,  # type: ignore[arg-type]
            output=lambda _: None,
        )
