from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from src.agents import PromptProfile
from src.models import MockModelProvider
from src.storage import NotFoundError, Provenance, SQLiteEventStore
from src.tasks import Task
from src.tournament import (
    FixedTaskSource,
    TrialCompleteError,
    TrialError,
    TrialRunner,
)


CONFIG_JSON = '{"apparatus":"e00","schema_version":1}'
CONFIG_HASH = sha256(CONFIG_JSON.encode("utf-8")).hexdigest()
TIMESTAMP = "2026-08-21T06:30:00+00:00"


def profiles() -> dict[str, PromptProfile]:
    return {
        f"profile-{index:03d}": PromptProfile(
            f"profile-{index:03d}", {"setting": index}, "v1"
        )
        for index in range(1, 9)
    }


def tasks() -> FixedTaskSource:
    return FixedTaskSource(
        (
            Task(
                "task-001",
                "logic",
                "Choose the correct answer: A.",
                "A",
                "exact-match-v1",
            ),
            Task(
                "task-002",
                "logic",
                "Choose the correct answer: B.",
                "B",
                "exact-match-v1",
            ),
        )
    )


def create_store(path: Path) -> SQLiteEventStore:
    store = SQLiteEventStore(path)
    store.initialize()
    if not store.trial_exists("trial-001"):
        try:
            store.get_experiment("experiment-e00")
        except NotFoundError:
            store.create_experiment(
                experiment_id="experiment-e00",
                name="E00 trial",
                config_schema_version=1,
                config_hash=CONFIG_HASH,
                config_json=CONFIG_JSON,
                provenance=Provenance(
                    code_commit="test-commit",
                    python_version="3.12",
                    platform="test-platform",
                    provider_name="mock",
                    model_name="deterministic-v1",
                    created_at=TIMESTAMP,
                ),
            )
    return store


def runner(
    store: SQLiteEventStore,
    *,
    condition: str = "peer_vote",
    trial_seed: int = 42,
    total_rounds: int = 10,
    config_hash: str = CONFIG_HASH,
) -> TrialRunner:
    return TrialRunner(
        experiment_id="experiment-e00",
        trial_id="trial-001",
        trial_seed=trial_seed,
        condition=condition,  # type: ignore[arg-type]
        total_rounds=total_rounds,
        config_hash=config_hash,
        profiles=profiles(),
        task_source=tasks(),
        provider=MockModelProvider(),
        event_store=store,
    )


def test_fresh_trial_initializes_once_at_round_zero(tmp_path: Path) -> None:
    store = create_store(tmp_path / "fresh.sqlite")
    trial_runner = runner(store, total_rounds=3)

    first = trial_runner.initialize()
    second = trial_runner.initialize()

    assert first == second
    assert first.next_round_index == 0
    assert first.replacement_queue_position == 0
    assert len(first.population) == 8
    assert store.load_initial_agents("trial-001") == first.population.agents
    assert len(store.load_agents("trial-001")) == 8
    store.close()


def test_ten_round_trial_persists_complete_e00_record(tmp_path: Path) -> None:
    store = create_store(tmp_path / "ten-rounds.sqlite")

    state = runner(store).run()

    assert state.completed is True
    assert state.next_round_index == 10
    assert state.replacement_queue_position == 9
    assert store.last_committed_round("trial-001") == 9
    assert store.get_trial("trial-001").status == "completed"
    assert len(store.load_replacement_events("trial-001")) == 9
    assert len(store.load_agents("trial-001")) == 17
    assert len(store.active_population("trial-001")) == 8
    for round_index in range(10):
        result = store.load_round("trial-001", round_index)
        assert len(result.responses) == 8
        assert len(result.scores) == 8
        assert len(result.ballots) == 8
    with pytest.raises(TrialCompleteError):
        runner(store).run_next_round()
    store.close()


@pytest.mark.parametrize("condition", ["peer_vote", "objective", "random"])
def test_all_conditions_complete_with_same_queue_policy(
    tmp_path: Path, condition: str
) -> None:
    store = create_store(tmp_path / f"{condition}.sqlite")
    trial_runner = runner(store, condition=condition, total_rounds=3)
    expected_queue = trial_runner.replacement_queue.profile_ids

    state = trial_runner.run()

    assert state.completed
    assert tuple(
        event.profile_id for event in store.load_replacement_events("trial-001")
    ) == expected_queue
    assert all(
        store.load_round("trial-001", index).selection.mechanism == condition
        for index in range(3)
    )
    store.close()


def test_interrupted_resume_matches_uninterrupted_trial(tmp_path: Path) -> None:
    uninterrupted_path = tmp_path / "uninterrupted.sqlite"
    resumed_path = tmp_path / "resumed.sqlite"

    uninterrupted_store = create_store(uninterrupted_path)
    uninterrupted_runner = runner(uninterrupted_store, total_rounds=6)
    uninterrupted_states = []
    while not uninterrupted_runner.initialize().completed:
        uninterrupted_states.append(uninterrupted_runner.run_next_round().state)

    resumed_store = create_store(resumed_path)
    resumed_runner = runner(resumed_store, total_rounds=6)
    resumed_states = [
        resumed_runner.run_next_round().state,
        resumed_runner.run_next_round().state,
    ]
    committed_before_restart = tuple(
        resumed_store.load_round("trial-001", index) for index in range(2)
    )
    active_before_restart = resumed_store.active_population("trial-001")
    resumed_store.close()

    resumed_store = create_store(resumed_path)
    resumed_runner = runner(resumed_store, total_rounds=6)
    resumed_state = resumed_runner.initialize()
    assert resumed_state.next_round_index == 2
    assert resumed_state.replacement_queue_position == 2
    assert resumed_state.population == active_before_restart
    while not resumed_state.completed:
        step = resumed_runner.run_next_round()
        resumed_states.append(step.state)
        resumed_state = step.state

    assert tuple(
        resumed_store.load_round("trial-001", index) for index in range(2)
    ) == committed_before_restart
    assert tuple(
        uninterrupted_store.load_round("trial-001", index) for index in range(6)
    ) == tuple(
        resumed_store.load_round("trial-001", index) for index in range(6)
    )
    assert uninterrupted_store.load_replacement_events(
        "trial-001"
    ) == resumed_store.load_replacement_events("trial-001")
    assert uninterrupted_store.load_agents("trial-001") == resumed_store.load_agents(
        "trial-001"
    )
    assert [state.population for state in uninterrupted_states] == [
        state.population for state in resumed_states
    ]
    assert resumed_store.get_trial("trial-001").status == "completed"
    uninterrupted_store.close()
    resumed_store.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"trial_seed": 43},
        {"condition": "objective"},
        {"config_hash": "0" * 64},
        {"total_rounds": 4},
    ],
)
def test_resume_rejects_protocol_changes(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    store = create_store(tmp_path / "wrong-protocol.sqlite")
    original = runner(store, total_rounds=3)
    original.run_next_round()

    arguments: dict[str, object] = {"total_rounds": 3}
    arguments.update(changes)
    with pytest.raises(TrialError, match="does not match"):
        runner(store, **arguments).initialize()  # type: ignore[arg-type]
    store.close()
