from __future__ import annotations

import sqlite3
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from src.agents import PromptProfile
from src.domain import Response
from src.models import MockModelProvider
from src.population import Population
from src.storage import (
    AlreadyCommittedError,
    DATABASE_SCHEMA_VERSION,
    IntegrityError,
    NotFoundError,
    Provenance,
    SQLiteEventStore,
    SchemaVersionError,
)
from src.tasks import Task
from src.tournament import RoundContext, RoundEngine, RoundResult


FIXED_TIMESTAMP = "2026-08-21T06:30:00+00:00"
CONFIG_JSON = '{"experiment":"e00","schema_version":1}'
CONFIG_HASH = sha256(CONFIG_JSON.encode("utf-8")).hexdigest()


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(
        code_commit="0123456789abcdef",
        python_version="3.12.10",
        platform="test-platform",
        provider_name="mock",
        model_name="deterministic-v1",
        created_at=FIXED_TIMESTAMP,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteEventStore:
    event_store = SQLiteEventStore(tmp_path / "experiment.sqlite")
    event_store.initialize()
    yield event_store
    event_store.close()


def create_experiment(
    store: SQLiteEventStore, provenance: Provenance, experiment_id: str = "experiment-e00"
) -> None:
    store.create_experiment(
        experiment_id=experiment_id,
        name="E00 apparatus",
        config_schema_version=1,
        config_hash=CONFIG_HASH,
        config_json=CONFIG_JSON,
        provenance=provenance,
    )


def prepare_trial(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
) -> None:
    create_experiment(store, provenance)
    store.create_trial(
        trial_id="trial-001",
        experiment_id="experiment-e00",
        trial_seed=42,
        created_at=FIXED_TIMESTAMP,
    )
    store.register_agents("trial-001", population)


def make_round(
    *,
    round_index: int,
    population: Population,
    profiles: dict[str, PromptProfile],
    task: Task,
) -> tuple[RoundContext, RoundResult]:
    context = RoundContext(
        experiment_id="experiment-e00",
        trial_id="trial-001",
        round_index=round_index,
        condition="peer_vote",
        seed=42,
        task=task,
        population=population,
        profiles=profiles,
    )
    return context, RoundEngine().execute(context, MockModelProvider())


def table_count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_new_database_initializes_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "schema.sqlite"
    store = SQLiteEventStore(path)
    store.initialize()

    assert store.schema_version == DATABASE_SCHEMA_VERSION
    assert store.foreign_keys_enabled is True
    store.close()

    reopened = SQLiteEventStore(path)
    reopened.initialize()
    assert reopened.schema_version == DATABASE_SCHEMA_VERSION
    assert reopened.foreign_keys_enabled is True
    reopened.close()


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_metadata (singleton INTEGER PRIMARY KEY, schema_version INTEGER)"
        )
        connection.execute(
            "INSERT INTO schema_metadata (singleton, schema_version) VALUES (1, 999)"
        )

    with pytest.raises(SchemaVersionError, match="unsupported database schema version 999"):
        SQLiteEventStore(path)


def test_experiment_and_provenance_round_trip(
    store: SQLiteEventStore, provenance: Provenance
) -> None:
    create_experiment(store, provenance)

    experiment = store.get_experiment("experiment-e00")

    assert experiment.config_hash == CONFIG_HASH
    assert experiment.config_json == CONFIG_JSON
    assert experiment.database_schema_version == DATABASE_SCHEMA_VERSION
    assert experiment.provenance == provenance


def test_duplicate_experiment_is_rejected(
    store: SQLiteEventStore, provenance: Provenance
) -> None:
    create_experiment(store, provenance)

    with pytest.raises(IntegrityError, match="could not create experiment"):
        create_experiment(store, provenance)


def test_trial_creation_and_duplicate_protection(
    store: SQLiteEventStore, provenance: Provenance
) -> None:
    create_experiment(store, provenance)
    store.create_trial(
        trial_id="trial-001",
        experiment_id="experiment-e00",
        trial_seed=2**63 + 7,
        created_at=FIXED_TIMESTAMP,
    )

    assert store.get_trial("trial-001").trial_seed == 2**63 + 7
    with pytest.raises(IntegrityError, match="could not create trial"):
        store.create_trial(
            trial_id="trial-001",
            experiment_id="experiment-e00",
            trial_seed=1,
            created_at=FIXED_TIMESTAMP,
        )


def test_trial_requires_existing_experiment(store: SQLiteEventStore) -> None:
    with pytest.raises(IntegrityError, match="could not create trial"):
        store.create_trial(
            trial_id="trial-001",
            experiment_id="missing",
            trial_seed=42,
            created_at=FIXED_TIMESTAMP,
        )


def test_agent_registration_round_trip_and_duplicate_protection(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
) -> None:
    prepare_trial(store, provenance, population)

    assert store.load_agents("trial-001") == population.agents
    with pytest.raises(IntegrityError, match="could not register agents"):
        store.register_agents("trial-001", population)
    assert store.load_agents("trial-001") == population.agents


def test_agent_registration_requires_existing_trial(
    store: SQLiteEventStore, population: Population
) -> None:
    with pytest.raises(NotFoundError, match="trial not found"):
        store.register_agents("missing", population)


def test_complete_round_persists_and_reconstructs(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    prepare_trial(store, provenance, population)
    context, result = make_round(
        round_index=0,
        population=population,
        profiles=profiles,
        task=round_task,
    )

    store.commit_round(context, result)

    assert store.load_round("trial-001", 0) == result
    assert store.last_committed_round("trial-001") == 0
    assert table_count(store.path, "responses") == 8
    assert table_count(store.path, "scores") == 8
    assert table_count(store.path, "ballots") == 8
    assert table_count(store.path, "selection_events") == 1
    assert table_count(store.path, "tasks") == 1


def test_nullable_and_optional_response_metadata_round_trips(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    prepare_trial(store, provenance, population)
    context, result = make_round(
        round_index=0,
        population=population,
        profiles=profiles,
        task=round_task,
    )
    first = replace(result.responses[0], latency_ms=1.25, token_count=7)
    modified = replace(result, responses=(first, *result.responses[1:]))

    store.commit_round(context, modified)
    loaded = store.load_round("trial-001", 0)

    assert loaded == modified
    assert loaded.responses[0].latency_ms == 1.25
    assert loaded.responses[1].latency_ms is None
    assert loaded.responses[1].token_count is None


def test_failed_round_transaction_rolls_back_all_records(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    prepare_trial(store, provenance, population)
    context, result = make_round(
        round_index=0,
        population=population,
        profiles=profiles,
        task=round_task,
    )
    duplicate = replace(
        result.responses[1], response_id=result.responses[0].response_id
    )
    invalid_result = replace(
        result,
        responses=(result.responses[0], duplicate, *result.responses[2:]),
    )

    with pytest.raises(IntegrityError, match="could not commit round"):
        store.commit_round(context, invalid_result)

    assert store.last_committed_round("trial-001") is None
    with pytest.raises(NotFoundError, match="committed round not found"):
        store.load_round("trial-001", 0)
    for table in ("tasks", "rounds", "responses", "scores", "ballots", "selection_events"):
        assert table_count(store.path, table) == 0


def test_duplicate_round_commit_is_strictly_rejected(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    prepare_trial(store, provenance, population)
    context, result = make_round(
        round_index=0,
        population=population,
        profiles=profiles,
        task=round_task,
    )
    store.commit_round(context, result)

    with pytest.raises(AlreadyCommittedError, match="round already committed"):
        store.commit_round(context, result)

    assert table_count(store.path, "responses") == 8
    assert table_count(store.path, "selection_events") == 1


def test_resume_boundary_for_contiguous_rounds(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    prepare_trial(store, provenance, population)

    assert store.last_committed_round("trial-001") is None
    assert store.next_round_index("trial-001") == 0
    for round_index in range(3):
        context, result = make_round(
            round_index=round_index,
            population=population,
            profiles=profiles,
            task=round_task,
        )
        store.commit_round(context, result)

    assert store.last_committed_round("trial-001") == 2
    assert store.next_round_index("trial-001") == 3


def test_resume_boundary_detects_round_gaps(
    store: SQLiteEventStore,
    provenance: Provenance,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    prepare_trial(store, provenance, population)
    for round_index in (0, 2):
        context, result = make_round(
            round_index=round_index,
            population=population,
            profiles=profiles,
            task=round_task,
        )
        store.commit_round(context, result)

    with pytest.raises(IntegrityError, match="non-contiguous committed rounds"):
        store.last_committed_round("trial-001")
