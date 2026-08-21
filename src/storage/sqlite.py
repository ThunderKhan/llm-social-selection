from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from ..agents import AgentIdentity
from ..domain import Ballot, Response, Score, SelectionEvent, SelectionMechanism
from ..population import Population
from ..tasks import Task
from ..tournament import RoundContext, RoundResult
from .errors import (
    AlreadyCommittedError,
    IntegrityError,
    NotFoundError,
    SchemaVersionError,
    StorageError,
)
from .models import ExperimentMetadata, Provenance, TrialMetadata
from .provenance import utc_now
from .schema import DATABASE_SCHEMA_VERSION, SCHEMA_SQL


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{field} must be a non-empty string")


def _require_utc_timestamp(value: str, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"{field} must be an ISO 8601 timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise IntegrityError(f"{field} must be timezone-aware UTC")


class SQLiteEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self._connection = sqlite3.connect(self.path, isolation_level=None)
        except sqlite3.Error as error:
            raise StorageError(f"could not open SQLite database {self.path}: {error}") from error
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            if self._has_schema_metadata():
                self._verify_schema_version()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> SQLiteEventStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def initialize(self) -> None:
        if self._has_schema_metadata():
            self._verify_schema_version()
            return
        try:
            self._connection.executescript(
                f"BEGIN IMMEDIATE;\n{SCHEMA_SQL}\nCOMMIT;"
            )
        except sqlite3.Error as error:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise StorageError(f"could not initialize SQLite schema: {error}") from error

    @property
    def schema_version(self) -> int:
        return self._verify_schema_version()

    @property
    def foreign_keys_enabled(self) -> bool:
        row = self._connection.execute("PRAGMA foreign_keys").fetchone()
        return bool(row[0])

    def create_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        config_schema_version: int,
        config_hash: str,
        config_json: str,
        provenance: Provenance,
    ) -> None:
        self._require_initialized()
        _require_non_empty(experiment_id, "experiment_id")
        _require_non_empty(name, "name")
        if (
            not isinstance(config_schema_version, int)
            or isinstance(config_schema_version, bool)
            or config_schema_version <= 0
        ):
            raise IntegrityError("config_schema_version must be a positive integer")
        self._validate_canonical_config(config_hash, config_json)
        _require_utc_timestamp(provenance.created_at, "provenance.created_at")
        for field, value in (
            ("python_version", provenance.python_version),
            ("platform", provenance.platform),
            ("provider_name", provenance.provider_name),
            ("model_name", provenance.model_name),
        ):
            _require_non_empty(value, f"provenance.{field}")

        try:
            with self._transaction():
                self._connection.execute(
                    """
                    INSERT INTO experiments (
                        experiment_id, name, config_schema_version, config_hash,
                        config_json, database_schema_version, created_at,
                        code_commit, python_version, platform, provider_name, model_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        name,
                        config_schema_version,
                        config_hash,
                        config_json,
                        DATABASE_SCHEMA_VERSION,
                        provenance.created_at,
                        provenance.code_commit,
                        provenance.python_version,
                        provenance.platform,
                        provenance.provider_name,
                        provenance.model_name,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise IntegrityError(f"could not create experiment {experiment_id}: {error}") from error

    def get_experiment(self, experiment_id: str) -> ExperimentMetadata:
        self._require_initialized()
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"experiment not found: {experiment_id}")
        return ExperimentMetadata(
            experiment_id=row["experiment_id"],
            name=row["name"],
            config_schema_version=row["config_schema_version"],
            config_hash=row["config_hash"],
            config_json=row["config_json"],
            database_schema_version=row["database_schema_version"],
            provenance=Provenance(
                code_commit=row["code_commit"],
                python_version=row["python_version"],
                platform=row["platform"],
                provider_name=row["provider_name"],
                model_name=row["model_name"],
                created_at=row["created_at"],
            ),
        )

    def create_trial(
        self,
        *,
        trial_id: str,
        experiment_id: str,
        trial_seed: int,
        created_at: str | None = None,
    ) -> None:
        self._require_initialized()
        _require_non_empty(trial_id, "trial_id")
        _require_non_empty(experiment_id, "experiment_id")
        if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
            raise IntegrityError("trial_seed must be an integer")
        timestamp = created_at or utc_now()
        _require_utc_timestamp(timestamp, "created_at")
        try:
            with self._transaction():
                self._connection.execute(
                    """
                    INSERT INTO trials (
                        trial_id, experiment_id, trial_seed, status, created_at
                    ) VALUES (?, ?, ?, 'created', ?)
                    """,
                    (trial_id, experiment_id, str(trial_seed), timestamp),
                )
        except sqlite3.IntegrityError as error:
            raise IntegrityError(f"could not create trial {trial_id}: {error}") from error

    def get_trial(self, trial_id: str) -> TrialMetadata:
        self._require_initialized()
        row = self._connection.execute(
            "SELECT * FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"trial not found: {trial_id}")
        return TrialMetadata(
            trial_id=row["trial_id"],
            experiment_id=row["experiment_id"],
            trial_seed=int(row["trial_seed"]),
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def register_agents(self, trial_id: str, population: Population) -> None:
        self._require_initialized()
        if self._connection.execute(
            "SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone() is None:
            raise NotFoundError(f"trial not found: {trial_id}")
        try:
            with self._transaction():
                self._connection.executemany(
                    """
                    INSERT INTO agent_instances (
                        trial_id, agent_id, profile_id, display_label, generation, ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            trial_id,
                            agent.agent_id,
                            agent.profile_id,
                            agent.display_label,
                            agent.generation,
                            ordinal,
                        )
                        for ordinal, agent in enumerate(population.agents)
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise IntegrityError(
                f"could not register agents for trial {trial_id}: {error}"
            ) from error

    def load_agents(self, trial_id: str) -> tuple[AgentIdentity, ...]:
        self._require_initialized()
        if self._connection.execute(
            "SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone() is None:
            raise NotFoundError(f"trial not found: {trial_id}")
        rows = self._connection.execute(
            """
            SELECT agent_id, profile_id, display_label, generation
            FROM agent_instances WHERE trial_id = ? ORDER BY ordinal
            """,
            (trial_id,),
        ).fetchall()
        return tuple(
            AgentIdentity(
                agent_id=row["agent_id"],
                profile_id=row["profile_id"],
                display_label=row["display_label"],
                generation=row["generation"],
            )
            for row in rows
        )

    def commit_round(self, context: RoundContext, result: RoundResult) -> None:
        self._require_initialized()
        try:
            with self._transaction():
                existing = self._connection.execute(
                    "SELECT status FROM rounds WHERE trial_id = ? AND round_index = ?",
                    (context.trial_id, context.round_index),
                ).fetchone()
                if existing is not None:
                    if existing["status"] == "complete":
                        raise AlreadyCommittedError(
                            f"round already committed: {context.trial_id}/{context.round_index}"
                        )
                    raise IntegrityError(
                        f"incomplete round record exists: {context.trial_id}/{context.round_index}"
                    )

                experiment = self._validate_round_payload(context, result)
                self._ensure_task(context.task)
                self._connection.execute(
                    """
                    INSERT INTO rounds (
                        trial_id, round_index, task_id, condition, round_seed, status
                    ) VALUES (?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        context.trial_id,
                        context.round_index,
                        context.task.task_id,
                        context.condition,
                        str(context.seed),
                    ),
                )
                self._insert_responses(result.responses)
                self._insert_scores(result.scores)
                self._insert_ballots(result.ballots)
                self._insert_selection(result.selection)
                committed_at = utc_now()
                self._connection.execute(
                    """
                    UPDATE rounds
                    SET status = 'complete', committed_at = ?
                    WHERE trial_id = ? AND round_index = ?
                    """,
                    (committed_at, context.trial_id, context.round_index),
                )
                self._connection.execute(
                    "UPDATE trials SET status = 'running' WHERE trial_id = ? AND status = 'created'",
                    (context.trial_id,),
                )
                if experiment["provider_name"] != result.responses[0].provider_name:
                    raise IntegrityError("experiment provider metadata changed during commit")
        except sqlite3.IntegrityError as error:
            raise IntegrityError(
                f"could not commit round {context.trial_id}/{context.round_index}: {error}"
            ) from error

    def load_round(self, trial_id: str, round_index: int) -> RoundResult:
        self._require_initialized()
        round_row = self._connection.execute(
            """
            SELECT * FROM rounds
            WHERE trial_id = ? AND round_index = ? AND status = 'complete'
            """,
            (trial_id, round_index),
        ).fetchone()
        if round_row is None:
            raise NotFoundError(f"committed round not found: {trial_id}/{round_index}")

        task_row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (round_row["task_id"],)
        ).fetchone()
        if task_row is None:
            raise IntegrityError(f"task missing for committed round: {trial_id}/{round_index}")
        task = Task(
            task_id=task_row["task_id"],
            family=task_row["family"],
            prompt=task_row["prompt"],
            expected_answer=task_row["expected_answer"],
            scorer_version=task_row["scorer_version"],
        )

        responses = tuple(
            Response(
                response_id=row["response_id"],
                trial_id=row["trial_id"],
                round_index=row["round_index"],
                task_id=row["task_id"],
                agent_id=row["agent_id"],
                content=row["content"],
                provider_name=row["provider_name"],
                model_name=row["model_name"],
                request_id=row["request_id"],
                seed=int(row["seed"]) if row["seed"] is not None else None,
                latency_ms=row["latency_ms"],
                token_count=row["token_count"],
            )
            for row in self._rows_for_round("responses", trial_id, round_index)
        )
        scores = tuple(
            Score(
                score_id=row["score_id"],
                trial_id=row["trial_id"],
                round_index=row["round_index"],
                task_id=row["task_id"],
                agent_id=row["agent_id"],
                value=row["value"],
                scorer_version=row["scorer_version"],
            )
            for row in self._rows_for_round("scores", trial_id, round_index)
        )
        ballots = tuple(
            Ballot(
                ballot_id=row["ballot_id"],
                trial_id=row["trial_id"],
                round_index=row["round_index"],
                voter_agent_id=row["voter_agent_id"],
                supported_agent_id=row["supported_agent_id"],
            )
            for row in self._rows_for_round("ballots", trial_id, round_index)
        )
        for name, records in (
            ("responses", responses),
            ("scores", scores),
            ("ballots", ballots),
        ):
            if len(records) != 8:
                raise IntegrityError(
                    f"committed round {trial_id}/{round_index} has {len(records)} {name}; expected 8"
                )
        selection_row = self._connection.execute(
            """
            SELECT * FROM selection_events
            WHERE trial_id = ? AND round_index = ?
            """,
            (trial_id, round_index),
        ).fetchone()
        if selection_row is None:
            raise IntegrityError(
                f"selection event missing for committed round: {trial_id}/{round_index}"
            )
        selection = SelectionEvent(
            selection_id=selection_row["selection_id"],
            trial_id=selection_row["trial_id"],
            round_index=selection_row["round_index"],
            mechanism=cast(SelectionMechanism, selection_row["mechanism"]),
            selected_agent_id=selection_row["selected_agent_id"],
            reason=selection_row["reason"],
        )
        return RoundResult(
            trial_id=trial_id,
            round_index=round_index,
            task=task,
            responses=responses,
            scores=scores,
            ballots=ballots,
            selection=selection,
        )

    def last_committed_round(self, trial_id: str) -> int | None:
        self._require_initialized()
        if self._connection.execute(
            "SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone() is None:
            raise NotFoundError(f"trial not found: {trial_id}")
        rows = self._connection.execute(
            "SELECT round_index, status FROM rounds WHERE trial_id = ? ORDER BY round_index",
            (trial_id,),
        ).fetchall()
        if not rows:
            return None
        if any(row["status"] != "complete" for row in rows):
            raise IntegrityError(f"trial {trial_id} contains an incomplete round")
        indexes = [row["round_index"] for row in rows]
        expected = list(range(indexes[-1] + 1))
        if indexes != expected:
            raise IntegrityError(
                f"trial {trial_id} has non-contiguous committed rounds: {indexes}"
            )
        return indexes[-1]

    def next_round_index(self, trial_id: str) -> int:
        last = self.last_committed_round(trial_id)
        return 0 if last is None else last + 1

    def _validate_round_payload(
        self, context: RoundContext, result: RoundResult
    ) -> sqlite3.Row:
        trial = self._connection.execute(
            """
            SELECT t.experiment_id, e.provider_name, e.model_name
            FROM trials AS t
            JOIN experiments AS e ON e.experiment_id = t.experiment_id
            WHERE t.trial_id = ?
            """,
            (context.trial_id,),
        ).fetchone()
        if trial is None:
            raise NotFoundError(f"trial not found: {context.trial_id}")
        if trial["experiment_id"] != context.experiment_id:
            raise IntegrityError("round context experiment does not match the persisted trial")
        if result.trial_id != context.trial_id or result.round_index != context.round_index:
            raise IntegrityError("round result identity does not match the round context")
        if result.task != context.task:
            raise IntegrityError("round result task does not match the round context")

        registered_rows = self._connection.execute(
            """
            SELECT agent_id, profile_id, display_label, generation
            FROM agent_instances WHERE trial_id = ? ORDER BY ordinal
            """,
            (context.trial_id,),
        ).fetchall()
        registered = tuple(
            (row["agent_id"], row["profile_id"], row["display_label"], row["generation"])
            for row in registered_rows
        )
        expected_agents = tuple(
            (agent.agent_id, agent.profile_id, agent.display_label, agent.generation)
            for agent in context.population.agents
        )
        if registered != expected_agents:
            raise IntegrityError("round population does not match registered trial agents")

        eligible = {agent.agent_id for agent in context.population.agents}
        self._validate_record_agents("responses", result.responses, eligible)
        self._validate_record_agents("scores", result.scores, eligible)
        if len(result.ballots) != len(eligible) or {
            ballot.voter_agent_id for ballot in result.ballots
        } != eligible:
            raise IntegrityError("ballots must contain exactly one record per eligible voter")

        for response in result.responses:
            if (
                response.trial_id != context.trial_id
                or response.round_index != context.round_index
                or response.task_id != context.task.task_id
            ):
                raise IntegrityError("response references do not match the round context")
            if (
                response.provider_name != trial["provider_name"]
                or response.model_name != trial["model_name"]
            ):
                raise IntegrityError("response provider metadata does not match the experiment")
        for score in result.scores:
            if (
                score.trial_id != context.trial_id
                or score.round_index != context.round_index
                or score.task_id != context.task.task_id
            ):
                raise IntegrityError("score references do not match the round context")
        for ballot in result.ballots:
            if (
                ballot.trial_id != context.trial_id
                or ballot.round_index != context.round_index
                or ballot.supported_agent_id not in eligible
            ):
                raise IntegrityError("support ballot references do not match the round context")

        selection = result.selection
        if (
            selection.trial_id != context.trial_id
            or selection.round_index != context.round_index
            or selection.mechanism != context.condition
            or selection.selected_agent_id not in eligible
        ):
            raise IntegrityError("selection event does not match the round context")
        return trial

    @staticmethod
    def _validate_record_agents(
        name: str, records: tuple[Response, ...] | tuple[Score, ...], eligible: set[str]
    ) -> None:
        if len(records) != len(eligible) or {record.agent_id for record in records} != eligible:
            raise IntegrityError(
                f"{name} must contain exactly one record per eligible agent"
            )

    def _ensure_task(self, task: Task) -> None:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task.task_id,)
        ).fetchone()
        values = (
            task.task_id,
            task.family,
            task.prompt,
            task.expected_answer,
            task.scorer_version,
        )
        if row is None:
            self._connection.execute(
                """
                INSERT INTO tasks (
                    task_id, family, prompt, expected_answer, scorer_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                values,
            )
            return
        stored = (
            row["task_id"],
            row["family"],
            row["prompt"],
            row["expected_answer"],
            row["scorer_version"],
        )
        if stored != values:
            raise IntegrityError(f"task metadata conflict for {task.task_id}")

    def _insert_responses(self, responses: tuple[Response, ...]) -> None:
        self._connection.executemany(
            """
            INSERT INTO responses (
                response_id, trial_id, round_index, task_id, agent_id, ordinal,
                content, provider_name, model_name, request_id, seed,
                latency_ms, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    response.response_id,
                    response.trial_id,
                    response.round_index,
                    response.task_id,
                    response.agent_id,
                    ordinal,
                    response.content,
                    response.provider_name,
                    response.model_name,
                    response.request_id,
                    str(response.seed) if response.seed is not None else None,
                    response.latency_ms,
                    response.token_count,
                )
                for ordinal, response in enumerate(responses)
            ),
        )

    def _insert_scores(self, scores: tuple[Score, ...]) -> None:
        self._connection.executemany(
            """
            INSERT INTO scores (
                score_id, trial_id, round_index, task_id, agent_id,
                ordinal, value, scorer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    score.score_id,
                    score.trial_id,
                    score.round_index,
                    score.task_id,
                    score.agent_id,
                    ordinal,
                    score.value,
                    score.scorer_version,
                )
                for ordinal, score in enumerate(scores)
            ),
        )

    def _insert_ballots(self, ballots: tuple[Ballot, ...]) -> None:
        self._connection.executemany(
            """
            INSERT INTO ballots (
                ballot_id, trial_id, round_index, voter_agent_id,
                supported_agent_id, ordinal
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    ballot.ballot_id,
                    ballot.trial_id,
                    ballot.round_index,
                    ballot.voter_agent_id,
                    ballot.supported_agent_id,
                    ordinal,
                )
                for ordinal, ballot in enumerate(ballots)
            ),
        )

    def _insert_selection(self, selection: SelectionEvent) -> None:
        self._connection.execute(
            """
            INSERT INTO selection_events (
                selection_id, trial_id, round_index, mechanism,
                selected_agent_id, reason
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                selection.selection_id,
                selection.trial_id,
                selection.round_index,
                selection.mechanism,
                selection.selected_agent_id,
                selection.reason,
            ),
        )

    def _rows_for_round(
        self, table: str, trial_id: str, round_index: int
    ) -> list[sqlite3.Row]:
        if table not in {"responses", "scores", "ballots"}:
            raise ValueError(f"unsupported round table: {table}")
        return self._connection.execute(
            f"SELECT * FROM {table} WHERE trial_id = ? AND round_index = ? ORDER BY ordinal",
            (trial_id, round_index),
        ).fetchall()

    @staticmethod
    def _validate_canonical_config(config_hash: str, config_json: str) -> None:
        try:
            value: Any = json.loads(config_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise IntegrityError("config_json must contain valid JSON") from error
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if canonical != config_json:
            raise IntegrityError("config_json must use canonical JSON serialization")
        actual_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        if config_hash != actual_hash:
            raise IntegrityError("config_hash does not match config_json")

    def _has_schema_metadata(self) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'schema_metadata'
            """
        ).fetchone()
        return row is not None

    def _verify_schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise SchemaVersionError("database schema version metadata is missing")
        version = row["schema_version"]
        if version != DATABASE_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported database schema version {version}; expected {DATABASE_SCHEMA_VERSION}"
            )
        return version

    def _require_initialized(self) -> None:
        if not self._has_schema_metadata():
            raise StorageError("database schema is not initialized")
        self._verify_schema_version()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
