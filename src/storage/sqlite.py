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
from ..domain import (
    Ballot,
    ReplacementEvent,
    Response,
    Score,
    SelectionEvent,
    SelectionMechanism,
)
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

    def trial_exists(self, trial_id: str) -> bool:
        self._require_initialized()
        return self._connection.execute(
            "SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone() is not None

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
        condition: SelectionMechanism | None = None,
        total_rounds: int | None = None,
        config_hash: str | None = None,
        profile_pool_hash: str | None = None,
        replacement_version: str | None = None,
    ) -> None:
        self._require_initialized()
        _require_non_empty(trial_id, "trial_id")
        _require_non_empty(experiment_id, "experiment_id")
        if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
            raise IntegrityError("trial_seed must be an integer")
        protocol = (
            condition,
            total_rounds,
            config_hash,
            profile_pool_hash,
            replacement_version,
        )
        if any(value is not None for value in protocol) and any(
            value is None for value in protocol
        ):
            raise IntegrityError("trial protocol metadata must be provided together")
        if condition is not None:
            if condition not in ("peer_vote", "objective", "random"):
                raise IntegrityError("invalid trial condition")
            if (
                not isinstance(total_rounds, int)
                or isinstance(total_rounds, bool)
                or total_rounds <= 0
            ):
                raise IntegrityError("total_rounds must be a positive integer")
            if not isinstance(config_hash, str) or len(config_hash) != 64:
                raise IntegrityError("config_hash must be a 64-character string")
            if not isinstance(profile_pool_hash, str) or len(profile_pool_hash) != 64:
                raise IntegrityError("profile_pool_hash must be a 64-character string")
            _require_non_empty(cast(str, replacement_version), "replacement_version")
        timestamp = created_at or utc_now()
        _require_utc_timestamp(timestamp, "created_at")
        try:
            with self._transaction():
                self._connection.execute(
                    """
                    INSERT INTO trials (
                        trial_id, experiment_id, trial_seed, status, created_at,
                        condition, total_rounds, config_hash, profile_pool_hash,
                        replacement_version
                    ) VALUES (?, ?, ?, 'created', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trial_id,
                        experiment_id,
                        str(trial_seed),
                        timestamp,
                        condition,
                        total_rounds,
                        config_hash,
                        profile_pool_hash,
                        replacement_version,
                    ),
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
            condition=row["condition"],
            total_rounds=row["total_rounds"],
            config_hash=row["config_hash"],
            profile_pool_hash=row["profile_pool_hash"],
            replacement_version=row["replacement_version"],
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
                        trial_id, agent_id, profile_id, display_label, generation,
                        ordinal, introduced_round
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
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

    def load_initial_agents(self, trial_id: str) -> tuple[AgentIdentity, ...]:
        self._require_initialized()
        if self._connection.execute(
            "SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone() is None:
            raise NotFoundError(f"trial not found: {trial_id}")
        rows = self._connection.execute(
            """
            SELECT agent_id, profile_id, display_label, generation
            FROM agent_instances
            WHERE trial_id = ? AND introduced_round IS NULL
            ORDER BY ordinal
            """,
            (trial_id,),
        ).fetchall()
        return tuple(self._agent_from_row(row) for row in rows)

    def load_replacement_events(self, trial_id: str) -> tuple[ReplacementEvent, ...]:
        self._require_initialized()
        if self._connection.execute(
            "SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone() is None:
            raise NotFoundError(f"trial not found: {trial_id}")
        rows = self._connection.execute(
            """
            SELECT * FROM replacement_events
            WHERE trial_id = ? ORDER BY round_index
            """,
            (trial_id,),
        ).fetchall()
        return tuple(
            ReplacementEvent(
                replacement_id=row["replacement_id"],
                trial_id=row["trial_id"],
                round_index=row["round_index"],
                removed_agent_id=row["removed_agent_id"],
                added_agent_id=row["added_agent_id"],
                profile_id=row["profile_id"],
                queue_index=row["queue_index"],
                reason=row["reason"],
            )
            for row in rows
        )

    def active_population(self, trial_id: str) -> Population:
        self._require_initialized()
        return Population(self._active_agents(trial_id))

    def commit_round(
        self,
        context: RoundContext,
        result: RoundResult,
        *,
        replacement_event: ReplacementEvent | None = None,
        replacement_agent: AgentIdentity | None = None,
    ) -> None:
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

                trial = self._validate_round_payload(context, result)
                self._validate_replacement_payload(
                    context,
                    result,
                    trial,
                    replacement_event,
                    replacement_agent,
                )
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
                if replacement_event is not None and replacement_agent is not None:
                    self._insert_replacement(replacement_event, replacement_agent)
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
        except sqlite3.IntegrityError as error:
            raise IntegrityError(
                f"could not commit round {context.trial_id}/{context.round_index}: {error}"
            ) from error

    def complete_trial(self, trial_id: str) -> None:
        self._require_initialized()
        trial = self.get_trial(trial_id)
        if trial.total_rounds is None:
            raise IntegrityError(f"trial {trial_id} has no configured round count")
        last = self.last_committed_round(trial_id)
        if last != trial.total_rounds - 1:
            raise IntegrityError(
                f"trial {trial_id} cannot complete at round {last}; expected {trial.total_rounds - 1}"
            )
        replacement_count = len(self.load_replacement_events(trial_id))
        if replacement_count != trial.total_rounds - 1:
            raise IntegrityError(
                f"trial {trial_id} has {replacement_count} replacement events; "
                f"expected {trial.total_rounds - 1}"
            )
        if trial.status == "completed":
            return
        with self._transaction():
            self._connection.execute(
                """
                UPDATE trials SET status = 'completed', completed_at = ?
                WHERE trial_id = ?
                """,
                (utc_now(), trial_id),
            )

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
            SELECT t.*, e.provider_name, e.model_name
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
        if trial["status"] == "completed":
            raise IntegrityError(f"trial is already completed: {context.trial_id}")
        if trial["condition"] is not None and trial["condition"] != context.condition:
            raise IntegrityError("round condition does not match the persisted trial protocol")
        if int(trial["trial_seed"]) != context.seed:
            raise IntegrityError("round seed does not match the persisted trial seed")
        if (
            trial["total_rounds"] is not None
            and context.round_index >= trial["total_rounds"]
        ):
            raise IntegrityError("round index exceeds the persisted trial protocol")
        if result.trial_id != context.trial_id or result.round_index != context.round_index:
            raise IntegrityError("round result identity does not match the round context")
        if result.task != context.task:
            raise IntegrityError("round result task does not match the round context")

        registered = tuple(
            (agent.agent_id, agent.profile_id, agent.display_label, agent.generation)
            for agent in self._active_agents(context.trial_id)
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

    def _validate_replacement_payload(
        self,
        context: RoundContext,
        result: RoundResult,
        trial: sqlite3.Row,
        event: ReplacementEvent | None,
        agent: AgentIdentity | None,
    ) -> None:
        if (event is None) != (agent is None):
            raise IntegrityError(
                "replacement_event and replacement_agent must be provided together"
            )
        total_rounds = trial["total_rounds"]
        if total_rounds is not None:
            replacement_required = context.round_index < total_rounds - 1
            if replacement_required and event is None:
                raise IntegrityError("non-final round requires a replacement event")
            if not replacement_required and event is not None:
                raise IntegrityError("final round must not contain a replacement event")
        if event is None or agent is None:
            return
        if (
            event.trial_id != context.trial_id
            or event.round_index != context.round_index
            or event.removed_agent_id != result.selection.selected_agent_id
            or event.added_agent_id != agent.agent_id
            or event.profile_id != agent.profile_id
        ):
            raise IntegrityError("replacement event does not match the round transition")
        if event.queue_index != context.round_index:
            raise IntegrityError("replacement queue index must match the round index")
        if agent.generation != 0:
            raise IntegrityError("fixed-profile replacement generation must be 0")
        if agent.agent_id in {item.agent_id for item in self.load_agents(context.trial_id)}:
            raise IntegrityError(f"replacement agent ID already exists: {agent.agent_id}")

    def _insert_replacement(
        self, event: ReplacementEvent, agent: AgentIdentity
    ) -> None:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal FROM agent_instances WHERE trial_id = ?",
            (event.trial_id,),
        ).fetchone()
        self._connection.execute(
            """
            INSERT INTO agent_instances (
                trial_id, agent_id, profile_id, display_label, generation,
                ordinal, introduced_round
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.trial_id,
                agent.agent_id,
                agent.profile_id,
                agent.display_label,
                agent.generation,
                row["next_ordinal"],
                event.round_index,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO replacement_events (
                replacement_id, trial_id, round_index, removed_agent_id,
                added_agent_id, profile_id, queue_index, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.replacement_id,
                event.trial_id,
                event.round_index,
                event.removed_agent_id,
                event.added_agent_id,
                event.profile_id,
                event.queue_index,
                event.reason,
            ),
        )

    def _active_agents(self, trial_id: str) -> tuple[AgentIdentity, ...]:
        initial = list(self.load_initial_agents(trial_id))
        if not initial:
            return ()
        agents_by_id = {
            agent.agent_id: agent for agent in self.load_agents(trial_id)
        }
        for event in self.load_replacement_events(trial_id):
            try:
                index = next(
                    index
                    for index, active_agent in enumerate(initial)
                    if active_agent.agent_id == event.removed_agent_id
                )
            except StopIteration as error:
                raise IntegrityError(
                    f"replacement event removes inactive agent: {event.removed_agent_id}"
                ) from error
            try:
                added = agents_by_id[event.added_agent_id]
            except KeyError as error:
                raise IntegrityError(
                    f"replacement agent record missing: {event.added_agent_id}"
                ) from error
            if added.profile_id != event.profile_id:
                raise IntegrityError(
                    f"replacement profile mismatch: {event.replacement_id}"
                )
            initial[index] = added
        if len(initial) != 8 or len({agent.agent_id for agent in initial}) != 8:
            raise IntegrityError(f"trial {trial_id} does not reconstruct to 8 active agents")
        return tuple(initial)

    @staticmethod
    def _agent_from_row(row: sqlite3.Row) -> AgentIdentity:
        return AgentIdentity(
            agent_id=row["agent_id"],
            profile_id=row["profile_id"],
            display_label=row["display_label"],
            generation=row["generation"],
        )

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
