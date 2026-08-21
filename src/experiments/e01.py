from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agents import PromptProfile
from ..ballots import LLMBallotProvider
from ..domain import SelectionMechanism
from ..models import ModelProvider, OllamaProvider
from ..replacement import FIXED_QUEUE_VERSION, profile_pool_hash
from ..seeding import derive_seed
from ..storage import NotFoundError, SQLiteEventStore, collect_provenance
from ..tasks.calibration import TaskSetArtifact, load_task_set
from ..tournament import FixedTaskSource, RoundEngine, TrialRunner


E01_PROTOCOL_VERSION = "e01-pilot-v1"
E01_CONFIG_SCHEMA_VERSION = 1
E01_CONDITIONS: tuple[SelectionMechanism, ...] = (
    "peer_vote",
    "objective",
    "random",
)
E01_DEFAULT_MODEL = "qwen3:0.6b"
E01_DEFAULT_MASTER_SEED = 20260821
E01_NUM_PREDICT = 32
E01_TEMPERATURE = 0.0
E01_IMPLEMENTATION_FILES = (
    "src/agents/prompting.py",
    "src/ballots/llm.py",
    "src/experiments/e01.py",
    "src/population/initial.py",
    "src/replacement/fixed_queue.py",
    "src/scoring/deterministic.py",
    "src/seeding.py",
    "src/selection/objective.py",
    "src/selection/peer.py",
    "src/selection/random.py",
    "src/tournament/round.py",
    "src/tournament/task_source.py",
    "src/tournament/trial.py",
)
Mode = Literal["smoke", "full"]


class E01Error(ValueError):
    """An E01 run request is unsafe or incompatible with persisted state."""


@dataclass(frozen=True)
class E01TrialPlan:
    trial_id: str
    identity_namespace: str
    replicate: int
    condition: SelectionMechanism
    seed: int


@dataclass(frozen=True)
class E01Plan:
    mode: Mode
    database_path: Path
    task_path: Path
    task_artifact: TaskSetArtifact
    task_hash: str
    profiles: dict[str, PromptProfile]
    trials: tuple[E01TrialPlan, ...]
    rounds_per_trial: int
    master_seed: int
    experiment_id: str
    config_json: str
    config_hash: str
    model: str
    base_url: str
    timeout_seconds: float

    @property
    def total_rounds(self) -> int:
        return len(self.trials) * self.rounds_per_trial

    @property
    def summary_path(self) -> Path:
        return self.database_path.with_suffix(".summary.json")


@dataclass(frozen=True)
class E01RunOutcome:
    committed_this_run: int
    committed_total: int
    total_rounds: int
    completed: bool
    controlled_stop: bool


def e01_profiles() -> dict[str, PromptProfile]:
    approaches = (
        "literal",
        "constraint-first",
        "pattern-first",
        "calculation-first",
        "independent-check",
        "skeptical",
        "deliberate",
        "concise",
    )
    return {
        f"e01-profile-{index:03d}": PromptProfile(
            profile_id=f"e01-profile-{index:03d}",
            parameters={"approach": approach},
            template_version="e01-profile-prompt-v1",
        )
        for index, approach in enumerate(approaches, start=1)
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _implementation_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    return {
        relative_path: hashlib.sha256(
            (repository / relative_path)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .encode("utf-8")
        ).hexdigest()
        for relative_path in E01_IMPLEMENTATION_FILES
    }


def build_e01_plan(
    *,
    mode: Mode,
    database_path: str | Path,
    task_path: str | Path,
    model: str = E01_DEFAULT_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 120.0,
    master_seed: int = E01_DEFAULT_MASTER_SEED,
) -> E01Plan:
    if mode not in ("smoke", "full"):
        raise E01Error("mode must be smoke or full")
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise E01Error("master_seed must be an integer")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise E01Error("timeout_seconds must be a positive finite number")
    task_source_path = Path(task_path).resolve()
    artifact = load_task_set(task_source_path)
    if artifact.status != "validated":
        raise E01Error("E01 requires a validated task-set artifact")
    if artifact.model_used_for_validation != model:
        raise E01Error(
            "task-set validation model does not match the requested E01 model"
        )
    task_hash = hashlib.sha256(task_source_path.read_bytes()).hexdigest()
    profiles = e01_profiles()
    replicate_count, rounds_per_trial = (2, 3) if mode == "smoke" else (10, 10)
    trials: list[E01TrialPlan] = []
    for replicate_index in range(replicate_count):
        replicate = replicate_index + 1
        seed = derive_seed(
            master_seed,
            replicate_index,
            "e01_replicate",
            E01_PROTOCOL_VERSION,
        )
        identity_namespace = f"e01-{mode}-replicate-{replicate:03d}"
        for condition in E01_CONDITIONS:
            trials.append(
                E01TrialPlan(
                    trial_id=f"e01-{mode}-{condition}-replicate-{replicate:03d}",
                    identity_namespace=identity_namespace,
                    replicate=replicate,
                    condition=condition,
                    seed=seed,
                )
            )
    config = {
        "apparatus": {
            "ballot_provider": "llm-json-choice-v1",
            "conditions": list(E01_CONDITIONS),
            "identity_namespace": "matched-replicate-v1",
            "population_size": 8,
            "protocol_version": E01_PROTOCOL_VERSION,
            "replacement_version": FIXED_QUEUE_VERSION,
        },
        "master_seed": master_seed,
        "mode": mode,
        "implementation_hashes": _implementation_hashes(),
        "model": {
            "base_url": base_url.rstrip("/"),
            "model": model,
            "num_predict": E01_NUM_PREDICT,
            "provider": "ollama",
            "temperature": E01_TEMPERATURE,
            "think": False,
            "timeout_seconds": float(timeout_seconds),
        },
        "profiles": {
            "hash": profile_pool_hash(profiles),
            "values": [
                {
                    "parameters": dict(profiles[profile_id].parameters),
                    "profile_id": profile_id,
                    "template_version": profiles[profile_id].template_version,
                }
                for profile_id in sorted(profiles)
            ],
        },
        "rounds_per_trial": rounds_per_trial,
        "task_set": {
            "hash": task_hash,
            "scorer_version": artifact.scorer_version,
            "task_set_version": artifact.task_set_version,
        },
        "trials_per_condition": replicate_count,
    }
    config_json = _canonical_json(config)
    return E01Plan(
        mode=mode,
        database_path=Path(database_path).resolve(),
        task_path=task_source_path,
        task_artifact=artifact,
        task_hash=task_hash,
        profiles=profiles,
        trials=tuple(trials),
        rounds_per_trial=rounds_per_trial,
        master_seed=master_seed,
        experiment_id=f"e01-{mode}-{E01_PROTOCOL_VERSION}",
        config_json=config_json,
        config_hash=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        model=model,
        base_url=base_url.rstrip("/"),
        timeout_seconds=float(timeout_seconds),
    )


class _RunLock:
    def __init__(self, database_path: Path) -> None:
        self.path = Path(f"{database_path}.lock")
        self._fd: int | None = None

    def __enter__(self) -> _RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as error:
                if attempt == 0 and self._remove_stale_lock():
                    continue
                raise E01Error(
                    f"E01 database is locked by another run: {self.path}"
                ) from error
        assert self._fd is not None
        os.write(self._fd, f"pid={os.getpid()}\n".encode("ascii"))
        return self

    def _remove_stale_lock(self) -> bool:
        try:
            text = self.path.read_text(encoding="ascii").strip()
            pid = int(text.removeprefix("pid="))
        except (OSError, ValueError):
            return False
        if _process_exists(pid):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if process:
            ctypes.windll.kernel32.CloseHandle(process)
            return True
        return ctypes.windll.kernel32.GetLastError() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _create_or_validate_experiment(
    store: SQLiteEventStore,
    plan: E01Plan,
    provider: ModelProvider,
    repository: Path,
) -> None:
    try:
        persisted = store.get_experiment(plan.experiment_id)
    except NotFoundError:
        store.create_experiment(
            experiment_id=plan.experiment_id,
            name=f"E01 {plan.mode} pilot",
            config_schema_version=E01_CONFIG_SCHEMA_VERSION,
            config_hash=plan.config_hash,
            config_json=plan.config_json,
            provenance=collect_provenance(
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                repository=repository,
            ),
        )
        return
    expected = (
        E01_CONFIG_SCHEMA_VERSION,
        plan.config_hash,
        plan.config_json,
        provider.provider_name,
        provider.model_name,
    )
    actual = (
        persisted.config_schema_version,
        persisted.config_hash,
        persisted.config_json,
        persisted.provenance.provider_name,
        persisted.provenance.model_name,
    )
    if actual != expected:
        raise E01Error("requested E01 protocol does not match persisted experiment metadata")


def _runner(
    store: SQLiteEventStore,
    plan: E01Plan,
    trial: E01TrialPlan,
    provider: ModelProvider,
) -> TrialRunner:
    return TrialRunner(
        experiment_id=plan.experiment_id,
        trial_id=trial.trial_id,
        trial_seed=trial.seed,
        condition=trial.condition,
        total_rounds=plan.rounds_per_trial,
        config_hash=plan.config_hash,
        profiles=plan.profiles,
        task_source=FixedTaskSource(plan.task_artifact.tasks),
        provider=provider,
        event_store=store,
        round_engine=RoundEngine(ballot_provider=LLMBallotProvider()),
        agent_id_namespace=trial.identity_namespace,
    )


def _summary(store: SQLiteEventStore, plan: E01Plan) -> dict[str, object]:
    trial_rows = []
    committed_total = 0
    completed_trials = 0
    response_total = 0
    score_total = 0
    ballot_total = 0
    ballot_valid = 0
    invalid_reasons: Counter[str] = Counter()
    for trial in plan.trials:
        metadata = store.get_trial(trial.trial_id)
        last_round = store.last_committed_round(trial.trial_id)
        committed = 0 if last_round is None else last_round + 1
        committed_total += committed
        completed_trials += int(metadata.status == "completed")
        for round_index in range(committed):
            result = store.load_round(trial.trial_id, round_index)
            response_total += len(result.responses)
            score_total += len(result.scores)
            ballot_total += len(result.ballots)
            ballot_valid += sum(evidence.valid for evidence in result.ballot_evidence)
            invalid_reasons.update(
                evidence.invalid_reason
                for evidence in result.ballot_evidence
                if evidence.invalid_reason is not None
            )
        trial_rows.append(
            {
                "committed_rounds": committed,
                "condition": trial.condition,
                "replicate": trial.replicate,
                "status": metadata.status,
                "total_rounds": plan.rounds_per_trial,
                "trial_id": trial.trial_id,
                "trial_seed": trial.seed,
            }
        )
    return {
        "apparatus": {
            "ballot_attempts": ballot_total,
            "ballot_invalid": ballot_total - ballot_valid,
            "ballot_invalid_reasons": dict(sorted(invalid_reasons.items())),
            "ballot_valid": ballot_valid,
            "committed_model_generations": response_total + ballot_total,
            "committed_responses": response_total,
            "committed_rounds": committed_total,
            "committed_scores": score_total,
            "completed_trials": completed_trials,
            "total_rounds": plan.total_rounds,
            "total_trials": len(plan.trials),
        },
        "database": str(plan.database_path),
        "experiment_id": plan.experiment_id,
        "mode": plan.mode,
        "protocol": {
            "config_hash": plan.config_hash,
            "profile_hash": profile_pool_hash(plan.profiles),
            "task_hash": plan.task_hash,
            "version": E01_PROTOCOL_VERSION,
        },
        "trials": trial_rows,
    }


def _write_summary(store: SQLiteEventStore, plan: E01Plan) -> None:
    target = plan.summary_path
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(_summary(store, plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def run_e01(
    plan: E01Plan,
    provider: ModelProvider,
    *,
    repository: str | Path,
    check_provider: Callable[[], str] | None = None,
    stop_after_commits: int | None = None,
    output: Callable[[str], None] = print,
) -> E01RunOutcome:
    if stop_after_commits is not None and (
        not isinstance(stop_after_commits, int)
        or isinstance(stop_after_commits, bool)
        or stop_after_commits <= 0
    ):
        raise E01Error("stop_after_commits must be a positive integer")
    if provider.provider_name != "ollama":
        raise E01Error("E01 requires the Ollama provider")
    if provider.model_name != plan.model:
        raise E01Error("provider model does not match the E01 plan")

    plan.database_path.parent.mkdir(parents=True, exist_ok=True)
    with _RunLock(plan.database_path), SQLiteEventStore(plan.database_path) as store:
        store.initialize()
        _create_or_validate_experiment(store, plan, provider, Path(repository))
        runners = tuple(_runner(store, plan, trial, provider) for trial in plan.trials)
        states = [runner.initialize() for runner in runners]
        committed_before = sum(state.next_round_index for state in states)
        output(
            f"E01 {plan.mode}: resuming at {committed_before}/{plan.total_rounds} "
            f"committed rounds ({sum(state.completed for state in states)}/"
            f"{len(states)} trials complete)"
        )
        for runner, state in zip(runners, states, strict=True):
            output(
                f"  {runner.trial_id}: next round {state.next_round_index}/"
                f"{plan.rounds_per_trial}"
            )
        _write_summary(store, plan)
        if all(state.completed for state in states):
            return E01RunOutcome(0, committed_before, plan.total_rounds, True, False)

        if check_provider is not None:
            version = check_provider()
            output(f"Ollama ready: version {version}; model {provider.model_name}")

        committed_this_run = 0
        try:
            while not all(state.completed for state in states):
                for index, runner in enumerate(runners):
                    if states[index].completed:
                        continue
                    step = runner.run_next_round()
                    states[index] = step.state
                    committed_this_run += 1
                    committed_total = committed_before + committed_this_run
                    output(
                        f"Committed {committed_total}/{plan.total_rounds}: "
                        f"{runner.trial_id} round {step.result.round_index + 1}/"
                        f"{plan.rounds_per_trial}"
                    )
                    _write_summary(store, plan)
                    if all(state.completed for state in states):
                        break
                    if (
                        stop_after_commits is not None
                        and committed_this_run >= stop_after_commits
                    ):
                        output("Controlled stop requested after committed round boundary.")
                        return E01RunOutcome(
                            committed_this_run,
                            committed_total,
                            plan.total_rounds,
                            False,
                            True,
                        )
        except KeyboardInterrupt:
            _write_summary(store, plan)
            output(
                f"Interrupted after {committed_before + committed_this_run}/"
                f"{plan.total_rounds} committed rounds. Database: {plan.database_path}"
            )
            raise
        return E01RunOutcome(
            committed_this_run,
            plan.total_rounds,
            plan.total_rounds,
            True,
            False,
        )


def build_ollama_provider(plan: E01Plan) -> OllamaProvider:
    return OllamaProvider(
        model=plan.model,
        base_url=plan.base_url,
        timeout_seconds=plan.timeout_seconds,
        temperature=E01_TEMPERATURE,
        num_predict=E01_NUM_PREDICT,
    )
