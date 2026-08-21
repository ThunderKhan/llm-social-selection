from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, cast

from ..agents import PromptProfile
from ..domain import ReplacementEvent, SelectionMechanism
from ..models import ModelProvider
from ..population import Population, build_initial_population
from ..replacement import (
    FIXED_QUEUE_VERSION,
    FixedReplacementQueue,
    profile_pool_hash,
    replace_selected_agent,
)
from .context import RoundContext
from .round import RoundEngine
from .state import TrialState, TrialStep
from .task_source import TaskSource

if TYPE_CHECKING:
    from ..storage import SQLiteEventStore


class TrialError(ValueError):
    """A trial cannot initialize, resume, or advance safely."""


class TrialCompleteError(TrialError):
    """No further round exists for a completed trial."""


class TrialRunner:
    def __init__(
        self,
        *,
        experiment_id: str,
        trial_id: str,
        trial_seed: int,
        condition: SelectionMechanism,
        total_rounds: int,
        config_hash: str,
        profiles: Mapping[str, PromptProfile],
        task_source: TaskSource,
        provider: ModelProvider,
        event_store: SQLiteEventStore,
        round_engine: RoundEngine | None = None,
        agent_id_namespace: str | None = None,
    ) -> None:
        if not isinstance(total_rounds, int) or isinstance(total_rounds, bool) or total_rounds <= 0:
            raise TrialError("total_rounds must be a positive integer")
        if condition not in ("peer_vote", "objective", "random"):
            raise TrialError("invalid trial condition")
        if not isinstance(config_hash, str) or len(config_hash) != 64:
            raise TrialError("config_hash must be a 64-character string")

        self.experiment_id = experiment_id
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.condition = condition
        self.total_rounds = total_rounds
        self.config_hash = config_hash
        self.profiles = MappingProxyType(dict(profiles))
        self.task_source = task_source
        self.provider = provider
        self.event_store = event_store
        self.round_engine = round_engine or RoundEngine()
        self.agent_id_namespace = agent_id_namespace or trial_id
        self.profile_pool_hash = profile_pool_hash(self.profiles)
        self.initial_population = build_initial_population(
            trial_id=trial_id,
            trial_seed=trial_seed,
            profiles=self.profiles,
            agent_id_namespace=self.agent_id_namespace,
        )
        self.replacement_queue = FixedReplacementQueue.build(
            trial_id=trial_id,
            trial_seed=trial_seed,
            profiles=self.profiles,
            count=max(total_rounds - 1, 0),
            agent_id_namespace=self.agent_id_namespace,
        )

    def initialize(self) -> TrialState:
        experiment = self.event_store.get_experiment(self.experiment_id)
        if experiment.config_hash != self.config_hash:
            raise TrialError("config hash does not match persisted experiment")
        if experiment.provenance.provider_name != self.provider.provider_name:
            raise TrialError("provider does not match persisted experiment")
        if experiment.provenance.model_name != self.provider.model_name:
            raise TrialError("model does not match persisted experiment")

        if not self.event_store.trial_exists(self.trial_id):
            self.event_store.create_trial(
                trial_id=self.trial_id,
                experiment_id=self.experiment_id,
                trial_seed=self.trial_seed,
                condition=self.condition,
                total_rounds=self.total_rounds,
                config_hash=self.config_hash,
                profile_pool_hash=self.profile_pool_hash,
                replacement_version=FIXED_QUEUE_VERSION,
            )
        trial = self.event_store.get_trial(self.trial_id)
        expected_protocol = (
            self.experiment_id,
            self.trial_seed,
            self.condition,
            self.total_rounds,
            self.config_hash,
            self.profile_pool_hash,
            FIXED_QUEUE_VERSION,
        )
        persisted_protocol = (
            trial.experiment_id,
            trial.trial_seed,
            trial.condition,
            trial.total_rounds,
            trial.config_hash,
            trial.profile_pool_hash,
            trial.replacement_version,
        )
        if persisted_protocol != expected_protocol:
            raise TrialError("requested trial protocol does not match persisted metadata")

        initial_agents = self.event_store.load_initial_agents(self.trial_id)
        if not initial_agents:
            if self.event_store.load_agents(self.trial_id):
                raise TrialError("trial has replacement agents but no initial population")
            self.event_store.register_agents(self.trial_id, self.initial_population)
            initial_agents = self.event_store.load_initial_agents(self.trial_id)
        if initial_agents != self.initial_population.agents:
            raise TrialError("initial population does not match persisted trial agents")

        last_round = self.event_store.last_committed_round(self.trial_id)
        next_round = 0 if last_round is None else last_round + 1
        if next_round > self.total_rounds:
            raise TrialError("persisted rounds exceed configured trial length")
        events = self.event_store.load_replacement_events(self.trial_id)
        expected_event_count = min(next_round, self.total_rounds - 1)
        if len(events) != expected_event_count:
            raise TrialError(
                "replacement event count does not match the committed round prefix"
            )
        agents_by_id = {
            agent.agent_id: agent for agent in self.event_store.load_agents(self.trial_id)
        }
        for index, event in enumerate(events):
            candidate = self.replacement_queue.replacement_for(index)
            if (
                event.round_index != index
                or event.queue_index != index
                or event.added_agent_id != candidate.agent.agent_id
                or event.profile_id != candidate.profile_id
                or agents_by_id.get(event.added_agent_id) != candidate.agent
            ):
                raise TrialError(f"replacement queue mismatch at position {index}")

        population = self.event_store.active_population(self.trial_id)
        if next_round == self.total_rounds and trial.status != "completed":
            self.event_store.complete_trial(self.trial_id)
            trial = self.event_store.get_trial(self.trial_id)
        if trial.status == "completed" and next_round != self.total_rounds:
            raise TrialError("trial is marked completed before all rounds are committed")
        return TrialState(
            experiment_id=self.experiment_id,
            trial_id=self.trial_id,
            trial_seed=self.trial_seed,
            condition=cast(SelectionMechanism, self.condition),
            total_rounds=self.total_rounds,
            next_round_index=next_round,
            population=population,
            replacement_queue_position=len(events),
            completed=trial.status == "completed",
        )

    def run_next_round(self) -> TrialStep:
        state = self.initialize()
        if state.completed:
            raise TrialCompleteError(f"trial is already completed: {self.trial_id}")
        round_index = state.next_round_index
        task = self.task_source.task_for(
            trial_seed=self.trial_seed,
            round_index=round_index,
        )
        context = RoundContext(
            experiment_id=self.experiment_id,
            trial_id=self.trial_id,
            round_index=round_index,
            condition=self.condition,
            seed=self.trial_seed,
            task=task,
            population=state.population,
            profiles=self.profiles,
        )
        result = self.round_engine.execute(context, self.provider)

        event: ReplacementEvent | None = None
        replacement_agent = None
        if round_index < self.total_rounds - 1:
            candidate = self.replacement_queue.replacement_for(
                state.replacement_queue_position
            )
            replacement_agent = candidate.agent
            event = ReplacementEvent(
                replacement_id=(
                    f"replacement-{self.trial_id}-r{round_index:03d}"
                ),
                trial_id=self.trial_id,
                round_index=round_index,
                removed_agent_id=result.selection.selected_agent_id,
                added_agent_id=replacement_agent.agent_id,
                profile_id=replacement_agent.profile_id,
                queue_index=candidate.queue_index,
            )
            replace_selected_agent(
                state.population,
                result.selection.selected_agent_id,
                replacement_agent,
            )

        self.event_store.commit_round(
            context,
            result,
            replacement_event=event,
            replacement_agent=replacement_agent,
        )
        if round_index == self.total_rounds - 1:
            self.event_store.complete_trial(self.trial_id)
        return TrialStep(result=result, replacement=event, state=self.initialize())

    def run(self) -> TrialState:
        state = self.initialize()
        while not state.completed:
            state = self.run_next_round().state
        return state
