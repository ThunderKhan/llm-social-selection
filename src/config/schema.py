from dataclasses import dataclass
from typing import Literal


SelectionMechanism = Literal["peer_vote", "objective", "random"]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    schema_version: int
    seed: int
    trials: int
    rounds: int


@dataclass(frozen=True)
class PopulationConfig:
    size: int
    profiles_file: str


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["mock"]
    model: str
    temperature: float


@dataclass(frozen=True)
class TaskConfig:
    source: str
    order: Literal["seeded"]


@dataclass(frozen=True)
class InformationConfig:
    response_authorship_visible: bool
    prior_votes_visible: bool


@dataclass(frozen=True)
class SelectionConfig:
    mechanism: SelectionMechanism
    tie_break: Literal["seeded_random"]
    self_vote: Literal["forbidden"]
    invalid_ballot: Literal["abstain"]


@dataclass(frozen=True)
class ReplacementConfig:
    mechanism: Literal["fixed_profile_pool"]


@dataclass(frozen=True)
class StorageConfig:
    sqlite_path: str


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig
    population: PopulationConfig
    model: ModelConfig
    task: TaskConfig
    information: InformationConfig
    selection: SelectionConfig
    replacement: ReplacementConfig
    storage: StorageConfig
