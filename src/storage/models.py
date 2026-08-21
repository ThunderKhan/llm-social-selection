from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    code_commit: str | None
    python_version: str
    platform: str
    provider_name: str
    model_name: str
    created_at: str


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment_id: str
    name: str
    config_schema_version: int
    config_hash: str
    config_json: str
    database_schema_version: int
    provenance: Provenance


@dataclass(frozen=True)
class TrialMetadata:
    trial_id: str
    experiment_id: str
    trial_seed: int
    status: str
    created_at: str
    completed_at: str | None
    condition: str | None
    total_rounds: int | None
    config_hash: str | None
    profile_pool_hash: str | None
    replacement_version: str | None
