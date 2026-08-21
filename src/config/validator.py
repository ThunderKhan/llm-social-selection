from collections.abc import Mapping
from math import isfinite
from typing import Any, Never, TypeVar, cast

from .schema import (
    AppConfig,
    ExperimentConfig,
    InformationConfig,
    ModelConfig,
    PopulationConfig,
    ReplacementConfig,
    SelectionConfig,
    SelectionMechanism,
    StorageConfig,
    TaskConfig,
)


class ConfigError(Exception):
    """A configuration could not be loaded or validated."""


T = TypeVar("T")


def _fail(field: str, message: str) -> Never:
    raise ConfigError(f"{field}: {message}")


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in data:
        _fail(name, "required section is missing")
    value = data[name]
    if not isinstance(value, Mapping):
        _fail(name, "expected a mapping")
    return value


def _required(section: Mapping[str, Any], section_name: str, name: str) -> Any:
    if name not in section:
        _fail(f"{section_name}.{name}", "required field is missing")
    return section[name]


def _string(section: Mapping[str, Any], section_name: str, name: str) -> str:
    value = _required(section, section_name, name)
    if not isinstance(value, str) or not value.strip():
        _fail(f"{section_name}.{name}", "expected a non-empty string")
    return value


def _integer(section: Mapping[str, Any], section_name: str, name: str) -> int:
    value = _required(section, section_name, name)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{section_name}.{name}", f"expected an integer, got {value!r}")
    return value


def _boolean(section: Mapping[str, Any], section_name: str, name: str) -> bool:
    value = _required(section, section_name, name)
    if not isinstance(value, bool):
        _fail(f"{section_name}.{name}", f"expected a boolean, got {value!r}")
    return value


def _literal(
    section: Mapping[str, Any],
    section_name: str,
    name: str,
    allowed: tuple[T, ...],
) -> T:
    value = _required(section, section_name, name)
    if value not in allowed or type(value) is not type(allowed[0]):
        expected = ", ".join(str(item) for item in allowed)
        _fail(
            f"{section_name}.{name}",
            f"expected one of {expected}; got {value}",
        )
    return cast(T, value)


def validate_config(data: object) -> AppConfig:
    """Validate parsed configuration data and return its typed representation."""
    if not isinstance(data, Mapping):
        raise ConfigError("configuration: expected a top-level mapping")

    experiment = _section(data, "experiment")
    population = _section(data, "population")
    model = _section(data, "model")
    task = _section(data, "task")
    information = _section(data, "information")
    selection = _section(data, "selection")
    replacement = _section(data, "replacement")
    storage = _section(data, "storage")

    schema_version = _integer(experiment, "experiment", "schema_version")
    if schema_version != 1:
        _fail("experiment.schema_version", f"expected 1, got {schema_version}")

    trials = _integer(experiment, "experiment", "trials")
    if trials <= 0:
        _fail("experiment.trials", f"expected a positive integer, got {trials}")

    rounds = _integer(experiment, "experiment", "rounds")
    if rounds <= 0:
        _fail("experiment.rounds", f"expected a positive integer, got {rounds}")

    population_size = _integer(population, "population", "size")
    if population_size != 8:
        _fail("population.size", f"expected 8, got {population_size}")

    temperature = _required(model, "model", "temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        _fail("model.temperature", f"expected a number, got {temperature!r}")
    if not isfinite(temperature) or temperature < 0:
        _fail("model.temperature", f"expected a non-negative finite number, got {temperature}")

    mechanism = _literal(
        selection,
        "selection",
        "mechanism",
        ("peer_vote", "objective", "random"),
    )

    return AppConfig(
        experiment=ExperimentConfig(
            name=_string(experiment, "experiment", "name"),
            schema_version=schema_version,
            seed=_integer(experiment, "experiment", "seed"),
            trials=trials,
            rounds=rounds,
        ),
        population=PopulationConfig(
            size=population_size,
            profiles_file=_string(population, "population", "profiles_file"),
        ),
        model=ModelConfig(
            provider=_literal(model, "model", "provider", ("mock",)),
            model=_string(model, "model", "model"),
            temperature=float(temperature),
        ),
        task=TaskConfig(
            source=_string(task, "task", "source"),
            order=_literal(task, "task", "order", ("seeded",)),
        ),
        information=InformationConfig(
            response_authorship_visible=_boolean(
                information, "information", "response_authorship_visible"
            ),
            prior_votes_visible=_boolean(
                information, "information", "prior_votes_visible"
            ),
        ),
        selection=SelectionConfig(
            mechanism=cast(SelectionMechanism, mechanism),
            tie_break=_literal(
                selection, "selection", "tie_break", ("seeded_random",)
            ),
            self_vote=_literal(selection, "selection", "self_vote", ("forbidden",)),
            invalid_ballot=_literal(
                selection, "selection", "invalid_ballot", ("abstain",)
            ),
        ),
        replacement=ReplacementConfig(
            mechanism=_literal(
                replacement, "replacement", "mechanism", ("fixed_profile_pool",)
            )
        ),
        storage=StorageConfig(
            sqlite_path=_string(storage, "storage", "sqlite_path")
        ),
    )
