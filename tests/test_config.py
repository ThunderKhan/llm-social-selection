from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import ConfigError, canonicalize, hash_config, load_config, validate_config
from src.config.schema import AppConfig


ROOT = Path(__file__).resolve().parents[1]
APPARATUS = ROOT / "configs" / "apparatus"


@pytest.fixture
def valid_data() -> dict[str, Any]:
    return {
        "experiment": {
            "name": "peer_selection_smoke",
            "schema_version": 1,
            "seed": 42,
            "trials": 1,
            "rounds": 10,
        },
        "population": {
            "size": 8,
            "profiles_file": "configs/profiles.yaml",
        },
        "model": {
            "provider": "mock",
            "model": "deterministic-v1",
            "temperature": 0,
        },
        "task": {"source": "fixtures/tasks.json", "order": "seeded"},
        "information": {
            "response_authorship_visible": False,
            "prior_votes_visible": False,
        },
        "selection": {
            "mechanism": "peer_vote",
            "tie_break": "seeded_random",
            "self_vote": "forbidden",
            "invalid_ballot": "abstain",
        },
        "replacement": {"mechanism": "fixed_profile_pool"},
        "storage": {"sqlite_path": "experiments/peer_selection_smoke.sqlite"},
    }


@pytest.mark.parametrize(
    ("filename", "mechanism"),
    [
        ("e00_peer.yaml", "peer_vote"),
        ("e00_objective.yaml", "objective"),
        ("e00_random.yaml", "random"),
    ],
)
def test_valid_apparatus_configs_load(filename: str, mechanism: str) -> None:
    config = load_config(APPARATUS / filename)

    assert isinstance(config, AppConfig)
    assert config.selection.mechanism == mechanism
    assert config.experiment.seed == 42
    assert config.population.size == 8
    assert config.model.temperature == 0.0
    assert config.information.response_authorship_visible is False


def test_missing_required_section(valid_data: dict[str, Any]) -> None:
    del valid_data["storage"]

    with pytest.raises(ConfigError, match="storage: required section is missing"):
        validate_config(valid_data)


def test_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("experiment: [\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="expected a top-level mapping"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("population", "size", 7, "population.size: expected 8, got 7"),
        (
            "selection",
            "mechanism",
            "unknown",
            "selection.mechanism: expected one of peer_vote, objective, random; got unknown",
        ),
        (
            "selection",
            "self_vote",
            "allowed",
            "selection.self_vote: expected one of forbidden; got allowed",
        ),
        ("experiment", "rounds", 0, "experiment.rounds: expected a positive integer"),
        ("experiment", "trials", -1, "experiment.trials: expected a positive integer"),
        ("model", "provider", "ollama", "model.provider: expected one of mock; got ollama"),
        (
            "information",
            "prior_votes_visible",
            "false",
            "information.prior_votes_visible: expected a boolean",
        ),
        (
            "replacement",
            "mechanism",
            "clone_winner",
            "replacement.mechanism: expected one of fixed_profile_pool; got clone_winner",
        ),
    ],
)
def test_semantically_invalid_values(
    valid_data: dict[str, Any],
    section: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    valid_data[section][field] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(valid_data)


def test_identical_configs_have_identical_hashes(valid_data: dict[str, Any]) -> None:
    first = validate_config(valid_data)
    second = validate_config(copy.deepcopy(valid_data))

    assert hash_config(first) == hash_config(second)
    assert len(hash_config(first)) == 64


def test_yaml_key_order_does_not_affect_hash(
    valid_data: dict[str, Any], tmp_path: Path
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(yaml.safe_dump(valid_data, sort_keys=False), encoding="utf-8")
    reversed_data = {
        key: dict(reversed(value.items()))
        for key, value in reversed(valid_data.items())
    }
    second_path.write_text(yaml.safe_dump(reversed_data, sort_keys=False), encoding="utf-8")

    assert hash_config(load_config(first_path)) == hash_config(load_config(second_path))


def test_yaml_whitespace_and_comments_do_not_affect_hash(
    valid_data: dict[str, Any], tmp_path: Path
) -> None:
    plain_path = tmp_path / "plain.yaml"
    commented_path = tmp_path / "commented.yaml"
    dumped = yaml.safe_dump(valid_data, sort_keys=False)
    plain_path.write_text(dumped, encoding="utf-8")
    commented_path.write_text(
        f"# Apparatus configuration\n\n{dumped.replace('rounds: 10', 'rounds: 10  # duration')}\n",
        encoding="utf-8",
    )

    assert hash_config(load_config(plain_path)) == hash_config(load_config(commented_path))


def test_semantic_change_changes_hash(valid_data: dict[str, Any]) -> None:
    original = validate_config(valid_data)
    changed_data = copy.deepcopy(valid_data)
    changed_data["experiment"]["rounds"] = 11
    changed = validate_config(changed_data)

    assert canonicalize(original) != canonicalize(changed)
    assert hash_config(original) != hash_config(changed)


def test_cli_valid_configuration() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src", "validate", str(APPARATUS / "e00_peer.yaml")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Configuration valid" in result.stdout
    assert "Selection mechanism: peer_vote" in result.stdout
    assert "Config hash:" in result.stdout
    assert result.stderr == ""


def test_cli_invalid_configuration(tmp_path: Path, valid_data: dict[str, Any]) -> None:
    valid_data["population"]["size"] = 7
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(valid_data), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "src", "validate", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == (
        "Configuration invalid:\n"
        "population.size: expected 8, got 7\n"
    )
    assert "Traceback" not in result.stderr
