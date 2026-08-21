from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from src.agents import AgentIdentity, PromptProfile
from src.ballots import (
    BallotParseError,
    LLMBallotProvider,
    anonymous_candidates,
    ballot_response_schema,
    parse_ballot_choice,
    render_ballot_prompt,
)
from src.domain import Ballot, Response
from src.models import ModelOutput, ModelProvider
from src.population import Population
from src.storage import Provenance, SQLiteEventStore
from src.tasks import Task
from src.tournament import RoundContext, RoundEngine


class BallotAwareProvider(ModelProvider):
    def __init__(self, ballot_output: str = '{"choice":"A"}') -> None:
        self.ballot_output = ballot_output
        self.prompts: list[str] = []
        self.response_schemas: list[Mapping[str, Any] | None] = []

    @property
    def provider_name(self) -> str:
        return "test-provider"

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(
        self,
        *,
        agent: AgentIdentity,
        task: Task,
        prompt: str,
        request_id: str,
        seed: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
    ) -> ModelOutput:
        del agent, task
        self.prompts.append(prompt)
        self.response_schemas.append(response_schema)
        content = (
            self.ballot_output
            if prompt.startswith("ANONYMOUS RESPONSE EVALUATION")
            else "observable answer"
        )
        return ModelOutput(
            content=content,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=request_id,
            seed=seed,
        )


def responses_for(population: Population, task: Task) -> tuple[Response, ...]:
    return tuple(
        Response(
            response_id=f"response-{agent.agent_id}",
            trial_id="trial-001",
            round_index=0,
            task_id=task.task_id,
            agent_id=agent.agent_id,
            content=f"Candidate answer {index}",
            provider_name="test-provider",
            model_name="test-model",
        )
        for index, agent in enumerate(population.agents, start=1)
    )


def context_for(
    condition: str,
    population: Population,
    profiles: dict[str, PromptProfile],
    task: Task,
) -> RoundContext:
    return RoundContext(
        experiment_id="experiment-001",
        trial_id="trial-001",
        round_index=0,
        condition=condition,  # type: ignore[arg-type]
        seed=42,
        task=task,
        population=population,
        profiles=profiles,
    )


def test_anonymous_mapping_has_seven_non_self_candidates(
    population: Population, round_task: Task
) -> None:
    responses = responses_for(population, round_task)
    voter = population.agents[0]

    first = anonymous_candidates(
        trial_seed=42,
        round_index=0,
        voter_agent_id=voter.agent_id,
        population=population,
        responses=responses,
    )
    second = anonymous_candidates(
        trial_seed=42,
        round_index=0,
        voter_agent_id=voter.agent_id,
        population=population,
        responses=tuple(reversed(responses)),
    )

    assert first == second
    assert tuple(candidate.label for candidate in first) == tuple("ABCDEFG")
    assert len(first) == 7
    assert {candidate.agent_id for candidate in first} == {
        agent.agent_id for agent in population.agents[1:]
    }
    assert voter.agent_id not in {candidate.agent_id for candidate in first}


def test_anonymous_order_is_voter_specific(
    population: Population, round_task: Task
) -> None:
    responses = responses_for(population, round_task)
    first = anonymous_candidates(
        trial_seed=42,
        round_index=0,
        voter_agent_id=population.agents[0].agent_id,
        population=population,
        responses=responses,
    )
    second = anonymous_candidates(
        trial_seed=42,
        round_index=0,
        voter_agent_id=population.agents[1].agent_id,
        population=population,
        responses=responses,
    )

    assert tuple(candidate.agent_id for candidate in first) != tuple(
        candidate.agent_id for candidate in second
    )


def test_rendered_prompt_contains_no_identity_metadata(
    population: Population, round_task: Task
) -> None:
    responses = responses_for(population, round_task)
    voter = population.agents[0]
    candidates = anonymous_candidates(
        trial_seed=42,
        round_index=0,
        voter_agent_id=voter.agent_id,
        population=population,
        responses=responses,
    )

    prompt = render_ballot_prompt(round_task, candidates, responses)

    assert "Response A:" in prompt
    assert 'one key named choice' in prompt
    assert not any(f'{{"choice":"{label}"}}' in prompt for label in "ABCDEFG")
    for agent in population.agents:
        assert agent.agent_id not in prompt
        assert agent.profile_id not in prompt
        assert agent.display_label not in prompt


def test_ballot_schema_allows_exactly_the_displayed_labels() -> None:
    schema = ballot_response_schema(("Q", "R", "S"))

    assert schema == {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": ["Q", "R", "S"]}
        },
        "required": ["choice"],
        "additionalProperties": False,
    }


def test_strict_parser_accepts_only_exact_choice_object() -> None:
    assert parse_ballot_choice('{"choice":"C"}', set("ABCDEFG")) == "C"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "empty_output"),
        ("not json", "invalid_json"),
        ("[]", "invalid_shape"),
        ("{}", "missing_choice"),
        ('{"choice":"A","note":"x"}', "extra_keys"),
        ('{"choice":""}', "invalid_choice"),
        ('{"choice":"Z"}', "invalid_label"),
        ('{"choice":"A","choice":"B"}', "duplicate_key"),
    ],
)
def test_strict_parser_rejects_invalid_output(raw: str, reason: str) -> None:
    with pytest.raises(BallotParseError) as caught:
        parse_ballot_choice(raw, set("ABCDEFG"))

    assert caught.value.reason == reason


def test_llm_ballot_preserves_mapping_and_provider_metadata(
    population: Population, round_task: Task
) -> None:
    provider = BallotAwareProvider('{"choice":"C"}')
    voter = population.agents[0]

    generated = LLMBallotProvider().generate_ballot(
        trial_id="trial-001",
        round_index=0,
        trial_seed=42,
        task=round_task,
        voter=voter,
        population=population,
        responses=responses_for(population, round_task),
        model_provider=provider,
    )

    assert generated.evidence is not None
    assert generated.evidence.valid is True
    assert generated.evidence.parsed_choice == "C"
    assert generated.evidence.raw_output == '{"choice":"C"}'
    assert generated.evidence.provider_name == "test-provider"
    assert generated.evidence.model_name == "test-model"
    assert provider.response_schemas == [ballot_response_schema(tuple("ABCDEFG"))]
    assert generated.ballot.supported_agent_id == generated.evidence.candidate_order[2].agent_id
    assert generated.ballot.supported_agent_id != voter.agent_id


def test_invalid_llm_output_becomes_auditable_abstention(
    population: Population, round_task: Task
) -> None:
    provider = BallotAwareProvider("not json")

    generated = LLMBallotProvider().generate_ballot(
        trial_id="trial-001",
        round_index=0,
        trial_seed=42,
        task=round_task,
        voter=population.agents[0],
        population=population,
        responses=responses_for(population, round_task),
        model_provider=provider,
    )

    assert generated.ballot.supported_agent_id is None
    assert generated.evidence is not None
    assert generated.evidence.valid is False
    assert generated.evidence.invalid_reason == "invalid_json"
    assert generated.evidence.parsed_choice is None


@pytest.mark.parametrize("condition", ["peer_vote", "objective", "random"])
def test_all_conditions_collect_equivalent_llm_ballot_evidence(
    condition: str,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    result = RoundEngine(ballot_provider=LLMBallotProvider()).execute(
        context_for(condition, population, profiles, round_task),
        BallotAwareProvider(),
    )

    assert len(result.ballots) == 8
    assert len(result.ballot_evidence) == 8
    assert all(evidence.valid for evidence in result.ballot_evidence)
    assert result.selection.mechanism == condition


def test_matched_conditions_share_llm_ballot_evidence(
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    results = tuple(
        RoundEngine(ballot_provider=LLMBallotProvider()).execute(
            context_for(condition, population, profiles, round_task),
            BallotAwareProvider(),
        )
        for condition in ("peer_vote", "objective", "random")
    )

    assert results[0].responses == results[1].responses == results[2].responses
    assert results[0].scores == results[1].scores == results[2].scores
    assert results[0].ballots == results[1].ballots == results[2].ballots
    assert (
        results[0].ballot_evidence
        == results[1].ballot_evidence
        == results[2].ballot_evidence
    )
    assert {result.selection.mechanism for result in results} == {
        "peer_vote",
        "objective",
        "random",
    }


def test_peer_selection_accepts_eight_invalid_attempts_as_abstentions(
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    result = RoundEngine(ballot_provider=LLMBallotProvider()).execute(
        context_for("peer_vote", population, profiles, round_task),
        BallotAwareProvider("not json"),
    )

    assert all(ballot.supported_agent_id is None for ballot in result.ballots)
    assert all(not evidence.valid for evidence in result.ballot_evidence)
    assert result.selection.reason == "fewest_support_votes;count=0"


def test_llm_ballot_evidence_persists_and_reconstructs(
    tmp_path: Path,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    config_json = '{"ballot_provider":"llm"}'
    config_hash = sha256(config_json.encode("utf-8")).hexdigest()
    store = SQLiteEventStore(tmp_path / "ballots.sqlite")
    store.initialize()
    store.create_experiment(
        experiment_id="experiment-001",
        name="LLM ballot test",
        config_schema_version=1,
        config_hash=config_hash,
        config_json=config_json,
        provenance=Provenance(
            code_commit=None,
            python_version="3.12",
            platform="test",
            provider_name="test-provider",
            model_name="test-model",
            created_at="2026-08-21T06:30:00+00:00",
        ),
    )
    store.create_trial(
        trial_id="trial-001",
        experiment_id="experiment-001",
        trial_seed=42,
        created_at="2026-08-21T06:30:00+00:00",
    )
    store.register_agents("trial-001", population)
    context = context_for("peer_vote", population, profiles, round_task)
    result = RoundEngine(ballot_provider=LLMBallotProvider()).execute(
        context,
        BallotAwareProvider("not json"),
    )

    store.commit_round(context, result)
    loaded = store.load_round("trial-001", 0)

    assert loaded == result
    assert len(loaded.ballot_evidence) == 8
    assert all(evidence.raw_output == "not json" for evidence in loaded.ballot_evidence)
    assert all(ballot.supported_agent_id is None for ballot in loaded.ballots)
    assert all(len(evidence.candidate_order) == 7 for evidence in loaded.ballot_evidence)
    store.close()
