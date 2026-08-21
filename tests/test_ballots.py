from __future__ import annotations

from src.domain import Response
from src.tournament import derive_seed, generate_support_ballot


def make_responses(agent_ids: tuple[str, ...]) -> tuple[Response, ...]:
    return tuple(
        Response(
            response_id=f"response-{agent_id}",
            trial_id="trial-001",
            round_index=2,
            task_id="task-001",
            agent_id=agent_id,
            content=f"content-{agent_id}",
            provider_name="mock",
            model_name="deterministic-v1",
        )
        for agent_id in agent_ids
    )


def test_support_ballot_is_deterministic_eligible_and_not_self() -> None:
    agent_ids = tuple(f"agent-{index:03d}" for index in range(1, 9))
    arguments = {
        "trial_id": "trial-001",
        "round_index": 2,
        "task_id": "task-001",
        "voter_agent_id": "agent-004",
        "eligible_agent_ids": agent_ids,
        "responses": make_responses(agent_ids),
        "seed": derive_seed(42, 2, "ballot", "agent-004"),
    }

    first = generate_support_ballot(**arguments)
    second = generate_support_ballot(**arguments)

    assert first == second
    assert first.voter_agent_id == "agent-004"
    assert first.supported_agent_id in agent_ids
    assert first.supported_agent_id != first.voter_agent_id


def test_candidate_and_response_order_do_not_change_ballot() -> None:
    agent_ids = tuple(f"agent-{index:03d}" for index in range(1, 9))
    responses = make_responses(agent_ids)
    common = {
        "trial_id": "trial-001",
        "round_index": 2,
        "task_id": "task-001",
        "voter_agent_id": "agent-004",
        "seed": 1234,
    }

    ordered = generate_support_ballot(
        **common, eligible_agent_ids=agent_ids, responses=responses
    )
    reversed_input = generate_support_ballot(
        **common,
        eligible_agent_ids=tuple(reversed(agent_ids)),
        responses=tuple(reversed(responses)),
    )

    assert ordered == reversed_input


def test_seed_derivation_is_stable_and_namespaced() -> None:
    assert derive_seed(42, 2, "ballot", "agent-004") == derive_seed(
        42, 2, "ballot", "agent-004"
    )
    assert derive_seed(42, 2, "ballot", "agent-004") != derive_seed(
        42, 2, "response", "agent-004"
    )
