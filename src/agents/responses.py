from ..domain import Response
from ..models import ModelProvider
from ..tasks import Task
from .models import AgentIdentity, PromptProfile
from .prompting import render_prompt


def generate_agent_response(
    *,
    response_id: str,
    trial_id: str,
    round_index: int,
    agent: AgentIdentity,
    profile: PromptProfile,
    task: Task,
    provider: ModelProvider,
    request_id: str,
    seed: int | None = None,
    prompt: str | None = None,
) -> Response:
    """Generate one typed response without round or population orchestration."""
    if agent.profile_id != profile.profile_id:
        raise ValueError(
            "agent.profile_id must match the supplied prompt profile's profile_id"
        )

    effective_prompt = prompt if prompt is not None else render_prompt(profile, task)
    output = provider.generate(
        agent=agent,
        task=task,
        prompt=effective_prompt,
        request_id=request_id,
        seed=seed,
    )
    return Response(
        response_id=response_id,
        trial_id=trial_id,
        round_index=round_index,
        task_id=task.task_id,
        agent_id=agent.agent_id,
        content=output.content,
        provider_name=output.provider_name,
        model_name=output.model_name,
        request_id=output.request_id,
        seed=output.seed,
    )
