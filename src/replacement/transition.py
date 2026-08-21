from ..agents import AgentIdentity
from ..population import Population
from .base import ReplacementError


def replace_selected_agent(
    population: Population,
    selected_agent_id: str,
    replacement_agent: AgentIdentity,
) -> Population:
    if replacement_agent.agent_id in {
        agent.agent_id for agent in population.agents
    }:
        raise ReplacementError(
            f"replacement agent ID already active: {replacement_agent.agent_id}"
        )
    try:
        index = next(
            index
            for index, agent in enumerate(population.agents)
            if agent.agent_id == selected_agent_id
        )
    except StopIteration as error:
        raise ReplacementError(f"selected agent is not active: {selected_agent_id}") from error

    agents = list(population.agents)
    agents[index] = replacement_agent
    return Population(tuple(agents))
