from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import (  # noqa: E402
    AgentIdentity,
    PromptProfile,
    generate_agent_response,
)
from src.models import ModelProviderError, OllamaProvider  # noqa: E402
from src.scoring import score_response  # noqa: E402
from src.storage import SQLiteEventStore, collect_provenance  # noqa: E402
from src.tasks import Task  # noqa: E402
from src.tournament import FixedTaskSource, TrialRunner  # noqa: E402


def smoke_task() -> Task:
    return Task(
        task_id="m2-smoke-arithmetic",
        family="arithmetic",
        prompt=(
            "Answer the task directly. Return only the final answer. "
            "Do not include explanation. What is 2 + 2?"
        ),
        expected_answer="4",
        scorer_version="exact-match-v1",
    )


def smoke_profiles() -> dict[str, PromptProfile]:
    return {
        f"m2-profile-{index:03d}": PromptProfile(
            profile_id=f"m2-profile-{index:03d}",
            parameters={"smoke_slot": index},
            template_version="m2-smoke-v1",
        )
        for index in range(1, 9)
    }


def build_runner(
    *,
    store: SQLiteEventStore,
    provider: OllamaProvider,
    profiles: dict[str, PromptProfile],
    task: Task,
    rounds: int,
    config_hash: str,
) -> TrialRunner:
    return TrialRunner(
        experiment_id="m2-smoke-experiment",
        trial_id="m2-smoke-trial",
        trial_seed=42,
        condition="objective",
        total_rounds=rounds,
        config_hash=config_hash,
        profiles=profiles,
        task_source=FixedTaskSource((task,)),
        provider=provider,
        event_store=store,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local M2 Ollama smoke path.")
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, choices=range(1, 4), default=2)
    args = parser.parse_args()

    provider = OllamaProvider(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=0,
        num_predict=32,
    )
    try:
        version = provider.check_health()
        provider.ensure_model_available()
        print(f"Ollama health: available (version {version})")
        print(f"Model available: {provider.model_name}")

        task = smoke_task()
        direct_profile = PromptProfile("m2-direct-profile", {}, "m2-smoke-v1")
        direct_agent = AgentIdentity(
            "m2-direct-agent", direct_profile.profile_id, "Participant 1", 0
        )
        direct_response = generate_agent_response(
            response_id="m2-direct-response",
            trial_id="m2-direct-smoke",
            round_index=0,
            agent=direct_agent,
            profile=direct_profile,
            task=task,
            provider=provider,
            request_id="m2-direct-request",
            seed=42,
        )
        direct_score = score_response(direct_response, task)
        print(
            "Single response: "
            f"content={direct_response.content!r}, exact_match={direct_score.value == 1.0}, "
            f"tokens={direct_response.token_count}, latency_ms={direct_response.latency_ms}"
        )

        profiles = smoke_profiles()
        config_json = json.dumps(
            {
                "condition": "objective",
                "model": {
                    "base_url": provider.base_url,
                    "model": provider.model_name,
                    "num_predict": provider.num_predict,
                    "provider": provider.provider_name,
                    "temperature": provider.temperature,
                    "timeout_seconds": provider.timeout_seconds,
                },
                "name": "M2 smoke",
                "rounds": args.rounds,
                "seed": 42,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        with TemporaryDirectory() as directory:
            database = Path(directory) / "m2-smoke.sqlite"
            store = SQLiteEventStore(database)
            store.initialize()
            store.create_experiment(
                experiment_id="m2-smoke-experiment",
                name="M2 smoke",
                config_schema_version=1,
                config_hash=config_hash,
                config_json=config_json,
                provenance=collect_provenance(
                    provider_name=provider.provider_name,
                    model_name=provider.model_name,
                    repository=ROOT,
                ),
            )
            runner = build_runner(
                store=store,
                provider=provider,
                profiles=profiles,
                task=task,
                rounds=args.rounds,
                config_hash=config_hash,
            )
            first = runner.run_next_round()
            extra_prose = sum(
                response.content.strip() != "4" for response in first.result.responses
            )
            print(
                "Single round: "
                f"responses={len(first.result.responses)}, "
                f"scores={len(first.result.scores)}, "
                f"ballots={len(first.result.ballots)}, "
                f"exact_matches={sum(score.value == 1.0 for score in first.result.scores)}, "
                f"unexpected_extra_prose={extra_prose}, "
                f"selected={first.result.selection.selected_agent_id}"
            )
            store.close()

            store = SQLiteEventStore(database)
            store.initialize()
            final_state = build_runner(
                store=store,
                provider=provider,
                profiles=profiles,
                task=task,
                rounds=args.rounds,
                config_hash=config_hash,
            ).run()
            first_persisted = store.load_round("m2-smoke-trial", 0)
            print(
                "Tiny trial: "
                f"rounds={args.rounds}, last_committed={store.last_committed_round('m2-smoke-trial')}, "
                f"status={store.get_trial('m2-smoke-trial').status}, "
                f"completed={final_state.completed}"
            )
            print(
                "Persistence: "
                f"first_round_equal={first_persisted == first.result}, "
                f"provider={first_persisted.responses[0].provider_name}, "
                f"model={first_persisted.responses[0].model_name}"
            )
            store.close()
    except ModelProviderError as error:
        print(f"Ollama smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
