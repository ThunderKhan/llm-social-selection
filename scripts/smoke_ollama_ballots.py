from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import PromptProfile  # noqa: E402
from src.ballots import LLMBallotProvider  # noqa: E402
from src.models import ModelProviderError, OllamaProvider  # noqa: E402
from src.storage import SQLiteEventStore, collect_provenance  # noqa: E402
from src.tasks import Task  # noqa: E402
from src.tournament import FixedTaskSource, RoundEngine, TrialRunner  # noqa: E402


def profiles() -> dict[str, PromptProfile]:
    return {
        f"m2-ballot-profile-{index:03d}": PromptProfile(
            profile_id=f"m2-ballot-profile-{index:03d}",
            parameters={"smoke_slot": index},
            template_version="m2-ballot-smoke-v1",
        )
        for index in range(1, 9)
    }


def task() -> Task:
    return Task(
        task_id="m2-ballot-smoke-task",
        family="arithmetic",
        prompt=(
            "Answer directly. Return only the final answer. "
            "Do not include explanation. What is 2 + 2?"
        ),
        expected_answer="4",
        scorer_version="exact-match-v1",
    )


def runner(
    *,
    store: SQLiteEventStore,
    provider: OllamaProvider,
    profile_pool: dict[str, PromptProfile],
    smoke_task: Task,
    rounds: int,
    config_hash: str,
) -> TrialRunner:
    return TrialRunner(
        experiment_id="m2-ballot-smoke-experiment",
        trial_id="m2-ballot-smoke-trial",
        trial_seed=42,
        condition="peer_vote",
        total_rounds=rounds,
        config_hash=config_hash,
        profiles=profile_pool,
        task_source=FixedTaskSource((smoke_task,)),
        provider=provider,
        event_store=store,
        round_engine=RoundEngine(ballot_provider=LLMBallotProvider()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local M2 anonymous LLM-ballot smoke path."
    )
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=1)
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
        print(f"Model: {provider.model_name}")

        profile_pool = profiles()
        smoke_task = task()
        config_json = json.dumps(
            {
                "ballot": {
                    "candidate_order": "voter-seeded-v1",
                    "invalid_policy": "abstain",
                    "output_format": "json-choice-v1",
                    "provider": "llm",
                },
                "condition": "peer_vote",
                "model": {
                    "base_url": provider.base_url,
                    "model": provider.model_name,
                    "num_predict": provider.num_predict,
                    "provider": provider.provider_name,
                    "temperature": provider.temperature,
                    "timeout_seconds": provider.timeout_seconds,
                },
                "name": "M2 anonymous ballot smoke",
                "rounds": args.rounds,
                "seed": 42,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        with TemporaryDirectory() as directory:
            database = Path(directory) / "m2-ballot-smoke.sqlite"
            store = SQLiteEventStore(database)
            store.initialize()
            store.create_experiment(
                experiment_id="m2-ballot-smoke-experiment",
                name="M2 anonymous ballot smoke",
                config_schema_version=1,
                config_hash=config_hash,
                config_json=config_json,
                provenance=collect_provenance(
                    provider_name=provider.provider_name,
                    model_name=provider.model_name,
                    repository=ROOT,
                ),
            )
            trial_runner = runner(
                store=store,
                provider=provider,
                profile_pool=profile_pool,
                smoke_task=smoke_task,
                rounds=args.rounds,
                config_hash=config_hash,
            )
            first = trial_runner.run_next_round()
            evidence = first.result.ballot_evidence
            invalid_reasons = Counter(
                item.invalid_reason for item in evidence if not item.valid
            )
            valid_count = sum(item.valid for item in evidence)
            print("Round 0:")
            print(f"responses: {len(first.result.responses)}")
            print(f"ballot attempts: {len(evidence)}")
            print(f"valid ballots: {valid_count}")
            print(f"abstentions: {len(evidence) - valid_count}")
            print(f"invalid reasons: {dict(sorted(invalid_reasons.items()))}")
            for item in tuple(item for item in evidence if not item.valid)[:3]:
                print(
                    f"invalid raw sample ({item.voter_agent_id}): "
                    f"{item.raw_output[:200]!r}"
                )
            print(f"peer-selected: {first.result.selection.selected_agent_id}")
            store.close()

            store = SQLiteEventStore(database)
            store.initialize()
            final = runner(
                store=store,
                provider=provider,
                profile_pool=profile_pool,
                smoke_task=smoke_task,
                rounds=args.rounds,
                config_hash=config_hash,
            ).run()
            loaded = store.load_round("m2-ballot-smoke-trial", 0)
            print("Persistence:")
            print(f"round reload: {loaded == first.result}")
            print(
                "raw ballot evidence preserved: "
                f"{loaded.ballot_evidence == first.result.ballot_evidence}"
            )
            print(
                f"Tiny trial: rounds={args.rounds}, "
                f"last_committed={store.last_committed_round('m2-ballot-smoke-trial')}, "
                f"status={store.get_trial('m2-ballot-smoke-trial').status}, "
                f"completed={final.completed}"
            )
            store.close()
    except ModelProviderError as error:
        print(f"Ollama ballot smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
