from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import AgentIdentity, PromptProfile  # noqa: E402
from src.ballots import (  # noqa: E402
    BallotParseError,
    LLMBallotProvider,
    anonymous_candidates,
    ballot_response_schema,
    parse_ballot_choice,
    render_ballot_prompt,
)
from src.diagnostics import (  # noqa: E402
    analyze_repeat_display,
    analyze_rounds,
    audit_identity_leakage,
    evaluate_sanity_gates,
    render_markdown_report,
)
from src.domain import BallotEvidence, Response  # noqa: E402
from src.models import (  # noqa: E402
    ModelOutput,
    ModelProvider,
    ModelProviderError,
    OllamaProvider,
    OllamaTimeoutError,
)
from src.population import Population  # noqa: E402
from src.storage import SQLiteEventStore, collect_provenance  # noqa: E402
from src.tasks import Task  # noqa: E402
from src.tournament import FixedTaskSource, RoundEngine, TrialRunner  # noqa: E402


class CountingProvider(ModelProvider):
    def __init__(self, provider: OllamaProvider) -> None:
        self.provider = provider
        self.requests = Counter[str]()
        self.failures = Counter[str]()
        self.structured_requests = Counter[str]()

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name

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
        kind = (
            "ballot"
            if prompt.startswith("ANONYMOUS RESPONSE EVALUATION")
            else "response"
        )
        self.requests[kind] += 1
        if response_schema is not None:
            self.structured_requests[kind] += 1
        try:
            return self.provider.generate(
                agent=agent,
                task=task,
                prompt=prompt,
                request_id=request_id,
                seed=seed,
                response_schema=response_schema,
            )
        except OllamaTimeoutError:
            self.failures[f"{kind}_timeout"] += 1
            raise
        except ModelProviderError:
            self.failures[f"{kind}_provider"] += 1
            raise


def diagnostic_profiles() -> dict[str, PromptProfile]:
    return {
        f"m2-diagnostic-profile-{index:03d}": PromptProfile(
            profile_id=f"m2-diagnostic-profile-{index:03d}",
            parameters={"diagnostic_slot": index},
            template_version="m2-diagnostic-v1",
        )
        for index in range(1, 9)
    }


def diagnostic_tasks() -> tuple[Task, ...]:
    return (
        Task(
            "m2-diagnostic-arithmetic-1",
            "arithmetic",
            "What is 9 + 6? Return only the integer.",
            "15",
            "exact-match-v1",
        ),
        Task(
            "m2-diagnostic-format-1",
            "format",
            "Return only the uppercase word ORANGE.",
            "ORANGE",
            "exact-match-v1",
        ),
        Task(
            "m2-diagnostic-sequence-1",
            "sequence",
            "What lowercase letter comes immediately after c? Return only that letter.",
            "d",
            "exact-match-v1",
        ),
    )


def controlled_population() -> tuple[Population, dict[str, PromptProfile]]:
    agents = tuple(
        AgentIdentity(
            f"CONTROL_AGENT_{index}",
            f"CONTROL_PROFILE_{index}",
            f"Control Participant {index}",
            0,
        )
        for index in range(1, 9)
    )
    profiles = {
        agent.profile_id: PromptProfile(agent.profile_id, {}, "control-v1")
        for agent in agents
    }
    return Population(agents), profiles


def controlled_responses(population: Population, task: Task) -> tuple[Response, ...]:
    contents = (
        "VOTER RESPONSE",
        "4",
        "5",
        "The final answer is 4.",
        "3",
        "four",
        "2 + 2 = 4",
        "22",
    )
    return tuple(
        Response(
            response_id=f"control-response-{index}",
            trial_id="control-display-trial",
            round_index=0,
            task_id=task.task_id,
            agent_id=agent.agent_id,
            content=content,
            provider_name="fixture",
            model_name="fixed-content-v1",
        )
        for index, (agent, content) in enumerate(
            zip(population.agents, contents, strict=True), start=1
        )
    )


def run_repeat_display_probe(
    provider: CountingProvider,
    *,
    repeats: int,
) -> tuple[dict[str, object], list[BallotEvidence]]:
    population, _ = controlled_population()
    task = Task(
        "control-ballot-task",
        "arithmetic",
        "What is 2 + 2?",
        "4",
        "exact-match-v1",
    )
    responses = controlled_responses(population, task)
    voter = population.agents[0]
    evidence_rows: list[BallotEvidence] = []
    for index in range(repeats):
        candidates = anonymous_candidates(
            trial_seed=10_000 + index,
            round_index=0,
            voter_agent_id=voter.agent_id,
            population=population,
            responses=responses,
        )
        prompt = render_ballot_prompt(task, candidates, responses)
        try:
            output = provider.generate(
                agent=voter,
                task=task,
                prompt=prompt,
                request_id=f"control-display-{index:03d}",
                seed=4242,
                response_schema=ballot_response_schema(
                    tuple(candidate.label for candidate in candidates)
                ),
            )
        except ModelProviderError:
            continue
        choice = None
        invalid_reason = None
        try:
            choice = parse_ballot_choice(
                output.content, {candidate.label for candidate in candidates}
            )
        except BallotParseError as error:
            invalid_reason = error.reason
        evidence_rows.append(
            BallotEvidence(
                ballot_id=f"control-ballot-{index:03d}",
                trial_id="control-display-trial",
                round_index=0,
                task_id=task.task_id,
                voter_agent_id=voter.agent_id,
                provider_name=output.provider_name,
                model_name=output.model_name,
                request_id=output.request_id,
                seed=output.seed,
                raw_output=output.content,
                parsed_choice=choice,
                valid=choice is not None,
                invalid_reason=invalid_reason,
                candidate_order=candidates,
                latency_ms=output.latency_ms,
                token_count=output.token_count,
            )
        )
    analysis = analyze_repeat_display(evidence_rows)
    analysis["attempts"] = repeats
    analysis["provider_failures"] = repeats - len(evidence_rows)
    return analysis, evidence_rows


def run_nondeterminism_probe(
    provider: CountingProvider,
    *,
    repeats: int,
) -> dict[str, object]:
    population, _ = controlled_population()
    task = Task(
        "control-repeatability-task",
        "arithmetic",
        "Return only the final answer to 2 + 2.",
        "4",
        "exact-match-v1",
    )
    responses = controlled_responses(population, task)
    voter = population.agents[0]
    candidates = anonymous_candidates(
        trial_seed=42,
        round_index=0,
        voter_agent_id=voter.agent_id,
        population=population,
        responses=responses,
    )
    ballot_prompt = render_ballot_prompt(task, candidates, responses)
    response_prompt = "Return only the final answer to 2 + 2."
    response_outputs: list[str] = []
    ballot_outputs: list[str] = []
    for _ in range(repeats):
        try:
            response_outputs.append(
                provider.generate(
                    agent=voter,
                    task=task,
                    prompt=response_prompt,
                    request_id="fixed-response-repeatability",
                    seed=42,
                ).content
            )
            ballot_outputs.append(
                provider.generate(
                    agent=voter,
                    task=task,
                    prompt=ballot_prompt,
                    request_id="fixed-ballot-repeatability",
                    seed=42,
                    response_schema=ballot_response_schema(
                        tuple(candidate.label for candidate in candidates)
                    ),
                ).content
            )
        except ModelProviderError:
            continue
    return {
        "requested_repeats": repeats,
        "response_completed": len(response_outputs),
        "response_unique_outputs": len(set(response_outputs)),
        "response_outputs": response_outputs,
        "ballot_completed": len(ballot_outputs),
        "ballot_unique_outputs": len(set(ballot_outputs)),
        "ballot_outputs": ballot_outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run descriptive M2 anonymous-ballot engineering diagnostics."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--repeat-displays", type=int, default=14)
    parser.add_argument("--repeatability", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "diagnostics"
    )
    args = parser.parse_args()
    if args.trials <= 0 or args.rounds <= 0:
        parser.error("--trials and --rounds must be positive")
    if args.repeat_displays < 14:
        parser.error("--repeat-displays must be at least 14")
    if args.repeatability <= 0:
        parser.error("--repeatability must be positive")

    started = perf_counter()
    provider = OllamaProvider(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=0,
        num_predict=32,
    )
    try:
        ollama_version = provider.check_health()
        provider.ensure_model_available()
    except ModelProviderError as error:
        print(f"M2 sanity check failed before execution: {error}", file=sys.stderr)
        return 1
    counted = CountingProvider(provider)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"m2_sanity_{timestamp}"
    database_path = args.output_dir / f"{stem}.sqlite"
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    profile_pool = diagnostic_profiles()
    tasks = diagnostic_tasks()
    config_json = json.dumps(
        {
            "ballot": {
                "candidate_order": "voter-seeded-v1",
                "invalid_policy": "abstain",
                "output_format": "json-choice-v1",
                "structured_output": "ollama-json-schema-v1",
                "provider": "llm",
            },
            "condition": "random",
            "model": {
                "base_url": provider.base_url,
                "model": provider.model_name,
                "num_predict": provider.num_predict,
                "provider": provider.provider_name,
                "temperature": provider.temperature,
                "timeout_seconds": provider.timeout_seconds,
            },
            "name": "M2 behavioral sanity check",
            "rounds": args.rounds,
            "trials": args.trials,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    store = SQLiteEventStore(database_path)
    store.initialize()
    experiment_id = f"m2-sanity-{timestamp}"
    store.create_experiment(
        experiment_id=experiment_id,
        name="M2 behavioral sanity check",
        config_schema_version=1,
        config_hash=config_hash,
        config_json=config_json,
        provenance=collect_provenance(
            provider_name=provider.provider_name,
            model_name=provider.model_name,
            repository=ROOT,
        ),
    )

    persistence_mismatches = 0
    trial_ids: list[str] = []
    execution_errors: list[str] = []
    for trial_number in range(args.trials):
        trial_id = f"m2-sanity-trial-{trial_number + 1:03d}"
        trial_ids.append(trial_id)
        trial_runner = TrialRunner(
            experiment_id=experiment_id,
            trial_id=trial_id,
            trial_seed=42 + trial_number,
            condition="random",
            total_rounds=args.rounds,
            config_hash=config_hash,
            profiles=profile_pool,
            task_source=FixedTaskSource(tasks),
            provider=counted,
            event_store=store,
            round_engine=RoundEngine(ballot_provider=LLMBallotProvider()),
        )
        try:
            state = trial_runner.initialize()
            while not state.completed:
                step = trial_runner.run_next_round()
                if store.load_round(trial_id, step.result.round_index) != step.result:
                    persistence_mismatches += 1
                state = step.state
        except ModelProviderError as error:
            execution_errors.append(f"{trial_id}: {error}")

    persisted_rounds = []
    all_agents: list[AgentIdentity] = []
    for trial_id in trial_ids:
        all_agents.extend(store.load_agents(trial_id))
        last = store.last_committed_round(trial_id)
        if last is not None:
            persisted_rounds.extend(
                store.load_round(trial_id, round_index)
                for round_index in range(last + 1)
            )

    summary = analyze_rounds(persisted_rounds)
    identity_audit = audit_identity_leakage(persisted_rounds, all_agents)
    repeat_display, repeat_evidence = run_repeat_display_probe(
        counted, repeats=args.repeat_displays
    )
    nondeterminism = run_nondeterminism_probe(
        counted, repeats=args.repeatability
    )
    provider_failures = sum(counted.failures.values())
    provider_requests = sum(counted.requests.values())
    gates = evaluate_sanity_gates(
        summary,
        identity_audit=identity_audit,
        persistence_mismatches=persistence_mismatches,
        provider_failures=provider_failures,
        provider_requests=provider_requests,
        repeat_display=repeat_display,
    )
    duration = perf_counter() - started
    report = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ollama_version": ollama_version,
            "model": provider.model_name,
            "trials": args.trials,
            "rounds_per_trial": args.rounds,
            "condition": "random",
            "duration_seconds": duration,
            "database": str(database_path),
            "provider_requests": dict(sorted(counted.requests.items())),
            "provider_failures": dict(sorted(counted.failures.items())),
            "execution_errors": execution_errors,
        },
        "summary": summary,
        "identity_audit": identity_audit,
        "persistence": {"mismatches": persistence_mismatches},
        "repeat_display": repeat_display,
        "structured_output": {
            "mode": "ollama-json-schema-v1",
            "ballot_requests": counted.structured_requests["ballot"],
            "all_requests": dict(sorted(counted.structured_requests.items())),
            "main_strict_valid": summary["ballots"]["valid"],
            "main_attempts": summary["ballots"]["attempts"],
        },
        "repeat_display_evidence": [
            {
                "raw_output": item.raw_output,
                "parsed_choice": item.parsed_choice,
                "valid": item.valid,
                "invalid_reason": item.invalid_reason,
                "candidate_order": [
                    {
                        "label": candidate.label,
                        "agent_id": candidate.agent_id,
                        "response_id": candidate.response_id,
                    }
                    for candidate in item.candidate_order
                ],
            }
            for item in repeat_evidence
        ],
        "nondeterminism": nondeterminism,
        "gates": gates,
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    store.close()

    print("M2 Behavioral Sanity Check")
    print(f"Model: {provider.model_name} (Ollama {ollama_version})")
    print(f"Trials: {args.trials}; rounds/trial: {args.rounds}")
    print(
        f"Responses: {summary['responses']['total']} generated, "
        f"{summary['responses']['exact_match']} exact-match, "
        f"{summary['responses']['extra_prose']} with answer plus extra prose"
    )
    print(
        f"Ballots: {summary['ballots']['valid']}/{summary['ballots']['attempts']} valid, "
        f"abstentions={summary['ballots']['abstentions']}"
    )
    print(
        "Objective-score agreement: "
        f"{summary['objective_agreement']['rate']} "
        f"(tie-aware chance={summary['objective_agreement']['mean_tie_aware_chance_baseline']})"
    )
    print(f"Position support: {summary['positions']['supported_counts']}")
    print(
        "Content/position concentration: "
        f"content={summary['content_vs_position']['max_supported_content_share']}, "
        f"position={summary['positions']['max_supported_position_share']}"
    )
    print(f"Identity leakage audit: {identity_audit['passed']}")
    print(f"Candidate mapping audit: {summary['candidate_mapping']['valid']}")
    print(f"Persistence mismatches: {persistence_mismatches}")
    print(
        f"Repeat-display consistency: {repeat_display['choice_consistency']}; "
        f"unique supported={repeat_display['unique_supported_responses']}"
    )
    print(
        f"Nondeterminism probe: responses unique={nondeterminism['response_unique_outputs']}, "
        f"ballots unique={nondeterminism['ballot_unique_outputs']}"
    )
    print(f"Duration: {duration:.2f}s; provider failures={provider_failures}")
    print(f"Report: {json_path}")
    print(f"Recommendation: {gates['recommendation']}")
    if gates["failures"]:
        print(f"Failure reasons: {gates['failures']}")
    if gates["warnings"]:
        print(f"Warnings: {gates['warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
