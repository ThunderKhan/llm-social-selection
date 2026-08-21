from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean, median
from typing import Any

from ..agents import AgentIdentity
from ..ballots import render_ballot_prompt
from ..domain import BallotEvidence
from ..tournament import RoundResult


LABELS = tuple("ABCDEFG")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
    }


def analyze_rounds(rounds: Sequence[RoundResult]) -> dict[str, Any]:
    response_total = 0
    response_non_empty = 0
    response_exact = 0
    response_extra_prose = 0
    ballot_total = 0
    ballot_valid = 0
    invalid_reasons: Counter[str] = Counter()
    supported_best = 0
    objective_comparable = 0
    chance_baselines: list[float] = []
    position_choices: Counter[str] = Counter()
    position_exposure: Counter[str] = Counter()
    supported_contents: Counter[str] = Counter()
    candidate_positions: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_lengths: list[int] = []
    supported_lengths: list[int] = []
    supported_length_ranks: list[int] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []
    response_latencies: list[float] = []
    ballot_latencies: list[float] = []
    response_tokens: list[int] = []
    ballot_tokens: list[int] = []
    mapping_issues: list[str] = []

    for result in rounds:
        response_by_id = {response.response_id: response for response in result.responses}
        response_by_agent = {response.agent_id: response for response in result.responses}
        score_by_agent = {score.agent_id: score.value for score in result.scores}
        expected = result.task.expected_answer
        expected_normalized = (
            " ".join(expected.split()).casefold() if expected is not None else None
        )
        for response in result.responses:
            response_total += 1
            if response.content.strip():
                response_non_empty += 1
            if response.latency_ms is not None:
                response_latencies.append(response.latency_ms)
            if response.token_count is not None:
                response_tokens.append(response.token_count)
            score = score_by_agent[response.agent_id]
            length = len(response.content)
            if score == 1.0:
                response_exact += 1
                correct_lengths.append(length)
            else:
                incorrect_lengths.append(length)
                normalized = " ".join(response.content.split()).casefold()
                if expected_normalized and expected_normalized in normalized:
                    response_extra_prose += 1

        ballot_by_id = {ballot.ballot_id: ballot for ballot in result.ballots}
        eligible_agents = set(response_by_agent)
        for evidence in result.ballot_evidence:
            ballot_total += 1
            if evidence.latency_ms is not None:
                ballot_latencies.append(evidence.latency_ms)
            if evidence.token_count is not None:
                ballot_tokens.append(evidence.token_count)
            labels = tuple(candidate.label for candidate in evidence.candidate_order)
            candidate_agents = {
                candidate.agent_id for candidate in evidence.candidate_order
            }
            candidate_response_ids = {
                candidate.response_id for candidate in evidence.candidate_order
            }
            if labels != LABELS:
                mapping_issues.append(f"{evidence.ballot_id}: labels are not A-G")
            if candidate_agents != eligible_agents - {evidence.voter_agent_id}:
                mapping_issues.append(f"{evidence.ballot_id}: candidate agent set mismatch")
            if not candidate_response_ids <= set(response_by_id):
                mapping_issues.append(f"{evidence.ballot_id}: unknown response reference")
            if evidence.voter_agent_id in candidate_agents:
                mapping_issues.append(f"{evidence.ballot_id}: self response exposed")

            candidate_by_label = {
                candidate.label: candidate for candidate in evidence.candidate_order
            }
            lengths = []
            for candidate in evidence.candidate_order:
                position_exposure[candidate.label] += 1
                candidate_positions[candidate.agent_id][candidate.label] += 1
                response = response_by_id[candidate.response_id]
                lengths.append(len(response.content))
                candidate_lengths.append(len(response.content))

            if not evidence.valid:
                invalid_reasons[evidence.invalid_reason or "unknown"] += 1
                continue
            ballot_valid += 1
            choice = evidence.parsed_choice
            if choice is None or choice not in candidate_by_label:
                mapping_issues.append(f"{evidence.ballot_id}: valid choice is unmapped")
                continue
            position_choices[choice] += 1
            chosen = candidate_by_label[choice]
            chosen_response = response_by_id[chosen.response_id]
            supported_contents[" ".join(chosen_response.content.split())] += 1
            chosen_length = len(chosen_response.content)
            supported_lengths.append(chosen_length)
            supported_length_ranks.append(
                1 + sum(length < chosen_length for length in lengths)
            )

            ballot = ballot_by_id[evidence.ballot_id]
            if ballot.supported_agent_id != chosen.agent_id:
                mapping_issues.append(f"{evidence.ballot_id}: parsed/final choice mismatch")
                continue
            if any(
                candidate.agent_id not in score_by_agent
                for candidate in evidence.candidate_order
            ):
                continue
            candidate_scores = [
                score_by_agent[candidate.agent_id]
                for candidate in evidence.candidate_order
            ]
            best_score = max(candidate_scores)
            best_count = sum(score == best_score for score in candidate_scores)
            chance_baselines.append(best_count / len(candidate_scores))
            objective_comparable += 1
            if score_by_agent[chosen.agent_id] == best_score:
                supported_best += 1

    candidate_position_concentration = [
        max(counts.values()) / sum(counts.values())
        for counts in candidate_positions.values()
        if sum(counts.values()) >= 2
    ]
    return {
        "rounds": len(rounds),
        "responses": {
            "total": response_total,
            "non_empty": response_non_empty,
            "non_empty_rate": _rate(response_non_empty, response_total),
            "exact_match": response_exact,
            "exact_match_rate": _rate(response_exact, response_total),
            "extra_prose": response_extra_prose,
        },
        "ballots": {
            "attempts": ballot_total,
            "valid": ballot_valid,
            "abstentions": ballot_total - ballot_valid,
            "valid_rate": _rate(ballot_valid, ballot_total),
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
        },
        "objective_agreement": {
            "comparable_ballots": objective_comparable,
            "supported_best_score": supported_best,
            "rate": _rate(supported_best, objective_comparable),
            "mean_tie_aware_chance_baseline": (
                mean(chance_baselines) if chance_baselines else None
            ),
        },
        "positions": {
            "supported_counts": {
                label: position_choices[label] for label in LABELS
            },
            "supported_rates": {
                label: _rate(position_choices[label], ballot_valid)
                for label in LABELS
            },
            "max_supported_position_share": (
                max(position_choices.values()) / ballot_valid
                if ballot_valid and position_choices
                else None
            ),
            "exposure_counts": {
                label: position_exposure[label] for label in LABELS
            },
            "max_candidate_position_share": (
                max(candidate_position_concentration)
                if candidate_position_concentration
                else None
            ),
        },
        "lengths": {
            "supported": _numeric_summary(supported_lengths),
            "candidate": _numeric_summary(candidate_lengths),
            "supported_length_rank": _numeric_summary(supported_length_ranks),
            "correct_responses": _numeric_summary(correct_lengths),
            "incorrect_responses": _numeric_summary(incorrect_lengths),
        },
        "content_vs_position": {
            "supported_content_counts": dict(
                sorted(supported_contents.items(), key=lambda item: (-item[1], item[0]))
            ),
            "unique_supported_contents": len(supported_contents),
            "max_supported_content_share": (
                max(supported_contents.values()) / ballot_valid
                if ballot_valid and supported_contents
                else None
            ),
            "max_supported_position_share": (
                max(position_choices.values()) / ballot_valid
                if ballot_valid and position_choices
                else None
            ),
        },
        "runtime": {
            "response_latency_ms": _numeric_summary(response_latencies),
            "ballot_latency_ms": _numeric_summary(ballot_latencies),
            "response_token_count": _numeric_summary(response_tokens),
            "ballot_token_count": _numeric_summary(ballot_tokens),
        },
        "candidate_mapping": {
            "valid": not mapping_issues,
            "issue_count": len(mapping_issues),
            "issues": mapping_issues,
        },
    }


def audit_identity_leakage(
    rounds: Sequence[RoundResult], agents: Sequence[AgentIdentity]
) -> dict[str, Any]:
    identifiers = {
        value
        for agent in agents
        for value in (agent.agent_id, agent.profile_id, agent.display_label)
        if value
    }
    protocol_terms = {
        "peer_vote",
        "objective selection",
        "random selection",
        "selection mechanism",
        "prior votes",
        "replacement history",
    }
    leaks: list[dict[str, str]] = []
    for result in rounds:
        for evidence in result.ballot_evidence:
            prompt = render_ballot_prompt(
                result.task, evidence.candidate_order, result.responses
            )
            for value in sorted(identifiers | protocol_terms):
                if value.casefold() in prompt.casefold():
                    leaks.append({"ballot_id": evidence.ballot_id, "value": value})
    return {"passed": not leaks, "leak_count": len(leaks), "leaks": leaks}


def analyze_repeat_display(evidence: Sequence[BallotEvidence]) -> dict[str, Any]:
    supported_agents: Counter[str] = Counter()
    selected_positions: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    raw_outputs: list[str] = []
    for item in evidence:
        raw_outputs.append(item.raw_output)
        if not item.valid or item.parsed_choice is None:
            invalid_reasons[item.invalid_reason or "unknown"] += 1
            continue
        candidate = next(
            candidate
            for candidate in item.candidate_order
            if candidate.label == item.parsed_choice
        )
        supported_agents[candidate.agent_id] += 1
        selected_positions[item.parsed_choice] += 1
    valid = sum(supported_agents.values())
    dominant_count = max(supported_agents.values(), default=0)
    dominant_position_count = max(selected_positions.values(), default=0)
    return {
        "attempts": len(evidence),
        "valid": valid,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "unique_supported_responses": len(supported_agents),
        "choice_consistency": _rate(dominant_count, valid),
        "response_consistency": _rate(dominant_count, valid),
        "max_selected_position_share": _rate(dominant_position_count, valid),
        "supported_agent_counts": dict(sorted(supported_agents.items())),
        "selected_position_counts": {
            label: selected_positions[label] for label in LABELS
        },
        "unique_raw_outputs": len(set(raw_outputs)),
        "raw_outputs": raw_outputs,
    }


def evaluate_sanity_gates(
    summary: dict[str, Any],
    *,
    identity_audit: dict[str, Any],
    persistence_mismatches: int,
    provider_failures: int,
    provider_requests: int,
    repeat_display: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    valid_rate = summary["ballots"]["valid_rate"] or 0.0
    provider_failure_rate = _rate(provider_failures, provider_requests) or 0.0
    if valid_rate < 0.95:
        failures.append("ballot validity is below 95%")
    if provider_failure_rate > 0.05:
        failures.append("provider failure/timeout rate exceeds 5%")
    if not identity_audit["passed"]:
        failures.append("model-visible ballot prompts contain internal identity metadata")
    if not summary["candidate_mapping"]["valid"]:
        failures.append("candidate mapping integrity failed")
    if persistence_mismatches:
        failures.append("persisted round reconstruction mismatched committed results")
    exact_match_rate = summary["responses"]["exact_match_rate"] or 0.0
    if exact_match_rate < 0.50:
        warnings.append("response exact-match rate is below 50%")
    chance_baseline = summary["objective_agreement"][
        "mean_tie_aware_chance_baseline"
    ]
    if chance_baseline == 1.0:
        warnings.append("objective agreement is uninformative because all candidates tie")
    elif chance_baseline is not None and chance_baseline >= 0.50:
        warnings.append("objective agreement has a tie-aware chance baseline of at least 50%")
    position_shares = [summary["positions"]["max_supported_position_share"]]
    position_shares.append(repeat_display.get("max_selected_position_share"))
    max_position_share = max(
        (share for share in position_shares if share is not None), default=None
    )
    if max_position_share is not None and max_position_share >= 0.80:
        failures.append("one anonymous display position received at least 80% of support")
    elif max_position_share is not None and max_position_share >= 0.60:
        warnings.append("one anonymous display position received 60% to 80% of support")
    reorder_valid_rate = _rate(repeat_display["valid"], repeat_display["attempts"]) or 0.0
    if reorder_valid_rate < 0.90:
        failures.append("controlled reorder ballot validity is below 90%")
    return {
        "recommendation": "REVISE BEFORE E01" if failures else "READY FOR E01",
        "failures": failures,
        "warnings": warnings,
        "ballot_validity_rate": valid_rate,
        "provider_failure_rate": provider_failure_rate,
        "controlled_reorder_validity_rate": reorder_valid_rate,
        "max_supported_position_share": max_position_share,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    gates = report["gates"]
    positions = summary["positions"]["supported_counts"]
    invalid = summary["ballots"]["invalid_reasons"]
    lines = [
        "# M2 Behavioral Sanity Check",
        "",
        "Engineering diagnostics only. No research claims are made from this report.",
        "",
        f"**Recommendation: {gates['recommendation']}**",
        "",
        "## Dataset",
        "",
        f"- Model: `{report['run']['model']}`",
        f"- Trials: {report['run']['trials']}",
        f"- Rounds per trial: {report['run']['rounds_per_trial']}",
        f"- Persisted rounds analyzed: {summary['rounds']}",
        "",
        "## Schema Compliance",
        "",
        f"- Responses: {summary['responses']['total']}",
        f"- Non-empty responses: {summary['responses']['non_empty']}",
        f"- Exact-match responses: {summary['responses']['exact_match']}",
        f"- Ballot attempts: {summary['ballots']['attempts']}",
        f"- Valid ballots: {summary['ballots']['valid']}",
        f"- Strict schema compliance rate: {summary['ballots']['valid_rate']}",
        f"- Abstentions: {summary['ballots']['abstentions']}",
        f"- Invalid reasons: `{invalid}`",
        "",
        "## Objective-Score Agreement",
        "",
        f"- Supported highest available score: {summary['objective_agreement']['rate']}",
        f"- Tie-aware chance baseline: {summary['objective_agreement']['mean_tie_aware_chance_baseline']}",
        "",
        "## Anonymous Position Distribution",
        "",
        *[f"- {label}: {positions[label]}" for label in LABELS],
        f"- Maximum selected-position share: {summary['positions']['max_supported_position_share']}",
        f"- Unique supported response contents: {summary['content_vs_position']['unique_supported_contents']}",
        f"- Maximum supported-content share: {summary['content_vs_position']['max_supported_content_share']}",
        "",
        "## Length And Runtime",
        "",
        f"- Supported response length: `{summary['lengths']['supported']}`",
        f"- Candidate response length: `{summary['lengths']['candidate']}`",
        f"- Correct response length: `{summary['lengths']['correct_responses']}`",
        f"- Incorrect response length: `{summary['lengths']['incorrect_responses']}`",
        f"- Response latency ms: `{summary['runtime']['response_latency_ms']}`",
        f"- Ballot latency ms: `{summary['runtime']['ballot_latency_ms']}`",
        "",
        "## Audits And Probes",
        "",
        f"- Identity leakage: {report['identity_audit']['passed']}",
        f"- Candidate mapping: {summary['candidate_mapping']['valid']}",
        f"- Persistence mismatches: {report['persistence']['mismatches']}",
        f"- Repeat-display choice consistency: {report['repeat_display']['choice_consistency']}",
        f"- Repeat-display validity: {report['repeat_display']['valid']}/{report['repeat_display']['attempts']}",
        f"- Repeat-display maximum position share: {report['repeat_display']['max_selected_position_share']}",
        f"- Native-schema ballot requests: {report['structured_output']['ballot_requests']}",
        f"- Fixed response prompt unique outputs: {report['nondeterminism']['response_unique_outputs']}",
        f"- Fixed ballot prompt unique outputs: {report['nondeterminism']['ballot_unique_outputs']}",
        "",
        "## Gate Reasons",
        "",
        f"- Failures: `{gates['failures']}`",
        f"- Warnings: `{gates['warnings']}`",
        "",
    ]
    return "\n".join(lines)
