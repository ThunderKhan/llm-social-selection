from __future__ import annotations

from dataclasses import replace

from src.agents import AgentIdentity
from src.diagnostics import (
    analyze_repeat_display,
    analyze_rounds,
    audit_identity_leakage,
    evaluate_sanity_gates,
    render_markdown_report,
)
from src.domain import (
    Ballot,
    BallotCandidate,
    BallotEvidence,
    Response,
    Score,
    SelectionEvent,
)
from src.tasks import Task
from src.tournament import RoundResult


def agents() -> tuple[AgentIdentity, ...]:
    return tuple(
        AgentIdentity(
            f"SECRET_AGENT_{index}",
            f"PROFILE_DO_NOT_LEAK_{index}",
            f"Hidden Display {index}",
            0,
        )
        for index in range(1, 9)
    )


def diagnostic_round(
    *, invalid_count: int = 0, leak: bool = False, corrupt_mapping: bool = False
) -> RoundResult:
    identities = agents()
    task = Task("task-001", "arithmetic", "What is 2 + 2?", "4", "exact-match-v1")
    contents = ["4", "4", "5", "The answer is 4.", "3", "four", "0", "22"]
    if leak:
        contents[2] = "SECRET_AGENT_1"
    responses = tuple(
        Response(
            response_id=f"response-{index}",
            trial_id="trial-001",
            round_index=0,
            task_id=task.task_id,
            agent_id=identity.agent_id,
            content=content,
            provider_name="ollama",
            model_name="qwen3:0.6b",
            latency_ms=10.0 + index,
            token_count=2,
        )
        for index, (identity, content) in enumerate(
            zip(identities, contents, strict=True), start=1
        )
    )
    scores = tuple(
        Score(
            score_id=f"score-{index}",
            trial_id="trial-001",
            round_index=0,
            task_id=task.task_id,
            agent_id=identity.agent_id,
            value=1.0 if index <= 2 else 0.0,
            scorer_version="exact-match-v1",
        )
        for index, identity in enumerate(identities, start=1)
    )
    ballots = []
    evidence = []
    for voter_index, voter in enumerate(identities):
        candidates = [identity for identity in identities if identity != voter]
        candidate_order = tuple(
            BallotCandidate(
                label=chr(ord("A") + index),
                agent_id=(
                    "UNKNOWN_AGENT"
                    if corrupt_mapping and voter_index == 0 and index == 6
                    else candidate.agent_id
                ),
                response_id=f"response-{identities.index(candidate) + 1}",
            )
            for index, candidate in enumerate(candidates)
        )
        valid = voter_index >= invalid_count
        choice = "A" if valid else None
        supported = candidate_order[0].agent_id if valid else None
        ballot_id = f"ballot-{voter_index + 1}"
        ballots.append(
            Ballot(
                ballot_id,
                "trial-001",
                0,
                voter.agent_id,
                supported,
            )
        )
        evidence.append(
            BallotEvidence(
                ballot_id=ballot_id,
                trial_id="trial-001",
                round_index=0,
                task_id=task.task_id,
                voter_agent_id=voter.agent_id,
                provider_name="ollama",
                model_name="qwen3:0.6b",
                request_id=f"request-{voter_index + 1}",
                seed=42,
                raw_output='{"choice":"A"}' if valid else "not json",
                parsed_choice=choice,
                valid=valid,
                invalid_reason=None if valid else "invalid_json",
                candidate_order=candidate_order,
                latency_ms=20.0 + voter_index,
                token_count=5,
            )
        )
    return RoundResult(
        trial_id="trial-001",
        round_index=0,
        task=task,
        responses=responses,
        scores=scores,
        ballots=tuple(ballots),
        selection=SelectionEvent(
            "selection-001", "trial-001", 0, "random", identities[3].agent_id, "test"
        ),
        ballot_evidence=tuple(evidence),
    )


def test_analyze_rounds_computes_compliance_exposure_and_agreement() -> None:
    summary = analyze_rounds((diagnostic_round(invalid_count=1),))

    assert summary["responses"]["total"] == 8
    assert summary["responses"]["exact_match"] == 2
    assert summary["responses"]["extra_prose"] == 1
    assert summary["ballots"]["attempts"] == 8
    assert summary["ballots"]["valid"] == 7
    assert summary["ballots"]["invalid_reasons"] == {"invalid_json": 1}
    assert summary["objective_agreement"]["supported_best_score"] == 7
    assert summary["positions"]["supported_counts"]["A"] == 7
    assert set(summary["positions"]["exposure_counts"].values()) == {8}
    assert summary["lengths"]["supported"]["count"] == 7
    assert summary["lengths"]["candidate"]["count"] == 56
    assert summary["runtime"]["response_latency_ms"]["count"] == 8
    assert summary["runtime"]["ballot_latency_ms"]["count"] == 8
    assert summary["candidate_mapping"]["valid"] is True


def test_candidate_mapping_audit_detects_unknown_candidate() -> None:
    summary = analyze_rounds((diagnostic_round(corrupt_mapping=True),))

    assert summary["candidate_mapping"]["valid"] is False
    assert summary["candidate_mapping"]["issue_count"] > 0


def test_identity_leakage_audit_passes_clean_prompt_and_detects_content_leak() -> None:
    clean = audit_identity_leakage((diagnostic_round(),), agents())
    leaked = audit_identity_leakage((diagnostic_round(leak=True),), agents())

    assert clean["passed"] is True
    assert leaked["passed"] is False
    assert any(item["value"] == "SECRET_AGENT_1" for item in leaked["leaks"])


def test_repeat_display_analysis_tracks_internal_choice_not_label() -> None:
    base = diagnostic_round().ballot_evidence[0]
    rows = []
    for index, label in enumerate(("A", "C", "G")):
        candidates = list(base.candidate_order)
        target_index = next(
            position
            for position, candidate in enumerate(candidates)
            if candidate.agent_id == "SECRET_AGENT_2"
        )
        label_index = ord(label) - ord("A")
        candidates[target_index], candidates[label_index] = (
            candidates[label_index],
            candidates[target_index],
        )
        candidates = [
            BallotCandidate(
                chr(ord("A") + position),
                candidate.agent_id,
                candidate.response_id,
            )
            for position, candidate in enumerate(candidates)
        ]
        rows.append(
            replace(
                base,
                ballot_id=f"repeat-{index}",
                request_id=f"repeat-{index}",
                parsed_choice=label,
                candidate_order=tuple(candidates),
            )
        )

    result = analyze_repeat_display(rows)

    assert result["valid"] == 3
    assert result["unique_supported_responses"] == 1
    assert result["choice_consistency"] == 1.0
    assert result["selected_position_counts"] == {
        "A": 1,
        "B": 0,
        "C": 1,
        "D": 0,
        "E": 0,
        "F": 0,
        "G": 1,
    }


def test_sanity_gates_apply_explicit_failure_rules() -> None:
    ready_summary = analyze_rounds((diagnostic_round(),))
    ready_summary["positions"]["supported_counts"] = {
        label: 1 if label != "G" else 2 for label in "ABCDEFG"
    }
    ready_summary["positions"]["supported_rates"] = {
        label: count / 8
        for label, count in ready_summary["positions"]["supported_counts"].items()
    }
    ready = evaluate_sanity_gates(
        ready_summary,
        identity_audit={"passed": True},
        persistence_mismatches=0,
        provider_failures=0,
        provider_requests=100,
    )
    revise = evaluate_sanity_gates(
        analyze_rounds((diagnostic_round(invalid_count=2),)),
        identity_audit={"passed": False},
        persistence_mismatches=1,
        provider_failures=6,
        provider_requests=100,
    )

    assert ready["recommendation"] == "READY FOR E01"
    assert revise["recommendation"] == "REVISE BEFORE E01"
    assert len(revise["failures"]) == 4


def test_markdown_report_contains_diagnostic_sections() -> None:
    summary = analyze_rounds((diagnostic_round(),))
    gates = evaluate_sanity_gates(
        summary,
        identity_audit={"passed": True},
        persistence_mismatches=0,
        provider_failures=0,
        provider_requests=16,
    )
    report = {
        "run": {"model": "qwen3:0.6b", "trials": 1, "rounds_per_trial": 1},
        "summary": summary,
        "identity_audit": {"passed": True},
        "persistence": {"mismatches": 0},
        "repeat_display": {"choice_consistency": 1.0},
        "nondeterminism": {
            "response_unique_outputs": 1,
            "ballot_unique_outputs": 1,
        },
        "gates": gates,
    }

    markdown = render_markdown_report(report)

    assert "# M2 Behavioral Sanity Check" in markdown
    assert "Objective-Score Agreement" in markdown
    assert "Anonymous Position Distribution" in markdown
    assert "REVISE BEFORE E01" in markdown
