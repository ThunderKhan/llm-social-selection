# Experimental Protocol Decisions

**Project:** LLM Social Selection  
**Protocol version:** `v0.1`  
**Status:** Accepted for deterministic-simulator implementation  
**Scope:** M1 apparatus and M2 pilot preparation  

## 1. Purpose

This document freezes the behavioral protocol needed to implement the deterministic simulator. It resolves ambiguities left intentionally open in `RESEARCH.md`, `PRD.md`, `MVP.md`, `METHODOLOGY.md`, and `EXPERIMENTS.md`.

These decisions are not permanent scientific truths. They are the explicit rules for protocol `v0.1`. Any later change must create a new protocol version and be recorded in the change log; completed data from different protocol versions must never be silently combined.

## 2. Frozen decisions at a glance

| Area | v0.1 decision |
|---|---|
| Population | Eight active agents at the beginning of every round |
| Underlying model | One provider/model shared by all agents in a trial |
| Agent profiles | Eight distinct parameterized prompt configurations |
| Task | All active agents receive the same task in a round |
| Ballot meaning | Vote for the best response—the response the voter recommends keeping |
| Ballots collected | One ballot from every valid voter in every condition |
| Self-voting | Forbidden |
| Peer elimination | Agent with the fewest valid support votes is eliminated |
| Objective elimination | Agent with the lowest external task score is eliminated |
| Random elimination | One active agent is sampled uniformly |
| Tie-breaking | Seeded random choice among tied eligible agents |
| Population size | Held constant at eight through non-evolutionary replacement |
| Replacement | Next profile instance from a preregistered, seeded replacement queue |
| Identity during voting | Stable pseudonymous agent ID is visible with each response |
| Previous ballot history | Full directed ballot history from prior rounds is public |
| Previous response text | Not included in later-round context |
| Private communication | Prohibited |
| Agent memory | Only the experiment-provided public history; no private persistent memory |
| Evolution | Excluded from v0.1 |
| Primary pilot purpose | Validate apparatus and estimate feasibility—not test confirmatory claims |

## 3. Population and agent identity

1. Each trial begins with exactly eight active agent instances.
2. Every agent receives:
   - a trial-unique immutable internal ID;
   - a stable public pseudonym such as `Agent A`, `Agent B`, and so on;
   - one versioned parameterized prompt profile;
   - access to the same underlying model provider and inference settings.
3. Public pseudonyms are randomly assigned to profile instances using a recorded child seed.
4. Profile parameter values and descriptive labels are hidden from other agents.
5. An agent is a prompt-and-state configuration, not a separate model or psychological person.
6. A replacement receives a new immutable internal ID and a previously unused public pseudonym. IDs are never recycled within a trial.

### Rationale

Stable visible identities are necessary to make reciprocal and reputation-sensitive voting possible. This introduces identity and reputation effects intentionally. A later experiment will compare this protocol with anonymous authorship; anonymity is not mixed into the v0.1 design.

## 4. Experimental conditions

Every trial uses exactly one selection condition:

### 4.1 Peer selection — `peer_vote`

All agents vote for the response they consider best. The agent receiving the fewest valid support votes is eliminated. A ballot is therefore a positive support decision, not a vote to eliminate.

### 4.2 Objective selection — `objective`

All responses are evaluated by the frozen external scorer. The agent receiving the lowest objective score is eliminated. Peer ballots are still collected and recorded but cannot affect elimination.

### 4.3 Random selection — `random`

One active agent is sampled uniformly from all eight agents. Peer ballots and objective scores are still collected and recorded but cannot affect elimination.

### 4.4 Condition disclosure

Agents are told which rule determines elimination in their current trial. They are also told that peer ballots are recorded in all conditions. The exact condition-description text must be versioned and identical across agents within a condition.

### Rationale

Collecting ballots and objective scores in all three conditions produces comparable measurements. Only the causal selection rule changes. Disclosing the rule allows selection pressure to affect behavior, which is the phenomenon under study; it also means conclusions apply specifically to systems where agents know the rule.

## 5. Task and response protocol

1. One task is selected per round using a recorded seeded task order.
2. All eight active agents receive the same task text and task metadata permitted by the protocol.
3. Tasks must have stable IDs, versions, provenance, and deterministic answer keys or scoring functions.
4. Agents answer independently and cannot inspect current-round answers before submitting their own.
5. A response contains:
   - the final answer in a schema-constrained field;
   - an optional bounded explanation;
   - no ballot or evaluation of other agents.
6. Temperature and all sampling parameters are identical across agents in matched trials.
7. Response generation is sequential in the MVP, but no current-round output is added to another agent's response context.

## 6. Evaluation and ballot protocol

### 6.1 Objective scoring

1. Every response in every condition is scored before ballots are collected.
2. The external scorer uses only the task, answer key, and submitted response.
3. The scorer does not receive the trial condition, public agent identity, profile, ballot history, or elimination outcome.
4. The raw score, normalized score, scorer version, and scoring failure are recorded.
5. Objective scores are not shown to agents before they vote.

### 6.2 Ballot presentation

1. After all responses are submitted, every active agent receives the eight current responses.
2. Each response is labeled with its author's stable public pseudonym.
3. Response display order is independently shuffled for each voter using a recorded child seed.
4. The voter is instructed to select the single best response based on correctness, reasoning quality, and relevance.
5. The voter cannot edit its original response after viewing the candidates.

### 6.3 Ballot validity

A valid ballot:

- selects exactly one active candidate;
- does not select the voter;
- uses the required schema;
- refers to a candidate eligible in the current round.

Self-votes, multiple selections, unknown IDs, malformed output, missing ballots, and ballots produced after retry exhaustion are invalid. Invalid ballots become abstentions; they are never guessed or repaired semantically.

### 6.4 Retries

One schema-repair retry is allowed for malformed responses or ballots. The retry prompt may explain the formatting error but must not suggest a substantive answer or candidate. Both attempts are stored. A failed retry follows the declared invalid-output rule.

## 7. Elimination rules

### 7.1 Peer condition

Count valid incoming support votes for every active agent. Eliminate the agent with the smallest count.

### 7.2 Objective condition

Eliminate the agent with the smallest valid normalized objective score. A scoring failure is assigned the frozen minimum score for selection purposes and separately recorded as a failure.

### 7.3 Random condition

Sample uniformly from all eight active agents using the condition-specific selection seed. Ballots, scores, profile, and prior survival do not affect the draw.

### 7.4 Ties

All elimination ties are resolved by seeded uniform random choice among tied agents. The candidate set, seed reference, and selected result are recorded. Alphabetical order, execution order, and database order must never act as implicit tie-breakers.

### 7.5 Complete ballot failure

If the peer condition receives zero valid ballots, elimination falls back to seeded uniform random selection among all active agents. The round is marked `peer_ballot_failure`, and the fallback is visible in exports. A high fallback rate is a pilot stop/revise signal.

## 8. Information and memory policy

Before producing a response in round `t`, each active agent receives:

- its own stable public pseudonym;
- the current condition and elimination rule;
- the current task;
- the complete public event history from rounds `1` through `t−1`, containing:
  - directed valid ballots and abstentions;
  - the eliminated agent;
  - the rule and tie/fallback status used;
  - the replacement agent's public pseudonym.

Agents do not receive:

- previous response text or explanations;
- hidden profile parameters or profile labels;
- objective scores from prior rounds;
- internal prompts belonging to other agents;
- private messages;
- unrecorded scratch memory;
- tools, web access, or external state.

The history is serialized from structured events using a versioned deterministic template. With only ten rounds, the full history is retained rather than summarized. If context limits later require truncation, that change creates a new protocol version.

### Rationale

Public directed ballot history makes lagged reciprocity and persistent voting patterns behaviorally possible. Excluding previous response text reduces context growth and prevents imitation of earlier wording from being mistaken for conformity.

## 9. Round lifecycle

Each round follows this exact order:

1. Verify that eight active agents exist.
2. Derive and record all round-specific child seeds.
3. Select the common task.
4. Construct each agent's response context.
5. Generate and persist all independent responses.
6. Score every response objectively.
7. Construct each voter's candidate list and seeded display order.
8. Collect and validate one ballot per active agent.
9. Apply only the configured condition's elimination rule.
10. Persist the complete selection decision and its reconstruction data.
11. Reveal ballots, abstentions, eliminated agent, and replacement identity as public history.
12. Insert one replacement agent from the fixed queue.
13. Atomically mark the round complete.

If the process fails before step 13, the round is incomplete and must be safely retried from its checkpoint without duplicating committed events.

## 10. Replacement policy

1. Population size remains eight throughout a trial.
2. Before the trial begins, create a seeded replacement queue from a frozen pool of profile instances.
3. The queue is independent of condition outcomes and saved in the trial manifest.
4. Matched trials across conditions use the same initial profile assignment and replacement queue.
5. At the end of each round, the next unused queue entry becomes active.
6. A replacement begins with no private memory and no inherited traits.
7. The replacement receives the same public history available to all other agents in the next round.
8. Replacement profiles may repeat a configuration, but each profile instance and public identity is unique.
9. No survivor is copied, mutated, or treated as a parent.

### Known limitation

Because different agents may be eliminated under different conditions, the same queue position can enter different population compositions. This is an unavoidable part of the selected fixed-size design and must be considered in analysis.

## 11. Randomness and matched trials

One root trial seed deterministically derives separate namespaced child seeds for:

- profile-to-pseudonym assignment;
- task order;
- provider sampling per agent and round;
- response display order per voter and round;
- peer tie-breaking;
- objective tie-breaking;
- random-condition elimination;
- replacement-queue construction.

Changing one stochastic component must not shift unrelated random sequences. Matched conditions share root seeds and all applicable child-seed inputs, while condition-specific selection randomness remains namespaced.

## 12. Pilot measurements

The v0.1 pilot records and reports:

- objective score and accuracy;
- valid-response, valid-ballot, abstention, retry, and failure rates;
- incoming support counts;
- normalized vote entropy;
- descriptive lag-one reciprocal-support rate;
- survival duration;
- response-length and exact-match diversity diagnostics;
- runtime, memory use, and inference metadata.

These are descriptive and exploratory during the pilot. Coalition detection, semantic-embedding conformity, mixed-effects inference, and publication claims remain outside the MVP implementation.

## 13. Data and audit requirements

Every elimination must be reconstructible from stored raw inputs. At minimum, persist:

- experiment, protocol, trial, round, task, agent, and request IDs;
- root seed and relevant child-seed references;
- exact configuration and its canonical hash;
- prompt-template and profile hashes;
- raw and parsed responses and ballots;
- candidate display order for each voter;
- scores and scorer version;
- ballot validation results;
- tie set, fallback status, and final elimination;
- replacement queue position and activated identity;
- model/provider/version and sampling parameters;
- software version or Git commit;
- retry and failure events.

Raw records are append-oriented. Corrected, parsed, redacted, and derived data remain linked to—not substituted for—the originals.

## 14. Pilot stop/revise criteria

The protocol must be revised before a main study if any of the following occur frequently enough to threaten interpretation:

- malformed responses or ballots;
- peer rounds requiring complete-ballot fallback;
- objective-score floor or ceiling effects;
- agents voting primarily by fixed identity rather than current response quality;
- replacement churn overwhelming within-agent observations;
- context limits truncating public history;
- unintended differences between matched condition contexts;
- results driven by a few tasks, positions, profiles, or seeds.

Thresholds for “frequently enough” must be set before the real-model pilot is evaluated.

## 15. Explicitly deferred decisions

The following are not part of protocol `v0.1` and require separate specifications:

- anonymous-authorship and hidden-history factorial conditions;
- multiple base models or quantization comparisons;
- LLM-as-judge scoring;
- semantic embedding selection and coalition thresholds;
- private communication;
- long-term summarized memory;
- inheritance, mutation, evolution, and lineage;
- main-study sample size and confirmatory statistical model.

## 16. Change-control rule

After the first real-model pilot begins, changes to ballot meaning, visibility, history, retry policy, elimination, scoring, replacement, or randomness require:

1. a new protocol version;
2. a dated change-log entry with rationale;
3. new configuration hashes;
4. fresh pilot runs where validity may be affected;
5. separate reporting of data generated under each protocol.

## 17. Implementation gate

This document authorizes implementation of the deterministic configuration and simulator layers only. The next task is to implement:

1. a typed, versioned configuration schema;
2. semantic cross-field validation;
3. canonical serialization and stable hashing;
4. three matched fixture configurations differing only where condition logic requires;
5. tests for invalid combinations and deterministic seed derivation.

Real-model inference begins only after deterministic E00 apparatus tests pass.
