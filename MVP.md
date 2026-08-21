# Minimum Viable Product

## 1. MVP purpose

The MVP proves that the experimental apparatus is deterministic, auditable, resumable, and capable of expressing the three selection conditions. It is not intended to demonstrate emergent social behavior or produce publishable findings.

## 2. Scope

### Population and execution

- Eight configured agent identities sharing one `MockModelProvider`.
- Distinct parameterized role prompts, labeled as prompt configurations rather than psychological personalities.
- One common task per round.
- Ten rounds per pilot trial by default; smaller fixtures in tests.
- Sequential execution with a deterministic event order.

### Conditions

- `peer_vote`, with a frozen ballot, eligibility, tie, and invalid-output rule.
- `objective`, using tasks with deterministic answer keys or scorers.
- `random`, using a recorded seeded generator.
- One common, non-evolutionary replacement policy across conditions, or a no-replacement fixed-horizon design if approved in the research gate.

### Infrastructure

- Validated YAML configuration.
- Agent, population, task, provider, evaluator, selection, replacement, and runner interfaces.
- Deterministic mock responses and ballots.
- SQLite event store with transactional round commits.
- Checkpoint/resume at completed-round boundaries.
- Structured logs and typed failure records.
- CSV and JSON export with provenance manifest.
- Basic descriptive metrics and test coverage.

## 3. Explicitly excluded

- Ollama, llama.cpp, Transformers, vLLM, or remote API integration.
- Evolution, inheritance, mutation, lineage, learned strategies, or generated offspring prompts.
- Private messaging, unrestricted memory, tools, or internet access.
- Multiple base models, parallel GPU execution, or distributed runs.
- Automated coalition labels or intention inference.
- Publication claims, preregistered main-study analysis, or a dashboard.

## 4. Required configuration fields

```yaml
experiment:
  name: peer_selection_smoke
  schema_version: 1
  seed: 42
  trials: 1
  rounds: 10

population:
  size: 8
  profiles_file: configs/profiles.yaml

model:
  provider: mock
  model: deterministic-v1
  temperature: 0

task:
  source: fixtures/tasks.json
  order: seeded

information:
  response_authorship_visible: false
  prior_votes_visible: false

selection:
  mechanism: peer_vote
  tie_break: seeded_random
  self_vote: forbidden
  invalid_ballot: abstain

replacement:
  mechanism: fixed_profile_pool

storage:
  sqlite_path: experiments/peer_selection_smoke.sqlite
```

Equivalent objective and random configs shall differ only in condition-specific fields.

## 5. Minimum metrics

- task accuracy/score by condition, agent, task family, and round;
- valid/invalid response and ballot counts;
- vote shares and normalized entropy per round;
- descriptive lag-one reciprocal-support rate;
- survival duration;
- response-length and exact-match diversity diagnostics.

These are apparatus checks. Confirmatory mixed-effects and temporal-network analyses are deferred to the research analysis milestone.

## 6. Test matrix

- Config validation and canonical hashing.
- Seed derivation and deterministic replay.
- Self-vote, duplicate-vote, ineligible-vote, missing-vote, and malformed-output handling.
- Peer, objective, and random selection with ties.
- Transaction rollback during a round.
- Resume after a completed round and after an interrupted round.
- Export reconstruction of selection decisions.
- Condition parity: matched configs expose identical information except the declared manipulation.

## 7. Definition of done

v0.1 is done when:

- [ ] The research acceptance gate in `RESEARCH.md` is approved.
- [ ] One command validates a config without running it.
- [ ] One command runs a deterministic mock experiment.
- [ ] All three conditions complete 8-agent, 10-round smoke trials.
- [ ] Re-running a normalized fixture with the same seed produces identical data.
- [ ] Every elimination can be reconstructed from stored inputs and the recorded rule.
- [ ] An interrupted run resumes without lost or duplicate rounds.
- [ ] CSV/JSON exports include a provenance manifest.
- [ ] Unit and integration tests pass on Windows and Linux.
- [ ] Documentation clearly separates descriptive metrics from research conclusions.
- [ ] No real-model, evolution, or private-communication feature is included.

## 8. First coding task after approval

Write a one-page `DECISIONS.md` freezing the round protocol, ballot semantics, tie/invalid policies, information exposure, and replacement rule. Then implement the typed configuration schema and validator with canonical serialization and config hashing. This is the first code task because every later component and every reproducibility guarantee depends on an unambiguous executable protocol.
