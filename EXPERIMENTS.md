# Experiment Registry and Roadmap

## 1. Purpose

This registry separates apparatus tests, pilot exploration, and confirmatory research. An experiment ID denotes a frozen protocol version; a trial ID denotes one independently initialized replicate.

## 2. Status labels

- **Planned:** specification incomplete or not yet executed.
- **Apparatus:** validates software and data, not a behavioral claim.
- **Pilot:** estimates feasibility and variance; analyses are exploratory.
- **Confirmatory:** protocol, outcomes, exclusions, sample size, and analysis frozen in advance.
- **Superseded:** retained for provenance but not used as primary evidence.

## 3. Experiment matrix

| ID | Status | Provider | Conditions | Trials × rounds | Purpose |
|---|---|---|---|---|---|
| E00 | Apparatus | Deterministic mock | peer/objective/random | 3 fixtures × 3–10 | Validate state transitions, selection, logging, replay, resume, and export |
| E01 | Planned pilot | One small local model | peer/objective/random | 10 × 10 per condition | Measure schema compliance, runtime, score variance, and obvious confounds |
| E02 | Planned manipulation check | Same model | profile prompts plus neutral control | To be powered | Test whether prompt profiles induce stable, measurable differences |
| E03 | Planned confirmatory | Frozen model/version | peer/objective/random | Power-determined | Test the approved primary hypothesis |
| E04 | Planned robustness | Second model or quantization | Frozen E03 protocol | Precision-determined | Estimate model dependence |
| E05 | Future exploratory | Identity/history visibility factorial | Selected conditions | TBD | Separate reputation-capable behavior from anonymous evaluation |
| E06 | Future | Evolutionary replacement | Baselines required | TBD | Study inheritance and mutation under a new specification |

`10 trials × 10 rounds` is not assumed to be statistically adequate. `100+ trials × 20+ rounds` is a compute estimate, not a sample-size justification.

## 4. E00 — Deterministic apparatus validation

### Inputs

- Eight fixed profile fixtures.
- Deterministic mock task set and answer key.
- Scripted valid, tied, missing, and malformed ballots.
- A known selection sequence for every condition.

### Acceptance tests

- Identical config and seed reproduce normalized event data.
- Each selection is reconstructible from stored records.
- Atomic rollback and resume tests pass.
- Condition configs differ only in declared mechanism fields.
- Exports contain schema, config, prompt, task, provider, and code provenance.

### Interpretation

E00 can establish software correctness against fixtures. It cannot establish LLM social behavior.

## 5. E01 — Real-model pilot

### Provisional design

- One small quantized instruction model shared sequentially by all agents.
- Eight profiles, one fixed replacement queue, and three selection conditions.
- Ten fresh trial seeds per condition and ten rounds per trial.
- Deterministic objective tasks from multiple families.

### Questions

- Can the model reliably produce responses and ballots in the required schema?
- Are objective scores sufficiently variable?
- Is the identity/history policy adequate for the proposed construct?
- How costly and nondeterministic are runs on the target laptop or free notebook?
- Which metrics have enough variation for power simulation?

### Outputs

Failure-rate report, runtime and memory report, manipulation checks, descriptive plots, variance estimates, confound log, and proposed changes. Interesting transcripts are illustrative only.

### Stop/revise criteria

Revise before E03 if invalid ballots are frequent, objective scores have floor/ceiling effects, information exposure differs unintentionally, profile effects are absent or directively confounded, or results depend heavily on a few tasks or positions.

## 6. E02 — Profile manipulation check

Compare each parameterized profile with neutral and paraphrased controls on tasks that do not involve survival voting. Evaluate output-style and decision differences with frozen measurements. Reject or revise profile labels that are unreliable. This experiment tests prompt effects, not real personality.

## 7. E03 — Main confirmatory study

E03 may start only after:

- the literature gap is documented;
- one primary outcome and contrast are chosen;
- the round, ballot, identity, history, tie, invalid-output, and replacement rules are frozen;
- fresh tasks and seeds are reserved;
- simulation-based power or precision analysis sets trial count;
- analysis code passes on synthetic/blinded data;
- the protocol is time-stamped or preregistered.

The report must include all conditions, exclusions, null results, effect sizes, uncertainty, and deviations.

## 8. Configuration family

```text
configs/
  apparatus/
    e00_peer.yaml
    e00_objective.yaml
    e00_random.yaml
  pilot/
    e01_peer.yaml
    e01_objective.yaml
    e01_random.yaml
  confirmatory/
    e03_peer.yaml
    e03_objective.yaml
    e03_random.yaml
```

Matched configurations should be generated from one base config plus an explicit condition override, then materialized and hashed before execution.

## 9. Experiment record template

For every experiment, record:

- ID, title, status, owner, dates, and protocol version;
- research question and hypothesis class;
- provider/model/version/quantization;
- task set and scoring version;
- population, profile assignment, information policy, and replacement policy;
- conditions and randomization;
- trial/round count and justification;
- primary/secondary outcomes;
- exclusions, retry, missing-data, and stopping rules;
- analysis plan and multiplicity treatment;
- config, prompt, task, code, data, and environment hashes;
- deviations, failures, and interpretation limits.

## 10. Concise milestone roadmap

1. **M0 — Specification:** Approve the five documents, complete literature review, and freeze unresolved protocol decisions.
2. **M1 — Deterministic simulator:** Build config validation, typed domain records, mock provider, round engine, SQLite events, replay/resume, and tests.
3. **M2 — Local inference:** Add one local provider and complete performance/schema smoke tests on the target hardware.
4. **M3 — Pilot:** Run E01/E02, audit validity, estimate variance, and revise transparently.
5. **M4 — Main study:** Preregister E03, run power-determined fresh trials, and publish reproducible analysis.
6. **M5 — Robustness:** Repeat across another model or quantization and test moderator conditions.
7. **M6 — Evolution:** Only then specify inheritance, mutation, lineage, and new controls as a separate study.

## 11. First implementation task

After M0 approval, freeze unresolved protocol choices in `DECISIONS.md`, then implement the versioned configuration schema, semantic validator, canonical serializer, and stable configuration hash. Add fixtures showing that only the declared selection field differs among matched conditions.

## 12. Major methodological weaknesses to resolve

1. **Identity paradox:** Reciprocity requires stable recognition or history, but anonymity is needed to reduce reputation and label bias. This should become an explicit manipulation, not an accidental compromise.
2. **Selection/replacement confounding:** Condition effects can reflect who remains or enters rather than agents changing behavior.
3. **Peer votes conflate quality and social preference:** Analyses must account for objective response quality and shared judgments.
4. **Non-independent data:** Eight agents and many rounds do not equal many independent samples; trial-level replication is essential.
5. **Prompt-induced behavior:** “Personalities” may directly encode the measured outcome or fail to produce stable differences.
6. **Single-model generalization:** One 4-bit model supports only model-specific conclusions.
7. **Judge validity:** LLM-based scoring could reproduce model bias; deterministic tasks are preferable.
8. **Coalition metric flexibility:** Network thresholds and null models can easily be chosen after seeing results.
9. **Survival framing:** Explicit survival language may manufacture strategic behavior and limit ecological validity.
10. **Compute constraints:** Free/local compute may restrict sample size; the claim scope must shrink rather than treating nested observations as extra replication.
