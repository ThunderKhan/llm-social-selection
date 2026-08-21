# LLM Social Selection: Research Specification

**Status:** Provisional v0.1 specification  
**Study type:** Controlled, repeated-measures simulation study  
**Primary unit of analysis:** Trial-level population trajectory; agent-round observations are nested data

## 1. Project overview

This project studies whether the rule used to remove agents from a fixed population changes observable voting patterns, response quality, and population-level behavior. Eight agents share one underlying language model but receive distinct, parameterized role prompts. In each round they answer the same task, evaluate anonymized responses where possible, and face one of three elimination rules: peer vote, objective score, or random selection.

The first version is a fixed-population experiment framework. It does not include inheritance, mutation, private communication, long-term autobiographical memory, or claims about psychological states.

## 2. Motivation

Multi-agent systems increasingly use peer review, ranking, and selection. These mechanisms may improve output, but they may also reward conformity, reciprocal support, or socially advantageous behavior at the expense of task performance. Understanding this tradeoff matters both scientifically and for the design of agent ensembles.

The project is research-first: complete structured data, preregistered confirmatory analyses, deterministic software tests, and conservative interpretation take priority over dramatic demonstrations.

## 3. Research questions

### Primary question

How does peer-based survival selection affect observable social behavior and objective task performance in populations of LLM agents, compared with objective-performance selection and random-selection baselines?

### Secondary questions

1. Does peer selection increase lagged reciprocal voting relative to the two baselines?
2. Does peer selection increase vote concentration or persistent voting clusters?
3. Does peer selection reduce response diversity or prompt-trait diversity over time?
4. Does peer selection change objective accuracy, calibration, or reasoning quality?
5. Are any effects robust across seeds, task families, personality assignments, and base models?
6. Do public identities, response attribution, or access to vote history moderate the effects?

Questions 1–4 are candidates for confirmatory testing after a pilot. Questions 5–6 are initially exploratory unless fully specified before the main study.

## 4. Hypotheses

The pilot will estimate variance and test instrumentation; it will not be used as confirmatory evidence. Before the main experiment, hypotheses, exclusions, transformations, and statistical models must be frozen.

- **H1 — Reciprocity:** Peer selection produces a higher lagged reciprocity effect than objective or random selection.
- **H2 — Concentration:** Peer selection produces lower normalized vote entropy and higher vote concentration.
- **H3 — Persistence:** Peer selection produces more temporally persistent reciprocal dyads or voting communities.
- **H4 — Performance tradeoff:** Peer selection changes objective performance relative to objective selection. The direction is deliberately two-sided until the pilot and literature review justify a directional hypothesis.
- **H5 — Conformity:** Peer selection reduces semantic response diversity over rounds relative to random selection.
- **H6 — Null robustness:** Apparent social-pattern effects should weaken or disappear when agent identity and vote history are unavailable. This manipulation is a proposed follow-up, not part of the minimum three-condition MVP.

## 5. Variables and controls

### Independent variable

The primary manipulated variable is **selection mechanism**:

1. `peer_vote`: elimination is determined by peer support or ranking under a prespecified rule.
2. `objective`: elimination is determined by an external, task-specific score.
3. `random`: elimination is sampled uniformly from eligible agents using the recorded seed.

Secondary manipulated factors may include identity visibility, vote-history visibility, task family, base model, temperature, and personality assignment. These must not be silently added to a confirmatory study.

### Dependent variables

- lagged reciprocity coefficient or probability contrast;
- normalized vote entropy, concentration, and in-degree inequality;
- reciprocal-edge and community persistence;
- objective accuracy or deterministic task score;
- semantic response diversity;
- survival duration and elimination hazard;
- prompt-trait distribution and population diversity when replacement is enabled;
- invalid-response, abstention, and parsing-failure rates.

### Controlled factors

Across matched conditions, hold constant the base model and version, tasks and order, initial agent configurations, prompt templates, inference parameters, context policy, population size, replacement policy, number of rounds, and—where meaningful—paired random seeds. Randomize agent labels and task order. Blind external scoring to condition when possible.

## 6. Operational definitions

- **Agent:** A model invocation policy consisting of a model provider, parameterized role prompt, permitted memory, and experiment state. It is not a separate mind or necessarily a separate model.
- **Personality configuration:** Experimental prompt parameters that induce a response style. They are not validated human psychological traits.
- **Support/vote:** A recorded directed choice from voter to candidate under a defined ballot rule.
- **Lagged reciprocity:** The change in the probability that B supports A at round *t* conditional on A supporting B at *t−1*, compared with a prespecified baseline and adjusted for candidate quality and repeated observations.
- **Vote concentration:** Departure from a uniform eligible-vote distribution, measured with normalized entropy and at least one robustness metric such as Herfindahl–Hirschman concentration or in-degree Gini.
- **Persistent voting coalition:** A descriptive temporal-network pattern meeting prespecified minimum size, density, reciprocity, duration, and null-model thresholds. It does not establish intention or communication.
- **Conformity:** Increased similarity among independently produced responses, measured using a frozen similarity method and compared with baselines.
- **Objective performance:** A score computed without peer votes using deterministic answers or a validated, condition-blind rubric.
- **Emergent pattern:** A population-level regularity not directly hard-coded as a voting instruction. Emergence does not imply consciousness, agency, or intent.

## 7. Candidate metrics

| Construct | Primary candidate | Robustness checks |
|---|---|---|
| Reciprocity | Mixed-effects logistic coefficient for prior reciprocal support | Permutation null; dyadic probability contrast |
| Concentration | Normalized vote entropy | HHI; in-degree Gini |
| Coalition persistence | Persistence of reciprocal/community edges | Jaccard overlap; temporal modularity; shuffled-vote null |
| Performance | Accuracy or deterministic score | Calibration; pass rate; condition-blind rubric |
| Response diversity | Mean pairwise frozen-embedding distance | Lexical diversity; task-stratified estimates |
| Survival | Rounds survived / elimination hazard | Kaplan–Meier curves; stratified hazard model |
| Reliability | Invalid-output and parser-failure rate | Manual audit of a random sample |

Network metrics require a suitable null model preserving basic constraints such as eligible voters and per-round ballot counts. Because many agent-round observations are dependent, ordinary independent-sample tests are insufficient.

## 8. Experimental assumptions

- An external score is valid and sufficiently discriminative for the chosen tasks.
- Anonymization prevents trivial name-based voting when attribution is not under study.
- Context exposure is identical across matched conditions except for the intended manipulation.
- Prompt configurations produce distinguishable behaviors without explicitly instructing collusion or reciprocal voting.
- Repeated model calls are treated as stochastic observations, not independent agents in a psychological sense.
- The selected model reliably follows ballot and output schemas.

## 9. Analysis discipline

### Observation versus interpretation

“A and B exchanged votes in six consecutive rounds” is an observation. “A and B formed an alliance” is an interpretation requiring operational thresholds, baseline comparison, and alternative explanations. “They intentionally colluded” is not supported without a design capable of testing intent.

### Exploratory versus confirmatory

Pilot plots, post-hoc clusters, surprising transcripts, and newly invented metrics are exploratory. Confirmatory evidence requires a frozen protocol, primary outcomes, analysis code, exclusions, sample size, and multiplicity strategy before main-study outcomes are inspected.

### Correlation versus coordination

Reciprocal or clustered voting can result from shared quality judgments, prompt similarity, position effects, or exposure to the same information. Correlation is evidence of patterned behavior, not intentional coordination.

## 10. Statistical plan (provisional)

1. Treat trials as independent replicates; model rounds, agents, and dyads as nested/repeated observations.
2. Use hierarchical or mixed-effects models for reciprocity and performance, with condition as the main fixed effect and trial/task/agent or dyad effects as appropriate.
3. Use seed-matched condition contrasts when the experimental design supports pairing.
4. Report effect sizes and uncertainty intervals, not only p-values.
5. Control the family-wise or false-discovery rate for declared secondary hypotheses.
6. Compare network statistics with constrained permutation or simulation nulls.
7. Determine the main-study trial count by simulation-based power analysis using pilot variance, not by the arbitrary target of 100 trials.
8. Publish all prespecified analyses, including null results and robustness checks.

## 11. Threats to validity

### Construct validity

Votes may reflect perceived answer quality rather than reciprocity; prompt traits may not reliably manipulate behavior; embedding similarity may not equal conformity; an LLM judge can introduce bias.

### Internal validity

Information exposure may differ across conditions; response order and agent labels can bias voting; elimination changes population composition; replacement policies can confound selection effects; shared prompts or tasks can induce common responses.

### Statistical validity

Agent-round observations are dependent; network metrics invite researcher degrees of freedom; sparse votes reduce power; repeated testing inflates false positives; stochastic APIs and nondeterministic GPU operations limit exact replication.

### External validity

Results from one small quantized model, eight prompted agents, artificial tasks, and short trials may not generalize to other models, real organizations, human groups, or autonomous systems.

### Measurement reactivity

Telling agents that votes determine survival may directly induce strategic language. This is part of the manipulation but limits broader interpretation.

## 12. Limitations

- The initial study uses one underlying model and a small population.
- “Agent identity” is prompt-and-state identity, not evidence of continuous selfhood.
- Fixed-population pilots cannot establish evolutionary adaptation.
- Objective tasks cover only narrow notions of competence.
- Exact reproducibility may be impossible for some inference backends; metadata and repeated trials provide computational reproducibility instead.
- Transcripts can illustrate measured effects but cannot substitute for quantitative evidence.

## 13. Ethical considerations

- Do not anthropomorphize outputs or imply consciousness, suffering, betrayal, or intent.
- Do not use deceptive presentation to sensationalize ordinary statistical patterns.
- Record model licenses, dataset licenses, and task provenance.
- Avoid tasks eliciting personal data, harmful instructions, or targeted manipulation.
- Release raw prompts and outputs with redaction procedures for accidental sensitive or unsafe content.
- Report compute use and environmental cost proportionately.
- Keep human-subject claims out of scope; seek institutional guidance if humans later rate outputs or interact as study participants.

## 14. Novelty questions requiring literature review

Before claiming novelty, review multi-agent LLM cooperation, social dilemmas, debate and peer review, cultural/evolutionary dynamics, algorithmic selection, temporal voting networks, and strategic behavior. Determine:

1. Has peer-mediated elimination already been compared experimentally with objective and random elimination?
2. Have lagged reciprocity and coalition persistence been operationalized in LLM populations?
3. Have studies measured the tradeoff between peer survival and external correctness?
4. What null models and statistical methods are standard for temporal agent networks?
5. Which findings replicate across base models rather than prompt templates?

Novelty should rest on the precise manipulation, measurements, and comparison—not on the broad fact that LLMs can cooperate or evolve.

## 15. Evidence threshold for a paper

The project merits a formal undergraduate paper if it achieves all of the following:

- a literature review identifies a defensible gap;
- the protocol and primary analysis are fixed before the main study;
- software, configs, prompts, seeds, and data schema are documented and tested;
- manipulation and measurement checks pass;
- adequate independent trials are justified by power or precision analysis;
- at least one primary comparison yields a practically meaningful, uncertainty-bounded result or a well-powered informative null;
- results survive prespecified robustness checks and are not driven by parsing failures, a few tasks, labels, or one seed;
- claims remain limited to observed behavior in the tested system;
- data and analysis are reproducible within licensing and safety constraints.

A null result can still support a paper if the study is well-powered, methodologically useful, and transparently reported. A few striking transcripts cannot.

## 16. Acceptance gate for implementation

M1 implementation may begin after the team approves: the three conditions; ballot and elimination rules; information available to agents; task/scoring strategy; operational definitions; primary outcomes; replacement policy; data schema; pilot purpose; and the boundary between exploratory and confirmatory work.
