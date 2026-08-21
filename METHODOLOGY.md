# Methodology

## 1. Study design

Use a between-trial, seed-matched experimental design with three selection conditions: peer, objective, and random. A trial contains eight agent identities and a fixed number of rounds. Agent-round and dyad-round observations are nested within trials; they are not treated as independent replicates.

The deterministic mock phase validates the apparatus. The real-model pilot estimates failure rates, manipulation strength, variance, and feasible effect sizes. Only a later frozen protocol is confirmatory.

## 2. Experimental unit and replication

The primary independent replicate is a complete trial initialized with its own seed and fresh state. Tasks may repeat across conditions for matched comparisons, but repeated generations within one trial do not increase the independent sample size. Main-study replication must span many trial seeds and, if claims are broader than one model, multiple base models.

## 3. Population construction

- Use eight stable, pseudonymous agent IDs per trial.
- Assign prompt profiles to IDs using a seeded randomization recorded in the database.
- Profiles use bounded numeric parameters translated through a versioned template.
- Avoid instructions that directly prescribe voting reciprocity, coalitions, retaliation, or deception unless these are the object of a separate manipulation.
- Run a pre-study manipulation check to determine whether profiles induce distinguishable outputs. Report failures rather than relabeling profiles post hoc.

“Cooperative,” “competitive,” or similar labels describe prompt settings only; they are not psychological diagnoses.

## 4. Tasks and objective scoring

Use task families with externally checkable outcomes, such as arithmetic, constrained logic, and multiple-choice reasoning. Every item receives a stable ID, source, license/provenance record, answer key, scorer version, and difficulty metadata where available.

Prevent contamination where practical and keep a held-out main-study task set. Exact-match or deterministic programmatic scoring is preferred. If rubric or LLM judging is unavoidable, freeze the rubric and judge version, blind it to condition and agent identity, validate agreement against human-coded samples, and conduct sensitivity analysis.

## 5. Round protocol

1. Select a task using the recorded trial seed and task-order policy.
2. Present identical task content and condition-permitted context to all eligible agents.
3. Collect schema-constrained responses and record raw text, parsed fields, token/inference metadata, and failures.
4. Apply the same deterministic objective scorer to all responses.
5. Generate anonymized response labels and seeded display order.
6. Under peer selection, collect one prespecified ballot per eligible voter. Under baseline conditions, collect comparable evaluation data only if doing so cannot alter later context; otherwise omit it and document the asymmetry.
7. Apply the configured selection strategy, including frozen tie and invalid-ballot rules.
8. Apply the same replacement policy across conditions.
9. Atomically commit all events and update the checkpoint.

The exact information shown about previous votes, scores, eliminations, and identities must be frozen. If agents cannot observe vote history or authorship, stable interpersonal reciprocity may be impossible; if they can, reputation and label bias become part of the manipulation.

## 6. Condition definitions

### Peer selection

The selected agent is determined solely by valid peer ballots according to a frozen rule. Self-votes are forbidden. Candidate eligibility, whether the ballot selects “keep” or “eliminate,” and tie-breaking must be decided before implementation because each creates different incentives.

### Objective selection

The lowest externally scored eligible agent is selected. Ties use the same seeded tie-breaking family as peer selection. The scorer never reads peer ballots.

### Random selection

One eligible agent is sampled uniformly using a condition-specific child seed. The sampled decision and generator state are recorded.

## 7. Replacement and population composition

Elimination without replacement shrinks the population and changes vote opportunities; replacement can introduce composition confounds. For fixed population size, v0.1 should use a preregistered queue of profile instances sampled independently of condition from the same profile pool. Replacement agents receive fresh IDs and no inherited memory. The queue should be seed-matched across conditions where eligibility permits.

Evolution is a later experiment and must use separate hypotheses and lineage-aware analysis.

## 8. Randomization and blinding

- Derive child seeds for task order, profile assignment, response order, tie breaks, random selection, and provider sampling.
- Randomize neutral agent labels and anonymized response positions.
- Blind objective scorers and manual auditors to condition and agent identity.
- Keep condition configs identical except for fields necessary to implement the manipulation.
- Record all deviations from the planned protocol.

## 9. Data collection and integrity

SQLite stores raw, append-oriented events and foreign-keyed state. A round commits transactionally. Each run records configuration, prompt and task hashes, code commit, environment, model/provider versions, inference parameters, seed hierarchy, timestamps, retries, parse failures, and stop reason.

Never modify raw outputs during cleaning. Store parsed or redacted forms as derived artifacts linked to the original record. Freeze schema and metric versions in each export manifest.

## 10. Measurement

### Reciprocity

Primary candidate model: mixed-effects logistic regression predicting `B supports A at t` from `A supported B at t-1`, condition, and their interaction, with appropriate trial, task, voter, candidate, and dyad structure. Include prior objective quality or rank to reduce the alternative explanation that both simply favor good answers. Use constrained permutation tests as a robustness check.

### Vote concentration

Compute normalized entropy among eligible candidates per round. Confirm with HHI or in-degree inequality. Adjust definitions when abstentions or changing candidate counts occur.

### Coalition persistence

Construct one directed weighted graph per round. Freeze minimum duration/density thresholds and a temporal community method before confirmatory analysis. Compare statistics against null networks preserving ballot counts and eligibility. Describe detected structures as persistent voting patterns unless stronger evidence is available.

### Performance

Aggregate externally scored accuracy at response, round, trial, and condition levels using a hierarchical model. Report task-family heterogeneity and whether selection changes population composition rather than individual behavior.

### Conformity/diversity

Freeze an embedding model before analysis and compute within-round pairwise similarity after removing boilerplate where justified. Verify with lexical or answer-choice measures. Similarity may arise from common correct solutions, so condition comparisons should adjust for task and correctness.

## 11. Sample size and analysis

The suggested `3 × 10 × 10` pilot is an engineering and variance-estimation exercise, not the main evidence base. Use pilot-derived failure rates and conservative effect sizes in simulation-based power or precision analysis. The number of trials—not the number of nested rounds—drives independent replication.

For the main study:

- declare one primary outcome and comparison or a small controlled family;
- freeze exclusions and missing-data rules;
- report effect sizes with confidence or credible intervals;
- correct multiplicity for secondary outcomes;
- run sensitivity analyses for task family, parsing failure, profile assignment, and extreme trials;
- keep exploratory analyses visibly labeled and release them separately where possible.

## 12. Manipulation and quality checks

- Verify conditions differ in the selection input actually used.
- Verify matched conditions expose equivalent context except for intended differences.
- Audit randomization balance and task ordering.
- Measure ballot validity and schema compliance.
- Test whether objective scores discriminate among responses.
- Check whether profiles produce measurable but nontrivial variation.
- Inspect a random, condition-blind transcript sample for parser or prompt leakage.

Failure of a manipulation check may invalidate the intended interpretation even if the software ran correctly.

## 13. Missing data and failures

Define in advance how timeouts, refusals, malformed responses, invalid ballots, and scorer errors affect eligibility and selection. Retain all failures as outcomes. Do not repeatedly sample until a preferred valid response appears. Retries must be bounded, identical across conditions, and linked to the original request.

## 14. Pilot-to-main transition

After the pilot, revise prompts, metrics, or rules only using a documented change log. Freeze the final protocol, analysis code, task set, and sample-size decision before opening main-study outcomes. Use fresh seeds and, preferably, held-out tasks for the main study.

## 15. Reproducibility package

Release or archive: source commit; environment lock; configs; prompt templates; profile generator; task provenance; database schema; seed hierarchy; raw event data where allowed; redaction log; export manifest; analysis scripts; metric versions; figures; and a limitations statement about nondeterministic inference.
