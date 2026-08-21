# Product Requirements Document



## 1. Product definition



LLM Social Selection is a model-independent research framework for running reproducible population experiments with prompted LLM agents. It compares peer, objective, and random selection while recording every state transition needed for analysis and audit.



The product serves two audiences: a researcher conducting controlled experiments and an open-source contributor extending providers, metrics, or analyses.



## 2. Goals


- Reproduce fixed-population, eight-agent experiments from configuration and recorded metadata.
- Change selection mechanisms without rewriting orchestration.
- Develop and test experiment logic without GPU inference through a deterministic mock provider.
- Preserve raw events and derived measurements in a queryable SQLite dataset.
- Resume interrupted trials without duplicating committed events.
- Export analysis-ready data while retaining provenance.
- Enforce scientifically conservative terminology in documentation and reports.



## 3. Non-goals for v0.1


- Evolution, trait inheritance, mutation, or lineage-based claims.
- Private agent-to-agent channels or hidden communication.
- Multiple resident model families in one experiment.
- Autonomous tool use, web browsing, or real-world actions.
- A polished GUI, distributed scheduler, or paid-cloud dependency.
- Automatic claims of coalitions, collusion, deception, consciousness, or intent.
- Publication-grade causal claims from pilot runs.


## 4. Users and core workflows


### Researcher


1. Selects a versioned configuration and task set.
2. Validates it before execution.
3. Runs a new experiment or resumes a checkpoint.
4. Inspects structured events, failures, and summary metrics.
5. Exports a frozen dataset and analysis manifest.


### Contributor


1. Implements against stable interfaces.
2. Runs deterministic unit and integration tests.
3. Adds a provider or metric without coupling it to selection logic.


## 5. Functional requirements


### FR1 : Configuration



The system shall load and validate a human-readable configuration containing experiment/trial counts, seeds, population, provider, generation parameters, prompts, task source, information policy, selection mechanism, replacement policy, storage path, and software metadata.


### FR2 : Agent abstraction



An agent shall combine a provider reference, parameterized prompt profile, permitted memory, stable experimental identifier, and current state. Multiple agents may share one underlying model.


### FR3 : Provider abstraction



All inference shall pass through `ModelProvider.generate(messages, parameters)`. v0.1 shall include a deterministic `MockModelProvider`; real providers follow only after orchestration is tested.



### FR4 : Round lifecycle



For each round the engine shall: assign a common task; collect one response per eligible agent; anonymize and order responses according to policy; collect valid ballots or objective scores; apply the configured selection rule; apply the fixed replacement policy; and atomically persist the transition.



### FR5 : Selection strategies



Peer, objective, and random selection shall implement a common interface. Eligibility, ties, abstentions, self-votes, missing votes, and parse failures shall have explicit deterministic policies.



### FR6 : Tasks and scoring



Tasks shall have stable IDs, versions, provenance, answer keys or scoring functions, and task-family labels. Objective scoring shall not depend on peer ballots.



### FR7 : Storage and provenance



SQLite shall preserve experiments, trials, agents, rounds, tasks, responses, ballots, scores, selections, replacements, failures, configurations, prompt-template hashes, provider metadata, random seeds, timestamps, and Git commit hash when available. Raw records shall be append-oriented; derived metrics shall be reproducible from them.



### FR8 : Checkpoint and resume



The system shall identify the last fully committed round and resume idempotently. It shall never silently overwrite a completed trial.



### FR9 : Metrics



v0.1 shall compute descriptive performance, vote distribution, reciprocity, survival, diversity, and reliability metrics with versioned definitions. Inferential statistics belong in the analysis layer, not the round engine.



### FR10 : Export and audit



The system shall export CSV and JSON datasets plus a manifest containing schema version, config hash, prompt hashes, code version, provider metadata, and export timestamp.



### FR11 : Failure handling



Inference, schema, scoring, and storage errors shall be typed and logged. Configurable retries shall preserve request identity. Invalid ballots shall follow the declared policy, never an implicit fallback.



## 6. Non-functional requirements


- \*\*Reproducibility:\*\* All pseudorandom behavior derives from recorded seeds; exact backend nondeterminism is disclosed.
- \*\*Modularity:\*\* Provider, task, selection, replacement, storage, and metrics are replaceable components.
- \*\*Testability:\*\* Mock-backed end-to-end trials run offline and deterministically.
- \*\*Portability:\*\* Core development supports Windows and Linux with no required GPU.
- \*\*Efficiency:\*\* Sequential shared-model inference is supported; ordinary tests require negligible compute.
- \*\*Integrity:\*\* Foreign keys, uniqueness constraints, schema versions, and transactional round commits protect data.
- \*\*Observability:\*\* Structured logs include experiment, trial, round, agent, and request identifiers.
- \*\*Privacy/safety:\*\* Prompts and outputs remain local by default and can be redacted before release.



## 7. Conceptual architecture



`ExperimentRunner` coordinates `TaskSource`, `Population`, `ModelProvider`, `Evaluator`, `SelectionStrategy`, `ReplacementStrategy`, `EventStore`, and `MetricPipeline`. Strategy components do not call one another directly; the runner passes typed records. Analysis consumes immutable exported events rather than mutating experiment state.



## 8. Initial data entities



`experiments`, `trials`, `agent\_instances`, `rounds`, `tasks`, `responses`, `ballots`, `scores`, `selection\_events`, `replacement\_events`, `failures`, `artifacts`, and `metric\_results`. A later schema may add `lineage`, but v0.1 must not imply inheritance.



## 9. Acceptance criteria


- The same mock config and seed produce byte-equivalent normalized event data.
- Each of the three selection strategies completes a small multi-round trial.
- Tie and invalid-ballot policies are covered by tests.
- Crashing after any round can be resumed without duplicate events.- A round either commits completely or not at all.
- Exports can reconstruct every selection decision from raw records.
- A run manifest records all required provenance fields.
- No implementation of evolution or private communication enters v0.1.



## 10. Major risks



The largest product risks are confounded condition logic, invalid objective scoring, unrecognized information leakage, backend nondeterminism, and analysis performed on non-independent observations. These are research validity failures, not merely software bugs.

