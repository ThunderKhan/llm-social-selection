DATABASE_SCHEMA_VERSION = 1


SCHEMA_SQL = f"""
CREATE TABLE schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL
);

INSERT INTO schema_metadata (singleton, schema_version)
VALUES (1, {DATABASE_SCHEMA_VERSION});

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    config_schema_version INTEGER NOT NULL,
    config_hash TEXT NOT NULL CHECK (length(config_hash) = 64),
    config_json TEXT NOT NULL,
    database_schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    code_commit TEXT,
    python_version TEXT NOT NULL,
    platform TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL
);

CREATE TABLE trials (
    trial_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    trial_seed TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('created', 'running', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    condition TEXT CHECK (condition IS NULL OR condition IN ('peer_vote', 'objective', 'random')),
    total_rounds INTEGER CHECK (total_rounds IS NULL OR total_rounds > 0),
    config_hash TEXT CHECK (config_hash IS NULL OR length(config_hash) = 64),
    profile_pool_hash TEXT CHECK (profile_pool_hash IS NULL OR length(profile_pool_hash) = 64),
    replacement_version TEXT,
    CHECK (
        (condition IS NULL AND total_rounds IS NULL AND config_hash IS NULL
            AND profile_pool_hash IS NULL AND replacement_version IS NULL)
        OR
        (condition IS NOT NULL AND total_rounds IS NOT NULL AND config_hash IS NOT NULL
            AND profile_pool_hash IS NOT NULL AND replacement_version IS NOT NULL)
    ),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE agent_instances (
    trial_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    display_label TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    introduced_round INTEGER CHECK (introduced_round IS NULL OR introduced_round >= 0),
    PRIMARY KEY (trial_id, agent_id),
    UNIQUE (trial_id, ordinal),
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);

CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected_answer TEXT,
    scorer_version TEXT NOT NULL
);

CREATE TABLE rounds (
    trial_id TEXT NOT NULL,
    round_index INTEGER NOT NULL CHECK (round_index >= 0),
    task_id TEXT NOT NULL,
    condition TEXT NOT NULL CHECK (condition IN ('peer_vote', 'objective', 'random')),
    round_seed TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'complete')),
    committed_at TEXT,
    PRIMARY KEY (trial_id, round_index),
    UNIQUE (trial_id, round_index, task_id),
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE responses (
    response_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    content TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    request_id TEXT,
    seed TEXT,
    latency_ms REAL CHECK (latency_ms IS NULL OR latency_ms >= 0),
    token_count INTEGER CHECK (token_count IS NULL OR token_count >= 0),
    UNIQUE (trial_id, round_index, agent_id),
    UNIQUE (trial_id, round_index, ordinal),
    FOREIGN KEY (trial_id, round_index, task_id)
        REFERENCES rounds(trial_id, round_index, task_id),
    FOREIGN KEY (trial_id, agent_id)
        REFERENCES agent_instances(trial_id, agent_id)
);

CREATE TABLE scores (
    score_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    value REAL NOT NULL,
    scorer_version TEXT NOT NULL,
    UNIQUE (trial_id, round_index, agent_id),
    UNIQUE (trial_id, round_index, ordinal),
    FOREIGN KEY (trial_id, round_index, task_id)
        REFERENCES rounds(trial_id, round_index, task_id),
    FOREIGN KEY (trial_id, agent_id)
        REFERENCES agent_instances(trial_id, agent_id)
);

CREATE TABLE ballots (
    ballot_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    voter_agent_id TEXT NOT NULL,
    supported_agent_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    CHECK (voter_agent_id <> supported_agent_id),
    UNIQUE (trial_id, round_index, voter_agent_id),
    UNIQUE (trial_id, round_index, ordinal),
    FOREIGN KEY (trial_id, round_index)
        REFERENCES rounds(trial_id, round_index),
    FOREIGN KEY (trial_id, voter_agent_id)
        REFERENCES agent_instances(trial_id, agent_id),
    FOREIGN KEY (trial_id, supported_agent_id)
        REFERENCES agent_instances(trial_id, agent_id)
);

CREATE TABLE selection_events (
    selection_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    mechanism TEXT NOT NULL CHECK (mechanism IN ('peer_vote', 'objective', 'random')),
    selected_agent_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    UNIQUE (trial_id, round_index),
    FOREIGN KEY (trial_id, round_index)
        REFERENCES rounds(trial_id, round_index),
    FOREIGN KEY (trial_id, selected_agent_id)
        REFERENCES agent_instances(trial_id, agent_id)
);

CREATE TABLE replacement_events (
    replacement_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    removed_agent_id TEXT NOT NULL,
    added_agent_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    queue_index INTEGER NOT NULL CHECK (queue_index >= 0),
    reason TEXT NOT NULL CHECK (reason = 'fixed_profile_pool'),
    UNIQUE (trial_id, round_index),
    UNIQUE (trial_id, queue_index),
    UNIQUE (trial_id, added_agent_id),
    CHECK (removed_agent_id <> added_agent_id),
    FOREIGN KEY (trial_id, round_index)
        REFERENCES rounds(trial_id, round_index),
    FOREIGN KEY (trial_id, removed_agent_id)
        REFERENCES agent_instances(trial_id, agent_id),
    FOREIGN KEY (trial_id, added_agent_id)
        REFERENCES agent_instances(trial_id, agent_id)
);
"""
