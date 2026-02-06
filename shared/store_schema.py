"""SQL schema and column metadata for StateStore."""

# ── JSON columns per table ───────────────────────────────────

JSON_COLS: dict[str, set[str]] = {
    "projects": {"evaluation_criteria", "research_domains"},
    "seeds": {"tags"},
    "insights": {"tags", "evaluation_scores", "review_record"},
    "sections": {"establishes", "requires"},
    "events": {"payload"},
    "metrics": {"labels"},
}

# ── datetime columns per table ───────────────────────────────

DT_COLS: dict[str, set[str]] = {
    "system_settings": {"updated_at"},
    "users": {"created_at"},
    "projects": {"created_at", "updated_at"},
    "seeds": {"created_at", "updated_at"},
    "threads": {"created_at", "updated_at", "retired_at"},
    "exchanges": {"created_at"},
    "insights": {"created_at", "updated_at", "blessed_at", "incorporated_at"},
    "insight_comments": {"created_at"},
    "seed_modifications": {"created_at", "resolved_at"},
    "sections": {"created_at", "updated_at", "last_reviewed_at"},
    "annotations": {"created_at", "resolved_at", "last_attempted_at"},
    "sources": {"created_at", "evaluated_at"},
    "findings": {"created_at"},
    "events": {"created_at", "timestamp"},
    "metrics": {"recorded_at"},
}

# ── Tables with auto-updated updated_at ──────────────────────

TABLES_WITH_UPDATED_AT = {"projects", "seeds", "threads", "insights", "sections"}

# ── Schema SQL ───────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    owner_id TEXT REFERENCES users(id),
    name TEXT NOT NULL,
    goal TEXT NOT NULL,
    context TEXT NOT NULL,
    evaluation_criteria TEXT NOT NULL,
    exploration_guidance TEXT,
    document_guidance TEXT,
    seed_modification_enabled BOOLEAN DEFAULT TRUE,
    seed_modification_require_approval BOOLEAN DEFAULT FALSE,
    research_domains TEXT DEFAULT '[]',
    status TEXT NOT NULL CHECK(status IN ('active','paused','archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seeds (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    text TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('axiom','conjecture','question')),
    priority INTEGER NOT NULL DEFAULT 5 CHECK(priority BETWEEN 1 AND 10),
    tags TEXT NOT NULL DEFAULT '[]',
    confidence TEXT CHECK(confidence IN ('high','medium','low')),
    source TEXT,
    notes TEXT,
    status TEXT NOT NULL CHECK(status IN ('active','explored','evolved','retired')),
    parent_seed_id TEXT REFERENCES seeds(id),
    modification_reason TEXT,
    exploration_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    seed_id TEXT NOT NULL REFERENCES seeds(id),
    status TEXT NOT NULL CHECK(status IN ('active','synthesizing','extracted','stalled','retired')),
    priority INTEGER NOT NULL DEFAULT 5,
    exchange_count INTEGER DEFAULT 0,
    last_completed_sequence INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retired_at TEXT,
    retire_reason TEXT
);

CREATE TABLE IF NOT EXISTS exchanges (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id),
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('explorer','critic','synthesizer')),
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    text TEXT NOT NULL,
    confidence TEXT NOT NULL CHECK(confidence IN ('high','medium','low')),
    tags TEXT NOT NULL DEFAULT '[]',
    source_thread_id TEXT REFERENCES threads(id),
    extraction_model TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'extracted','reviewing','attested','discarded','disputed',
        'interesting','incorporating','incorporated','shelved','transient_failure'
    )),
    evaluation_scores TEXT DEFAULT '{}',
    dispute_count INTEGER DEFAULT 0,
    transient_failure_count INTEGER DEFAULT 0,
    review_record TEXT,
    blessed_at TEXT,
    incorporated_at TEXT,
    incorporated_in_section TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS insight_dependencies (
    insight_id TEXT NOT NULL REFERENCES insights(id),
    depends_on_id TEXT NOT NULL REFERENCES insights(id),
    PRIMARY KEY (insight_id, depends_on_id)
);

CREATE TABLE IF NOT EXISTS insight_comments (
    id TEXT PRIMARY KEY,
    insight_id TEXT NOT NULL REFERENCES insights(id),
    comment_type TEXT NOT NULL CHECK(comment_type IN ('endorse','dismiss','reconsider','general')),
    content TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seed_modifications (
    id TEXT PRIMARY KEY,
    seed_id TEXT NOT NULL REFERENCES seeds(id),
    original_text TEXT NOT NULL,
    proposed_text TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    proposing_thread_id TEXT NOT NULL REFERENCES threads(id),
    agreement_ratio REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')),
    child_seed_id TEXT REFERENCES seeds(id),
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('provisional','stable','needs_work')),
    order_index INTEGER NOT NULL DEFAULT 0,
    establishes TEXT NOT NULL DEFAULT '[]',
    requires TEXT NOT NULL DEFAULT '[]',
    source_insight_id TEXT REFERENCES insights(id),
    review_count INTEGER DEFAULT 0,
    last_reviewed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    established_in_section TEXT REFERENCES sections(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    domain TEXT,
    PRIMARY KEY (canonical_name, project_id)
);

CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    type TEXT NOT NULL CHECK(type IN ('comment','protected','suggestion')),
    section_id TEXT REFERENCES sections(id),
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','attempted','resolved','needs_human_review')),
    attempt_count INTEGER DEFAULT 0,
    last_attempted_at TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    url TEXT NOT NULL,
    domain TEXT,
    title TEXT,
    trust_score INTEGER DEFAULT 0,
    trust_tier TEXT,
    content_hash TEXT,
    evaluated_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    source_id TEXT REFERENCES sources(id),
    finding_type TEXT,
    summary TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    domain TEXT,
    related_insight_id TEXT REFERENCES insights(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    payload TEXT NOT NULL,
    correlation_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT REFERENCES projects(id),
    name TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT DEFAULT '{}',
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS project_access (
    project_id TEXT NOT NULL REFERENCES projects(id),
    user_id TEXT NOT NULL REFERENCES users(id),
    role TEXT NOT NULL CHECK(role IN ('owner','editor','viewer')),
    granted_at TEXT NOT NULL,
    PRIMARY KEY (project_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_seeds_project ON seeds(project_id, status);
CREATE INDEX IF NOT EXISTS idx_threads_project ON threads(project_id, status);
CREATE INDEX IF NOT EXISTS idx_insights_project ON insights(project_id, status);
CREATE INDEX IF NOT EXISTS idx_insights_status ON insights(status);
CREATE INDEX IF NOT EXISTS idx_sections_project ON sections(project_id);
CREATE INDEX IF NOT EXISTS idx_concepts_canonical ON concepts(canonical_name);
CREATE INDEX IF NOT EXISTS idx_annotations_project ON annotations(project_id, status);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_insight_comments ON insight_comments(insight_id);
CREATE INDEX IF NOT EXISTS idx_findings_insight ON findings(related_insight_id);
"""
