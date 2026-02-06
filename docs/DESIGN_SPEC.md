# Research Project Assistant — Low-Level Design Specification

> **Status**: Proposal (pending review)
> **Date**: 2026-02-06
> **Parent**: [ARCHITECTURE_V2.md](./ARCHITECTURE_V2.md)
> **Purpose**: Implementation-level specification for parallel agent development

---

## Table of Contents

1. [Parallel Development Guide](#1-parallel-development-guide)
2. [Shared Types and Interfaces](#2-shared-types-and-interfaces)
3. [Module: shared/store — StateStore](#3-module-sharedstore--statestore)
4. [Module: shared/events — EventBus](#4-module-sharedevents--eventbus)
5. [Module: shared/config — Config](#5-module-sharedconfig--config)
6. [Module: llm/client — LLMClient](#6-module-llmclient--llmclient)
7. [Module: llm/consensus — ConsensusEngine](#7-module-llmconsensus--consensusengine)
8. [Module: explorer — Explorer Engine](#8-module-explorer--explorer-engine)
9. [Module: documenter — Documenter Engine](#9-module-documenter--documenter-engine)
10. [Module: researcher — Researcher Engine](#10-module-researcher--researcher-engine)
11. [Module: orchestrator — Orchestrator](#11-module-orchestrator--orchestrator)
12. [Module: control — Control Panel](#12-module-control--control-panel)

---

## 1. Parallel Development Guide

### Module Dependency Graph

```
Layer 0 (no dependencies):     shared/models
Layer 1 (depends on Layer 0):  shared/store, shared/events, shared/config
Layer 2 (depends on Layer 1):  llm/client
Layer 3 (depends on Layer 2):  llm/consensus
Layer 4 (depends on Layer 1-3): explorer, documenter, researcher
Layer 5 (depends on Layer 4):  orchestrator, control
```

### Agent Assignment

Each module can be developed by an independent agent. The agent needs to read:
1. **Section 2** (Shared Types) — always
2. **Their module's section** — for implementation details
3. **Interface signatures of dependencies** — but NOT the internals

| Agent | Module | Reads | Mocks |
|-------|--------|-------|-------|
| A1 | `shared/models` + `shared/store` + `shared/events` + `shared/config` | Section 2, 3, 4, 5 | Nothing (foundation layer) |
| A2 | `llm/client` | Section 2, 6 | `shared/config` |
| A3 | `llm/consensus` | Section 2, 7 | `llm/client` |
| A4 | `explorer` | Section 2, 8 | `shared/store`, `shared/events`, `llm/client`, `llm/consensus` |
| A5 | `documenter` | Section 2, 9 | `shared/store`, `shared/events`, `llm/client`, `llm/consensus` |
| A6 | `researcher` | Section 2, 10 | `shared/store`, `shared/events`, `llm/client`, `llm/consensus` |
| A7 | `orchestrator` | Section 2, 11 | Engine interfaces (A4-A6) |
| A8 | `control` | Section 2, 12 | `shared/store`, `shared/events`, engine interfaces |

**Agents A1 through A8 can all start simultaneously.** Each agent mocks its dependencies using the interfaces defined in Section 2. Integration happens when modules are assembled.

### File Locations

Every file has a single, defined location. No ambiguity.

```
fano/
├── shared/
│   ├── __init__.py
│   ├── models.py              # Section 2: All shared data types
│   ├── store.py               # Section 3: StateStore
│   ├── events.py              # Section 4: EventBus
│   └── config.py              # Section 5: Config
├── llm/
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── client.py           # Section 6: LLMClient
│       └── consensus/
│           ├── __init__.py
│           ├── engine.py       # Section 7: ConsensusEngine
│           └── parsing.py      # Section 7: Response parsing
├── explorer/
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── engine.py           # Section 8: ExplorerEngine (facade)
│       ├── thread_manager.py   # Section 8: Thread lifecycle
│       ├── seed_manager.py     # Section 8: Seed lifecycle
│       ├── insight_extractor.py # Section 8: Extraction + dedup
│       ├── review_panel.py     # Section 8: Review using ConsensusEngine
│       └── prompts.py          # Section 8: Prompt templates
├── documenter/
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── engine.py           # Section 9: DocumenterEngine (facade)
│       ├── planner.py          # Section 9: Work planning
│       ├── processor.py        # Section 9: Insight processing pipeline
│       ├── annotations.py      # Section 9: Annotation handling
│       ├── context.py          # Section 9: Context window builder
│       ├── renderer.py         # Section 9: Markdown rendering
│       └── prompts.py          # Section 9: Prompt templates
├── researcher/
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── engine.py           # Section 10: ResearcherEngine (facade)
│       ├── questions.py        # Section 10: Question generation
│       ├── searcher.py         # Section 10: Search execution
│       ├── extractor.py        # Section 10: Finding extraction
│       ├── trust.py            # Section 10: Trust evaluation
│       └── prompts.py          # Section 10: Prompt templates
├── orchestrator/
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── main.py             # Section 11: Orchestrator (facade)
│       ├── scheduler.py        # Section 11: Task scheduling
│       ├── quota.py            # Section 11: Budget allocation
│       └── recovery.py         # Section 11: WAL-based recovery
├── control/
│   ├── __init__.py
│   ├── server.py               # Section 12: Flask app
│   └── blueprints/
│       ├── __init__.py
│       ├── projects.py         # Section 12: Project CRUD
│       ├── seeds.py            # Section 12: Seed management
│       ├── insights.py         # Section 12: Insight feed
│       ├── document.py         # Section 12: Document viewer
│       ├── research.py         # Section 12: Research status
│       └── status.py           # Section 12: System health
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── unit/
│   │   ├── shared/             # Tests for A1
│   │   ├── llm/                # Tests for A2 + A3
│   │   ├── explorer/           # Tests for A4
│   │   ├── documenter/         # Tests for A5
│   │   ├── researcher/         # Tests for A6
│   │   └── orchestrator/       # Tests for A7
│   ├── integration/            # Cross-module tests (post-assembly)
│   └── contract/               # LLM response format validation
├── config.yaml                 # Infrastructure config
└── projects/
    ├── fano-mathematics.yaml   # Project config: math research (primary)
    └── ev-charging-europe.yaml # Project config: business planning (validates generalization)
```

---

## 2. Shared Types and Interfaces

**File**: `shared/models.py`

This file defines every type that crosses module boundaries. It is the contract language. All modules import from here.

### 2.1 Enumerations

```python
from enum import Enum

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"

class SeedStatus(str, Enum):
    ACTIVE = "active"
    EXPLORED = "explored"
    EVOLVED = "evolved"
    RETIRED = "retired"

class SeedType(str, Enum):
    AXIOM = "axiom"
    CONJECTURE = "conjecture"
    QUESTION = "question"

class ThreadStatus(str, Enum):
    ACTIVE = "active"
    SYNTHESIZING = "synthesizing"
    EXTRACTED = "extracted"
    STALLED = "stalled"
    RETIRED = "retired"

class ExchangeRole(str, Enum):
    EXPLORER = "explorer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"

class InsightStatus(str, Enum):
    EXTRACTED = "extracted"
    REVIEWING = "reviewing"
    ATTESTED = "attested"
    DISCARDED = "discarded"
    DISPUTED = "disputed"
    INTERESTING = "interesting"
    INCORPORATING = "incorporating"
    INCORPORATED = "incorporated"
    SHELVED = "shelved"
    TRANSIENT_FAILURE = "transient_failure"

class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SectionStatus(str, Enum):
    PROVISIONAL = "provisional"
    STABLE = "stable"
    NEEDS_WORK = "needs_work"

class AnnotationType(str, Enum):
    COMMENT = "comment"
    PROTECTED = "protected"
    SUGGESTION = "suggestion"

class AnnotationStatus(str, Enum):
    OPEN = "open"
    ATTEMPTED = "attempted"
    RESOLVED = "resolved"
    NEEDS_HUMAN_REVIEW = "needs_human_review"

class CommentType(str, Enum):
    ENDORSE = "endorse"
    DISMISS = "dismiss"
    RECONSIDER = "reconsider"
    GENERAL = "general"

class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"

class BackendState(str, Enum):
    CLOSED = "closed"       # Healthy
    OPEN = "open"           # Failed, rejecting requests
    HALF_OPEN = "half_open" # Testing recovery
```

### 2.2 Data Models

All models are frozen dataclasses. Mutable state lives in the StateStore; models are snapshots.

```python
from dataclasses import dataclass, field
from datetime import datetime

# ---------- Configuration types ----------

@dataclass(frozen=True)
class EvaluationCriterion:
    name: str
    description: str
    weight: float  # 0.0 to 1.0

@dataclass(frozen=True)
class ResearchDomain:
    name: str
    keywords: list[str]
    source_types: list[str]
    extraction_patterns: list[str] = field(default_factory=list)

# ---------- Core entities ----------

@dataclass(frozen=True)
class Project:
    id: str
    name: str
    goal: str
    context: str
    evaluation_criteria: list[EvaluationCriterion]
    exploration_guidance: str
    document_guidance: str
    seed_modification_enabled: bool
    seed_modification_require_approval: bool
    research_domains: list[ResearchDomain]
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class Seed:
    id: str
    project_id: str
    text: str
    type: SeedType
    priority: int               # 1-10
    tags: list[str]
    confidence: Confidence | None
    source: str | None
    notes: str | None
    status: SeedStatus
    parent_seed_id: str | None
    modification_reason: str | None
    exploration_count: int
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class Thread:
    id: str
    project_id: str
    seed_id: str
    status: ThreadStatus
    priority: int
    exchange_count: int
    last_completed_sequence: int    # Last exchange sequence completed (for resume)
    created_at: datetime
    updated_at: datetime
    retired_at: datetime | None
    retire_reason: str | None

@dataclass(frozen=True)
class Exchange:
    id: str
    thread_id: str
    sequence: int
    role: ExchangeRole
    model: str
    prompt: str
    response: str
    created_at: datetime

@dataclass(frozen=True)
class Insight:
    id: str
    project_id: str
    text: str
    confidence: Confidence
    tags: list[str]
    source_thread_id: str | None
    extraction_model: str | None
    status: InsightStatus
    evaluation_scores: dict[str, float]  # {criterion_name: score}
    dispute_count: int
    transient_failure_count: int
    review_record: dict | None           # Full review history
    blessed_at: datetime | None
    incorporated_at: datetime | None
    incorporated_in_section: str | None
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class InsightComment:
    id: str
    insight_id: str
    comment_type: CommentType
    content: str | None
    created_at: datetime

@dataclass(frozen=True)
class Section:
    id: str
    project_id: str
    title: str
    content: str
    status: SectionStatus
    order_index: int
    establishes: list[str]      # Concept names this section defines
    requires: list[str]         # Concept names this section depends on
    source_insight_id: str | None
    review_count: int
    last_reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class Concept:
    name: str
    canonical_name: str
    established_in_section: str | None
    project_id: str
    domain: str | None

@dataclass(frozen=True)
class Annotation:
    id: str
    project_id: str
    type: AnnotationType
    section_id: str | None
    content: str
    status: AnnotationStatus
    attempt_count: int
    last_attempted_at: datetime | None
    created_at: datetime
    resolved_at: datetime | None

@dataclass(frozen=True)
class Source:
    id: str
    project_id: str
    url: str
    domain: str | None
    title: str | None
    trust_score: int            # 0-100
    trust_tier: str | None
    content_hash: str | None
    evaluated_at: datetime | None
    created_at: datetime

@dataclass(frozen=True)
class Finding:
    id: str
    project_id: str
    source_id: str | None
    finding_type: str | None
    summary: str
    confidence: float           # 0.0-1.0
    domain: str | None
    related_insight_id: str | None
    created_at: datetime

# ---------- LLM types ----------

@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

@dataclass(frozen=True)
class LLMResponse:
    success: bool
    text: str
    backend: str
    model: str
    token_usage: TokenUsage | None
    error: str | None

@dataclass(frozen=True)
class BackendStatus:
    name: str
    state: BackendState
    requests_per_minute: int
    avg_latency_ms: float
    failure_count: int
    last_failure_at: datetime | None

# ---------- Consensus types ----------

@dataclass(frozen=True)
class ParsedResponse:
    """Result of parsing one LLM's review response."""
    is_valid: bool
    verdict: Verdict | None
    scores: dict[str, float]     # {criterion_name: score}
    reasoning: str | None
    error: str | None

@dataclass(frozen=True)
class ValidatedResponse:
    """A validated, parsed response from one reviewer."""
    backend: str
    verdict: Verdict
    scores: dict[str, float]
    reasoning: str
    raw_text: str

@dataclass(frozen=True)
class RoundResult:
    round_num: int
    responses: list[ValidatedResponse]
    is_converged: bool
    verdict: Verdict | None      # None if not converged

@dataclass(frozen=True)
class ConsensusResult:
    verdict: Verdict
    confidence: float            # 0.0-1.0
    scores: dict[str, float]     # Averaged across valid responses
    rounds_completed: int
    round_history: list[RoundResult]
    valid_vote_count: int

# ---------- Event type ----------

@dataclass(frozen=True)
class Event:
    topic: str
    timestamp: datetime
    source: str                  # Module that published
    payload: dict
    correlation_id: str

# ---------- Orchestrator types ----------

@dataclass(frozen=True)
class HealthStatus:
    module: str
    healthy: bool
    message: str
    details: dict = field(default_factory=dict)

# ---------- Search types ----------

@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    domain: str | None

# ---------- Consensus configuration ----------

@dataclass(frozen=True)
class ConsensusConfig:
    backends: list[str]                # ["gemini", "chatgpt", "claude"]
    max_rounds: int                    # 1-4, default 4
    min_valid_responses: int           # Minimum backends that must respond, default 2
    max_retries_per_backend: int       # Per-backend retry count per round, default 2
    convergence_threshold: float       # 0.0-1.0, default 0.7
    decision_method: str               # "majority" | "supermajority" | "unanimous"
    minimum_agreement: float           # 0.0-1.0, default 0.66

# ---------- Work planning ----------

@dataclass(frozen=True)
class WorkItem:
    type: str               # "annotation", "insight", "review", "suggestion"
    entity_id: str          # annotation_id, insight_id, or section_id
    priority: int           # Higher = do first

# ---------- Seed modification proposal ----------

@dataclass(frozen=True)
class SeedModification:
    id: str
    seed_id: str
    original_text: str
    proposed_text: str
    reasoning: str
    proposing_thread_id: str
    agreement_ratio: float       # 0.0-1.0
    status: str                  # "pending", "approved", "rejected"
    child_seed_id: str | None    # Set when approved and child seed created
    created_at: datetime
    resolved_at: datetime | None
```

### 2.3 Abstract Interfaces

These are the contracts that modules depend on. Agents mock these during development.

```python
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator

# ---------- Module interface (all engines implement this) ----------

class ModuleInterface(ABC):
    """Every engine module (Explorer, Documenter, Researcher) implements this."""

    @property
    @abstractmethod
    def module_name(self) -> str: ...

    @abstractmethod
    async def initialize(self) -> bool:
        """Set up resources. Return True if ready."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Begin processing. Called after initialize()."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Graceful shutdown. Finish in-flight work, release resources."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Report current health. Called periodically by orchestrator."""
        ...

# ---------- StateStore interface ----------

class StateStoreInterface(ABC):
    """Data access layer. All state reads/writes go through this."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Atomic transaction. Yields control, commits on success, rolls back on exception."""
        ...

    # -- Projects --
    @abstractmethod
    async def create_project(self, project: Project) -> None: ...
    @abstractmethod
    async def get_project(self, project_id: str) -> Project | None: ...
    @abstractmethod
    async def update_project(self, project_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_projects(self, status: ProjectStatus | None = None) -> list[Project]: ...

    # -- Seeds --
    @abstractmethod
    async def create_seed(self, seed: Seed) -> None: ...
    @abstractmethod
    async def get_seed(self, seed_id: str) -> Seed | None: ...
    @abstractmethod
    async def update_seed(self, seed_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_seeds(self, project_id: str, status: SeedStatus | None = None) -> list[Seed]: ...
    @abstractmethod
    async def get_seed_lineage(self, seed_id: str) -> list[Seed]: ...

    # -- Threads --
    @abstractmethod
    async def create_thread(self, thread: Thread) -> None: ...
    @abstractmethod
    async def get_thread(self, thread_id: str) -> Thread | None: ...
    @abstractmethod
    async def update_thread(self, thread_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_threads(self, project_id: str, status: ThreadStatus | None = None) -> list[Thread]: ...

    # -- Exchanges --
    @abstractmethod
    async def create_exchange(self, exchange: Exchange) -> None: ...
    @abstractmethod
    async def get_exchanges(self, thread_id: str) -> list[Exchange]: ...

    # -- Insights --
    @abstractmethod
    async def create_insight(self, insight: Insight) -> None: ...
    @abstractmethod
    async def get_insight(self, insight_id: str) -> Insight | None: ...
    @abstractmethod
    async def update_insight(self, insight_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_insights(self, project_id: str, status: InsightStatus | None = None) -> list[Insight]: ...
    @abstractmethod
    async def get_insights_for_thread(self, thread_id: str) -> list[Insight]: ...

    # -- Insight Comments --
    @abstractmethod
    async def create_insight_comment(self, comment: InsightComment) -> None: ...
    @abstractmethod
    async def get_insight_comments(self, insight_id: str) -> list[InsightComment]: ...

    # -- Seed Modifications --
    @abstractmethod
    async def create_seed_modification(self, modification: SeedModification) -> None: ...
    @abstractmethod
    async def get_seed_modification(self, modification_id: str) -> SeedModification | None: ...
    @abstractmethod
    async def update_seed_modification(self, modification_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_pending_modifications(self, project_id: str) -> list[SeedModification]: ...

    # -- Sections --
    @abstractmethod
    async def create_section(self, section: Section) -> None: ...
    @abstractmethod
    async def get_section(self, section_id: str) -> Section | None: ...
    @abstractmethod
    async def update_section(self, section_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_sections(self, project_id: str) -> list[Section]: ...
    @abstractmethod
    async def get_recent_sections(self, project_id: str, limit: int = 2) -> list[Section]: ...
    @abstractmethod
    async def get_sections_establishing(self, concept_names: list[str]) -> list[Section]: ...
    @abstractmethod
    async def get_sections_by_tags(self, tags: list[str], limit: int = 3) -> list[Section]: ...

    # -- Concepts --
    @abstractmethod
    async def create_concept(self, concept: Concept) -> None: ...
    @abstractmethod
    async def get_concept(self, canonical_name: str, project_id: str) -> Concept | None: ...
    @abstractmethod
    async def list_concepts(self, project_id: str) -> list[Concept]: ...

    # -- Annotations --
    @abstractmethod
    async def create_annotation(self, annotation: Annotation) -> None: ...
    @abstractmethod
    async def get_annotation(self, annotation_id: str) -> Annotation | None: ...
    @abstractmethod
    async def update_annotation(self, annotation_id: str, **fields) -> None: ...
    @abstractmethod
    async def list_annotations(self, project_id: str, status: AnnotationStatus | None = None) -> list[Annotation]: ...

    # -- Sources --
    @abstractmethod
    async def create_source(self, source: Source) -> None: ...
    @abstractmethod
    async def get_source(self, source_id: str) -> Source | None: ...
    @abstractmethod
    async def get_source_by_url(self, url: str, project_id: str) -> Source | None: ...
    @abstractmethod
    async def list_sources(self, project_id: str) -> list[Source]: ...

    # -- Findings --
    @abstractmethod
    async def create_finding(self, finding: Finding) -> None: ...
    @abstractmethod
    async def get_finding(self, finding_id: str) -> Finding | None: ...
    @abstractmethod
    async def list_findings(self, project_id: str, related_insight_id: str | None = None) -> list[Finding]: ...

    # -- Events --
    @abstractmethod
    async def persist_event(self, event: Event) -> None: ...
    @abstractmethod
    async def list_events(self, since: datetime | None = None, topic: str | None = None) -> list[Event]: ...

    # -- Metrics --
    @abstractmethod
    async def record_metric(self, project_id: str, name: str, value: float, labels: dict | None = None) -> None: ...
    @abstractmethod
    async def query_metrics(self, name: str, since: datetime | None = None) -> list[dict]: ...

# ---------- EventBus interface ----------

from typing import Callable, Awaitable

EventHandler = Callable[[Event], Awaitable[None]]

class EventBusInterface(ABC):
    """Async pub/sub event bus."""

    @abstractmethod
    def subscribe(self, topic_pattern: str, handler: EventHandler) -> None:
        """Subscribe to events matching pattern. Supports wildcards: 'explorer.*'"""
        ...

    @abstractmethod
    def unsubscribe(self, topic_pattern: str, handler: EventHandler) -> None: ...

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Persist event to store, then deliver to matching subscribers."""
        ...

    @abstractmethod
    async def replay(self, since: datetime) -> None:
        """Replay persisted events since timestamp. For crash recovery."""
        ...

# ---------- LLM Client interface ----------

class LLMClientInterface(ABC):
    """LLM access layer. Rate-limited, tracked, circuit-broken."""

    @abstractmethod
    async def send(self, backend: str, prompt: str, **kwargs) -> LLMResponse:
        """Send prompt to backend. Raises LLMError on failure after retries."""
        ...

    @abstractmethod
    async def send_structured(self, backend: str, prompt: str, schema: dict) -> LLMResponse:
        """Send prompt expecting JSON response matching schema."""
        ...

    @abstractmethod
    def get_available_backends(self) -> list[str]:
        """Return backends that are not circuit-broken."""
        ...

    @abstractmethod
    def get_backend_status(self, backend: str) -> BackendStatus: ...

# ---------- Consensus interface ----------

class ConsensusTaskInterface(ABC):
    """A task to be evaluated by the consensus engine.

    Implementors: InsightReviewTask, PlanningTask, TrustEvaluationTask.
    """

    @abstractmethod
    def get_prompt(self, round_num: int, prior_rounds: list[RoundResult], backend: str) -> str:
        """Build the prompt for this round and backend."""
        ...

    @abstractmethod
    def parse_response(self, text: str) -> ParsedResponse:
        """Parse one LLM's response into structured form."""
        ...

    @abstractmethod
    def get_evaluation_criteria(self) -> list[EvaluationCriterion]:
        """Return the criteria being evaluated."""
        ...

class ConsensusEngineInterface(ABC):
    """Multi-LLM agreement engine."""

    @abstractmethod
    async def run(self, task: ConsensusTaskInterface, backends: list[str] | None = None) -> ConsensusResult:
        """Execute multi-round consensus. Returns final result."""
        ...
```

### 2.4 Error Types

```python
# shared/errors.py

class ResearchAssistantError(Exception):
    """Base error for all project errors."""
    pass

class StoreError(ResearchAssistantError):
    """Database operation failed."""
    pass

class ConfigError(ResearchAssistantError):
    """Configuration invalid or missing."""
    pass

class LLMError(ResearchAssistantError):
    """LLM call failed after retries."""
    def __init__(self, message: str, backend: str, is_transient: bool):
        super().__init__(message)
        self.backend = backend
        self.is_transient = is_transient

class ConsensusError(ResearchAssistantError):
    """Consensus could not be reached."""
    def __init__(self, message: str, rounds_completed: int, partial_result: dict | None = None):
        super().__init__(message)
        self.rounds_completed = rounds_completed
        self.partial_result = partial_result

class InsufficientResponsesError(ConsensusError):
    """Too many backends failed to get a valid consensus round."""
    pass

class BudgetExhaustedError(ResearchAssistantError):
    """Module has exceeded its LLM budget."""
    def __init__(self, module: str, spent: float, limit: float):
        super().__init__(f"{module} budget exhausted: ${spent:.2f} / ${limit:.2f}")
        self.module = module
        self.spent = spent
        self.limit = limit
```

### 2.5 Event Catalog

Every event in the system. Modules publish and subscribe by topic string.

| Topic | Source Module | Payload Fields | Subscribers |
|-------|-------------|----------------|-------------|
| `explorer.thread.created` | Explorer | `thread_id, seed_id, priority` | Control, Metrics |
| `explorer.thread.retired` | Explorer | `thread_id, reason, insights_produced` | Control, Metrics |
| `explorer.insight.extracted` | Explorer | `insight_id, text, thread_id` | Dedup |
| `explorer.insight.attested` | Explorer | `insight_id, text, tags, scores` | Documenter, Researcher, Control |
| `explorer.insight.discarded` | Explorer | `insight_id, reason, round` | Metrics |
| `explorer.insight.disputed` | Explorer | `insight_id, text, disagreement` | Control |
| `explorer.seed.modification.proposed` | Explorer | `seed_id, original, proposed, reason, agreement_ratio` | Control |
| `documenter.section.added` | Documenter | `section_id, title, establishes` | Researcher, Control |
| `documenter.section.updated` | Documenter | `section_id, reason` | Control |
| `documenter.insight.incorporated` | Documenter | `insight_id, section_id` | Explorer, Control |
| `documenter.insight.disputed` | Documenter | `insight_id, reason` | Control |
| `documenter.research.requested` | Documenter | `topic, context, urgency, requesting_section_id` | Researcher |
| `researcher.finding.stored` | Researcher | `finding_id, topic, summary, source_id` | Documenter, Control |
| `researcher.evidence.supports` | Researcher | `insight_id, evidence_summary, confidence` | Explorer |
| `researcher.evidence.contradicts` | Researcher | `insight_id, evidence_summary, confidence` | Explorer |
| `user.seed.created` | Control | `seed_id, text, priority` | Explorer |
| `user.seed.prioritized` | Control | `seed_id, new_priority` | Explorer |
| `user.seed.retired` | Control | `seed_id` | Explorer |
| `user.seed.modification.approved` | Control | `seed_id, modification_id` | Explorer |
| `user.seed.modification.rejected` | Control | `seed_id, modification_id` | Explorer |
| `user.insight.endorsed` | Control | `insight_id, comment` | Explorer |
| `user.insight.dismissed` | Control | `insight_id, comment` | Explorer |
| `user.insight.reconsider` | Control | `insight_id, reasoning` | Explorer |
| `user.annotation.created` | Control | `annotation_id, type, section_id, content` | Documenter |
| `user.research.requested` | Control | `topic, context` | Researcher |
| `llm.request.completed` | LLMClient | `backend, model, tokens, cost_usd, latency_ms, module` | Metrics, Quota |
| `llm.request.failed` | LLMClient | `backend, error_type, module` | CircuitBreaker |
| `system.module.started` | Orchestrator | `module_name` | Control |
| `system.module.stopped` | Orchestrator | `module_name, reason` | Control |
| `system.budget.warning` | Quota | `module, spent, limit, percent` | Control |

### 2.6 ID Generation

All entity IDs are ULIDs (Universally Unique Lexicographically Sortable Identifiers). They sort chronologically and are globally unique.

```python
import ulid

def generate_id() -> str:
    """Generate a new ULID string."""
    return str(ulid.new())
```

**Dependency**: `python-ulid` package.

### 2.7 Content Hashing

All content hashing uses SHA-256 with normalization. Used for deduplication and change detection.

```python
import hashlib
import re

def content_hash(text: str) -> str:
    """Deterministic content hash. Normalizes whitespace and case."""
    normalized = text.strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
```

---

## 3. Module: shared/store — StateStore

**File**: `shared/store.py`
**Implements**: `StateStoreInterface`
**Dependencies**: `aiosqlite`, `shared/models.py`

### 3.1 Summary

Single SQLite database with WAL mode. All system state flows through this module. Provides ACID transactions, concurrent read access, and typed CRUD operations.

### 3.2 Constructor

```python
class StateStore(StateStoreInterface):
    def __init__(self, db_path: str | Path):
        """
        Args:
            db_path: Path to SQLite file, or ":memory:" for testing.
        """
```

### 3.3 Connection Lifecycle

```python
async def connect(self) -> None:
    """Open database connection. Create tables if they don't exist.

    Executes:
        PRAGMA journal_mode=WAL
        PRAGMA foreign_keys=ON
        PRAGMA busy_timeout=5000
        CREATE TABLE IF NOT EXISTS ... (all tables from schema)
    """

async def close(self) -> None:
    """Close database connection gracefully."""
```

### 3.4 Transaction Semantics

```python
@asynccontextmanager
async def transaction(self) -> AsyncIterator[None]:
    """Atomic transaction block.

    Usage:
        async with store.transaction():
            await store.update_insight(id, status="attested")
            await store.create_section(section)
            # Both succeed or both fail

    On exception: ROLLBACK
    On success: COMMIT
    """
```

### 3.5 Schema

The full SQL schema is defined in `ARCHITECTURE_V2.md` Section 11. The `StateStore` creates all tables on `connect()` using `CREATE TABLE IF NOT EXISTS`.

Tables: `projects`, `seeds`, `seed_comments`, `seed_modifications`, `threads`, `exchanges`, `insights`, `insight_dependencies`, `insight_comments`, `sections`, `concepts`, `annotations`, `sources`, `findings`, `events`, `metrics`.

Additional tables beyond architecture doc:
- `seed_comments` — user comments on seeds (parallels `insight_comments`)
- `seed_modifications` — modification proposals with status tracking

Indexes: On all foreign keys, status columns, and common query patterns (see architecture doc).

### 3.6 CRUD Method Details

Each entity follows the same pattern. Taking `Seed` as the example:

```python
async def create_seed(self, seed: Seed) -> None:
    """Insert a new seed. Raises StoreError if id already exists."""

async def get_seed(self, seed_id: str) -> Seed | None:
    """Return seed by ID, or None if not found."""

async def update_seed(self, seed_id: str, **fields) -> None:
    """Update specified fields. Automatically sets updated_at=now().
    Raises StoreError if seed_id not found.

    Allowed fields: text, type, priority, tags, confidence, source,
                    notes, status, parent_seed_id, modification_reason,
                    exploration_count
    """

async def list_seeds(
    self, project_id: str, status: SeedStatus | None = None
) -> list[Seed]:
    """List seeds for project, optionally filtered by status.
    Ordered by priority DESC, created_at ASC.
    """

async def get_seed_lineage(self, seed_id: str) -> list[Seed]:
    """Return the chain: seed → parent → grandparent → ... root.
    Ordered from newest to oldest.
    """
```

All other entities follow this pattern. Specific extra queries are documented in the interface (Section 2.3).

### 3.7 JSON Serialization

Columns storing JSON (tags, evaluation_criteria, evaluation_scores, research_domains, etc.) are automatically serialized/deserialized:

- On write: `json.dumps(value)` if value is list or dict
- On read: `json.loads(value)` to reconstruct the Python type
- Stored as `TEXT` in SQLite

### 3.8 Date Handling

All datetime columns store ISO 8601 strings (`datetime.now(timezone.utc).isoformat()`). On read, parse back to `datetime` using `datetime.fromisoformat()`.

### 3.9 Test Requirements

| Test | Description |
|------|-------------|
| `test_connect_creates_tables` | Fresh DB has all tables after connect() |
| `test_connect_idempotent` | Multiple connect() calls don't fail |
| `test_crud_all_entities` | Create, get, update, list for every entity type |
| `test_transaction_commit` | Changes visible after successful transaction |
| `test_transaction_rollback` | Changes reverted on exception |
| `test_concurrent_reads` | Multiple readers don't block |
| `test_get_nonexistent_returns_none` | get_seed("nonexistent") returns None |
| `test_update_nonexistent_raises` | update_seed("nonexistent") raises StoreError |
| `test_json_roundtrip` | list/dict fields survive write→read cycle |
| `test_list_filtering` | Status filter, project filter work correctly |
| `test_seed_lineage` | Multi-generation lineage returns correct chain |
| `test_foreign_key_enforcement` | Can't create seed with nonexistent project_id |

---

## 4. Module: shared/events — EventBus

**File**: `shared/events.py`
**Implements**: `EventBusInterface`
**Dependencies**: `shared/store.py` (for event persistence), `shared/models.py`

### 4.1 Summary

In-process async pub/sub with topic-based routing and wildcard support. Events are persisted to the StateStore before delivery, enabling replay after crash.

### 4.2 Constructor

```python
class EventBus(EventBusInterface):
    def __init__(self, store: StateStoreInterface):
        """
        Args:
            store: StateStore for event persistence.
        """
```

### 4.3 Topic Matching

Topics use dot-separated hierarchy. Wildcards match one level (`*`) or all remaining levels (`**`).

```
"explorer.insight.attested"  matches  "explorer.insight.attested"   ✓
"explorer.*"                 matches  "explorer.insight.attested"   ✗ (only one level)
"explorer.**"                matches  "explorer.insight.attested"   ✓
"*.insight.*"                matches  "explorer.insight.attested"   ✓
"user.**"                    matches  "user.seed.created"           ✓
```

Implementation: compile patterns to regex on subscribe, match against event topic on publish.

### 4.4 Publish Flow

```
publish(event) →
    1. Persist event to store (await store.persist_event(event))
    2. Find all handlers matching event.topic
    3. For each handler, call handler(event)
    4. If a handler raises, log the error and continue to next handler
       (one handler failure must not block other handlers)
```

### 4.5 Replay

```python
async def replay(self, since: datetime) -> None:
    """Load events from store since timestamp, deliver to current subscribers.
    Used after crash recovery to catch up on missed events.
    """
```

### 4.6 Thread Safety

The EventBus is single-threaded async. All publish/subscribe operations happen in the same event loop. No locks needed.

### 4.7 Test Requirements

| Test | Description |
|------|-------------|
| `test_subscribe_and_publish` | Handler receives published event |
| `test_wildcard_matching` | `explorer.**` matches `explorer.insight.attested` |
| `test_single_level_wildcard` | `*` matches exactly one level |
| `test_handler_error_isolation` | Failing handler doesn't prevent other handlers |
| `test_event_persisted` | Published event appears in store.list_events() |
| `test_replay` | Events published before subscriber added are delivered on replay |
| `test_unsubscribe` | Unsubscribed handler no longer receives events |
| `test_multiple_subscribers` | Multiple handlers on same topic all receive event |
| `test_no_match_no_delivery` | Non-matching topic doesn't trigger handler |
| `test_publish_ordering` | Events delivered in publish order |

---

## 5. Module: shared/config — Config

**File**: `shared/config.py`
**Implements**: Config (no abstract interface; consumed directly)
**Dependencies**: `pyyaml`, `shared/models.py`, `shared/errors.py`

### 5.1 Summary

Two-level configuration: infrastructure (`config.yaml`) + project (`projects/<name>.yaml`). Supports environment variable overrides with `FANO_` prefix.

### 5.2 Constructor and Loading

```python
class Config:
    def __init__(
        self,
        config_path: Path | None = None,
        project_path: Path | None = None,
    ):
        """
        Args:
            config_path: Path to infrastructure config. Default: FANO_ROOT / "config.yaml"
            project_path: Path to project config. Optional.
        """

    def load(self) -> None:
        """Load and validate config files. Apply environment variable overrides.
        Raises ConfigError if validation fails.
        """
```

### 5.3 Access Methods

```python
def get(self, dotted_key: str, default=None) -> Any:
    """Access nested infrastructure config.

    Examples:
        config.get("llm.models.gemini")  → "google/gemini-2.0-flash-..."
        config.get("explorer.max_active_threads")  → 3
        config.get("nonexistent.key", 42)  → 42
    """

@property
def project(self) -> Project | None:
    """Parsed project configuration as Project model, or None if no project loaded."""
```

### 5.4 Environment Variable Overrides

Pattern: `FANO_<DOTTED_PATH_WITH_DOUBLE_UNDERSCORE>` overrides config values.

```
FANO_LLM__MODELS__GEMINI="google/gemini-pro"
→ overrides config.get("llm.models.gemini")

FANO_EXPLORER__MAX_ACTIVE_THREADS=5
→ overrides config.get("explorer.max_active_threads")
```

Type coercion: `"true"/"false"` → bool, numeric strings → int/float, everything else → str.

### 5.5 Validation

On `load()`, validate:

1. Required keys exist: `llm.api_key_env`, `llm.models`, `consensus.backends`
2. `llm.api_key_env` references an environment variable that is set
3. All backends in `consensus.backends` exist in `llm.models`
4. Numeric values are in valid ranges (priorities 1-10, thresholds 0-1, etc.)
5. If project config loaded: `goal` and `evaluation_criteria` are non-empty

Raise `ConfigError` with specific message on failure.

### 5.6 Test Requirements

| Test | Description |
|------|-------------|
| `test_load_infrastructure_config` | Loads config.yaml, all keys accessible |
| `test_load_project_config` | Project model parsed correctly |
| `test_dotted_key_access` | Nested keys via `get("a.b.c")` |
| `test_default_value` | Missing key returns default |
| `test_env_override` | Environment variable overrides YAML value |
| `test_env_type_coercion` | "true" → bool, "42" → int, "3.14" → float |
| `test_validation_missing_required` | ConfigError on missing `llm.api_key_env` |
| `test_validation_invalid_backend` | ConfigError if consensus backend not in llm.models |
| `test_no_project_config` | config.project is None when no project loaded |

---

## 6. Module: llm/client — LLMClient

**File**: `llm/src/client.py`
**Implements**: `LLMClientInterface`
**Dependencies**: `aiohttp`, `shared/config.py`, `shared/events.py`, `shared/models.py`

### 6.1 Summary

OpenRouter API client with per-backend rate limiting, retry with exponential backoff, circuit breaker, and token/cost tracking. Every request publishes metrics events.

### 6.2 Constructor

```python
class LLMClient(LLMClientInterface):
    def __init__(
        self,
        config: Config,
        event_bus: EventBusInterface,
    ):
        """
        Reads from config:
            llm.api_key_env       → env var name for API key
            llm.endpoint          → OpenRouter URL
            llm.models            → {backend_name: model_id}
            llm.rate_limits       → {backend_name: requests_per_minute}
            llm.default_timeout_seconds
            llm.max_retries
        """
```

### 6.3 Core Methods

```python
async def send(
    self,
    backend: str,
    prompt: str,
    *,
    module: str = "unknown",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> LLMResponse:
    """Send prompt to backend via OpenRouter.

    Flow:
        1. Check circuit breaker → raise LLMError(is_transient=True) if open
        2. Wait for rate limiter slot
        3. POST to OpenRouter with model mapping
        4. On success: record_success on circuit breaker, publish llm.request.completed
        5. On failure: retry up to max_retries with exponential backoff
        6. After all retries exhausted: record_failure on circuit breaker,
           publish llm.request.failed, raise LLMError

    Args:
        backend: Logical name ("gemini", "chatgpt", "claude")
        prompt: The prompt text
        module: Calling module name (for cost tracking)
        temperature: Sampling temperature
        max_tokens: Max completion tokens (None = model default)
        timeout: Override default timeout in seconds

    Returns: LLMResponse with text, token usage, cost estimate

    Raises: LLMError (is_transient=True for retryable, False for permanent)
    """

async def send_structured(
    self,
    backend: str,
    prompt: str,
    schema: dict,
    *,
    module: str = "unknown",
) -> LLMResponse:
    """Send prompt expecting JSON response matching schema.

    Adds response_format={"type": "json_object"} to the API call
    if the backend supports it. Falls back to regular send + JSON
    extraction if not.
    """
```

### 6.4 Internal Components

#### RateLimiter

```python
class RateLimiter:
    """Token bucket rate limiter, per-backend."""

    def __init__(self, limits: dict[str, int]):
        """Args: limits = {"gemini": 10, "chatgpt": 60, ...} (requests/minute)"""

    async def acquire(self, backend: str) -> None:
        """Wait until a request slot is available for this backend."""
```

#### CircuitBreaker

```python
class CircuitBreaker:
    """Per-backend circuit breaker. States: closed → open → half_open → closed."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: int = 60):
        ...

    def can_request(self, backend: str) -> bool:
        """Returns True if requests should be attempted for this backend."""

    def record_success(self, backend: str) -> None:
        """Reset failure count, transition to closed."""

    def record_failure(self, backend: str) -> None:
        """Increment failure count. If >= threshold, transition to open."""
```

#### CostEstimator

```python
class CostEstimator:
    """Estimates USD cost from token counts and model pricing."""

    # Pricing table (updated periodically)
    MODEL_PRICING: dict[str, tuple[float, float]]  # model_id → (input_$/1M, output_$/1M)

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return estimated cost in USD."""
```

### 6.5 Test Requirements

| Test | Description |
|------|-------------|
| `test_send_success` | Mock HTTP 200 → returns LLMResponse with text |
| `test_send_retry_on_500` | Mock HTTP 500 → retries → succeeds on 2nd try |
| `test_send_raises_after_max_retries` | All retries fail → LLMError raised |
| `test_rate_limiter_blocks` | Exceeding rate limit causes wait |
| `test_circuit_breaker_opens` | 5 failures → circuit opens → immediate LLMError |
| `test_circuit_breaker_half_open` | After recovery timeout → allows test request |
| `test_circuit_breaker_recovers` | Success in half_open → back to closed |
| `test_token_usage_tracked` | Response includes TokenUsage with correct counts |
| `test_cost_estimation` | Estimated cost matches expected for known model pricing |
| `test_event_published_on_success` | `llm.request.completed` event published |
| `test_event_published_on_failure` | `llm.request.failed` event published |
| `test_available_backends_excludes_open_circuits` | get_available_backends() skips broken backends |

---

## 7. Module: llm/consensus — ConsensusEngine

**Files**: `llm/src/consensus/engine.py`, `llm/src/consensus/parsing.py`
**Implements**: `ConsensusEngineInterface`
**Dependencies**: `llm/src/client.py`, `shared/config.py`, `shared/models.py`

### 7.1 Summary

Multi-LLM agreement engine. Runs 1-4 rounds of structured debate to reach consensus on a task. Used by Explorer (insight attestation), Documenter (planning), and Researcher (trust evaluation). Each use case provides a `ConsensusTaskInterface` implementation.

### 7.2 Configuration

Read from `config.yaml` under `consensus:`:

```python
@dataclass
class ConsensusConfig:
    backends: list[str]             # ["gemini", "chatgpt", "claude"]
    max_rounds: int                 # 1-4, default 4
    min_valid_responses: int        # Minimum backends that must respond, default 2
    max_retries_per_backend: int    # Per-backend retry count per round, default 2
    convergence_threshold: float    # 0.0-1.0, default 0.7
    decision_method: str            # "majority" | "supermajority" | "unanimous"
    minimum_agreement: float        # 0.0-1.0, default 0.66
```

### 7.3 Engine: run()

```python
async def run(
    self,
    task: ConsensusTaskInterface,
    backends: list[str] | None = None,
) -> ConsensusResult:
    """Execute multi-round consensus.

    Algorithm:
        for round_num in 1..max_rounds:
            1. Build prompt for each backend using task.get_prompt(round_num, prior_rounds, backend)
            2. Send prompts in parallel (asyncio.gather)
            3. Parse each response using task.parse_response(text)
            4. Filter: only ParsedResponse.is_valid == True count as votes
            5. If valid_count < min_valid_responses:
                 retry the round (up to max_retries_per_backend per backend)
                 if still insufficient: raise InsufficientResponsesError
            6. Check convergence:
                 - If all verdicts agree → converged
                 - If agreement ratio >= minimum_agreement and decision_method allows → converged
                 - Otherwise → continue to next round
            7. Store RoundResult

        After all rounds:
            Compile ConsensusResult from round history.
            If never converged: verdict = UNCERTAIN, confidence = agreement ratio

    Round prompt building:
        Round 1: Independent evaluation (no knowledge of others)
        Round 2+: Include prior round results (each backend sees what others said)
        This is handled entirely by task.get_prompt() — the engine just passes prior_rounds.
    """
```

### 7.4 Convergence Detection

```python
def _check_convergence(
    self,
    responses: list[ValidatedResponse],
    round_num: int,
) -> tuple[bool, Verdict | None]:
    """Determine if consensus has been reached.

    Decision logic (configurable via decision_method):

    "unanimous":
        All verdicts must match → converged with that verdict

    "majority":
        Count verdicts. If most common >= minimum_agreement * total → converged.

    "supermajority":
        Most common >= 0.66 * total → converged.

    Returns: (is_converged, winning_verdict_or_none)
    """
```

### 7.5 Score Aggregation

When consensus is reached, scores are aggregated across valid responses:

```python
def _aggregate_scores(
    self, responses: list[ValidatedResponse]
) -> dict[str, float]:
    """Average scores across valid responses, per criterion.

    For each criterion name:
        score = mean of all responses that provided a score for that criterion
        (skip responses that didn't score a particular criterion)
    """
```

### 7.6 Response Parsing Module

**File**: `llm/src/consensus/parsing.py`

```python
def parse_review_response(
    text: str,
    criteria_names: list[str],
) -> ParsedResponse:
    """Multi-strategy parser for LLM review responses.

    Strategy 1 — JSON block:
        Look for ```json ... ``` in response.
        Expected schema: {"verdict": "accept"|"reject"|"uncertain",
                          "scores": {"criterion_name": float}, "reasoning": "..."}

    Strategy 2 — Regex:
        Verdict: look for ⚡|accept|ACCEPT → accept, ✗|reject|REJECT → reject,
                 ?|uncertain|UNCERTAIN → uncertain  (case-insensitive)
        Scores: look for "criterion_name: N/10" or "criterion_name: N" patterns
        Reasoning: entire text (minus extracted fields)

    Strategy 3 — Unrecognized:
        Return ParsedResponse(is_valid=False, error="Could not parse verdict")

    CRITICAL: Never return is_valid=True for empty responses, error messages
    ("I apologize", "I cannot"), or responses that don't contain any identifiable verdict.
    """

def normalize_verdict(raw: str) -> Verdict | None:
    """Normalize verdict string to Verdict enum.

    Handles: "⚡", "accept", "Accept", "ACCEPT", "bless", "approve" → Verdict.ACCEPT
             "✗", "reject", "Reject", "REJECT", "discard" → Verdict.REJECT
             "?", "uncertain", "Uncertain", "UNCERTAIN", "unsure" → Verdict.UNCERTAIN

    Returns None if not recognized.
    """
```

### 7.7 Test Requirements

| Test | Description |
|------|-------------|
| `test_unanimous_accept_round1` | All backends accept → attested in round 1 |
| `test_unanimous_reject_round1` | All backends reject → discarded in round 1 |
| `test_split_proceeds_to_round2` | Mixed verdicts → round 2 with prior context |
| `test_majority_converges` | 2/3 agree in round 2 → converged |
| `test_error_response_excluded` | Backend returns error → not counted as vote |
| `test_insufficient_responses_retries` | Too many failures → round retried |
| `test_insufficient_responses_raises` | All retries exhausted → InsufficientResponsesError |
| `test_score_aggregation` | Scores averaged correctly across valid responses |
| `test_max_rounds_reached` | 4 rounds without convergence → uncertain verdict |
| `test_variable_backend_count` | Works with 2, 3, 4, 5 backends |
| `test_parse_json_response` | JSON block parsed correctly |
| `test_parse_regex_response` | Symbol/text verdict + scores parsed |
| `test_parse_empty_response` | Empty string → is_valid=False |
| `test_parse_error_message` | "I apologize..." → is_valid=False |
| `test_parse_case_insensitive` | "ACCEPT", "accept", "Accept" all → Verdict.ACCEPT |
| `test_property_deterministic` | Same votes → same result (property-based) |
| `test_property_unanimous_accept` | All-accept always converges (property-based) |

---

## 8. Module: explorer — Explorer Engine

**Files**: `explorer/src/engine.py`, `explorer/src/thread_manager.py`, `explorer/src/seed_manager.py`, `explorer/src/insight_extractor.py`, `explorer/src/review_panel.py`, `explorer/src/prompts.py`
**Implements**: `ModuleInterface`
**Dependencies**: `shared/store`, `shared/events`, `shared/config`, `llm/client`, `llm/consensus`

### 8.1 Summary

Pure LLM reasoning engine. Takes seeds, creates exploration threads, generates multi-turn debates between LLMs, extracts insights from debates, and runs consensus-based review to attest/discard/dispute each insight.

### 8.2 ExplorerEngine (Facade)

**File**: `explorer/src/engine.py`

```python
class ExplorerEngine(ModuleInterface):
    """Main entry point for the Explorer module."""

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface,
        config: Config,
    ): ...

    @property
    def module_name(self) -> str:
        return "explorer"

    async def initialize(self) -> bool:
        """Load active project. Set up internal components.
        Subscribe to events: user.seed.*, user.insight.*, researcher.evidence.*
        """

    async def start(self) -> None:
        """Begin exploration loop:
            while running:
                1. Select highest-priority seed that hasn't exceeded max exploration count
                2. Create or resume a thread for that seed
                3. Run exchanges until chunk_ready (min_exchanges reached)
                4. Extract insights from thread
                5. Review each insight via consensus
                6. Transition insights to attested/discarded/disputed/interesting
                7. Check for seed modification proposals
                8. Sleep briefly, repeat
        """

    async def stop(self) -> None:
        """Save thread state atomically. Mark module as stopped."""

    async def health_check(self) -> HealthStatus:
        """Return active thread count, last insight time, queue depth."""
```

### 8.3 ThreadManager

**File**: `explorer/src/thread_manager.py`

Manages thread lifecycle: creation, exchange execution, priority adjustment, retirement.

```python
class ThreadManager:
    def __init__(
        self,
        store: StateStoreInterface,
        llm_client: LLMClientInterface,
        event_bus: EventBusInterface,
        config: Config,
    ): ...

    async def create_thread(self, seed: Seed, project: Project) -> Thread:
        """Create a new thread for exploring a seed.
        Publishes: explorer.thread.created
        Reads config: explorer.max_active_threads
        """

    async def run_exchange(self, thread: Thread, project: Project) -> Exchange:
        """Run one exploration exchange.

        Role rotation:
            exchange 1, 4, 7, ...: explorer (creative reasoning)
            exchange 2, 5, 8, ...: critic (challenge/pushback)
            exchange 3, 6, 9, ...: synthesizer (identify core insights)

        Backend selection:
            Weighted random from config.explorer.model_weights[role]

        Prompt built by: prompts.build_exchange_prompt(project, thread, exchanges, role)

        Returns the new Exchange, persisted to store.
        """

    async def is_chunk_ready(self, thread: Thread) -> bool:
        """True if thread has >= min_exchanges_for_chunk since last extraction."""

    async def should_retire(self, thread: Thread) -> tuple[bool, str | None]:
        """Check retirement conditions:
        - exchange_count >= max_exchanges_per_thread → "max_exchanges"
        - idle for > max_idle_hours → "idle_timeout"
        - all insights from last extraction were discarded → "low_quality"
        Returns (should_retire, reason)
        """

    async def retire_thread(self, thread: Thread, reason: str) -> None:
        """Transition to RETIRED. Publish explorer.thread.retired."""

    async def adjust_priority(self, thread_id: str, delta: int) -> None:
        """Adjust thread priority by delta (clamped to 1-10).
        Called when user endorses/dismisses related insights.
        """

    async def get_active_threads(self, project_id: str) -> list[Thread]:
        """Return threads with status ACTIVE, ordered by priority DESC."""
```

**Configuration read**:
- `explorer.max_active_threads` (default 3)
- `explorer.min_exchanges_for_chunk` (default 4)
- `explorer.max_exchanges_per_thread` (default 12)
- `explorer.thread_retirement.max_idle_hours` (default 48)
- `explorer.model_weights.exploration` — backend weights for explorer role
- `explorer.model_weights.critique` — backend weights for critic role

### 8.4 SeedManager

**File**: `explorer/src/seed_manager.py`

Manages seed lifecycle and modification proposals.

```python
class SeedManager:
    def __init__(self, store: StateStoreInterface, event_bus: EventBusInterface, config: Config): ...

    async def select_next_seed(self, project_id: str) -> Seed | None:
        """Select the highest-priority active seed that hasn't exceeded max exploration count.
        Returns None if no seeds available.

        Query: WHERE project_id=? AND status='active'
               AND exploration_count < max_reexplore_count
               ORDER BY priority DESC, created_at ASC
               LIMIT 1
        """

    async def record_exploration(self, seed_id: str) -> None:
        """Increment exploration_count. If >= max, transition to EXPLORED."""

    async def propose_modification(
        self,
        seed: Seed,
        proposed_text: str,
        reasoning: str,
        proposing_thread_id: str,
        agreement_ratio: float,
    ) -> SeedModification:
        """Create a seed modification proposal.

        If agreement_ratio < 0.66 (supermajority): reject automatically.
        If project.seed_modification_require_approval: queue for user.
        If auto-approve: immediately create child seed and publish event.

        Publishes: explorer.seed.modification.proposed (if queued for user)
        """

    async def approve_modification(self, seed_id: str, modification: SeedModification) -> Seed:
        """Apply an approved modification.
        1. Mark original seed as EVOLVED
        2. Create new child seed with parent_seed_id=original, status=ACTIVE
        3. Return the new child seed
        """

    async def handle_user_event(self, event: Event) -> None:
        """Event handler for user.seed.* events.
        - user.seed.created → create seed in store
        - user.seed.prioritized → update priority
        - user.seed.retired → mark as RETIRED
        - user.seed.modification.approved → approve_modification()
        - user.seed.modification.rejected → log, keep original
        """
```

**Configuration read**:
- `explorer.thread_retirement.max_reexplore_count` (default 3)
- Project: `seed_modification_enabled`, `seed_modification_require_approval`

### 8.5 InsightExtractor

**File**: `explorer/src/insight_extractor.py`

Extracts atomic insights from thread exchanges and deduplicates them.

```python
class InsightExtractor:
    def __init__(
        self,
        store: StateStoreInterface,
        llm_client: LLMClientInterface,
        event_bus: EventBusInterface,
        config: Config,
    ): ...

    async def extract(self, thread: Thread, project: Project) -> list[Insight]:
        """Extract atomic insights from thread exchanges.

        1. Load all exchanges for this thread
        2. Build extraction prompt using prompts.build_extraction_prompt()
        3. Send to LLM (use the synthesizer backend)
        4. Parse response into list of raw insight texts
        5. For each raw insight:
           a. Compute content_hash
           b. Check against existing insights for same thread (dedup within thread)
           c. Check against all project insights (dedup across threads)
           d. If unique: create Insight with status=EXTRACTED
           e. Publish explorer.insight.extracted
        6. Return list of new (non-duplicate) insights

        Parsing format expected from LLM:
            INSIGHT 1: <text>
            CONFIDENCE: high|medium|low
            TAGS: tag1, tag2, tag3

            INSIGHT 2: <text>
            ...

        If parsing fails, fall back to splitting on double-newline
        and treating each block as a low-confidence insight.
        """

    async def _is_duplicate(self, text: str, project_id: str, thread_id: str) -> bool:
        """Check if insight text is a duplicate.
        1. Content hash match against project insights → definite duplicate
        2. If no hash match but text is suspiciously similar (>0.8 similarity),
           optionally use LLM to confirm → probable duplicate
        """
```

### 8.6 ReviewPanel

**File**: `explorer/src/review_panel.py`

Runs consensus-based review of extracted insights using the shared ConsensusEngine.

```python
class ReviewPanel:
    def __init__(
        self,
        consensus: ConsensusEngineInterface,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
    ): ...

    async def review(self, insight: Insight, project: Project) -> InsightStatus:
        """Review an insight via multi-round consensus.

        1. Update insight status to REVIEWING
        2. Create InsightReviewTask(insight, project)
        3. Run consensus engine
        4. Map ConsensusResult to InsightStatus:
             verdict=ACCEPT, confidence >= 0.8 → ATTESTED
             verdict=ACCEPT, confidence < 0.8  → INTERESTING
             verdict=REJECT                    → DISCARDED
             verdict=UNCERTAIN                 → DISPUTED
        5. Update insight with scores, review_record, new status
        6. Publish appropriate event (attested/discarded/disputed)
        7. Check for seed modification proposals in review reasoning
        8. Return new status

        On ConsensusError: mark as TRANSIENT_FAILURE, schedule retry
        """

    async def _check_for_modifications(
        self, insight: Insight, result: ConsensusResult, project: Project
    ) -> None:
        """Scan review reasoning for proposed seed modifications.
        If reviewers suggest the seed should be reformulated,
        pass to SeedManager.propose_modification().
        """


class InsightReviewTask(ConsensusTaskInterface):
    """Consensus task for reviewing an insight against project criteria."""

    def __init__(self, insight: Insight, project: Project): ...

    def get_prompt(
        self, round_num: int, prior_rounds: list[RoundResult], backend: str
    ) -> str:
        """Build review prompt.

        Round 1: Independent evaluation
            "You are reviewing a proposed insight for a research project.
             PROJECT GOAL: {project.goal}
             PROJECT CONTEXT: {project.context}
             EVALUATION CRITERIA: {formatted_criteria}
             EXPLORATION GUIDANCE: {project.exploration_guidance}
             The insight to review: {insight.text}
             Rate on each criterion (1-10) with reasoning.
             Give overall verdict: accept / reject / uncertain.
             Respond in JSON: {\"verdict\": ..., \"scores\": {...}, \"reasoning\": ...}"

        Round 2+: With prior context
            Append: "Previous round results: {formatted_prior_rounds}
                     Reconsider your position in light of other reviewers' reasoning."
        """

    def parse_response(self, text: str) -> ParsedResponse:
        """Delegate to parsing.parse_review_response()"""

    def get_evaluation_criteria(self) -> list[EvaluationCriterion]:
        """Return project.evaluation_criteria"""
```

### 8.7 Prompt Templates

**File**: `explorer/src/prompts.py`

All prompts are functions that take project config and return strings. Zero hardcoded domain content.

```python
def build_exchange_prompt(
    project: Project,
    thread: Thread,
    exchanges: list[Exchange],
    role: ExchangeRole,
    seed: Seed,
) -> str:
    """Build prompt for one exploration exchange.

    Structure:
        SYSTEM CONTEXT: {project.context}
        RESEARCH GOAL: {project.goal}
        EVALUATION CRITERIA: {formatted_criteria}
        GUIDANCE: {project.exploration_guidance}
        SEED BEING EXPLORED: {seed.text}
        CONVERSATION SO FAR: {formatted_exchanges}
        YOUR ROLE: {role_description}
        RESPOND:

    Role descriptions:
        EXPLORER: "Build on this conversation. Develop the idea further.
                   Follow your curiosity. Propose novel connections."
        CRITIC: "Challenge the reasoning. Find weaknesses. Push for rigor.
                 Is this inevitable or forced?"
        SYNTHESIZER: "Identify the core insight emerging. What's the atomic claim?
                      What needs more development?"
    """

def build_extraction_prompt(
    project: Project,
    exchanges: list[Exchange],
) -> str:
    """Build prompt for extracting atomic insights from a thread.

    "Given this research conversation, extract the atomic insights — claims
     that can be independently evaluated. For each insight, specify
     confidence (high/medium/low) and relevant tags.

     PROJECT GOAL: {project.goal}
     CONVERSATION: {formatted_exchanges}

     Format each insight as:
     INSIGHT N: <concise, self-contained claim>
     CONFIDENCE: high|medium|low
     TAGS: tag1, tag2, ..."
    """
```

### 8.8 Event Subscriptions

| Event | Handler | Action |
|-------|---------|--------|
| `user.seed.created` | SeedManager.handle_user_event | Create seed in store |
| `user.seed.prioritized` | SeedManager.handle_user_event | Update seed priority |
| `user.seed.retired` | SeedManager.handle_user_event | Mark seed RETIRED |
| `user.seed.modification.approved` | SeedManager.handle_user_event | Create child seed |
| `user.seed.modification.rejected` | SeedManager.handle_user_event | Log, keep original |
| `user.insight.endorsed` | ExplorerEngine._on_insight_endorsed | Boost source thread priority |
| `user.insight.dismissed` | ExplorerEngine._on_insight_dismissed | Reduce source thread priority |
| `user.insight.reconsider` | ExplorerEngine._on_insight_reconsider | Re-queue for review |
| `researcher.evidence.supports` | ExplorerEngine._on_evidence | Boost thread priority |
| `researcher.evidence.contradicts` | ExplorerEngine._on_evidence | Reduce thread priority |

### 8.9 Test Requirements

| Test | Description |
|------|-------------|
| `test_select_seed_highest_priority` | Highest-priority active seed selected |
| `test_select_seed_skips_exhausted` | Seeds at max exploration_count skipped |
| `test_thread_creation` | New thread created with correct seed link |
| `test_exchange_role_rotation` | Roles cycle: explorer, critic, synthesizer |
| `test_chunk_ready_threshold` | True only after min_exchanges reached |
| `test_thread_retirement_max_exchanges` | Retires at max_exchanges |
| `test_thread_retirement_idle` | Retires after max_idle_hours |
| `test_insight_extraction_basic` | Extracts insights from exchange text |
| `test_insight_dedup_within_thread` | Duplicate within same thread rejected |
| `test_insight_dedup_across_threads` | Duplicate across threads rejected |
| `test_review_attested` | Unanimous accept → ATTESTED status |
| `test_review_discarded` | Unanimous reject → DISCARDED status |
| `test_review_disputed` | Split → DISPUTED status |
| `test_review_transient_failure` | ConsensusError → TRANSIENT_FAILURE |
| `test_seed_modification_auto_approve` | High agreement + no approval required → child created |
| `test_seed_modification_needs_approval` | Queued for user when require_approval=true |
| `test_user_endorse_boosts_thread` | user.insight.endorsed → thread priority increased |
| `test_user_dismiss_reduces_thread` | user.insight.dismissed → thread priority decreased |
| `test_exploration_loop_integration` | Full cycle: seed → thread → exchanges → extract → review |

---

## 9. Module: documenter — Documenter Engine

**Files**: `documenter/src/engine.py`, `documenter/src/planner.py`, `documenter/src/processor.py`, `documenter/src/annotations.py`, `documenter/src/context.py`, `documenter/src/renderer.py`, `documenter/src/prompts.py`
**Implements**: `ModuleInterface`
**Dependencies**: `shared/store`, `shared/events`, `shared/config`, `llm/client`, `llm/consensus`

### 9.1 Summary

Takes attested insights and synthesizes them into a coherent document. Manages document structure, concept tracking, user annotations, and a work planning cycle that balances new material incorporation with existing content review.

### 9.2 DocumenterEngine (Facade)

**File**: `documenter/src/engine.py`

```python
class DocumenterEngine(ModuleInterface):
    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface,
        config: Config,
    ): ...

    @property
    def module_name(self) -> str:
        return "documenter"

    async def initialize(self) -> bool:
        """Subscribe to events. Load document state from store.
        Create initial document structure if project is new.
        """

    async def start(self) -> None:
        """Begin documenter work loop:
            while running:
                1. Plan: choose work items (annotations first, then insights, then review)
                2. Execute: process each work item
                3. Render: regenerate markdown from database state
                4. Snapshot: if structural changes occurred, archive current version
                5. Sleep, repeat
        """

    async def stop(self) -> None: ...
    async def health_check(self) -> HealthStatus: ...
```

### 9.3 Planner

**File**: `documenter/src/planner.py`

Decides what work to do in each cycle.

```python
class Planner:
    def __init__(self, store: StateStoreInterface, config: Config): ...

    async def plan_cycle(self, project_id: str) -> list[WorkItem]:
        """Plan the next work cycle.

        Priority order:
            1. Open annotations (type=comment) — user requests are highest priority
            2. Attested insights not yet incorporated — new material
            3. Sections needing review — existing content maintenance
            4. Suggestions (type=suggestion) — optional enhancements

        Work allocation:
            Track actual LLM calls spent on categories.
            If new_material_ratio drifts beyond config target ± 10%, rebalance.

        Returns ordered list of WorkItem(type, entity_id, priority).
        """

@dataclass
class WorkItem:
    type: str               # "annotation", "insight", "review", "suggestion"
    entity_id: str          # annotation_id, insight_id, or section_id
    priority: int           # Higher = do first
```

### 9.4 Processor

**File**: `documenter/src/processor.py`

Processes individual work items: incorporates insights, addresses annotations, reviews sections.

```python
class Processor:
    def __init__(
        self,
        store: StateStoreInterface,
        llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface,
        event_bus: EventBusInterface,
        context_builder: ContextBuilder,
        config: Config,
    ): ...

    async def incorporate_insight(self, insight: Insight, project: Project) -> None:
        """Incorporate an attested insight into the document.

        Pipeline (all within a transaction):
            1. DEDUP: Check if insight is already represented in existing sections
               → If duplicate: mark INCORPORATED, skip remaining stages
            2. PREREQUISITES: Identify concepts this insight depends on
               → If missing: create prerequisite section first
            3. DRAFT: Generate section content using LLM
               → Prompt includes: insight text, relevant context, project guidance
            4. EVALUATE: Use consensus to evaluate draft quality
               → On dispute (substantive): increment insight.dispute_count
               → On transient failure: increment insight.transient_failure_count, retry later
            5. ADD: Insert section into document at appropriate position
               → Update insight status to INCORPORATED
               → Register new concepts in concept table
               → Publish documenter.insight.incorporated

        CRITICAL: Dispute vs failure separation.
            - LLMError(is_transient=True) → transient_failure_count, retry later
            - Consensus verdict=REJECT → dispute_count, if >= max → SHELVED
        """

    async def address_annotation(self, annotation: Annotation, project: Project) -> None:
        """Address a user annotation.

        For type=COMMENT:
            1. Load the section referenced by annotation.section_id
            2. Build context with section content + annotation content
            3. Generate revised section content via LLM
            4. Use consensus to evaluate revision
            5. If accepted: update section, mark annotation RESOLVED
            6. If rejected: increment annotation.attempt_count
            7. If attempt_count >= max_attempts: mark NEEDS_HUMAN_REVIEW

        For type=SUGGESTION:
            Similar to COMMENT but lower priority and more exploratory tone.

        For type=PROTECTED:
            Not processed — protection markers are read by the renderer.
        """

    async def review_section(self, section: Section, project: Project) -> None:
        """Review an existing section for quality and consistency.

        1. Build context with surrounding sections
        2. Ask LLM: "Does this section still accurately represent the project's findings?
           Is it consistent with the current document? Any improvements?"
        3. If improvements suggested: generate revision, evaluate via consensus
        4. Update section.review_count and last_reviewed_at
        """
```

### 9.5 ContextBuilder

**File**: `documenter/src/context.py`

Builds focused context for LLM prompts instead of including entire document.

```python
class ContextBuilder:
    def __init__(self, store: StateStoreInterface, config: Config): ...

    async def build_for_insight(
        self,
        insight: Insight,
        project: Project,
        max_tokens: int | None = None,
    ) -> str:
        """Build focused context relevant to an insight being incorporated.

        Includes:
            1. Research goal (always)
            2. 2 most recent sections (narrative continuity)
            3. Sections establishing concepts this insight depends on
            4. Sections with overlapping tags (thematic relevance)

        Deduplicates sections. Truncates to token budget.
        Token budget from config: documenter.context.max_tokens (default 8000)
        """

    async def build_for_annotation(
        self,
        annotation: Annotation,
        project: Project,
    ) -> str:
        """Build context for addressing an annotation.

        Includes:
            1. Research goal
            2. The annotated section (full content)
            3. Adjacent sections (before and after)
            4. The annotation content itself
        """

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count. Uses tiktoken if available, falls back to len(text)//4."""
```

### 9.6 Annotations Handler

**File**: `documenter/src/annotations.py`

```python
class AnnotationHandler:
    def __init__(self, store: StateStoreInterface, event_bus: EventBusInterface): ...

    async def handle_new_annotation(self, event: Event) -> None:
        """Event handler for user.annotation.created.
        Creates annotation in store. If type=COMMENT, marks as highest priority.
        """

    async def get_protected_sections(self, project_id: str) -> set[str]:
        """Return section_ids that have an active PROTECTED annotation.
        The renderer and processor check this to avoid modifying protected content.
        """
```

### 9.7 Renderer

**File**: `documenter/src/renderer.py`

Generates markdown document from database state.

```python
class Renderer:
    def __init__(self, store: StateStoreInterface, config: Config): ...

    async def render(self, project: Project) -> str:
        """Render the complete document as markdown.

        1. Load all sections ordered by order_index
        2. Generate document header from project.name and project.goal
        3. For each section: render content
        4. Annotations are NOT baked into the markdown
           (they're displayed by the control panel at view time)
        5. Return complete markdown string
        """

    async def save(self, project: Project, output_dir: Path) -> None:
        """Render and write to output_dir/main.md.
        Also archives to output_dir/archive/ with timestamp if content changed.
        """
```

### 9.8 Concept Canonicalization

```python
def canonicalize(concept_name: str) -> str:
    """Normalize concept names for reliable matching.

    'Fano plane' → 'fano_plane'
    'The Fano Plane' → 'fano_plane'
    'market entry strategy' → 'market_entry_strategy'
    'Market-Entry Strategy' → 'market_entry_strategy'

    Steps:
        1. Strip whitespace, lowercase
        2. Remove leading articles (the, a, an)
        3. Replace non-alphanumeric with underscore
        4. Strip leading/trailing underscores
    """
```

Defined in `documenter/src/context.py` (used by ContextBuilder and Processor).

### 9.9 Event Subscriptions

| Event | Handler | Action |
|-------|---------|--------|
| `explorer.insight.attested` | DocumenterEngine._on_insight_attested | Queue insight for incorporation |
| `user.annotation.created` | AnnotationHandler.handle_new_annotation | Create annotation, prioritize |
| `researcher.finding.stored` | DocumenterEngine._on_finding | Enrich context for related insights |

### 9.10 Test Requirements

| Test | Description |
|------|-------------|
| `test_plan_annotations_first` | Annotations prioritized above insights |
| `test_plan_work_allocation` | Actual ratio tracked and rebalanced |
| `test_incorporate_insight_full_pipeline` | Extract → prerequisites → draft → evaluate → add |
| `test_incorporate_duplicate_skipped` | Already-represented insight marked INCORPORATED without new section |
| `test_incorporate_transient_failure_retries` | API error → transient_failure_count, retried later |
| `test_incorporate_dispute_shelves` | Max disputes → SHELVED status |
| `test_address_comment` | Annotation → revised section → annotation RESOLVED |
| `test_address_comment_max_attempts` | 3 failed attempts → NEEDS_HUMAN_REVIEW |
| `test_protected_section_not_modified` | Processor skips sections with PROTECTED annotation |
| `test_context_builder_token_budget` | Context truncated to max_tokens |
| `test_context_builder_dedup` | Same section not included twice |
| `test_render_ordered` | Sections rendered in order_index sequence |
| `test_render_no_inline_annotations` | Annotations not baked into markdown |
| `test_concept_canonicalization` | "The Fano Plane" → "fano_plane" |
| `test_initial_document_from_project` | New project → document skeleton from document_guidance |
| `test_transaction_rollback` | Failure mid-incorporate → all changes reverted |

---

## 10. Module: researcher — Researcher Engine

**Files**: `researcher/src/engine.py`, `researcher/src/questions.py`, `researcher/src/searcher.py`, `researcher/src/extractor.py`, `researcher/src/trust.py`, `researcher/src/prompts.py`
**Implements**: `ModuleInterface`
**Dependencies**: `shared/store`, `shared/events`, `shared/config`, `llm/client`, `llm/consensus`

### 10.1 Summary

Goes outside the system to find external evidence. Generates research questions from attested insights, searches for sources, evaluates source trustworthiness via consensus, extracts structured findings, and links them back to insights.

### 10.2 ResearcherEngine (Facade)

**File**: `researcher/src/engine.py`

```python
class ResearcherEngine(ModuleInterface):
    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface,
        config: Config,
    ): ...

    @property
    def module_name(self) -> str:
        return "researcher"

    async def initialize(self) -> bool:
        """Subscribe to events: explorer.insight.attested, documenter.research.requested,
        user.research.requested
        """

    async def start(self) -> None:
        """Research loop:
            while running:
                1. Check for directed research requests (highest priority)
                2. Check for new attested insights needing research
                3. For each item:
                   a. Generate research questions
                   b. Execute searches
                   c. For each source: evaluate trust, extract findings
                   d. Link findings to insights
                   e. Publish evidence events
                4. Sleep for idle_polling_interval, repeat
        """
```

### 10.3 QuestionGenerator

**File**: `researcher/src/questions.py`

```python
class QuestionGenerator:
    def __init__(self, llm_client: LLMClientInterface, config: Config): ...

    async def generate(
        self,
        insight: Insight,
        project: Project,
        max_questions: int | None = None,
    ) -> list[str]:
        """Generate research questions for an attested insight.

        Prompt:
            "Given this insight from a research project, generate questions
             that would help find external evidence to support, refute, or
             extend it.

             PROJECT GOAL: {project.goal}
             PROJECT CONTEXT: {project.context}
             INSIGHT: {insight.text}
             RESEARCH DOMAINS: {formatted_domains}

             Generate up to {max_questions} specific, searchable questions."

        max_questions from config: researcher.max_questions_per_insight (default 10)

        CRITICAL: No hardcoded domain fallbacks. If the prompt fails,
        use project.goal as generic context, never domain-specific strings.
        """

    async def generate_directed(
        self,
        topic: str,
        context: str,
        project: Project,
    ) -> list[str]:
        """Generate questions for a directed research request (from Documenter or user)."""
```

### 10.4 SearchExecutor

**File**: `researcher/src/searcher.py`

```python
class SearchExecutor:
    def __init__(self, llm_client: LLMClientInterface, config: Config): ...

    async def search(
        self,
        question: str,
        domains: list[ResearchDomain],
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """Execute web searches for a research question.

        Strategy:
            1. Reformulate question into search queries using LLM
            2. Execute searches (implementation depends on available search APIs)
            3. Return deduplicated results

        max_results from config: researcher.max_searches_per_question (default 5)

        NOTE: Actual search implementation (web API, academic API, etc.)
        is pluggable. The initial implementation uses LLM-generated
        summaries of what would be found. Full web search is Phase 6+.
        """

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    domain: str | None
```

### 10.5 TrustEvaluator

**File**: `researcher/src/trust.py`

```python
class TrustEvaluator:
    def __init__(
        self,
        consensus: ConsensusEngineInterface,
        store: StateStoreInterface,
        config: Config,
    ): ...

    async def evaluate(
        self,
        source: Source,
        content_summary: str,
        project: Project,
    ) -> int:
        """Evaluate source trustworthiness using consensus.

        1. Check cache: if source.url exists with matching content_hash,
           return cached trust_score
        2. Create TrustEvaluationTask(source, content_summary, project)
        3. Run consensus engine (min 2 backends)
        4. Convert consensus scores to trust_score (0-100)
        5. Determine trust_tier: "authoritative" (80+), "reliable" (60-79),
           "uncertain" (40-59), "unreliable" (<40)
        6. Persist to store
        7. Return trust_score

        Config: researcher.trust.min_trust_score (default 50)
        Sources below this threshold are not used for findings.
        """


class TrustEvaluationTask(ConsensusTaskInterface):
    """Consensus task for evaluating source trustworthiness."""

    def __init__(self, source: Source, content_summary: str, project: Project): ...

    def get_prompt(self, round_num: int, prior_rounds: list[RoundResult], backend: str) -> str:
        """
        "Evaluate the trustworthiness of this source for a research project.

         PROJECT GOAL: {project.goal}
         SOURCE URL: {source.url}
         SOURCE TITLE: {source.title}
         CONTENT SUMMARY: {content_summary}

         Rate on:
         - Authority: Is the author/publisher credible in this domain?
         - Accuracy: Does the content appear factually sound?
         - Relevance: How relevant is this to the research goal?
         - Recency: Is the information current?

         Overall trust score: 0-100
         Verdict: accept (trustworthy) / reject (not trustworthy) / uncertain"
        """
```

### 10.6 FindingExtractor

**File**: `researcher/src/extractor.py`

```python
class FindingExtractor:
    def __init__(self, llm_client: LLMClientInterface, config: Config): ...

    async def extract(
        self,
        source: Source,
        content: str,
        insight: Insight,
        project: Project,
    ) -> list[Finding]:
        """Extract structured findings from source content.

        Pipeline:
            1. Pre-filter with regex patterns from project.research_domains[].extraction_patterns
               (fast, catches obvious matches — numbers, key terms, etc.)
            2. LLM extraction (primary method):
               "Given this source and the related insight, extract specific findings.
                For each finding: summary, confidence (0-1), type (supports/refutes/extends)"
            3. Create Finding objects, link to source and insight
            4. Persist to store

        Returns list of new findings.
        """
```

### 10.7 Evidence Linking

After findings are extracted, the engine links evidence back to insights:

```python
# In ResearcherEngine
async def _publish_evidence(
    self, findings: list[Finding], insight: Insight
) -> None:
    """Analyze findings and publish evidence events.

    If findings predominantly support the insight:
        Publish researcher.evidence.supports
    If findings predominantly contradict:
        Publish researcher.evidence.contradicts
    Also publish researcher.finding.stored for each finding.
    """
```

### 10.8 Event Subscriptions

| Event | Handler | Action |
|-------|---------|--------|
| `explorer.insight.attested` | ResearcherEngine._on_insight_attested | Queue insight for research |
| `documenter.research.requested` | ResearcherEngine._on_research_request | Directed research (high priority) |
| `user.research.requested` | ResearcherEngine._on_user_request | Directed research (high priority) |

### 10.9 Configuration Read

- `researcher.max_questions_per_insight` (default 10)
- `researcher.max_searches_per_question` (default 5)
- `researcher.max_findings_per_source` (default 20)
- `researcher.idle_polling_interval_seconds` (default 300)
- `researcher.trust.min_trust_score` (default 50)

### 10.10 Test Requirements

| Test | Description |
|------|-------------|
| `test_question_generation` | Generates relevant questions from insight |
| `test_question_generation_no_domain_fallback` | No hardcoded domain strings in questions |
| `test_trust_evaluation_uses_consensus` | Multiple LLMs evaluate source, not single call |
| `test_trust_caching` | Same URL returns cached score without re-evaluation |
| `test_trust_cache_invalidation` | Changed content_hash triggers re-evaluation |
| `test_finding_extraction` | Findings extracted and linked to source + insight |
| `test_evidence_supports_event` | Supporting findings → researcher.evidence.supports published |
| `test_evidence_contradicts_event` | Contradicting findings → researcher.evidence.contradicts published |
| `test_directed_research_priority` | Directed requests processed before autonomous |
| `test_configurable_limits` | Config values respected for question/search counts |
| `test_low_trust_source_skipped` | Source below min_trust_score → no findings extracted |

---

## 11. Module: orchestrator — Orchestrator

**Files**: `orchestrator/src/main.py`, `orchestrator/src/scheduler.py`, `orchestrator/src/quota.py`, `orchestrator/src/recovery.py`
**Dependencies**: `shared/store`, `shared/events`, `shared/config`, all engine modules

### 11.1 Summary

Coordinates all modules. Manages lifecycle (start, stop, health), task scheduling, budget allocation, and crash recovery.

### 11.2 Orchestrator (Facade)

**File**: `orchestrator/src/main.py`

```python
class Orchestrator:
    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        config: Config,
    ): ...

    def register_module(self, module: ModuleInterface) -> None:
        """Register an engine module. Called during setup."""

    async def start(self) -> None:
        """
        1. Initialize all registered modules (in dependency order)
        2. Run crash recovery (replay missed events)
        3. Start all modules
        4. Begin health check loop
        5. Begin quota tracking
        """

    async def stop(self) -> None:
        """Stop all modules in reverse order. Save state."""

    async def get_status(self) -> dict:
        """Return health status of all modules, quota usage, event stats."""
```

### 11.3 QuotaAllocator

**File**: `orchestrator/src/quota.py`

```python
class QuotaAllocator:
    def __init__(self, config: Config, event_bus: EventBusInterface): ...

    async def initialize(self) -> None:
        """Subscribe to llm.request.completed to track spend."""

    def get_remaining(self, module: str) -> float:
        """Return remaining budget in USD for this module today."""

    def is_within_budget(self, module: str) -> bool:
        """True if module hasn't exceeded its daily allocation."""

    async def _on_llm_request(self, event: Event) -> None:
        """Track spend per module. Publish system.budget.warning at alert threshold."""
```

**Configuration read**:
- `quotas.daily_budget_usd`
- `quotas.per_module_weights` → {module: weight_percent}
- `quotas.alert_at_percent`

### 11.4 Recovery

**File**: `orchestrator/src/recovery.py`

```python
class RecoveryManager:
    def __init__(self, store: StateStoreInterface, event_bus: EventBusInterface): ...

    async def recover(self) -> None:
        """Replay events since last known checkpoint.

        1. Read last checkpoint timestamp from store
        2. Replay all events since that timestamp
        3. Update checkpoint
        """

    async def checkpoint(self) -> None:
        """Record current timestamp as checkpoint. Called periodically."""
```

### 11.5 Transient Failure Retry Scheduling

The Orchestrator is responsible for retrying insights/annotations that entered `transient_failure` state.

```python
class RetryScheduler:
    """Periodically scans for transient failures and re-queues them."""

    BACKOFF_BASE_SECONDS = 300  # 5 minutes
    MAX_BACKOFF_SECONDS = 3600  # 1 hour
    MAX_TRANSIENT_RETRIES = 5   # After this, mark as SHELVED

    async def scan_and_retry(self, project_id: str) -> None:
        """Called periodically by the orchestrator main loop.

        1. Query insights WHERE status='transient_failure'
           AND transient_failure_count < MAX_TRANSIENT_RETRIES
        2. For each: check if enough time has elapsed since last attempt
           backoff = min(BACKOFF_BASE * (2 ** failure_count), MAX_BACKOFF)
        3. If ready: transition back to REVIEWING (for Explorer)
           or INCORPORATING (for Documenter) and publish retry event
        4. If transient_failure_count >= MAX_TRANSIENT_RETRIES:
           transition to SHELVED
        """
```

Similarly handles annotation retries (annotations with status='attempted' and attempt_count < max).

### 11.6 Data Retention

The Orchestrator runs a daily cleanup task:

```python
class RetentionManager:
    """Enforces data retention policies from config."""

    async def cleanup(self) -> None:
        """Run daily. Reads from config:
            retention.events_max_age_days    → delete old events
            retention.metrics_max_age_days   → delete old metrics
            retention.snapshots_max_count    → prune old document archives
            retention.logs_max_age_days      → delete old log files
            retention.logs_max_size_mb       → delete if total size exceeds
        """
```

### 11.7 CLI Entry Points

The system is launched via CLI commands defined in `pyproject.toml`:

```toml
[project.scripts]
research-assistant = "orchestrator.src.main:cli_main"  # Full system
ra-control = "control.server:cli_main"                  # Control panel only
```

`cli_main()` parses arguments: `--config <path>`, `--project <name>`, `--port <int>`.

### 11.8 Test Requirements

| Test | Description |
|------|-------------|
| `test_module_registration` | Modules registered and accessible by name |
| `test_start_initializes_all` | All modules initialized on start |
| `test_stop_in_reverse_order` | Modules stopped in reverse registration order |
| `test_health_check_propagates` | Orchestrator health reflects module health |
| `test_quota_tracking` | Spend tracked correctly per module |
| `test_quota_warning_at_threshold` | system.budget.warning published at alert_at_percent |
| `test_recovery_replays_events` | Events since checkpoint replayed on start |

---

## 12. Module: control — Control Panel

**Files**: `control/server.py`, `control/blueprints/projects.py`, `control/blueprints/seeds.py`, `control/blueprints/insights.py`, `control/blueprints/document.py`, `control/blueprints/research.py`, `control/blueprints/status.py`
**Dependencies**: `flask`, `shared/store`, `shared/events`, `shared/config`

### 12.1 Summary

Flask web application providing REST API and dashboard UI. All user interactions flow through here as events. The control panel does NOT contain business logic — it translates HTTP requests to events and database reads to JSON responses.

### 12.2 Server Setup

**File**: `control/server.py`

```python
def create_app(
    store: StateStoreInterface,
    event_bus: EventBusInterface,
    config: Config,
) -> Flask:
    """Create Flask app with all blueprints registered.

    Blueprints:
        /api/projects   → projects.py
        /api/seeds      → seeds.py
        /api/insights   → insights.py
        /api/document   → document.py
        /api/research   → research.py
        /api/status     → status.py
    """
```

### 12.3 Blueprint: Projects

**File**: `control/blueprints/projects.py`

```
GET    /api/projects              → list all projects
GET    /api/projects/:id          → get project details
POST   /api/projects              → create project (from uploaded YAML or JSON body)
PUT    /api/projects/:id          → update project settings
PUT    /api/projects/:id/activate → set as active project
```

### 12.4 Blueprint: Seeds

**File**: `control/blueprints/seeds.py`

```
GET    /api/seeds                          → list seeds for active project (with status filter)
GET    /api/seeds/:id                      → get seed with lineage
POST   /api/seeds                          → create seed → publishes user.seed.created
PUT    /api/seeds/:id                      → update seed text/tags/priority → publishes user.seed.prioritized if priority changed
DELETE /api/seeds/:id                      → retire seed → publishes user.seed.retired
GET    /api/seeds/:id/lineage              → get seed evolution chain
POST   /api/seeds/:id/comment              → add user comment on a seed
GET    /api/seeds/:id/comments             → list comments on a seed
POST   /api/seeds/:id/approve-modification → approve modification → publishes user.seed.modification.approved
POST   /api/seeds/:id/reject-modification  → reject modification → publishes user.seed.modification.rejected
GET    /api/seeds/pending-modifications    → list seeds with pending modification proposals
```

### 12.5 Blueprint: Insights

**File**: `control/blueprints/insights.py`

```
GET    /api/insights              → list insights (filterable by status, project, seed)
GET    /api/insights/:id          → full insight detail with review record
POST   /api/insights/:id/comment  → add user comment → publishes user.insight.endorsed/dismissed/reconsider
GET    /api/insights/:id/comments → list comments for insight
GET    /api/insights/:id/lineage  → trace back to seed and thread
```

Request body for comment:
```json
{
    "type": "endorse | dismiss | reconsider | general",
    "content": "Optional text explanation"
}
```

### 12.6 Blueprint: Document

**File**: `control/blueprints/document.py`

```
GET    /api/document              → current document markdown (annotations served separately, NOT inline)
GET    /api/document/sections     → list sections with metadata
GET    /api/document/sections/:id → single section detail
POST   /api/annotations          → create annotation → publishes user.annotation.created
PUT    /api/annotations/:id      → update annotation content or status
DELETE /api/annotations/:id      → delete annotation
GET    /api/annotations          → list annotations (with status filter, includes section context)
GET    /api/document/history     → list archived versions with timestamps
GET    /api/document/history/:ts → get archived version at timestamp
```

**Annotation rendering clarification**: Annotations are stored in the database and served via `GET /api/annotations`. They are NOT baked into the document markdown. The control panel UI overlays annotations onto the document at display time, keyed by `section_id`. This avoids the v1 divergence problem where inline markers and JSON annotations got out of sync.

Request body for annotation:
```json
{
    "type": "comment | protected | suggestion",
    "section_id": "section-uuid",
    "content": "The annotation text"
}
```

### 12.7 Blueprint: Research

**File**: `control/blueprints/research.py`

```
GET    /api/research/findings     → list findings (filterable by domain, insight)
GET    /api/research/sources      → list sources with trust scores
POST   /api/research/request      → directed research request → publishes user.research.requested
GET    /api/research/status       → research queue depth, active tasks, recent findings
```

### 12.8 Blueprint: Status

**File**: `control/blueprints/status.py`

```
GET    /api/status                → system health: module states, backend health, quota usage
GET    /api/status/metrics        → metrics dashboard data (costs, throughput, consensus quality)
GET    /api/status/events         → recent events stream (with optional topic filter)
```

### 12.9 JSON Response Format

All API responses follow a consistent envelope:

```json
// Success
{
    "ok": true,
    "data": { ... }
}

// Error
{
    "ok": false,
    "error": "Human-readable error message",
    "code": "VALIDATION_ERROR"
}

// List with pagination
{
    "ok": true,
    "data": [ ... ],
    "total": 42,
    "offset": 0,
    "limit": 20
}
```

### 12.10 Event Publishing Pattern

Every user action that modifies state goes through the EventBus:

```python
@seeds_bp.route("/api/seeds", methods=["POST"])
async def create_seed():
    data = request.get_json()
    seed = Seed(id=generate_id(), project_id=active_project_id, ...)
    await store.create_seed(seed)
    await event_bus.publish(Event(
        topic="user.seed.created",
        source="control",
        payload={"seed_id": seed.id, "text": seed.text, "priority": seed.priority},
        ...
    ))
    return jsonify({"ok": True, "data": asdict(seed)})
```

The control panel WRITES to the store (for immediate UI consistency) AND publishes an event (for engine reaction). Engines subscribe to the event, not to store changes.

### 12.11 Test Requirements

| Test | Description |
|------|-------------|
| `test_list_seeds` | GET /api/seeds returns seeds for active project |
| `test_create_seed_publishes_event` | POST /api/seeds → user.seed.created event |
| `test_insight_comment_publishes_event` | POST comment → correct event type published |
| `test_annotation_create` | POST annotation → stored + event published |
| `test_document_render` | GET /api/document returns markdown |
| `test_status_endpoint` | GET /api/status returns module health |
| `test_error_response_format` | Invalid request → error envelope with code |
| `test_pagination` | List endpoints support offset/limit |
| `test_filtering` | List endpoints support status/project filters |

---

## Appendix A: Configuration Schema

Complete `config.yaml` with all keys referenced by all modules:

```yaml
llm:
  api_key_env: "OPENROUTER_API_KEY"
  endpoint: "https://openrouter.ai/api/v1"
  models:
    gemini: "google/gemini-2.0-flash-thinking-exp-01-21"
    chatgpt: "openai/gpt-4o"
    claude: "anthropic/claude-sonnet-4-20250514"
    deepseek: "deepseek/deepseek-r1"
  rate_limits:       # Requests per minute
    gemini: 10
    chatgpt: 60
    claude: 50
    deepseek: 10
  default_timeout_seconds: 300
  max_retries: 3

consensus:
  backends: [gemini, chatgpt, claude]
  max_rounds: 4
  min_valid_responses: 2
  max_retries_per_backend: 2
  convergence_threshold: 0.7
  decision_method: "majority"     # majority | supermajority | unanimous
  minimum_agreement: 0.66

explorer:
  max_active_threads: 3
  min_exchanges_for_chunk: 4
  max_exchanges_per_thread: 12
  thread_retirement:
    max_idle_hours: 48
    max_reexplore_count: 3
  model_weights:
    exploration: { gemini: 60, chatgpt: 40 }
    critique: { gemini: 30, chatgpt: 70 }
    synthesis: { claude: 60, chatgpt: 40 }

documenter:
  document_dir: "data/document"
  work_allocation:
    new_material_percent: 70
    review_existing_percent: 30
  context:
    max_tokens: 8000
  max_annotation_attempts: 3
  max_disputes_before_shelve: 3

researcher:
  max_questions_per_insight: 10
  max_searches_per_question: 5
  max_findings_per_source: 20
  idle_polling_interval_seconds: 300
  trust:
    min_trust_score: 50

quotas:
  daily_budget_usd: 10.00
  per_module_weights:
    explorer: 50
    documenter: 35
    researcher: 15
  alert_at_percent: 80

logging:
  level: "INFO"
  directory: "./logs"
  max_bytes: 10485760
  backup_count: 10

control:
  host: "127.0.0.1"
  port: 8080

database:
  path: "data/research.db"

retention:
  events_max_age_days: 90
  metrics_max_age_days: 365
  snapshots_max_count: 90
  logs_max_age_days: 30
  logs_max_size_mb: 500

active_project: "fano-mathematics"
```

---

## Appendix B: Python Dependencies

### Runtime Dependencies

| Package | Version | Purpose | Why This One |
|---------|---------|---------|-------------|
| `aiosqlite` | >= 0.20.0 | Async SQLite (StateStore) | Wraps stdlib sqlite3 in async; de facto standard; WAL works on Windows NTFS |
| `aiohttp` | >= 3.9.0 | Async HTTP client (LLMClient → OpenRouter) | Already in v1; better async throughput than httpx; mature connection pooling |
| `tenacity` | >= 8.2.0 | Retry logic with exponential backoff | Already in v1; clean decorator-based retries |
| `flask` | >= 3.0.0 | Web framework (Control Panel) | Already in v1; control panel is sync (human-speed requests); no need for async web framework |
| `pyyaml` | >= 6.0.0 | YAML loading | Already in v1; standard YAML parser |
| `pydantic` | >= 2.8.0 | Data validation, settings | Already in v1; validates config at load time, not at use time |
| `pydantic-settings` | >= 2.5.0 | Config with env var overrides | Typed config models; `FANO_LLM__RATE_LIMITS__GEMINI=20` overrides; complements existing pydantic |
| `python-ulid` | >= 3.0.0 | ID generation (ULID) | Chronologically sortable, globally unique; v1's `uuid4()[:12]` has collision risk |
| `tiktoken` | >= 0.7.0 | Token estimation | Pre-flight prompt size checks; ~10-20% accurate for non-OpenAI models; actual counts from API preferred |
| `tavily-python` | >= 0.5.0 | Web search (Researcher) | Purpose-built for AI/RAG; structured results; 1000 free credits/month; replaces Playwright scraping |
| `markdown` | >= 3.5.0 | Markdown → HTML rendering | Already in v1; for control panel document display |

### Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 8.0.0 | Test framework |
| `pytest-asyncio` | >= 0.23.0 | Async test support (asyncio_mode=auto) |
| `hypothesis` | >= 6.100.0 | Property-based testing (convergence, dedup, canonicalization) |

### What We Keep From v1

| Component | Decision | Rationale |
|-----------|----------|-----------|
| `shared/logging/` | **Keep custom logger** | Already outputs JSON Lines, has correlation IDs via contextvars, dot-separated event naming. Switching to structlog would rewrite 30+ files for no gain. |
| `orchestrator/scheduler.py` | **Port algorithm** | Priority-based scheduling with multi-factor scoring, backlog pressure, starvation prevention. Reimplement cleanly but preserve the algorithm. |
| Content hashing | **Use `hashlib` (stdlib)** | SHA-256 via `hashlib.sha256()` — C-implemented, no faster alternative exists for content identity. |
| Markdown generation | **String concatenation** | Documenter generates markdown via `"\n".join(lines)`. No AST library needed — we write sections, not restructure documents. |

### What We Do NOT Use

| Library | Why Not |
|---------|---------|
| `httpx` | v1 uses aiohttp which has better async throughput; httpx's HTTP/2 advantage irrelevant for OpenRouter |
| `quart` / `fastapi` | Control panel doesn't need async; Flask is simpler and already familiar from v1 |
| `dynaconf` | Multi-environment layer system is overkill for single-machine deployment |
| `omegaconf` | Designed for ML experiment configs (Hydra); lacks pydantic's runtime validation |
| `structlog` | Custom logger already does JSON Lines + correlation IDs; switching cost > benefit |
| `apscheduler` | Designed for cron-like jobs; our orchestrator is a priority queue, not a timer |
| `ulid-py` | Not updated since 2020; python-ulid is actively maintained |

### pyproject.toml

```toml
[project]
dependencies = [
    "aiosqlite>=0.20.0",
    "aiohttp>=3.9.0",
    "tenacity>=8.2.0",
    "flask>=3.0.0",
    "pyyaml>=6.0.0",
    "pydantic>=2.8.0",
    "pydantic-settings>=2.5.0",
    "python-ulid>=3.0.0",
    "tiktoken>=0.7.0",
    "tavily-python>=0.5.0",
    "markdown>=3.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "hypothesis>=6.100.0",
]

[project.scripts]
research-assistant = "orchestrator.src.main:cli_main"
ra-control = "control.server:cli_main"
```

---

## Appendix C: Integration Test Scenarios

These tests require multiple modules assembled together. Run after individual module tests pass.

| Scenario | Modules | Description |
|----------|---------|-------------|
| `test_seed_to_insight` | Store + EventBus + Explorer | Create seed → exploration → extraction → review → attested |
| `test_insight_to_document` | Store + EventBus + Explorer + Documenter | Attested insight → documenter picks up → section added |
| `test_insight_to_research` | Store + EventBus + Explorer + Researcher | Attested insight → researcher generates questions → findings stored |
| `test_directed_research` | Store + EventBus + Documenter + Researcher | Documenter requests → researcher delivers → finding linked |
| `test_user_endorses_insight` | Store + EventBus + Control + Explorer | User endorses → event → thread priority boosted |
| `test_user_annotation_addressed` | Store + EventBus + Control + Documenter | User comments → event → documenter revises section |
| `test_seed_modification_flow` | Store + EventBus + Explorer + Control | LLM proposes → user approves → child seed created |
| `test_transient_failure_recovery` | Store + EventBus + Explorer + Documenter | API failure → transient_failure → retry → success |
| `test_budget_exhaustion` | Store + EventBus + Orchestrator + Explorer | Explorer exceeds budget → requests queued |
| `test_full_pipeline` | All modules | Seed → explore → attest → research → document → user annotates → revision |
| `test_second_project_config` | All modules | Same engine with business planning config produces appropriate output |
