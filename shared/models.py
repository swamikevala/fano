"""Shared types and interfaces for the Research Project Assistant.

This file defines every type that crosses module boundaries.
All modules import from here. Models are frozen dataclasses (immutable snapshots).
Mutable state lives in the StateStore.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

import ulid

if TYPE_CHECKING:
    pass


# ============================================================
# ID Generation
# ============================================================

def generate_id() -> str:
    """Generate a new ULID string (chronologically sortable, globally unique)."""
    return str(ulid.ULID())


# ============================================================
# Content Hashing
# ============================================================

def content_hash(text: str) -> str:
    """Deterministic content hash. Normalizes whitespace and case."""
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ============================================================
# Enumerations
# ============================================================

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
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ============================================================
# Configuration Types
# ============================================================

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


# ============================================================
# Core Entities
# ============================================================

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
    priority: int  # 1-10
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
    last_completed_sequence: int
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
    evaluation_scores: dict[str, float]
    dispute_count: int
    transient_failure_count: int
    review_record: dict | None
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
    establishes: list[str]
    requires: list[str]
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
    trust_score: int  # 0-100
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
    confidence: float  # 0.0-1.0
    domain: str | None
    related_insight_id: str | None
    created_at: datetime


# ============================================================
# LLM Types
# ============================================================

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


# ============================================================
# Consensus Types
# ============================================================

@dataclass(frozen=True)
class ParsedResponse:
    """Result of parsing one LLM's review response."""
    is_valid: bool
    verdict: Verdict | None
    scores: dict[str, float]
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
    verdict: Verdict | None


@dataclass(frozen=True)
class ConsensusResult:
    verdict: Verdict
    confidence: float  # 0.0-1.0
    scores: dict[str, float]
    rounds_completed: int
    round_history: list[RoundResult]
    valid_vote_count: int


@dataclass(frozen=True)
class ConsensusConfig:
    backends: list[str]
    max_rounds: int = 4
    min_valid_responses: int = 2
    max_retries_per_backend: int = 2
    convergence_threshold: float = 0.7
    decision_method: str = "majority"
    minimum_agreement: float = 0.66


# ============================================================
# Event Type
# ============================================================

@dataclass(frozen=True)
class Event:
    topic: str
    timestamp: datetime
    source: str
    payload: dict
    correlation_id: str


# ============================================================
# Orchestrator Types
# ============================================================

@dataclass(frozen=True)
class HealthStatus:
    module: str
    healthy: bool
    message: str
    details: dict = field(default_factory=dict)


# ============================================================
# Search Types
# ============================================================

@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    domain: str | None


# ============================================================
# Work Planning
# ============================================================

@dataclass(frozen=True)
class WorkItem:
    type: str  # "annotation", "insight", "review", "suggestion"
    entity_id: str
    priority: int


# ============================================================
# Seed Modification Proposal
# ============================================================

@dataclass(frozen=True)
class SeedModification:
    id: str
    seed_id: str
    original_text: str
    proposed_text: str
    reasoning: str
    proposing_thread_id: str
    agreement_ratio: float  # 0.0-1.0
    status: str  # "pending", "approved", "rejected"
    child_seed_id: str | None
    created_at: datetime
    resolved_at: datetime | None


# ============================================================
# Abstract Interfaces
# ============================================================

class ModuleInterface(ABC):
    """Every engine module (Explorer, Documenter, Researcher) implements this."""

    @property
    @abstractmethod
    def module_name(self) -> str: ...

    @abstractmethod
    async def initialize(self) -> bool: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def health_check(self) -> HealthStatus: ...


class StateStoreInterface(ABC):
    """Data access layer. All state reads/writes go through this."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield  # pragma: no cover

    # -- Projects --
    @abstractmethod
    async def create_project(self, project: Project) -> None: ...
    @abstractmethod
    async def get_project(self, project_id: str) -> Project | None: ...
    @abstractmethod
    async def update_project(self, project_id: str, **fields: object) -> None: ...
    @abstractmethod
    async def list_projects(self, status: ProjectStatus | None = None) -> list[Project]: ...

    # -- Seeds --
    @abstractmethod
    async def create_seed(self, seed: Seed) -> None: ...
    @abstractmethod
    async def get_seed(self, seed_id: str) -> Seed | None: ...
    @abstractmethod
    async def update_seed(self, seed_id: str, **fields: object) -> None: ...
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
    async def update_thread(self, thread_id: str, **fields: object) -> None: ...
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
    async def update_insight(self, insight_id: str, **fields: object) -> None: ...
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
    async def update_seed_modification(self, modification_id: str, **fields: object) -> None: ...
    @abstractmethod
    async def list_pending_modifications(self, project_id: str) -> list[SeedModification]: ...

    # -- Sections --
    @abstractmethod
    async def create_section(self, section: Section) -> None: ...
    @abstractmethod
    async def get_section(self, section_id: str) -> Section | None: ...
    @abstractmethod
    async def update_section(self, section_id: str, **fields: object) -> None: ...
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
    async def update_annotation(self, annotation_id: str, **fields: object) -> None: ...
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


# ============================================================
# EventBus Interface
# ============================================================

EventHandler = Callable[[Event], Awaitable[None]]


class EventBusInterface(ABC):
    """Async pub/sub event bus."""

    @abstractmethod
    def subscribe(self, topic_pattern: str, handler: EventHandler) -> None: ...

    @abstractmethod
    def unsubscribe(self, topic_pattern: str, handler: EventHandler) -> None: ...

    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    async def replay(self, since: datetime) -> None: ...


# ============================================================
# LLM Client Interface
# ============================================================

class LLMClientInterface(ABC):
    """LLM access layer. Rate-limited, tracked, circuit-broken."""

    @abstractmethod
    async def send(self, backend: str, prompt: str, **kwargs: object) -> LLMResponse: ...

    @abstractmethod
    async def send_structured(self, backend: str, prompt: str, schema: dict) -> LLMResponse: ...

    @abstractmethod
    def get_available_backends(self) -> list[str]: ...

    @abstractmethod
    def get_backend_status(self, backend: str) -> BackendStatus: ...


# ============================================================
# Consensus Interfaces
# ============================================================

class ConsensusTaskInterface(ABC):
    """A task to be evaluated by the consensus engine."""

    @abstractmethod
    def get_prompt(self, round_num: int, prior_rounds: list[RoundResult], backend: str) -> str: ...

    @abstractmethod
    def parse_response(self, text: str) -> ParsedResponse: ...

    @abstractmethod
    def get_evaluation_criteria(self) -> list[EvaluationCriterion]: ...


class ConsensusEngineInterface(ABC):
    """Multi-LLM agreement engine."""

    @abstractmethod
    async def run(self, task: ConsensusTaskInterface, backends: list[str] | None = None) -> ConsensusResult: ...
