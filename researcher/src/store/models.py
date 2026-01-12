"""
Data models for the Researcher module.

Defines the core entities: Source, Finding, Concept, Number, and their relationships.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json
import uuid


class FindingType(str, Enum):
    """Type of research finding."""
    SUPPORTING = "supporting"      # Supports current exploration
    CHALLENGING = "challenging"    # Challenges/contradicts
    ALTERNATIVE = "alternative"    # Alternative interpretation
    CONTEXTUAL = "contextual"      # Background/context


class MentionType(str, Enum):
    """How a number is mentioned in a source."""
    INCIDENTAL = "incidental"      # "he waited 7 days"
    ENUMERATIVE = "enumerative"    # "the 7 svaras are..."
    STRUCTURAL = "structural"      # "reality is organized in 7 layers"
    SYMBOLIC = "symbolic"          # "7 represents completeness"


class SourceTier(int, Enum):
    """Trust tier for sources."""
    TIER_1 = 1  # Highly trusted (academic, primary texts)
    TIER_2 = 2  # Good (established sites, translations)
    TIER_3 = 3  # Use with caution (verify claims)
    UNKNOWN = 0  # Not yet evaluated


@dataclass
class Source:
    """A web source that has been fetched and evaluated."""
    id: str
    url: str
    domain: str
    title: str

    # Trust evaluation
    trust_score: int  # 0-100
    trust_tier: SourceTier
    evaluation_reasoning: str
    evaluated_at: Optional[datetime]

    # Content info
    content_hash: str  # SHA256 of content
    content_type: str  # "html", "pdf", etc.
    last_fetched: datetime
    fetch_count: int

    # Observable features (for grounded trust evaluation)
    has_sanskrit_citations: bool
    has_verse_references: bool
    has_bibliography: bool
    is_academic_domain: bool

    # Topic-specific trust (domain -> score)
    topic_trust: dict = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "trust_score": self.trust_score,
            "trust_tier": self.trust_tier.value,
            "evaluation_reasoning": self.evaluation_reasoning,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "content_hash": self.content_hash,
            "content_type": self.content_type,
            "last_fetched": self.last_fetched.isoformat(),
            "fetch_count": self.fetch_count,
            "has_sanskrit_citations": self.has_sanskrit_citations,
            "has_verse_references": self.has_verse_references,
            "has_bibliography": self.has_bibliography,
            "is_academic_domain": self.is_academic_domain,
            "topic_trust": self.topic_trust,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        return cls(
            id=data["id"],
            url=data["url"],
            domain=data["domain"],
            title=data["title"],
            trust_score=data["trust_score"],
            trust_tier=SourceTier(data["trust_tier"]),
            evaluation_reasoning=data["evaluation_reasoning"],
            evaluated_at=datetime.fromisoformat(data["evaluated_at"]) if data.get("evaluated_at") else None,
            content_hash=data["content_hash"],
            content_type=data["content_type"],
            last_fetched=datetime.fromisoformat(data["last_fetched"]),
            fetch_count=data["fetch_count"],
            has_sanskrit_citations=data["has_sanskrit_citations"],
            has_verse_references=data["has_verse_references"],
            has_bibliography=data["has_bibliography"],
            is_academic_domain=data["is_academic_domain"],
            topic_trust=data.get("topic_trust", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    @classmethod
    def create(cls, url: str, title: str = "", content_hash: str = "",
               content_type: str = "html", trust_score: int = 0,
               trust_tier: SourceTier = SourceTier.UNKNOWN) -> "Source":
        """Create a new source with default values."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        return cls(
            id=str(uuid.uuid4())[:12],
            url=url,
            domain=domain,
            title=title or url,
            trust_score=trust_score,
            trust_tier=trust_tier,
            evaluation_reasoning="",
            evaluated_at=None,
            content_hash=content_hash,
            content_type=content_type,
            last_fetched=datetime.now(),
            fetch_count=1,
            has_sanskrit_citations=False,
            has_verse_references=False,
            has_bibliography=False,
            is_academic_domain=False,
        )


@dataclass
class Finding:
    """A research finding extracted from a source."""
    id: str
    source_id: str
    finding_type: FindingType

    # Content
    summary: str              # Brief summary of the finding
    original_quote: str       # Exact quote from source
    source_location: str      # Chapter, verse, page reference

    # Extracted entities
    concepts: list[str]       # Concept IDs linked to this finding
    numbers: list[str]        # Number IDs linked to this finding

    # Relevance to current exploration
    relevance_score: float    # 0-1, how relevant to current context
    relevance_reasoning: str  # Why it's relevant

    # Confidence
    confidence: float         # 0-1, extraction confidence

    # Cross-references
    related_findings: list[str]  # IDs of related findings

    # Provenance
    extracted_at: datetime
    extraction_model: str     # Which LLM extracted this

    # Domain
    domain: str               # Research domain (from config)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "finding_type": self.finding_type.value,
            "summary": self.summary,
            "original_quote": self.original_quote,
            "source_location": self.source_location,
            "concepts": self.concepts,
            "numbers": self.numbers,
            "relevance_score": self.relevance_score,
            "relevance_reasoning": self.relevance_reasoning,
            "confidence": self.confidence,
            "related_findings": self.related_findings,
            "extracted_at": self.extracted_at.isoformat(),
            "extraction_model": self.extraction_model,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            finding_type=FindingType(data["finding_type"]),
            summary=data["summary"],
            original_quote=data["original_quote"],
            source_location=data["source_location"],
            concepts=data["concepts"],
            numbers=data["numbers"],
            relevance_score=data["relevance_score"],
            relevance_reasoning=data["relevance_reasoning"],
            confidence=data["confidence"],
            related_findings=data["related_findings"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            extraction_model=data["extraction_model"],
            domain=data["domain"],
        )

    @classmethod
    def create(cls, source_id: str, finding_type: FindingType, summary: str,
               original_quote: str = "", domain: str = "",
               extraction_model: str = "unknown",
               source_location: str = "", confidence: float = 0.5) -> "Finding":
        """Create a new finding."""
        return cls(
            id=str(uuid.uuid4())[:12],
            source_id=source_id,
            finding_type=finding_type,
            summary=summary,
            original_quote=original_quote,
            source_location=source_location,
            concepts=[],
            numbers=[],
            relevance_score=0.0,
            relevance_reasoning="",
            confidence=confidence,
            related_findings=[],
            extracted_at=datetime.now(),
            extraction_model=extraction_model,
            domain=domain,
        )


@dataclass
class Concept:
    """A concept extracted from findings."""
    id: str
    canonical_name: str       # Normalized name
    display_name: str         # Human-readable name
    aliases: list[str]        # Alternative names/spellings
    domain: str               # Primary domain

    # Sanskrit handling
    sanskrit_forms: list[str]  # IAST, Devanagari, etc.

    # Description
    description: str

    # Relations
    related_concepts: list[str]  # IDs of related concepts

    # Statistics
    occurrence_count: int
    finding_ids: list[str]    # Findings that mention this concept

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "domain": self.domain,
            "sanskrit_forms": self.sanskrit_forms,
            "description": self.description,
            "related_concepts": self.related_concepts,
            "occurrence_count": self.occurrence_count,
            "finding_ids": self.finding_ids,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Concept":
        return cls(
            id=data["id"],
            canonical_name=data["canonical_name"],
            display_name=data["display_name"],
            aliases=data["aliases"],
            domain=data["domain"],
            sanskrit_forms=data.get("sanskrit_forms", []),
            description=data["description"],
            related_concepts=data["related_concepts"],
            occurrence_count=data["occurrence_count"],
            finding_ids=data["finding_ids"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    @classmethod
    def create(cls, canonical_name: str, display_name: str, domain: str,
               description: str = "") -> "Concept":
        """Create a new concept."""
        return cls(
            id=str(uuid.uuid4())[:12],
            canonical_name=canonical_name,
            display_name=display_name,
            aliases=[],
            domain=domain,
            sanskrit_forms=[],
            description=description,
            related_concepts=[],
            occurrence_count=0,
            finding_ids=[],
        )


@dataclass
class NumberMention:
    """A mention of a significant number in findings."""
    id: str
    value: int                # The number itself
    context: str              # Brief context of mention
    mention_type: MentionType
    significance: str         # What the number represents

    # Source info
    finding_id: str
    source_id: str
    domain: str

    # Cross-domain tracking
    domains_found_in: list[str]  # All domains where this number appears

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "value": self.value,
            "context": self.context,
            "mention_type": self.mention_type.value,
            "significance": self.significance,
            "finding_id": self.finding_id,
            "source_id": self.source_id,
            "domain": self.domain,
            "domains_found_in": self.domains_found_in,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NumberMention":
        return cls(
            id=data["id"],
            value=data["value"],
            context=data["context"],
            mention_type=MentionType(data["mention_type"]),
            significance=data["significance"],
            finding_id=data["finding_id"],
            source_id=data["source_id"],
            domain=data["domain"],
            domains_found_in=data["domains_found_in"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    @classmethod
    def create(cls, value: int, context: str, mention_type: MentionType,
               finding_id: str, source_id: str, domain: str,
               significance: str = "") -> "NumberMention":
        """Create a new number mention."""
        return cls(
            id=str(uuid.uuid4())[:12],
            value=value,
            context=context,
            mention_type=mention_type,
            significance=significance,
            finding_id=finding_id,
            source_id=source_id,
            domain=domain,
            domains_found_in=[domain],
        )


@dataclass
class CrossReference:
    """A cross-reference between findings."""
    id: str
    finding_id_1: str
    finding_id_2: str
    relationship_type: str    # "same_number", "same_concept", "contradicts", "supports"
    relationship_value: str   # The shared number/concept, or description
    strength: float           # 0-1, how strong the connection is

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "finding_id_1": self.finding_id_1,
            "finding_id_2": self.finding_id_2,
            "relationship_type": self.relationship_type,
            "relationship_value": self.relationship_value,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrossReference":
        return cls(
            id=data["id"],
            finding_id_1=data["finding_id_1"],
            finding_id_2=data["finding_id_2"],
            relationship_type=data["relationship_type"],
            relationship_value=data["relationship_value"],
            strength=data["strength"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class ResearchContext:
    """Current research context from observing explorer/documenter."""
    active_concepts: list[str]     # Concepts currently being explored
    active_numbers: list[int]      # Numbers of interest
    active_domains: list[str]      # Domains being explored
    recent_insights: list[dict]    # Recent explorer insights
    documenter_topics: list[str]   # Topics documenter is writing about

    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "active_concepts": self.active_concepts,
            "active_numbers": self.active_numbers,
            "active_domains": self.active_domains,
            "recent_insights": self.recent_insights,
            "documenter_topics": self.documenter_topics,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def empty(cls) -> "ResearchContext":
        """Create an empty context."""
        return cls(
            active_concepts=[],
            active_numbers=[],
            active_domains=[],
            recent_insights=[],
            documenter_topics=[],
        )
