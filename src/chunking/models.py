"""
Atomic Insight model.

An atomic insight is a single, standalone mathematical observation (1-3 sentences)
extracted from exploration threads. Multiple insights can be extracted per thread.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class InsightStatus(Enum):
    """Status of an atomic insight in the review pipeline."""
    PENDING = "pending"          # Awaiting review
    BLESSED = "blessed"          # ⚡ Approved as axiom
    INTERESTING = "interesting"  # ? Worth exploring more
    REJECTED = "rejected"        # ✗ Discarded


@dataclass
class AtomicInsight:
    """
    A single atomic insight extracted from exploration.

    Each insight is 1-3 sentences capturing ONE clear mathematical
    observation or connection. Multiple insights are extracted per thread.
    """
    # Core identity
    id: str
    insight: str                     # The atomic aphorism (1-3 sentences)
    confidence: str                  # "high" | "medium" | "low"
    tags: list[str]                  # Relevant concepts for searchability

    # Provenance
    source_thread_id: str            # Which exploration thread
    source_exchange_indices: list[int]  # Which exchanges contributed
    extraction_model: str            # Which LLM extracted this
    extracted_at: datetime

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # IDs of blessed insights
    pending_dependencies: list[str] = field(default_factory=list)  # Descriptions not yet matched

    # Review status
    status: InsightStatus = InsightStatus.PENDING
    rating: Optional[str] = None     # "⚡" | "?" | "✗" after review
    is_disputed: bool = False        # True if review panel had split decision
    reviewed_at: Optional[datetime] = None
    review_notes: str = ""
    priority: int = 5                # 1-10 (10 = highest priority, default 5)

    # Mathematical verification status (DeepSeek)
    math_verified: bool = False
    math_verification_result: Optional[dict] = None  # Serialized VerificationResult

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "insight": self.insight,
            "confidence": self.confidence,
            "tags": self.tags,
            "source_thread_id": self.source_thread_id,
            "source_exchange_indices": self.source_exchange_indices,
            "extraction_model": self.extraction_model,
            "extracted_at": self.extracted_at.isoformat(),
            "depends_on": self.depends_on,
            "pending_dependencies": self.pending_dependencies,
            "status": self.status.value,
            "rating": self.rating,
            "is_disputed": self.is_disputed,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "priority": self.priority,
            "math_verified": self.math_verified,
            "math_verification_result": self.math_verification_result,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AtomicInsight":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            insight=data["insight"],
            confidence=data["confidence"],
            tags=data["tags"],
            source_thread_id=data["source_thread_id"],
            source_exchange_indices=data.get("source_exchange_indices", []),
            extraction_model=data["extraction_model"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            depends_on=data.get("depends_on", []),
            pending_dependencies=data.get("pending_dependencies", []),
            status=InsightStatus(data.get("status", "pending")),
            rating=data.get("rating"),
            is_disputed=data.get("is_disputed", False),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
            review_notes=data.get("review_notes", ""),
            priority=data.get("priority", 5),
            math_verified=data.get("math_verified", False),
            math_verification_result=data.get("math_verification_result"),
        )

    @classmethod
    def create(
        cls,
        insight: str,
        confidence: str,
        tags: list[str],
        source_thread_id: str,
        extraction_model: str,
        depends_on: list[str] = None,
        pending_dependencies: list[str] = None,
        source_exchange_indices: list[int] = None,
    ) -> "AtomicInsight":
        """Factory method to create a new atomic insight."""
        return cls(
            id=str(uuid.uuid4())[:12],
            insight=insight,
            confidence=confidence,
            tags=tags,
            source_thread_id=source_thread_id,
            source_exchange_indices=source_exchange_indices or [],
            extraction_model=extraction_model,
            extracted_at=datetime.now(),
            depends_on=depends_on or [],
            pending_dependencies=pending_dependencies or [],
        )

    def apply_rating(self, rating: str, is_disputed: bool = False, notes: str = ""):
        """Apply review panel rating to this insight."""
        self.rating = rating
        self.is_disputed = is_disputed
        self.review_notes = notes
        self.reviewed_at = datetime.now()

        # Update status based on rating
        status_map = {
            "⚡": InsightStatus.BLESSED,
            "?": InsightStatus.INTERESTING,
            "✗": InsightStatus.REJECTED,
        }
        if rating in status_map:
            self.status = status_map[rating]

    def set_priority(self, priority: int):
        """Set the priority (1-10, clamped)."""
        self.priority = max(1, min(10, priority))

    def is_foundation_solid(self, blessed_ids: set[str]) -> bool:
        """Check if all dependencies are blessed."""
        return all(dep_id in blessed_ids for dep_id in self.depends_on)

    def save(self, base_dir: Path):
        """Save insight to appropriate directory based on status."""
        status_dirs = {
            InsightStatus.PENDING: "pending",
            InsightStatus.BLESSED: "blessed",
            InsightStatus.INTERESTING: "interesting",
            InsightStatus.REJECTED: "rejected",
        }

        current_status_dir = status_dirs[self.status]
        insight_dir = base_dir / "insights" / current_status_dir
        insight_dir.mkdir(parents=True, exist_ok=True)

        # Clean up files from OTHER status directories (prevent duplicates)
        for status, dir_name in status_dirs.items():
            if dir_name != current_status_dir:
                old_json = base_dir / "insights" / dir_name / f"{self.id}.json"
                old_md = base_dir / "insights" / dir_name / f"{self.id}.md"
                if old_json.exists():
                    old_json.unlink()
                if old_md.exists():
                    old_md.unlink()

        # Save JSON
        json_path = insight_dir / f"{self.id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # Save markdown for readability
        md_path = insight_dir / f"{self.id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    @classmethod
    def load(cls, json_path: Path) -> "AtomicInsight":
        """Load insight from JSON file."""
        with open(json_path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_markdown(self) -> str:
        """Export insight as markdown."""
        rating_display = self.rating or "(pending)"
        disputed_flag = " [DISPUTED]" if self.is_disputed else ""

        lines = [
            f"# {rating_display}{disputed_flag} Insight `{self.id}`",
            "",
            f"> {self.insight}",
            "",
            "---",
            "",
            f"**Confidence:** {self.confidence}",
            f"**Priority:** {self.priority}/10",
            f"**Tags:** {', '.join(self.tags)}",
            f"**Status:** {self.status.value}",
            f"**Extracted:** {self.extracted_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Model:** {self.extraction_model}",
            f"**Source Thread:** `{self.source_thread_id}`",
        ]

        if self.depends_on:
            lines.append(f"**Depends On:** {', '.join(self.depends_on)}")

        if self.pending_dependencies:
            lines.append(f"**Pending Dependencies:** {'; '.join(self.pending_dependencies)}")

        if self.reviewed_at:
            lines.append(f"**Reviewed:** {self.reviewed_at.strftime('%Y-%m-%d %H:%M')}")

        if self.review_notes:
            lines.extend(["", "**Review Notes:**", self.review_notes])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AtomicInsight({self.id}, confidence={self.confidence}, status={self.status.value})"


@dataclass
class InsightVersion:
    """
    A version of an insight's text (original or refined).

    Tracks the evolution of an insight through refinement rounds.
    """
    version: int                      # 1 = original, 2+ = refined
    insight: str                      # The insight text at this version
    author: str                       # "extraction" or "refinement"
    created_at: datetime
    review_round: int                 # Which review round evaluated this
    ratings: dict[str, str]           # {llm: rating} from the review

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "insight": self.insight,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "review_round": self.review_round,
            "ratings": self.ratings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InsightVersion":
        """Deserialize from dictionary."""
        return cls(
            version=data["version"],
            insight=data["insight"],
            author=data["author"],
            created_at=datetime.fromisoformat(data["created_at"]),
            review_round=data["review_round"],
            ratings=data.get("ratings", {}),
        )


@dataclass
class Refinement:
    """
    Record of a refinement operation on an insight.

    Created when Claude Opus rewrites an insight based on review critiques.
    """
    from_version: int                 # Source version number
    to_version: int                   # New version number
    original_insight: str             # Text before refinement
    refined_insight: str              # Text after refinement
    changes_made: list[str]           # List of changes made
    addressed_critiques: list[str]    # Which reviewer concerns were addressed
    unresolved_issues: list[str]      # Issues that couldn't be fixed
    refinement_confidence: str        # "high" | "medium" | "low"
    triggered_by_ratings: dict[str, str]  # Ratings that prompted refinement
    timestamp: datetime

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "original_insight": self.original_insight,
            "refined_insight": self.refined_insight,
            "changes_made": self.changes_made,
            "addressed_critiques": self.addressed_critiques,
            "unresolved_issues": self.unresolved_issues,
            "refinement_confidence": self.refinement_confidence,
            "triggered_by_ratings": self.triggered_by_ratings,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Refinement":
        """Deserialize from dictionary."""
        return cls(
            from_version=data["from_version"],
            to_version=data["to_version"],
            original_insight=data["original_insight"],
            refined_insight=data["refined_insight"],
            changes_made=data.get("changes_made", []),
            addressed_critiques=data.get("addressed_critiques", []),
            unresolved_issues=data.get("unresolved_issues", []),
            refinement_confidence=data.get("refinement_confidence", "medium"),
            triggered_by_ratings=data.get("triggered_by_ratings", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    @classmethod
    def create(
        cls,
        from_version: int,
        original_insight: str,
        refined_insight: str,
        changes_made: list[str],
        addressed_critiques: list[str],
        unresolved_issues: list[str],
        refinement_confidence: str,
        triggered_by_ratings: dict[str, str],
    ) -> "Refinement":
        """Factory method to create a new refinement record."""
        return cls(
            from_version=from_version,
            to_version=from_version + 1,
            original_insight=original_insight,
            refined_insight=refined_insight,
            changes_made=changes_made,
            addressed_critiques=addressed_critiques,
            unresolved_issues=unresolved_issues,
            refinement_confidence=refinement_confidence,
            triggered_by_ratings=triggered_by_ratings,
            timestamp=datetime.now(),
        )


@dataclass
class VersionedInsight:
    """
    An insight with full version history.

    Extends AtomicInsight with version tracking for refinement process.
    This is used when an insight goes through refinement rounds.
    """
    # Core insight data
    base: AtomicInsight

    # Version tracking
    versions: list[InsightVersion] = field(default_factory=list)
    refinements: list[Refinement] = field(default_factory=list)
    current_version: int = 1

    def __post_init__(self):
        """Initialize with original version if versions empty."""
        if not self.versions and self.base.insight:
            self.versions.append(InsightVersion(
                version=1,
                insight=self.base.insight,
                author="extraction",
                created_at=self.base.extracted_at,
                review_round=1,
                ratings={},
            ))

    @property
    def was_refined(self) -> bool:
        """Check if this insight has been refined."""
        return len(self.versions) > 1

    @property
    def original_insight(self) -> str:
        """Get the original (pre-refinement) insight text."""
        return self.versions[0].insight if self.versions else self.base.insight

    @property
    def current_insight(self) -> str:
        """Get the current (possibly refined) insight text."""
        if self.versions:
            return self.versions[self.current_version - 1].insight
        return self.base.insight

    def add_refinement(
        self,
        refined_insight: str,
        changes_made: list[str],
        addressed_critiques: list[str],
        unresolved_issues: list[str],
        refinement_confidence: str,
        triggered_by_ratings: dict[str, str],
    ) -> Refinement:
        """
        Add a refinement to this insight.

        Args:
            refined_insight: The refined text
            changes_made: List of changes made
            addressed_critiques: Which critiques were addressed
            unresolved_issues: Issues that couldn't be fixed
            refinement_confidence: How confident in the improvement
            triggered_by_ratings: The ratings that prompted refinement

        Returns:
            The created Refinement record
        """
        # Create refinement record
        refinement = Refinement.create(
            from_version=self.current_version,
            original_insight=self.current_insight,
            refined_insight=refined_insight,
            changes_made=changes_made,
            addressed_critiques=addressed_critiques,
            unresolved_issues=unresolved_issues,
            refinement_confidence=refinement_confidence,
            triggered_by_ratings=triggered_by_ratings,
        )
        self.refinements.append(refinement)

        # Create new version
        new_version = InsightVersion(
            version=refinement.to_version,
            insight=refined_insight,
            author="refinement",
            created_at=refinement.timestamp,
            review_round=len(self.versions) + 1,
            ratings={},
        )
        self.versions.append(new_version)
        self.current_version = new_version.version

        # Update base insight
        self.base.insight = refined_insight

        return refinement

    def record_ratings(self, ratings: dict[str, str]):
        """Record ratings for the current version."""
        if self.versions:
            self.versions[self.current_version - 1].ratings = ratings

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        data = self.base.to_dict()
        data["versions"] = [v.to_dict() for v in self.versions]
        data["refinements"] = [r.to_dict() for r in self.refinements]
        data["current_version"] = self.current_version
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VersionedInsight":
        """Deserialize from dictionary."""
        base = AtomicInsight.from_dict(data)
        versions = [InsightVersion.from_dict(v) for v in data.get("versions", [])]
        refinements = [Refinement.from_dict(r) for r in data.get("refinements", [])]
        current_version = data.get("current_version", 1)

        return cls(
            base=base,
            versions=versions,
            refinements=refinements,
            current_version=current_version,
        )

    @classmethod
    def from_insight(cls, insight: AtomicInsight) -> "VersionedInsight":
        """Create a VersionedInsight from an existing AtomicInsight."""
        return cls(base=insight)

    def save(self, base_dir: Path):
        """Save versioned insight (delegates to base)."""
        # Update base with current insight text
        self.base.insight = self.current_insight

        # Add version info to review notes
        if self.was_refined:
            version_note = f"\n\n[Refined from v{self.versions[0].version} to v{self.current_version}]"
            if version_note not in self.base.review_notes:
                self.base.review_notes += version_note

        self.base.save(base_dir)
