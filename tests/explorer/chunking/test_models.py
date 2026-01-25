"""
Tests for explorer chunking models.

Tests cover:
- AtomicInsight creation, serialization, and status management
- InsightVersion tracking
- Refinement records
- VersionedInsight with version history
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from explorer.src.chunking.models import (
    AtomicInsight,
    InsightStatus,
    InsightVersion,
    Refinement,
    VersionedInsight,
)


class TestAtomicInsightCreation:
    """Tests for AtomicInsight creation and factory method."""

    def test_create_generates_id(self):
        """create() generates a unique ID."""
        insight = AtomicInsight.create(
            insight="Test insight",
            confidence="high",
            tags=["test"],
            source_thread_id="thread-1",
            extraction_model="claude",
        )
        assert insight.id is not None
        assert len(insight.id) == 12

    def test_create_sets_defaults(self):
        """create() sets appropriate defaults."""
        insight = AtomicInsight.create(
            insight="Test insight",
            confidence="medium",
            tags=["test"],
            source_thread_id="thread-1",
            extraction_model="claude",
        )
        assert insight.status == InsightStatus.PENDING
        assert insight.rating is None
        assert insight.is_disputed is False
        assert insight.priority == 5
        assert insight.depends_on == []
        assert insight.pending_dependencies == []

    def test_create_with_dependencies(self):
        """create() accepts dependency lists."""
        insight = AtomicInsight.create(
            insight="Dependent insight",
            confidence="high",
            tags=["test"],
            source_thread_id="thread-1",
            extraction_model="claude",
            depends_on=["insight-1", "insight-2"],
            pending_dependencies=["Some concept"],
        )
        assert insight.depends_on == ["insight-1", "insight-2"]
        assert insight.pending_dependencies == ["Some concept"]

    def test_create_sets_extraction_time(self):
        """create() sets extraction timestamp."""
        before = datetime.now()
        insight = AtomicInsight.create(
            insight="Test",
            confidence="low",
            tags=[],
            source_thread_id="t1",
            extraction_model="gemini",
        )
        after = datetime.now()

        assert before <= insight.extracted_at <= after


class TestAtomicInsightSerialization:
    """Tests for AtomicInsight serialization."""

    def test_to_dict_includes_all_fields(self, sample_atomic_insight):
        """to_dict() includes all required fields."""
        data = sample_atomic_insight.to_dict()

        assert data["id"] == sample_atomic_insight.id
        assert data["insight"] == sample_atomic_insight.insight
        assert data["confidence"] == sample_atomic_insight.confidence
        assert data["tags"] == sample_atomic_insight.tags
        assert data["source_thread_id"] == sample_atomic_insight.source_thread_id
        assert data["extraction_model"] == sample_atomic_insight.extraction_model
        assert data["status"] == "pending"

    def test_from_dict_restores_insight(self, sample_atomic_insight):
        """from_dict() correctly restores an insight."""
        data = sample_atomic_insight.to_dict()
        restored = AtomicInsight.from_dict(data)

        assert restored.id == sample_atomic_insight.id
        assert restored.insight == sample_atomic_insight.insight
        assert restored.confidence == sample_atomic_insight.confidence
        assert restored.status == sample_atomic_insight.status

    def test_roundtrip_serialization(self, sample_atomic_insight):
        """Serialization roundtrip preserves all data."""
        data = sample_atomic_insight.to_dict()
        restored = AtomicInsight.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2

    def test_to_json_is_valid(self, sample_atomic_insight):
        """to_dict() produces valid JSON."""
        data = sample_atomic_insight.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed == data


class TestAtomicInsightRating:
    """Tests for AtomicInsight rating and status management."""

    def test_apply_rating_blessing(self, sample_atomic_insight):
        """apply_rating with ⚡ sets status to BLESSED."""
        sample_atomic_insight.apply_rating("⚡", notes="Approved")

        assert sample_atomic_insight.rating == "⚡"
        assert sample_atomic_insight.status == InsightStatus.BLESSED
        assert sample_atomic_insight.review_notes == "Approved"
        assert sample_atomic_insight.reviewed_at is not None

    def test_apply_rating_interesting(self, sample_atomic_insight):
        """apply_rating with ? sets status to INTERESTING."""
        sample_atomic_insight.apply_rating("?", notes="Needs more exploration")

        assert sample_atomic_insight.rating == "?"
        assert sample_atomic_insight.status == InsightStatus.INTERESTING

    def test_apply_rating_rejection(self, sample_atomic_insight):
        """apply_rating with ✗ sets status to REJECTED."""
        sample_atomic_insight.apply_rating("✗", notes="Not accurate")

        assert sample_atomic_insight.rating == "✗"
        assert sample_atomic_insight.status == InsightStatus.REJECTED

    def test_apply_rating_disputed(self, sample_atomic_insight):
        """apply_rating can mark as disputed."""
        sample_atomic_insight.apply_rating("⚡", is_disputed=True, notes="Split decision")

        assert sample_atomic_insight.is_disputed is True
        assert sample_atomic_insight.status == InsightStatus.BLESSED


class TestAtomicInsightPriority:
    """Tests for AtomicInsight priority management."""

    def test_set_priority_valid_range(self, sample_atomic_insight):
        """set_priority accepts values 1-10."""
        sample_atomic_insight.set_priority(10)
        assert sample_atomic_insight.priority == 10

        sample_atomic_insight.set_priority(1)
        assert sample_atomic_insight.priority == 1

    def test_set_priority_clamps_high(self, sample_atomic_insight):
        """set_priority clamps values above 10."""
        sample_atomic_insight.set_priority(15)
        assert sample_atomic_insight.priority == 10

    def test_set_priority_clamps_low(self, sample_atomic_insight):
        """set_priority clamps values below 1."""
        sample_atomic_insight.set_priority(0)
        assert sample_atomic_insight.priority == 1

        sample_atomic_insight.set_priority(-5)
        assert sample_atomic_insight.priority == 1


class TestAtomicInsightDependencies:
    """Tests for AtomicInsight dependency checking."""

    def test_is_foundation_solid_no_deps(self, sample_atomic_insight):
        """is_foundation_solid returns True with no dependencies."""
        assert sample_atomic_insight.is_foundation_solid(set()) is True

    def test_is_foundation_solid_all_blessed(self, sample_atomic_insight):
        """is_foundation_solid returns True when all deps are blessed."""
        sample_atomic_insight.depends_on = ["id-1", "id-2"]
        blessed_ids = {"id-1", "id-2", "id-3"}

        assert sample_atomic_insight.is_foundation_solid(blessed_ids) is True

    def test_is_foundation_solid_missing_dep(self, sample_atomic_insight):
        """is_foundation_solid returns False when a dep is missing."""
        sample_atomic_insight.depends_on = ["id-1", "id-2", "id-missing"]
        blessed_ids = {"id-1", "id-2"}

        assert sample_atomic_insight.is_foundation_solid(blessed_ids) is False


class TestAtomicInsightStorage:
    """Tests for AtomicInsight save/load operations."""

    def test_save_creates_json_file(self, sample_atomic_insight, temp_dir):
        """save() creates JSON file in correct directory."""
        sample_atomic_insight.save(temp_dir)

        expected_path = temp_dir / "insights" / "pending" / f"{sample_atomic_insight.id}.json"
        assert expected_path.exists()

    def test_save_creates_markdown_file(self, sample_atomic_insight, temp_dir):
        """save() creates markdown file alongside JSON."""
        sample_atomic_insight.save(temp_dir)

        md_path = temp_dir / "insights" / "pending" / f"{sample_atomic_insight.id}.md"
        assert md_path.exists()

    def test_save_moves_on_status_change(self, sample_atomic_insight, temp_dir):
        """save() moves file when status changes."""
        # First save as pending
        sample_atomic_insight.save(temp_dir)
        pending_path = temp_dir / "insights" / "pending" / f"{sample_atomic_insight.id}.json"
        assert pending_path.exists()

        # Change status and save
        sample_atomic_insight.apply_rating("⚡")
        sample_atomic_insight.save(temp_dir)

        blessed_path = temp_dir / "insights" / "blessed" / f"{sample_atomic_insight.id}.json"
        assert blessed_path.exists()
        assert not pending_path.exists()  # Old file removed

    def test_load_restores_insight(self, sample_atomic_insight, temp_dir):
        """load() correctly restores a saved insight."""
        sample_atomic_insight.save(temp_dir)
        json_path = temp_dir / "insights" / "pending" / f"{sample_atomic_insight.id}.json"

        loaded = AtomicInsight.load(json_path)

        assert loaded.id == sample_atomic_insight.id
        assert loaded.insight == sample_atomic_insight.insight
        assert loaded.tags == sample_atomic_insight.tags


class TestAtomicInsightMarkdown:
    """Tests for AtomicInsight markdown export."""

    def test_to_markdown_includes_insight(self, sample_atomic_insight):
        """to_markdown() includes the insight text."""
        md = sample_atomic_insight.to_markdown()
        assert sample_atomic_insight.insight in md

    def test_to_markdown_includes_metadata(self, sample_atomic_insight):
        """to_markdown() includes metadata fields."""
        md = sample_atomic_insight.to_markdown()

        assert sample_atomic_insight.confidence in md
        assert "Tags:" in md
        assert sample_atomic_insight.source_thread_id in md

    def test_to_markdown_shows_disputed_flag(self, sample_atomic_insight):
        """to_markdown() shows disputed flag when set."""
        sample_atomic_insight.apply_rating("⚡", is_disputed=True)
        md = sample_atomic_insight.to_markdown()

        assert "[DISPUTED]" in md


class TestInsightVersion:
    """Tests for InsightVersion tracking."""

    def test_create_version(self):
        """InsightVersion can be created with required fields."""
        version = InsightVersion(
            version=1,
            insight="Original insight",
            author="extraction",
            created_at=datetime.now(),
            review_round=1,
            ratings={},
        )
        assert version.version == 1
        assert version.insight == "Original insight"

    def test_version_serialization(self):
        """InsightVersion serializes correctly."""
        version = InsightVersion(
            version=1,
            insight="Test",
            author="extraction",
            created_at=datetime.now(),
            review_round=1,
            ratings={"claude": "⚡"},
        )

        data = version.to_dict()
        restored = InsightVersion.from_dict(data)

        assert restored.version == version.version
        assert restored.insight == version.insight
        assert restored.ratings == version.ratings


class TestRefinement:
    """Tests for Refinement records."""

    def test_create_refinement(self):
        """Refinement.create() generates correct record."""
        refinement = Refinement.create(
            from_version=1,
            original_insight="Original text",
            refined_insight="Refined text",
            changes_made=["Added precision", "Fixed terminology"],
            addressed_critiques=["Was too vague"],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings={"claude": "?"},
        )

        assert refinement.from_version == 1
        assert refinement.to_version == 2
        assert refinement.original_insight == "Original text"
        assert refinement.refined_insight == "Refined text"
        assert len(refinement.changes_made) == 2

    def test_refinement_serialization(self):
        """Refinement serializes correctly."""
        refinement = Refinement.create(
            from_version=1,
            original_insight="Original",
            refined_insight="Refined",
            changes_made=["Change 1"],
            addressed_critiques=["Critique 1"],
            unresolved_issues=["Issue 1"],
            refinement_confidence="medium",
            triggered_by_ratings={"gemini": "?"},
        )

        data = refinement.to_dict()
        restored = Refinement.from_dict(data)

        assert restored.from_version == refinement.from_version
        assert restored.to_version == refinement.to_version
        assert restored.refined_insight == refinement.refined_insight


class TestVersionedInsight:
    """Tests for VersionedInsight with version history."""

    def test_from_insight_creates_initial_version(self, sample_atomic_insight):
        """VersionedInsight.from_insight() creates initial version."""
        versioned = VersionedInsight.from_insight(sample_atomic_insight)

        assert versioned.current_version == 1
        assert len(versioned.versions) == 1
        assert versioned.versions[0].insight == sample_atomic_insight.insight

    def test_was_refined_false_initially(self, sample_versioned_insight):
        """was_refined is False for unrefined insight."""
        assert sample_versioned_insight.was_refined is False

    def test_original_insight_property(self, sample_versioned_insight):
        """original_insight returns first version."""
        assert sample_versioned_insight.original_insight == sample_versioned_insight.base.insight

    def test_add_refinement_increments_version(self, sample_versioned_insight):
        """add_refinement() increments version number."""
        sample_versioned_insight.add_refinement(
            refined_insight="Refined version of the insight",
            changes_made=["Made more precise"],
            addressed_critiques=["Vagueness"],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings={"claude": "?"},
        )

        assert sample_versioned_insight.current_version == 2
        assert sample_versioned_insight.was_refined is True

    def test_add_refinement_tracks_history(self, sample_versioned_insight):
        """add_refinement() preserves original version."""
        original = sample_versioned_insight.current_insight

        sample_versioned_insight.add_refinement(
            refined_insight="New text",
            changes_made=["Changed"],
            addressed_critiques=[],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings={},
        )

        assert sample_versioned_insight.original_insight == original
        assert sample_versioned_insight.current_insight == "New text"

    def test_record_ratings(self, sample_versioned_insight):
        """record_ratings() stores ratings for current version."""
        ratings = {"claude": "⚡", "gemini": "⚡"}
        sample_versioned_insight.record_ratings(ratings)

        assert sample_versioned_insight.versions[0].ratings == ratings

    def test_versioned_serialization(self, sample_versioned_insight):
        """VersionedInsight serializes correctly."""
        sample_versioned_insight.add_refinement(
            refined_insight="Refined",
            changes_made=["Change"],
            addressed_critiques=[],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings={},
        )

        data = sample_versioned_insight.to_dict()
        restored = VersionedInsight.from_dict(data)

        assert restored.current_version == sample_versioned_insight.current_version
        assert len(restored.versions) == len(sample_versioned_insight.versions)
        assert len(restored.refinements) == len(sample_versioned_insight.refinements)


class TestInsightStatus:
    """Tests for InsightStatus enum."""

    def test_status_values(self):
        """InsightStatus has expected values."""
        assert InsightStatus.PENDING.value == "pending"
        assert InsightStatus.BLESSED.value == "blessed"
        assert InsightStatus.INTERESTING.value == "interesting"
        assert InsightStatus.REJECTED.value == "rejected"

    def test_status_from_string(self):
        """InsightStatus can be created from string value."""
        assert InsightStatus("pending") == InsightStatus.PENDING
        assert InsightStatus("blessed") == InsightStatus.BLESSED
