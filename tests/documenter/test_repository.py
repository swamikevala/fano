"""
Tests for DocumentRepository - split document storage.
"""

import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest

from documenter.repository import (
    DocumentRepository,
    SectionContent,
    PlacementPlan,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def repo(temp_dir: Path) -> DocumentRepository:
    """Create a DocumentRepository with temporary storage."""
    repository = DocumentRepository(temp_dir / "document")
    repository.initialize()
    return repository


@pytest.fixture
def sample_section() -> SectionContent:
    """Create a sample section for testing."""
    return SectionContent(
        id="sec_01_01",
        title="Axioms of Incidence",
        content="The Fano plane has exactly 7 points and 7 lines...",
        summary="Defines the fundamental incidence relations.",
        concepts_defined=["fano_plane", "incidence_geometry"],
        concepts_required=[],
        chapter_id="ch_01",
    )


class TestRepositoryInitialization:
    """Tests for repository initialization."""

    def test_initialize_creates_directories(self, temp_dir: Path):
        """Initialize should create all required directories."""
        repo = DocumentRepository(temp_dir / "document")
        result = repo.initialize()

        assert result is True
        assert (temp_dir / "document").exists()
        assert (temp_dir / "document" / "sections").exists()
        assert (temp_dir / "document" / "assets").exists()
        assert (temp_dir / "document" / "comments").exists()

    def test_initialize_creates_seed_files(self, temp_dir: Path):
        """Initialize should create seed structure and notation files."""
        repo = DocumentRepository(temp_dir / "document")
        repo.initialize()

        assert (temp_dir / "document" / "structure.json").exists()
        assert (temp_dir / "document" / "definitions.json").exists()
        assert (temp_dir / "document" / "notation.json").exists()

    def test_initialize_idempotent(self, repo: DocumentRepository):
        """Multiple initialize calls should not corrupt data."""
        # Add some data first
        repo.add_chapter("ch_01", "Foundations", "Test chapter")

        # Re-initialize
        repo.initialize()

        # Data should still be there
        chapter = repo.get_chapter("ch_01")
        assert chapter is not None
        assert chapter["title"] == "Foundations"


class TestStructureOperations:
    """Tests for structure.json CRUD operations."""

    def test_load_structure_returns_seed(self, repo: DocumentRepository):
        """Load structure should return seed data for new repo."""
        structure = repo.load_structure()

        assert structure["title"] == "The Fano Discoveries"
        assert structure["version"] == "3.0"
        assert len(structure["chapters"]) == 1  # Preface

    def test_add_chapter(self, repo: DocumentRepository):
        """Add chapter should create new chapter entry."""
        chapter = repo.add_chapter("ch_01", "Foundations", "Base axioms")

        assert chapter["id"] == "ch_01"
        assert chapter["title"] == "Foundations"
        assert chapter["summary"] == "Base axioms"
        assert chapter["sections"] == []

    def test_get_chapter_returns_chapter(self, repo: DocumentRepository):
        """Get chapter should return the requested chapter."""
        repo.add_chapter("ch_01", "Foundations")

        chapter = repo.get_chapter("ch_01")
        assert chapter is not None
        assert chapter["id"] == "ch_01"

    def test_get_chapter_returns_none_for_missing(self, repo: DocumentRepository):
        """Get chapter should return None for non-existent chapter."""
        chapter = repo.get_chapter("ch_99")
        assert chapter is None

    def test_add_section(self, repo: DocumentRepository, sample_section: SectionContent):
        """Add section should create file and update structure."""
        repo.add_chapter("ch_01", "Foundations")
        result = repo.add_section("ch_01", sample_section)

        assert result is True

        # Check structure updated
        chapter = repo.get_chapter("ch_01")
        assert len(chapter["sections"]) == 1
        assert chapter["sections"][0]["id"] == "sec_01_01"

        # Check file created
        section_file = repo.sections_dir / "sec_01_01.md"
        assert section_file.exists()
        assert "Fano plane" in section_file.read_text()

    def test_add_section_at_position(self, repo: DocumentRepository):
        """Add section with position should insert at correct index."""
        repo.add_chapter("ch_01", "Foundations")

        # Add first section
        sec1 = SectionContent(
            id="sec_01_01", title="First", content="First content",
            summary="First section", chapter_id="ch_01",
        )
        repo.add_section("ch_01", sec1)

        # Add second section at position 0
        sec2 = SectionContent(
            id="sec_01_00", title="Before First", content="Before content",
            summary="Inserted section", chapter_id="ch_01",
        )
        repo.add_section("ch_01", sec2, position=0)

        chapter = repo.get_chapter("ch_01")
        assert chapter["sections"][0]["id"] == "sec_01_00"
        assert chapter["sections"][1]["id"] == "sec_01_01"

    def test_update_section(self, repo: DocumentRepository, sample_section: SectionContent):
        """Update section should modify content and metadata."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        # Update
        sample_section.content = "Updated content about the Fano plane."
        sample_section.summary = "Updated summary."
        result = repo.update_section(sample_section)

        assert result is True

        # Verify update
        loaded = repo.get_section_content("sec_01_01")
        assert "Updated content" in loaded.content
        assert loaded.summary == "Updated summary."

    def test_get_section_content(self, repo: DocumentRepository, sample_section: SectionContent):
        """Get section content should return full section data."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        loaded = repo.get_section_content("sec_01_01")

        assert loaded is not None
        assert loaded.id == "sec_01_01"
        assert loaded.title == "Axioms of Incidence"
        assert "Fano plane" in loaded.content
        assert loaded.chapter_id == "ch_01"

    def test_get_section_meta(self, repo: DocumentRepository, sample_section: SectionContent):
        """Get section meta should return chapter and section tuple."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        result = repo.get_section_meta("sec_01_01")

        assert result is not None
        chapter, section = result
        assert chapter["id"] == "ch_01"
        assert section["id"] == "sec_01_01"

    def test_get_adjacent_summaries(self, repo: DocumentRepository):
        """Get adjacent summaries should return prev/next context."""
        repo.add_chapter("ch_01", "Foundations")

        # Add three sections
        for i in range(3):
            sec = SectionContent(
                id=f"sec_01_{i:02d}",
                title=f"Section {i}",
                content=f"Content {i}",
                summary=f"Summary {i}",
                chapter_id="ch_01",
            )
            repo.add_section("ch_01", sec)

        # Middle section should have both
        prev_sum, next_sum = repo.get_adjacent_summaries("sec_01_01")
        assert prev_sum == "Summary 0"
        assert next_sum == "Summary 2"

        # First section should have no prev
        prev_sum, next_sum = repo.get_adjacent_summaries("sec_01_00")
        assert prev_sum == ""
        assert next_sum == "Summary 1"

        # Last section should have no next
        prev_sum, next_sum = repo.get_adjacent_summaries("sec_01_02")
        assert prev_sum == "Summary 1"
        assert next_sum == ""


class TestDefinitionsRegistry:
    """Tests for definitions.json operations."""

    def test_register_concept(self, repo: DocumentRepository):
        """Register concept should add to definitions."""
        repo.register_concept("fano_plane", "sec_01_01", "The 7-point projective plane")

        assert repo.is_concept_defined("fano_plane")
        assert repo.get_concept_section("fano_plane") == "sec_01_01"

    def test_is_concept_defined(self, repo: DocumentRepository):
        """Is concept defined should correctly check registry."""
        assert not repo.is_concept_defined("fano_plane")

        repo.register_concept("fano_plane", "sec_01_01")

        assert repo.is_concept_defined("fano_plane")

    def test_get_missing_prerequisites(self, repo: DocumentRepository):
        """Get missing prerequisites should return undefined concepts."""
        repo.register_concept("field", "sec_00_01")

        missing = repo.get_missing_prerequisites(["field", "vector_space", "ring"])

        assert "field" not in missing
        assert "vector_space" in missing
        assert "ring" in missing

    def test_add_section_registers_concepts(
        self, repo: DocumentRepository, sample_section: SectionContent
    ):
        """Adding section should auto-register its defined concepts."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        assert repo.is_concept_defined("fano_plane")
        assert repo.is_concept_defined("incidence_geometry")


class TestCompilation:
    """Tests for compile_book functionality."""

    def test_compile_book_creates_file(self, repo: DocumentRepository, sample_section: SectionContent):
        """Compile book should create compiled_book.md."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        compiled = repo.compile_book()

        assert repo.compiled_path.exists()
        assert "The Fano Discoveries" in compiled
        assert "Axioms of Incidence" in compiled
        assert "Fano plane" in compiled

    def test_compile_book_includes_toc(self, repo: DocumentRepository, sample_section: SectionContent):
        """Compiled book should include table of contents."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        compiled = repo.compile_book()

        assert "Table of Contents" in compiled
        assert "Axioms of Incidence" in compiled

    def test_get_structure_summary(self, repo: DocumentRepository, sample_section: SectionContent):
        """Get structure summary should return compact overview."""
        repo.add_chapter("ch_01", "Foundations")
        repo.add_section("ch_01", sample_section)

        summary = repo.get_structure_summary()

        assert "Structure Overview" in summary
        assert "ch_01" in summary
        assert "sec_01_01" in summary
        assert "fano_plane" in summary


class TestSectionIdGeneration:
    """Tests for section ID generation."""

    def test_generate_section_id_first(self, repo: DocumentRepository):
        """Generate section ID should return proper ID for empty chapter."""
        repo.add_chapter("ch_01", "Foundations")

        section_id = repo.generate_section_id("ch_01")

        assert section_id == "sec_01_01"

    def test_generate_section_id_increment(self, repo: DocumentRepository):
        """Generate section ID should increment for existing sections."""
        repo.add_chapter("ch_01", "Foundations")

        # Add first section
        sec1 = SectionContent(
            id="sec_01_01", title="First", content="Content",
            summary="Summary", chapter_id="ch_01",
        )
        repo.add_section("ch_01", sec1)

        # Generate next
        section_id = repo.generate_section_id("ch_01")

        assert section_id == "sec_01_02"


class TestSectionChecksum:
    """Tests for content checksum functionality."""

    def test_compute_checksum(self, sample_section: SectionContent):
        """Compute checksum should return consistent hash."""
        checksum1 = sample_section.compute_checksum()
        checksum2 = sample_section.compute_checksum()

        assert checksum1 == checksum2
        assert len(checksum1) == 16  # Truncated to 16 chars

    def test_checksum_changes_with_content(self, sample_section: SectionContent):
        """Checksum should change when content changes."""
        checksum1 = sample_section.compute_checksum()

        sample_section.content = "Different content"
        checksum2 = sample_section.compute_checksum()

        assert checksum1 != checksum2


class TestNotation:
    """Tests for notation guide functionality."""

    def test_load_notation_returns_seed(self, repo: DocumentRepository):
        """Load notation should return seed entries."""
        notation = repo.load_notation()

        assert "field" in notation
        assert notation["field"]["latex"] == "\\mathbb{F}"

    def test_get_notation_for_concepts(self, repo: DocumentRepository):
        """Get notation should filter to requested concepts."""
        notation = repo.get_notation_for_concepts(["field", "nonexistent"])

        assert "field" in notation
        assert "nonexistent" not in notation


class TestPlacementPlan:
    """Tests for PlacementPlan data class."""

    def test_placement_plan_insert(self):
        """PlacementPlan should support INSERT mode."""
        plan = PlacementPlan(
            mode="INSERT",
            target_chapter_id="ch_01",
            new_section_title="New Section",
            rationale="This is new content",
        )

        assert plan.mode == "INSERT"
        assert plan.target_section_id is None

    def test_placement_plan_merge(self):
        """PlacementPlan should support MERGE mode."""
        plan = PlacementPlan(
            mode="MERGE",
            target_section_id="sec_01_01",
            target_chapter_id="ch_01",
            rationale="Content should be merged into existing section",
        )

        assert plan.mode == "MERGE"
        assert plan.target_section_id == "sec_01_01"

    def test_placement_plan_conflict(self):
        """PlacementPlan should support conflict detection."""
        plan = PlacementPlan(
            mode="INSERT",
            target_chapter_id="ch_01",
            conflict_detected=True,
            conflict_description="This insight contradicts section sec_01_02",
        )

        assert plan.conflict_detected is True
        assert "contradicts" in plan.conflict_description
