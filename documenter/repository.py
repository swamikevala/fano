"""
Document Repository - Split storage management for the Living Textbook architecture.

Handles the graph-based Document Object Model with:
- structure.json: Table of contents with summaries (~2k tokens)
- definitions.json: Concept registry (where each is defined)
- notation.json: Style guide for LaTeX conventions
- sections/*.md: Atomic content files
- compiled_book.md: Read-only concatenation for humans
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from shared.logging import get_logger

log = get_logger("documenter", "repository")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class SectionMeta(TypedDict):
    """Metadata for a single section in structure.json."""
    id: str
    title: str
    file: str
    summary: str
    concepts_defined: list[str]
    concepts_required: list[str]
    checksum: str
    created: str
    last_modified: str
    status: str  # provisional | stable | needs_work


class ChapterMeta(TypedDict):
    """Metadata for a chapter in structure.json."""
    id: str
    title: str
    summary: str
    sections: list[SectionMeta]


class StructureDoc(TypedDict):
    """The structure.json schema."""
    title: str
    version: str
    chapters: list[ChapterMeta]
    last_compiled: str


class ConceptEntry(TypedDict):
    """Entry in definitions.json."""
    section_id: str
    defined_at: str
    summary: str


class NotationEntry(TypedDict):
    """Entry in notation.json."""
    symbol: str
    meaning: str
    latex: str
    example: str


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class SectionContent:
    """Full section data including content."""
    id: str
    title: str
    content: str
    summary: str
    concepts_defined: list[str] = field(default_factory=list)
    concepts_required: list[str] = field(default_factory=list)
    chapter_id: str = ""
    created: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    status: str = "provisional"

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class PlacementPlan:
    """Result of Architect's placement decision."""
    mode: str  # INSERT | MERGE | APPEND
    target_section_id: Optional[str] = None
    target_chapter_id: str = ""
    new_section_title: Optional[str] = None
    rationale: str = ""
    conflict_detected: bool = False
    conflict_description: str = ""


# -----------------------------------------------------------------------------
# Document Repository
# -----------------------------------------------------------------------------

class DocumentRepository:
    """
    Manages split document storage for the Living Textbook.

    File Structure:
        data/document/
        ├── structure.json      # TOC with summaries
        ├── definitions.json    # Concept registry
        ├── notation.json       # Style guide
        ├── compiled_book.md    # Read-only view
        ├── assets/             # Diagrams
        ├── sections/           # Content files
        └── comments/
            └── pending.json
    """

    def __init__(self, base_path: Path):
        """
        Initialize repository.

        Args:
            base_path: Base directory for document storage (e.g., data/document)
        """
        self.base_path = Path(base_path)
        self.structure_path = self.base_path / "structure.json"
        self.definitions_path = self.base_path / "definitions.json"
        self.notation_path = self.base_path / "notation.json"
        self.sections_dir = self.base_path / "sections"
        self.assets_dir = self.base_path / "assets"
        self.comments_dir = self.base_path / "comments"
        self.compiled_path = self.base_path / "compiled_book.md"

        # In-memory cache
        self._structure: Optional[StructureDoc] = None
        self._definitions: Optional[dict[str, ConceptEntry]] = None
        self._notation: Optional[dict[str, NotationEntry]] = None

        # Architect thread persistence
        self.architect_thread_id: Optional[str] = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Initialize repository structure.

        Creates directories and seed files if they don't exist.

        Returns:
            True if successful
        """
        try:
            # Create directories
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.sections_dir.mkdir(exist_ok=True)
            self.assets_dir.mkdir(exist_ok=True)
            self.comments_dir.mkdir(exist_ok=True)

            # Create seed structure if needed
            if not self.structure_path.exists():
                self._create_seed_structure()

            # Create seed definitions if needed
            if not self.definitions_path.exists():
                self._save_json(self.definitions_path, {})

            # Create seed notation if needed
            if not self.notation_path.exists():
                self._create_seed_notation()

            log.info(
                "repository.initialized",
                base_path=str(self.base_path),
            )
            return True

        except Exception as e:
            log.error("repository.init_error", error=str(e))
            return False

    def _create_seed_structure(self):
        """Create initial structure.json."""
        structure: StructureDoc = {
            "title": "The Fano Discoveries",
            "version": "3.0",
            "chapters": [
                {
                    "id": "ch_00",
                    "title": "Preface",
                    "summary": "Introduction and guiding principles.",
                    "sections": [],
                }
            ],
            "last_compiled": "",
        }
        self._save_json(self.structure_path, structure)
        log.info("repository.seed_structure_created")

    def _create_seed_notation(self):
        """Create initial notation.json with common mathematical symbols."""
        notation: dict[str, NotationEntry] = {
            "field": {
                "symbol": "𝔽",
                "meaning": "A field (algebraic structure)",
                "latex": "\\mathbb{F}",
                "example": "𝔽₂ is the field with two elements",
            },
            "vector_space": {
                "symbol": "V",
                "meaning": "A vector space",
                "latex": "V",
                "example": "V = 𝔽₂³",
            },
            "projective_plane": {
                "symbol": "PG(2,q)",
                "meaning": "Projective plane of order q",
                "latex": "\\text{PG}(2,q)",
                "example": "PG(2,2) is the Fano plane",
            },
        }
        self._save_json(self.notation_path, notation)
        log.info("repository.seed_notation_created")

    # -------------------------------------------------------------------------
    # Structure Operations
    # -------------------------------------------------------------------------

    def load_structure(self) -> StructureDoc:
        """
        Load structure.json.

        Returns:
            Structure document
        """
        if self._structure is None:
            self._structure = self._load_json(self.structure_path) or {
                "title": "The Fano Discoveries",
                "version": "3.0",
                "chapters": [],
                "last_compiled": "",
            }
        return self._structure

    def save_structure(self):
        """Save structure.json to disk."""
        if self._structure is not None:
            self._save_json(self.structure_path, self._structure)
            log.info("repository.structure_saved")

    def get_chapter(self, chapter_id: str) -> Optional[ChapterMeta]:
        """Get a chapter by ID."""
        structure = self.load_structure()
        for chapter in structure["chapters"]:
            if chapter["id"] == chapter_id:
                return chapter
        return None

    def get_section_meta(self, section_id: str) -> Optional[tuple[ChapterMeta, SectionMeta]]:
        """
        Get section metadata and its parent chapter.

        Returns:
            (chapter, section) tuple or None
        """
        structure = self.load_structure()
        for chapter in structure["chapters"]:
            for section in chapter["sections"]:
                if section["id"] == section_id:
                    return chapter, section
        return None

    def add_chapter(self, chapter_id: str, title: str, summary: str = "") -> ChapterMeta:
        """
        Add a new chapter.

        Args:
            chapter_id: Unique chapter identifier
            title: Chapter title
            summary: Brief description

        Returns:
            The created chapter
        """
        structure = self.load_structure()

        chapter: ChapterMeta = {
            "id": chapter_id,
            "title": title,
            "summary": summary,
            "sections": [],
        }
        structure["chapters"].append(chapter)
        self._structure = structure
        self.save_structure()

        log.info("repository.chapter_added", chapter_id=chapter_id, title=title)
        return chapter

    def add_section(
        self,
        chapter_id: str,
        section: SectionContent,
        position: Optional[int] = None,
    ) -> bool:
        """
        Add a new section to a chapter.

        Args:
            chapter_id: Parent chapter ID
            section: Section content and metadata
            position: Index to insert at (None = append)

        Returns:
            True if successful
        """
        structure = self.load_structure()

        # Find chapter
        chapter = None
        for ch in structure["chapters"]:
            if ch["id"] == chapter_id:
                chapter = ch
                break

        if chapter is None:
            log.error("repository.chapter_not_found", chapter_id=chapter_id)
            return False

        # Write section file
        filename = f"{section.id}.md"
        section_path = self.sections_dir / filename

        if not self._write_section_file(section_path, section):
            return False

        # Create metadata
        meta: SectionMeta = {
            "id": section.id,
            "title": section.title,
            "file": f"sections/{filename}",
            "summary": section.summary,
            "concepts_defined": section.concepts_defined,
            "concepts_required": section.concepts_required,
            "checksum": section.compute_checksum(),
            "created": section.created.isoformat(),
            "last_modified": section.last_modified.isoformat(),
            "status": section.status,
        }

        # Add to chapter
        if position is not None and 0 <= position < len(chapter["sections"]):
            chapter["sections"].insert(position, meta)
        else:
            chapter["sections"].append(meta)

        # Update definitions registry
        for concept in section.concepts_defined:
            self.register_concept(concept, section.id, section.summary)

        self._structure = structure
        self.save_structure()

        log.info(
            "repository.section_added",
            section_id=section.id,
            chapter_id=chapter_id,
            concepts_defined=section.concepts_defined,
        )
        return True

    def update_section(self, section: SectionContent) -> bool:
        """
        Update an existing section.

        Args:
            section: Updated section content

        Returns:
            True if successful
        """
        result = self.get_section_meta(section.id)
        if result is None:
            log.error("repository.section_not_found", section_id=section.id)
            return False

        chapter, meta = result

        # Update file
        section_path = self.base_path / meta["file"]
        if not self._write_section_file(section_path, section):
            return False

        # Update metadata
        meta["title"] = section.title
        meta["summary"] = section.summary
        meta["concepts_defined"] = section.concepts_defined
        meta["concepts_required"] = section.concepts_required
        meta["checksum"] = section.compute_checksum()
        meta["last_modified"] = datetime.now().isoformat()
        meta["status"] = section.status

        # Update definitions for new concepts
        for concept in section.concepts_defined:
            if not self.is_concept_defined(concept):
                self.register_concept(concept, section.id, section.summary)

        self.save_structure()

        log.info("repository.section_updated", section_id=section.id)
        return True

    def get_section_content(self, section_id: str) -> Optional[SectionContent]:
        """
        Load full section content.

        Args:
            section_id: Section to load

        Returns:
            Section with content or None
        """
        result = self.get_section_meta(section_id)
        if result is None:
            return None

        chapter, meta = result
        section_path = self.base_path / meta["file"]

        if not section_path.exists():
            log.error("repository.section_file_missing", path=str(section_path))
            return None

        content = section_path.read_text(encoding="utf-8")

        return SectionContent(
            id=meta["id"],
            title=meta["title"],
            content=content,
            summary=meta["summary"],
            concepts_defined=meta["concepts_defined"],
            concepts_required=meta["concepts_required"],
            chapter_id=chapter["id"],
            created=datetime.fromisoformat(meta["created"]),
            last_modified=datetime.fromisoformat(meta["last_modified"]),
            status=meta["status"],
        )

    def get_adjacent_summaries(self, section_id: str) -> tuple[str, str]:
        """
        Get summaries of previous and next sections for context.

        Args:
            section_id: Current section ID

        Returns:
            (prev_summary, next_summary) - empty strings if none
        """
        structure = self.load_structure()

        # Flatten all sections in order
        all_sections: list[SectionMeta] = []
        for chapter in structure["chapters"]:
            all_sections.extend(chapter["sections"])

        # Find position
        current_idx = None
        for i, sec in enumerate(all_sections):
            if sec["id"] == section_id:
                current_idx = i
                break

        if current_idx is None:
            return "", ""

        prev_summary = all_sections[current_idx - 1]["summary"] if current_idx > 0 else ""
        next_summary = all_sections[current_idx + 1]["summary"] if current_idx < len(all_sections) - 1 else ""

        return prev_summary, next_summary

    def _write_section_file(self, path: Path, section: SectionContent) -> bool:
        """Write section content to file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(section.content, encoding="utf-8")
            return True
        except Exception as e:
            log.error("repository.section_write_error", path=str(path), error=str(e))
            return False

    # -------------------------------------------------------------------------
    # Definitions Registry
    # -------------------------------------------------------------------------

    def load_definitions(self) -> dict[str, ConceptEntry]:
        """Load definitions.json."""
        if self._definitions is None:
            self._definitions = self._load_json(self.definitions_path) or {}
        return self._definitions

    def save_definitions(self):
        """Save definitions.json to disk."""
        if self._definitions is not None:
            self._save_json(self.definitions_path, self._definitions)
            log.info("repository.definitions_saved")

    def is_concept_defined(self, concept: str) -> bool:
        """Check if a concept is already defined."""
        return concept in self.load_definitions()

    def get_concept_section(self, concept: str) -> Optional[str]:
        """Get the section ID where a concept is defined."""
        definitions = self.load_definitions()
        entry = definitions.get(concept)
        return entry["section_id"] if entry else None

    def register_concept(self, concept: str, section_id: str, summary: str = ""):
        """
        Register a concept in the definitions registry.

        Args:
            concept: Concept name
            section_id: Section where it's defined
            summary: Brief description
        """
        definitions = self.load_definitions()

        if concept in definitions:
            log.warning(
                "repository.concept_already_defined",
                concept=concept,
                existing_section=definitions[concept]["section_id"],
                new_section=section_id,
            )
            return

        definitions[concept] = {
            "section_id": section_id,
            "defined_at": datetime.now().strftime("%Y-%m-%d"),
            "summary": summary,
        }

        self._definitions = definitions
        self.save_definitions()

        log.info("repository.concept_registered", concept=concept, section_id=section_id)

    def get_missing_prerequisites(self, requires: list[str]) -> list[str]:
        """
        Check which required concepts are not yet defined.

        Args:
            requires: List of required concepts

        Returns:
            List of missing concepts
        """
        definitions = self.load_definitions()
        return [c for c in requires if c not in definitions]

    # -------------------------------------------------------------------------
    # Notation Guide
    # -------------------------------------------------------------------------

    def load_notation(self) -> dict[str, NotationEntry]:
        """Load notation.json."""
        if self._notation is None:
            self._notation = self._load_json(self.notation_path) or {}
        return self._notation

    def get_notation_for_concepts(self, concepts: list[str]) -> dict[str, NotationEntry]:
        """Get notation entries for specified concepts."""
        notation = self.load_notation()
        return {c: notation[c] for c in concepts if c in notation}

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------

    def compile_book(self) -> str:
        """
        Compile all sections into a single markdown document.

        Returns:
            Complete book as markdown string
        """
        structure = self.load_structure()
        parts: list[str] = []

        # Title
        parts.append(f"# {structure['title']}\n")
        parts.append(f"*Version {structure['version']} - Compiled {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        parts.append("---\n")

        # Table of contents
        parts.append("## Table of Contents\n")
        for chapter in structure["chapters"]:
            parts.append(f"\n### {chapter['title']}\n")
            for section in chapter["sections"]:
                parts.append(f"- [{section['title']}](#{section['id']})\n")

        parts.append("\n---\n")

        # Content
        for chapter in structure["chapters"]:
            parts.append(f"\n# {chapter['title']}\n")
            if chapter["summary"]:
                parts.append(f"*{chapter['summary']}*\n")

            for section_meta in chapter["sections"]:
                section = self.get_section_content(section_meta["id"])
                if section:
                    parts.append(f"\n<a id=\"{section.id}\"></a>\n")
                    parts.append(f"## {section.title}\n")
                    parts.append(section.content)
                    parts.append("\n")

        compiled = "\n".join(parts)

        # Update compiled file
        try:
            self.compiled_path.write_text(compiled, encoding="utf-8")
            self._structure["last_compiled"] = datetime.now().isoformat()
            self.save_structure()
            log.info("repository.book_compiled", path=str(self.compiled_path))
        except Exception as e:
            log.error("repository.compile_error", error=str(e))

        return compiled

    def get_structure_summary(self) -> str:
        """
        Get a compact summary of structure for Architect context.

        This should fit in ~2k tokens.

        Returns:
            Markdown summary of document structure
        """
        structure = self.load_structure()
        lines: list[str] = []

        lines.append(f"# {structure['title']} - Structure Overview\n")
        lines.append(f"Version: {structure['version']}\n")

        total_sections = sum(len(ch["sections"]) for ch in structure["chapters"])
        lines.append(f"Total Chapters: {len(structure['chapters'])}, Sections: {total_sections}\n")

        for chapter in structure["chapters"]:
            lines.append(f"\n## {chapter['id']}: {chapter['title']}")
            if chapter["summary"]:
                lines.append(f"   {chapter['summary']}")

            for section in chapter["sections"]:
                concepts_str = ", ".join(section["concepts_defined"]) if section["concepts_defined"] else "—"
                lines.append(f"   - {section['id']}: {section['title']} [defines: {concepts_str}]")
                lines.append(f"     > {section['summary'][:100]}...")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Refactoring Support
    # -------------------------------------------------------------------------

    def needs_refactor(self, growth_threshold: float = 0.20) -> bool:
        """
        Check if document has grown enough to warrant refactoring.

        Args:
            growth_threshold: Percentage growth since last refactor

        Returns:
            True if refactor should be triggered
        """
        # For now, simple heuristic based on section count
        # TODO: Track last refactor state and compare
        structure = self.load_structure()
        total_sections = sum(len(ch["sections"]) for ch in structure["chapters"])

        # Trigger refactor if more than 20 sections and no recent refactor
        return total_sections > 20

    def get_sections_by_status(self, status: str) -> list[SectionMeta]:
        """Get all sections with a given status."""
        structure = self.load_structure()
        result: list[SectionMeta] = []

        for chapter in structure["chapters"]:
            for section in chapter["sections"]:
                if section["status"] == status:
                    result.append(section)

        return result

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _load_json(self, path: Path) -> Optional[dict]:
        """Load JSON file."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("repository.json_load_error", path=str(path), error=str(e))
            return None

    def _save_json(self, path: Path, data: dict):
        """Save JSON file with pretty formatting."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.error("repository.json_save_error", path=str(path), error=str(e))

    def generate_section_id(self, chapter_id: str) -> str:
        """
        Generate a unique section ID for a chapter.

        Args:
            chapter_id: Parent chapter (e.g., "ch_01")

        Returns:
            Section ID (e.g., "sec_01_05")
        """
        chapter = self.get_chapter(chapter_id)
        if chapter is None:
            return f"sec_00_001"

        # Extract chapter number
        ch_num = chapter_id.replace("ch_", "")

        # Find max section number in chapter
        max_num = 0
        for section in chapter["sections"]:
            # Parse section ID like "sec_01_05"
            parts = section["id"].split("_")
            if len(parts) >= 3:
                try:
                    sec_num = int(parts[2])
                    max_num = max(max_num, sec_num)
                except ValueError:
                    pass

        return f"sec_{ch_num}_{max_num + 1:02d}"
