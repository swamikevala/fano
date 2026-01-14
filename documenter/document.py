"""
Document handling - loading, saving, parsing, and metadata management.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from shared.logging import get_logger

if TYPE_CHECKING:
    from .versions import VersionManager

log = get_logger("documenter", "document")

# Pattern for annotation markers: <!-- @ann:c001 --> or <!-- @ann:p001 -->
ANNOTATION_MARKER_PATTERN = re.compile(r'<!-- @ann:(\w+) -->')


@dataclass
class Section:
    """A section of the document with metadata."""
    id: str
    content: str
    created: datetime
    last_reviewed: Optional[datetime] = None
    review_count: int = 0
    status: str = "provisional"  # stable | needs_work | provisional
    establishes: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)

    # Position in document (line numbers)
    start_line: int = 0
    end_line: int = 0

    def to_metadata_comment(self) -> str:
        """Generate metadata comment for this section."""
        establishes_str = ", ".join(self.establishes) if self.establishes else ""
        requires_str = ", ".join(self.requires) if self.requires else ""
        reviewed_str = self.last_reviewed.strftime("%Y-%m-%d") if self.last_reviewed else ""

        return f"""<!-- SECTION
id: {self.id}
created: {self.created.strftime("%Y-%m-%d")}
last_reviewed: {reviewed_str}
review_count: {self.review_count}
status: {self.status}
establishes: [{establishes_str}]
requires: [{requires_str}]
-->"""


class Document:
    """
    Manages the living mathematical document.

    Handles loading, saving, parsing sections, and tracking metadata.
    """

    def __init__(self, path: Path, enable_versioning: bool = True):
        """
        Initialize document manager.

        Args:
            path: Path to the markdown document
            enable_versioning: Whether to track version history
        """
        self.path = Path(path)
        self.content = ""
        self.sections: list[Section] = []
        self._summary: str = ""
        self._summary_stale = True
        self._version_manager: Optional["VersionManager"] = None
        self._enable_versioning = enable_versioning

    def load(self) -> bool:
        """Load document from disk."""
        if not self.path.exists():
            log.warning("documenter.document.not_found", path=str(self.path))
            return False

        try:
            self.content = self.path.read_text(encoding="utf-8")
            self._parse_sections()
            self._summary_stale = True
            log.info(
                "documenter.document.loaded",
                path=str(self.path),
                sections=len(self.sections),
                content_length=len(self.content),
            )
            return True
        except Exception as e:
            log.exception(e, "documenter.document.load_error", {"path": str(self.path)})
            return False

    @property
    def version_manager(self) -> "VersionManager":
        """Get or create the version manager."""
        if self._version_manager is None:
            from .versions import VersionManager
            self._version_manager = VersionManager(self.path)
        return self._version_manager

    def save(self, description: str = "") -> bool:
        """
        Save document to disk with optional version tracking.

        Args:
            description: Description of what changed (for version history)
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.content, encoding="utf-8")

            # Save version if enabled and there's a description
            if self._enable_versioning and description:
                self.version_manager.save_version(
                    self.content,
                    description,
                    sections_count=len(self.sections),
                )

            log.info(
                "documenter.document.saved",
                path=str(self.path),
                content_length=len(self.content),
                versioned=bool(description),
            )
            return True
        except Exception as e:
            log.exception(e, "documenter.document.save_error", {"path": str(self.path)})
            return False

    def _parse_sections(self):
        """Parse document into sections based on metadata comments."""
        self.sections = []

        # Pattern to match section metadata
        section_pattern = re.compile(
            r'<!-- SECTION\s*\n(.*?)-->',
            re.DOTALL
        )

        lines = self.content.split('\n')
        current_pos = 0

        for match in section_pattern.finditer(self.content):
            metadata_text = match.group(1)
            section = self._parse_section_metadata(metadata_text)

            if section:
                # Find line numbers
                start_pos = match.start()
                end_pos = match.end()

                # Find the next section or end of document
                next_match = section_pattern.search(self.content, end_pos)
                section_end = next_match.start() if next_match else len(self.content)

                # Extract section content
                section.content = self.content[end_pos:section_end].strip()

                # Calculate line numbers
                section.start_line = self.content[:start_pos].count('\n')
                section.end_line = self.content[:section_end].count('\n')

                self.sections.append(section)

        log.debug(
            "documenter.document.parsed",
            sections_found=len(self.sections),
        )

    def _parse_section_metadata(self, metadata_text: str) -> Optional[Section]:
        """Parse metadata text into a Section object."""
        try:
            # Parse key-value pairs
            data = {}
            for line in metadata_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()

            if 'id' not in data:
                return None

            # Parse lists
            establishes = []
            if data.get('establishes'):
                establishes_str = data['establishes'].strip('[]')
                if establishes_str:
                    establishes = [e.strip() for e in establishes_str.split(',')]

            requires = []
            if data.get('requires'):
                requires_str = data['requires'].strip('[]')
                if requires_str:
                    requires = [r.strip() for r in requires_str.split(',')]

            # Parse dates
            created = datetime.now()
            if data.get('created'):
                try:
                    created = datetime.strptime(data['created'], "%Y-%m-%d")
                except ValueError:
                    pass

            last_reviewed = None
            if data.get('last_reviewed'):
                try:
                    last_reviewed = datetime.strptime(data['last_reviewed'], "%Y-%m-%d")
                except ValueError:
                    pass

            return Section(
                id=data['id'],
                content="",  # Will be filled in later
                created=created,
                last_reviewed=last_reviewed,
                review_count=int(data.get('review_count', 0)),
                status=data.get('status', 'provisional'),
                establishes=establishes,
                requires=requires,
            )
        except Exception as e:
            log.warning("documenter.document.parse_metadata_error", error=str(e))
            return None

    def get_established_concepts(self) -> set[str]:
        """Get all concepts established in the document."""
        concepts = set()
        for section in self.sections:
            concepts.update(section.establishes)
        return concepts

    def get_section_by_id(self, section_id: str) -> Optional[Section]:
        """Get a section by its ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def get_section_establishing(self, concept: str) -> Optional[Section]:
        """Get the section that establishes a concept."""
        for section in self.sections:
            if concept in section.establishes:
                return section
        return None

    def find_unresolved_comments(self) -> list[tuple[str, int]]:
        """Find all unresolved author comments."""
        comments = []
        pattern = re.compile(r'<!-- COMMENT:\s*(.+?)\s*-->')

        for match in pattern.finditer(self.content):
            comment_text = match.group(1)
            # Check if already attempted
            if 'attempted: true' not in comment_text.lower():
                line_num = self.content[:match.start()].count('\n') + 1
                comments.append((comment_text, line_num))

        return comments

    def find_human_review_needed(self) -> list[str]:
        """Find items flagged for human review."""
        items = []
        pattern = re.compile(r'<!-- NEEDS_HUMAN_REVIEW:\s*(.+?)\s*-->')

        for match in pattern.finditer(self.content):
            items.append(match.group(1))

        return items

    def append_section(self, section: Section, content: str):
        """Append a new section to the document."""
        # Generate metadata comment
        metadata = section.to_metadata_comment()

        # Add to document
        addition = f"\n\n{metadata}\n\n{content}"
        self.content += addition

        # Update section info
        section.content = content
        section.start_line = self.content[:len(self.content) - len(addition)].count('\n')
        section.end_line = self.content.count('\n')
        self.sections.append(section)

        self._summary_stale = True
        log.info(
            "documenter.document.section_added",
            section_id=section.id,
            establishes=section.establishes,
        )

    def update_section(self, section_id: str, new_content: str):
        """Update an existing section's content."""
        section = self.get_section_by_id(section_id)
        if not section:
            log.warning("documenter.document.section_not_found", section_id=section_id)
            return

        # Find section by its metadata comment and replace content following it
        # This is more reliable than simple string replacement which could match
        # duplicated content in the wrong location
        pattern = re.compile(
            rf'(<!-- SECTION\s*\nid: {re.escape(section.id)}\n.*?-->\s*\n\n)'
            rf'(.*?)(?=<!-- SECTION|\Z)',
            re.DOTALL
        )

        match = pattern.search(self.content)
        if match:
            # Replace content after metadata comment
            start = match.start(2)
            end = match.end(2)
            self.content = self.content[:start] + new_content + self.content[end:]
            section.content = new_content
            self._summary_stale = True
            log.info("documenter.document.section_updated", section_id=section_id)
        else:
            # Fallback: try simple replacement if regex doesn't match
            # (handles legacy format without metadata comments)
            old_content = section.content
            if old_content in self.content:
                self.content = self.content.replace(old_content, new_content, 1)
                section.content = new_content
                self._summary_stale = True
                log.info("documenter.document.section_updated_fallback", section_id=section_id)
            else:
                log.warning(
                    "documenter.document.section_update_failed",
                    section_id=section_id,
                    reason="content_not_found",
                )

    def mark_section_reviewed(self, section_id: str):
        """Mark a section as reviewed."""
        section = self.get_section_by_id(section_id)
        if section:
            section.last_reviewed = datetime.now()
            section.review_count += 1
            # Update metadata in document
            self._update_section_metadata(section)
            log.info(
                "documenter.document.section_reviewed",
                section_id=section_id,
                review_count=section.review_count,
            )

    def _update_section_metadata(self, section: Section):
        """Update section metadata in the document."""
        # Find existing metadata comment
        pattern = re.compile(
            rf'<!-- SECTION\s*\nid: {re.escape(section.id)}\n.*?-->',
            re.DOTALL
        )
        new_metadata = section.to_metadata_comment()
        self.content = pattern.sub(new_metadata, self.content, count=1)

    def get_summary(self, max_tokens: int = 500) -> str:
        """Get a summary of the document for context."""
        if not self._summary_stale and self._summary:
            return self._summary

        # Build summary from sections
        parts = []
        parts.append(f"Document: {self.path.name}")
        parts.append(f"Sections: {len(self.sections)}")

        concepts = self.get_established_concepts()
        if concepts:
            parts.append(f"Established concepts: {', '.join(sorted(concepts))}")

        # Show ALL existing sections so LLMs know what content already exists
        if self.sections:
            parts.append("\nEXISTING SECTIONS (do not duplicate this content):")
            for section in self.sections:
                # Show section ID, what it establishes, and a content preview
                establishes_str = f" [establishes: {', '.join(section.establishes)}]" if section.establishes else ""
                preview = section.content[:200].replace('\n', ' ')
                if len(section.content) > 200:
                    preview += "..."
                parts.append(f"  - {section.id}{establishes_str}: {preview}")

        self._summary = "\n".join(parts)
        self._summary_stale = False
        return self._summary

    def generate_next_section_id(self) -> str:
        """Generate a unique section ID."""
        existing_ids = {s.id for s in self.sections}
        counter = 1
        while True:
            new_id = f"section_{counter:03d}"
            if new_id not in existing_ids:
                return new_id
            counter += 1

    # -------------------------------------------------------------------------
    # Annotation Marker Methods
    # -------------------------------------------------------------------------

    def insert_marker(self, marker_id: str, char_offset: int, search_text: str = "") -> bool:
        """
        Insert an annotation marker at the specified position.

        If search_text is provided, finds that text in the document and inserts
        the marker at the start of the match. This is more reliable than using
        char_offset directly since the offset from rendered HTML doesn't match
        the raw markdown positions.

        Args:
            marker_id: The annotation ID (e.g., "c001" or "p001")
            char_offset: Fallback character position (used if search_text not found)
            search_text: Text to search for - marker inserted at start of match

        Returns:
            True if successful
        """
        marker = f"<!-- @ann:{marker_id} -->"

        insert_pos = None

        # Try to find the search text in the document
        if search_text:
            # Normalize for searching - handle whitespace differences
            normalized_search = ' '.join(search_text.split())

            # Try exact match first
            pos = self.content.find(search_text)
            if pos >= 0:
                insert_pos = pos
                log.debug("documenter.document.marker_exact_match", marker_id=marker_id, pos=pos)
            else:
                # Try normalized search (collapse whitespace)
                # Build a mapping from normalized positions back to original
                normalized_content = ' '.join(self.content.split())
                norm_pos = normalized_content.find(normalized_search)

                if norm_pos >= 0:
                    # Map normalized position back to original content
                    # Count characters up to that position accounting for whitespace
                    insert_pos = self._map_normalized_pos_to_original(norm_pos, normalized_search)
                    log.debug("documenter.document.marker_normalized_match", marker_id=marker_id, pos=insert_pos)

                # Try finding a shorter unique prefix
                if insert_pos is None and len(normalized_search) > 20:
                    # Try progressively shorter prefixes
                    for length in [50, 40, 30, 20, 15, 10]:
                        if length >= len(normalized_search):
                            continue
                        prefix = normalized_search[:length]
                        pos = self.content.find(prefix)
                        if pos >= 0:
                            insert_pos = pos
                            log.debug("documenter.document.marker_prefix_match",
                                     marker_id=marker_id, pos=pos, prefix_len=length)
                            break

        # Fall back to char_offset if search failed
        if insert_pos is None:
            insert_pos = max(0, min(char_offset, len(self.content)))
            log.debug("documenter.document.marker_fallback_offset", marker_id=marker_id, pos=insert_pos)

        self.content = self.content[:insert_pos] + marker + self.content[insert_pos:]
        self._summary_stale = True

        log.info(
            "documenter.document.marker_inserted",
            marker_id=marker_id,
            position=insert_pos,
            used_search=search_text != "" and insert_pos != char_offset,
        )
        return True

    def _map_normalized_pos_to_original(self, norm_pos: int, search_text: str) -> Optional[int]:
        """Map a position in normalized (whitespace-collapsed) content back to original."""
        # Walk through original content, tracking normalized position
        orig_pos = 0
        norm_count = 0
        in_whitespace = False

        for i, ch in enumerate(self.content):
            if ch in ' \t\n\r':
                if not in_whitespace:
                    in_whitespace = True
                    if norm_count >= norm_pos:
                        return i
                    norm_count += 1  # Single space in normalized
            else:
                in_whitespace = False
                if norm_count >= norm_pos:
                    return i
                norm_count += 1

        return None

    def remove_marker(self, marker_id: str) -> bool:
        """
        Remove an annotation marker from the document.

        Args:
            marker_id: The annotation ID to remove

        Returns:
            True if marker was found and removed, False if not found
        """
        marker = f"<!-- @ann:{marker_id} -->"
        if marker in self.content:
            self.content = self.content.replace(marker, "", 1)
            self._summary_stale = True
            log.info("documenter.document.marker_removed", marker_id=marker_id)
            return True
        else:
            log.warning("documenter.document.marker_not_found", marker_id=marker_id)
            return False

    def find_marker_position(self, marker_id: str) -> Optional[int]:
        """
        Find the character position of an annotation marker.

        Args:
            marker_id: The annotation ID to find

        Returns:
            Character position of the marker, or None if not found
        """
        marker = f"<!-- @ann:{marker_id} -->"
        pos = self.content.find(marker)
        return pos if pos >= 0 else None

    def find_all_markers(self) -> list[str]:
        """
        Find all annotation marker IDs in the document.

        Returns:
            List of marker IDs found in document order
        """
        return ANNOTATION_MARKER_PATTERN.findall(self.content)

    def get_content_without_markers(self) -> str:
        """
        Get document content with all annotation markers removed.

        Useful for rendering where markers should become anchor elements.

        Returns:
            Document content with markers stripped
        """
        return ANNOTATION_MARKER_PATTERN.sub("", self.content)
