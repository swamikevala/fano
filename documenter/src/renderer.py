"""Renderer — generates markdown document from database state.

Annotations are NOT baked into the markdown. They are displayed
by the control panel at view time.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import Project, StateStoreInterface


class Renderer:
    """Renders the document as markdown from sections in the store."""

    def __init__(self, store: StateStoreInterface, config: Config) -> None:
        self._store = store
        self._config = config

    async def render(self, project: Project) -> str:
        """Render the complete document as markdown.

        1. Load all sections ordered by order_index
        2. Generate header from project name and goal
        3. For each section: render content
        4. Annotations are NOT inlined
        """
        sections = await self._store.list_sections(project.id)
        sections = sorted(sections, key=lambda s: s.order_index)

        parts: list[str] = []

        # Document header
        parts.append(f"# {project.name}")
        parts.append("")
        parts.append(f"> {project.goal}")
        parts.append("")

        # Sections
        for section in sections:
            parts.append(f"## {section.title}")
            parts.append("")
            parts.append(section.content)
            parts.append("")

        return "\n".join(parts)

    async def save(self, project: Project, output_dir: Path) -> None:
        """Render and write to output_dir/main.md.

        Also archives to output_dir/archive/ with timestamp if content changed.
        """
        md = await self.render(project)
        output_dir.mkdir(parents=True, exist_ok=True)
        main_path = output_dir / "main.md"

        # Check if content changed
        existing_hash = ""
        if main_path.exists():
            existing_hash = hashlib.sha256(
                main_path.read_text(encoding="utf-8").encode()
            ).hexdigest()

        new_hash = hashlib.sha256(md.encode()).hexdigest()

        main_path.write_text(md, encoding="utf-8")

        if new_hash != existing_hash:
            archive_dir = output_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_path = archive_dir / f"document_{ts}.md"
            archive_path.write_text(md, encoding="utf-8")
