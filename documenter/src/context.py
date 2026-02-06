"""ContextBuilder — builds focused context for LLM prompts.

Also defines canonicalize() for concept name normalization.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import Annotation, Insight, Project, Section, StateStoreInterface


_ARTICLES = {"the", "a", "an"}


def canonicalize(concept_name: str) -> str:
    """Normalize concept names for reliable matching.

    Steps:
        1. Strip whitespace, lowercase
        2. Remove leading articles (the, a, an)
        3. Replace non-alphanumeric with underscore
        4. Collapse multiple underscores, strip leading/trailing
    """
    text = concept_name.strip().lower()
    words = text.split()
    if words and words[0] in _ARTICLES:
        words = words[1:]
    text = " ".join(words)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


class ContextBuilder:
    """Builds focused context windows for LLM prompts."""

    def __init__(self, store: StateStoreInterface, config: Config) -> None:
        self._store = store
        self._config = config

    def _max_tokens(self) -> int:
        return self._config.get("documenter.context.max_tokens", 8000)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count. Fallback: len(text) // 4."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    async def build_for_insight(
        self,
        insight: Insight,
        project: Project,
        max_tokens: int | None = None,
    ) -> str:
        """Build focused context relevant to an insight being incorporated.

        Includes: goal, recent sections, dependency sections, tag-related sections.
        Deduplicates. Truncates to token budget.
        """
        budget = max_tokens or self._max_tokens()
        parts: list[str] = []
        seen_ids: set[str] = set()

        # 1. Research goal (always)
        parts.append(f"Research Goal: {project.goal}")

        # 2. Recent sections
        recent = await self._store.get_recent_sections(project.id, limit=2)
        for sec in recent:
            if sec.id not in seen_ids:
                seen_ids.add(sec.id)
                parts.append(self._format_section(sec))

        # 3. Sections establishing concepts this insight depends on
        if insight.tags:
            canonical_tags = [canonicalize(t) for t in insight.tags]
            dep_sections = await self._store.get_sections_establishing(canonical_tags)
            for sec in dep_sections:
                if sec.id not in seen_ids:
                    seen_ids.add(sec.id)
                    parts.append(self._format_section(sec))

        # 4. Sections with overlapping tags
        if insight.tags:
            tag_sections = await self._store.get_sections_by_tags(insight.tags, limit=3)
            for sec in tag_sections:
                if sec.id not in seen_ids:
                    seen_ids.add(sec.id)
                    parts.append(self._format_section(sec))

        return self._truncate("\n\n".join(parts), budget)

    async def build_for_annotation(
        self,
        annotation: Annotation,
        project: Project,
    ) -> str:
        """Build context for addressing an annotation.

        Includes: goal, annotated section, adjacent sections, annotation content.
        """
        parts: list[str] = [f"Research Goal: {project.goal}"]

        if annotation.section_id:
            section = await self._store.get_section(annotation.section_id)
            if section:
                parts.append(self._format_section(section))
                # Get adjacent sections
                all_sections = await self._store.list_sections(project.id)
                ordered = sorted(all_sections, key=lambda s: s.order_index)
                for i, s in enumerate(ordered):
                    if s.id == section.id:
                        if i > 0:
                            parts.append(self._format_section(ordered[i - 1]))
                        if i < len(ordered) - 1:
                            parts.append(self._format_section(ordered[i + 1]))
                        break

        parts.append(f"Annotation: {annotation.content}")
        budget = self._max_tokens()
        return self._truncate("\n\n".join(parts), budget)

    def _format_section(self, section: Section) -> str:
        return f"### {section.title}\n{section.content}"

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        current = self.estimate_tokens(text)
        if current <= max_tokens:
            return text
        # Binary-search-style truncation by character ratio
        ratio = max_tokens / max(current, 1)
        target_chars = int(len(text) * ratio * 0.95)  # 5% safety margin
        return text[:target_chars]
