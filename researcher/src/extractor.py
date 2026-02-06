"""FindingExtractor — LLM-based finding extraction from source content.

Extracts structured findings linked to both source and insight.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from shared.models import Finding, generate_id
from researcher.src.prompts import FINDING_EXTRACTION

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import Insight, LLMClientInterface, Project, Source

logger = logging.getLogger(__name__)


class FindingExtractor:
    """Extracts structured findings from source content using LLM."""

    def __init__(self, llm_client: LLMClientInterface, config: Config) -> None:
        self._llm = llm_client
        self._config = config

    async def extract(
        self,
        source: Source,
        content: str,
        insight: Insight,
        project: Project,
    ) -> list[Finding]:
        """Extract findings from source content, linked to insight."""
        max_findings = self._config.get("researcher.max_findings_per_source", 20)
        prompt = FINDING_EXTRACTION.format(
            project_goal=project.goal,
            insight_text=insight.text,
            source_title=source.title or "Unknown",
            content=content[:4000],  # Truncate to avoid token limits
            max_findings=max_findings,
        )
        backends = self._llm.get_available_backends()
        backend = backends[0] if backends else "claude"

        resp = await self._llm.send(backend, prompt=prompt)
        if not resp.success:
            logger.warning("Finding extraction failed: %s", resp.error)
            return []

        raw_findings = self._parse_findings(resp.text)
        now = datetime.now(timezone.utc)

        findings: list[Finding] = []
        for item in raw_findings[:max_findings]:
            findings.append(Finding(
                id=generate_id(),
                project_id=project.id,
                source_id=source.id,
                finding_type=item.get("type"),
                summary=item.get("summary", ""),
                confidence=float(item.get("confidence", 0.5)),
                domain=source.domain,
                related_insight_id=insight.id,
                created_at=now,
            ))
        return findings

    @staticmethod
    def _parse_findings(text: str) -> list[dict]:
        """Parse JSON array of finding objects from LLM response."""
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list):
                return [
                    f for f in parsed
                    if isinstance(f, dict) and "summary" in f
                ]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse findings from LLM response")
        return []
