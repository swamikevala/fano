"""QuestionGenerator — generates research questions from insights.

Uses LLM to produce searchable questions grounded in project context.
CRITICAL: No hardcoded domain fallbacks.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from researcher.src.prompts import (
    DIRECTED_QUESTION_GENERATION,
    QUESTION_GENERATION,
    QUESTION_GENERATION_FALLBACK,
)

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import Insight, LLMClientInterface, Project

logger = logging.getLogger(__name__)


def _format_domains(project: Project) -> str:
    """Format research domains for prompt injection."""
    if not project.research_domains:
        return "General research"
    parts = []
    for d in project.research_domains:
        kw = ", ".join(d.keywords) if d.keywords else "general"
        parts.append(f"{d.name} (keywords: {kw})")
    return "; ".join(parts)


class QuestionGenerator:
    """Generates research questions from attested insights or directed requests."""

    def __init__(self, llm_client: LLMClientInterface, config: Config) -> None:
        self._llm = llm_client
        self._config = config

    async def generate(
        self,
        insight: Insight,
        project: Project,
        max_questions: int | None = None,
    ) -> list[str]:
        """Generate research questions for an attested insight."""
        limit = max_questions or self._config.get(
            "researcher.max_questions_per_insight", 10,
        )
        prompt = QUESTION_GENERATION.format(
            project_goal=project.goal,
            project_context=project.context,
            insight_text=insight.text,
            formatted_domains=_format_domains(project),
            max_questions=limit,
        )
        backends = self._llm.get_available_backends()
        backend = backends[0] if backends else "claude"

        resp = await self._llm.send(backend, prompt=prompt)
        if resp.success:
            questions = self._parse_questions(resp.text)
            if questions:
                return questions[:limit]

        # Fallback: use project.goal as generic context (no domain strings)
        fallback_prompt = QUESTION_GENERATION_FALLBACK.format(
            project_goal=project.goal,
            insight_text=insight.text,
            max_questions=limit,
        )
        resp2 = await self._llm.send(backend, prompt=fallback_prompt)
        if resp2.success:
            questions = self._parse_questions(resp2.text)
            return questions[:limit]

        return []

    async def generate_directed(
        self,
        topic: str,
        context: str,
        project: Project,
    ) -> list[str]:
        """Generate questions for a directed research request."""
        limit = self._config.get("researcher.max_questions_per_insight", 10)
        prompt = DIRECTED_QUESTION_GENERATION.format(
            project_goal=project.goal,
            topic=topic,
            context=context,
            formatted_domains=_format_domains(project),
            max_questions=limit,
        )
        backends = self._llm.get_available_backends()
        backend = backends[0] if backends else "claude"

        resp = await self._llm.send(backend, prompt=prompt)
        if resp.success:
            questions = self._parse_questions(resp.text)
            return questions[:limit]
        return []

    @staticmethod
    def _parse_questions(text: str) -> list[str]:
        """Parse JSON array of question strings from LLM response."""
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list):
                return [str(q) for q in parsed if isinstance(q, str)]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse questions from LLM response")
        return []
