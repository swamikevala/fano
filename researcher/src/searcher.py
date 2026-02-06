"""SearchExecutor — executes searches for research questions.

Initial implementation uses LLM-generated simulated search results.
Full web search is Phase 6+.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from shared.models import SearchResult
from researcher.src.prompts import SEARCH_SIMULATION

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import LLMClientInterface, ResearchDomain

logger = logging.getLogger(__name__)


def _format_domains(domains: list[ResearchDomain]) -> str:
    if not domains:
        return "General"
    parts = []
    for d in domains:
        kw = ", ".join(d.keywords) if d.keywords else "general"
        parts.append(f"{d.name} ({kw})")
    return "; ".join(parts)


class SearchExecutor:
    """Executes web searches for research questions.

    The initial implementation uses LLM-generated simulated results.
    """

    def __init__(self, llm_client: LLMClientInterface, config: Config) -> None:
        self._llm = llm_client
        self._config = config

    async def search(
        self,
        question: str,
        domains: list[ResearchDomain],
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """Execute a search for a research question."""
        limit = max_results or self._config.get(
            "researcher.max_searches_per_question", 5,
        )
        prompt = SEARCH_SIMULATION.format(
            question=question,
            formatted_domains=_format_domains(domains),
            max_results=limit,
        )
        backends = self._llm.get_available_backends()
        backend = backends[0] if backends else "claude"

        resp = await self._llm.send(backend, prompt=prompt)
        if not resp.success:
            logger.warning("Search simulation failed: %s", resp.error)
            return []

        results = self._parse_results(resp.text)
        return results[:limit]

    @staticmethod
    def _parse_results(text: str) -> list[SearchResult]:
        """Parse JSON array of search result objects from LLM response."""
        try:
            parsed = json.loads(text.strip())
            if not isinstance(parsed, list):
                return []
            results = []
            for item in parsed:
                if isinstance(item, dict) and "url" in item and "title" in item:
                    results.append(SearchResult(
                        url=item["url"],
                        title=item["title"],
                        snippet=item.get("snippet", ""),
                        domain=item.get("domain"),
                    ))
            return results
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse search results from LLM response")
            return []
