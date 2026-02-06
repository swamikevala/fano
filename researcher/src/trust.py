"""TrustEvaluator — evaluates source trustworthiness via consensus.

Uses ConsensusEngine for multi-LLM evaluation. Caches results by
URL + content_hash to avoid redundant re-evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from shared.models import (
    ConsensusTaskInterface,
    EvaluationCriterion,
    ParsedResponse,
    Verdict,
    content_hash as compute_hash,
)
from researcher.src.prompts import TRUST_EVALUATION

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import (
        ConsensusEngineInterface,
        Project,
        RoundResult,
        Source,
        StateStoreInterface,
    )

logger = logging.getLogger(__name__)


def _trust_tier(score: int) -> str:
    if score >= 80:
        return "authoritative"
    if score >= 60:
        return "reliable"
    if score >= 40:
        return "uncertain"
    return "unreliable"


class TrustEvaluationTask(ConsensusTaskInterface):
    """Consensus task for evaluating source trustworthiness."""

    def __init__(
        self, source: Source, content_summary: str, project: Project,
    ) -> None:
        self._source = source
        self._content_summary = content_summary
        self._project = project

    def get_prompt(
        self,
        round_num: int,
        prior_rounds: list[RoundResult],
        backend: str,
    ) -> str:
        return TRUST_EVALUATION.format(
            project_goal=self._project.goal,
            source_url=self._source.url,
            source_title=self._source.title or "Unknown",
            content_summary=self._content_summary,
        )

    def parse_response(self, text: str) -> ParsedResponse:
        try:
            data = json.loads(text.strip())
            score = int(data.get("score", 0))
            verdict_str = data.get("verdict", "uncertain").lower()
            verdict_map = {
                "accept": Verdict.ACCEPT,
                "reject": Verdict.REJECT,
                "uncertain": Verdict.UNCERTAIN,
            }
            verdict = verdict_map.get(verdict_str, Verdict.UNCERTAIN)
            return ParsedResponse(
                is_valid=True,
                verdict=verdict,
                scores={"trust": score / 100},
                reasoning=data.get("reasoning"),
                error=None,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return ParsedResponse(
                is_valid=False, verdict=None, scores={},
                reasoning=None, error=str(exc),
            )

    def get_evaluation_criteria(self) -> list[EvaluationCriterion]:
        return [
            EvaluationCriterion("authority", "Source credibility", 0.3),
            EvaluationCriterion("accuracy", "Factual soundness", 0.3),
            EvaluationCriterion("relevance", "Research relevance", 0.25),
            EvaluationCriterion("recency", "Information currency", 0.15),
        ]


class TrustEvaluator:
    """Evaluates source trustworthiness using multi-LLM consensus."""

    def __init__(
        self,
        consensus: ConsensusEngineInterface,
        store: StateStoreInterface,
        config: Config,
    ) -> None:
        self._consensus = consensus
        self._store = store
        self._config = config
        # Cache: (url, content_hash) -> trust_score
        self._cache: dict[tuple[str, str | None], int] = {}

    async def evaluate(
        self,
        source: Source,
        content_summary: str,
        project: Project,
    ) -> int:
        """Evaluate source trust. Returns score 0-100."""
        cache_key = (source.url, source.content_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]

        task = TrustEvaluationTask(source, content_summary, project)
        result = await self._consensus.run(task)

        # Convert consensus confidence to 0-100 score
        score = int(result.confidence * 100)
        tier = _trust_tier(score)
        self._cache[cache_key] = score

        logger.info(
            "Trust evaluated: url=%s score=%d tier=%s",
            source.url, score, tier,
        )
        return score
