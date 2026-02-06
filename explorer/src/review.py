"""ReviewPanel — consensus-based review of extracted insights."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from shared.errors import ConsensusError
from shared.logging import get_logger
from shared.models import (
    ConsensusEngineInterface,
    ConsensusTaskInterface,
    EvaluationCriterion,
    Event,
    EventBusInterface,
    Insight,
    InsightStatus,
    ParsedResponse,
    Project,
    RoundResult,
    StateStoreInterface,
    Verdict,
)
from llm.src.consensus.parsing import parse_review_response

log = get_logger("explorer", "review_panel")


class InsightReviewTask(ConsensusTaskInterface):
    """Consensus task for reviewing an insight against project criteria."""

    def __init__(self, insight: Insight, project: Project) -> None:
        self._insight = insight
        self._project = project

    def get_prompt(
        self, round_num: int, prior_rounds: list[RoundResult], backend: str,
    ) -> str:
        """Build review prompt."""
        criteria_text = "\n".join(
            f"- {c.name} (weight {c.weight}): {c.description}"
            for c in self._project.evaluation_criteria
        )
        prompt = (
            "You are reviewing a proposed insight for a research project.\n\n"
            f"PROJECT GOAL: {self._project.goal}\n"
            f"PROJECT CONTEXT: {self._project.context}\n"
            f"EVALUATION CRITERIA:\n{criteria_text}\n"
            f"EXPLORATION GUIDANCE: {self._project.exploration_guidance}\n\n"
            f"The insight to review:\n{self._insight.text}\n\n"
            "Rate on each criterion (1-10) with reasoning.\n"
            "Give overall verdict: accept / reject / uncertain.\n"
            'Respond in JSON: {"verdict": ..., "scores": {...}, "reasoning": ...}'
        )
        if round_num > 1 and prior_rounds:
            prior_text = self._format_prior_rounds(prior_rounds)
            prompt += (
                f"\n\nPrevious round results:\n{prior_text}\n"
                "Reconsider your position in light of other reviewers' reasoning."
            )
        return prompt

    def parse_response(self, text: str) -> ParsedResponse:
        """Delegate to shared parsing."""
        criteria_names = [c.name for c in self._project.evaluation_criteria]
        return parse_review_response(text, criteria_names)

    def get_evaluation_criteria(self) -> list[EvaluationCriterion]:
        """Return project evaluation criteria."""
        return list(self._project.evaluation_criteria)

    @staticmethod
    def _format_prior_rounds(rounds: list[RoundResult]) -> str:
        parts = []
        for rr in rounds:
            for resp in rr.responses:
                parts.append(
                    f"[{resp.backend}] verdict={resp.verdict.value}, "
                    f"reasoning={resp.reasoning[:200]}"
                )
        return "\n".join(parts)


class ReviewPanel:
    """Runs consensus-based review of extracted insights."""

    def __init__(
        self,
        consensus: ConsensusEngineInterface,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
    ) -> None:
        self._consensus = consensus
        self._store = store
        self._bus = event_bus

    async def review(self, insight: Insight, project: Project) -> InsightStatus:
        """Review an insight via multi-round consensus."""
        await self._store.update_insight(insight.id, status=InsightStatus.REVIEWING)
        task = InsightReviewTask(insight, project)

        try:
            result = await self._consensus.run(task)
        except ConsensusError as exc:
            log.error("explorer.review.consensus_failed",
                      insight_id=insight.id, error=str(exc))
            await self._store.update_insight(
                insight.id, status=InsightStatus.TRANSIENT_FAILURE,
                transient_failure_count=insight.transient_failure_count + 1,
            )
            return InsightStatus.TRANSIENT_FAILURE

        # Map verdict to status
        if result.verdict == Verdict.ACCEPT:
            status = InsightStatus.ATTESTED if result.confidence >= 0.8 else InsightStatus.INTERESTING
        elif result.verdict == Verdict.REJECT:
            status = InsightStatus.DISCARDED
        else:
            status = InsightStatus.DISPUTED

        now = datetime.now(timezone.utc)
        update_fields: dict = {
            "status": status,
            "evaluation_scores": result.scores,
            "review_record": {
                "rounds": result.rounds_completed,
                "confidence": result.confidence,
                "verdict": result.verdict.value,
            },
        }
        if status == InsightStatus.ATTESTED:
            update_fields["blessed_at"] = now

        await self._store.update_insight(insight.id, **update_fields)

        event_suffix = {
            InsightStatus.ATTESTED: "attested",
            InsightStatus.INTERESTING: "interesting",
            InsightStatus.DISCARDED: "discarded",
            InsightStatus.DISPUTED: "disputed",
        }.get(status, "reviewed")

        await self._bus.publish(Event(
            topic=f"explorer.insight.{event_suffix}",
            timestamp=now, source="explorer.review_panel",
            payload={"insight_id": insight.id, "verdict": result.verdict.value},
            correlation_id=str(uuid.uuid4()),
        ))

        return status
