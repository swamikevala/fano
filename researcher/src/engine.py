"""ResearcherEngine — ModuleInterface facade for the research pipeline.

Orchestrates: question generation -> search -> trust evaluation ->
finding extraction -> evidence linking.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from shared.models import (
    Event,
    Finding,
    HealthStatus,
    ModuleInterface,
    Source,
    SearchResult,
    content_hash,
    generate_id,
)
from researcher.src.extractor import FindingExtractor
from researcher.src.questions import QuestionGenerator
from researcher.src.searcher import SearchExecutor
from researcher.src.trust import TrustEvaluator

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import (
        ConsensusEngineInterface,
        EventBusInterface,
        Insight,
        LLMClientInterface,
        StateStoreInterface,
    )

logger = logging.getLogger(__name__)


class ResearcherEngine(ModuleInterface):
    """Facade for the Researcher module. Implements ModuleInterface."""

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface,
        config: Config,
    ) -> None:
        self._store = store
        self._bus = event_bus
        self._llm = llm_client
        self._config = config
        self._running = False

        self._question_gen = QuestionGenerator(llm_client, config)
        self._searcher = SearchExecutor(llm_client, config)
        self._trust_evaluator = TrustEvaluator(consensus, store, config)
        self._extractor = FindingExtractor(llm_client, config)

        self._insight_queue: list[Insight] = []
        self._directed_queue: list[dict[str, Any]] = []

    @property
    def module_name(self) -> str:
        return "researcher"

    async def initialize(self) -> bool:
        """Subscribe to events."""
        self._bus.subscribe(
            "explorer.insight.attested", self._on_insight_attested,
        )
        self._bus.subscribe(
            "documenter.research.requested", self._on_research_request,
        )
        self._bus.subscribe(
            "user.research.requested", self._on_user_request,
        )
        logger.info("ResearcherEngine initialized")
        return True

    async def start(self) -> None:
        """Main research loop."""
        self._running = True
        interval = self._config.get(
            "researcher.idle_polling_interval_seconds", 300,
        )
        while self._running:
            work = self._get_next_work_item()
            if work is None:
                await asyncio.sleep(interval)
                continue
            try:
                await self._process_work_item(work)
            except Exception:
                logger.exception("Error processing work item")
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._running = False
        logger.info("ResearcherEngine stopped")

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            module="researcher",
            healthy=True,
            message="ok",
            details={
                "insight_queue": len(self._insight_queue),
                "directed_queue": len(self._directed_queue),
            },
        )

    # ── Work scheduling ───────────────────────────────────────

    def _get_next_work_item(self) -> dict[str, Any] | None:
        """Get next item. Directed requests have priority."""
        if self._directed_queue:
            item = self._directed_queue.pop(0)
            return {"type": "directed", **item}
        if self._insight_queue:
            insight = self._insight_queue.pop(0)
            return {"type": "insight", "insight": insight}
        return None

    async def _process_work_item(self, work: dict[str, Any]) -> None:
        """Route work to appropriate handler."""
        if work["type"] == "directed":
            await self._process_directed(work)
        else:
            await self._process_insight(work["insight"])

    # ── Event handlers ────────────────────────────────────────

    async def _on_insight_attested(self, event: Event) -> None:
        insight_id = event.payload.get("insight_id")
        if insight_id:
            insight = await self._store.get_insight(insight_id)
            if insight:
                self._insight_queue.append(insight)

    async def _on_research_request(self, event: Event) -> None:
        self._directed_queue.append({
            "topic": event.payload.get("topic", ""),
            "context": event.payload.get("context", ""),
            "project_id": event.payload.get("project_id", ""),
        })

    async def _on_user_request(self, event: Event) -> None:
        self._directed_queue.append({
            "topic": event.payload.get("topic", ""),
            "context": event.payload.get("context", ""),
            "project_id": event.payload.get("project_id", ""),
        })

    # ── Core pipeline ─────────────────────────────────────────

    async def _process_insight(self, insight: Insight) -> None:
        project = await self._store.get_project(insight.project_id)
        if not project:
            return
        questions = await self._question_gen.generate(insight, project)
        all_findings: list[Finding] = []
        for question in questions:
            results = await self._searcher.search(
                question, project.research_domains,
            )
            for sr in results:
                findings = await self._evaluate_and_extract(
                    sr, insight, project,
                )
                all_findings.extend(findings)
        if all_findings:
            await self._publish_evidence(all_findings, insight)

    async def _process_directed(self, work: dict[str, Any]) -> None:
        project = await self._store.get_project(work.get("project_id", ""))
        if not project:
            return
        questions = await self._question_gen.generate_directed(
            work["topic"], work["context"], project,
        )
        for question in questions:
            await self._searcher.search(question, project.research_domains)

    async def _evaluate_and_extract(
        self, sr: SearchResult, insight: Insight, project: Project,
    ) -> list[Finding]:
        """Evaluate trust, then extract findings if trusted."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=generate_id(),
            project_id=project.id,
            url=sr.url,
            domain=sr.domain,
            title=sr.title,
            trust_score=0,
            trust_tier=None,
            content_hash=content_hash(sr.snippet),
            evaluated_at=None,
            created_at=now,
        )
        min_trust = self._config.get("researcher.trust.min_trust_score", 50)
        score = await self._trust_evaluator.evaluate(
            source, sr.snippet, project,
        )
        if score < min_trust:
            logger.info("Source skipped (trust %d < %d): %s", score, min_trust, sr.url)
            return []
        return await self._extractor.extract(
            source, sr.snippet, insight, project,
        )

    async def _publish_evidence(
        self, findings: list[Finding], insight: Insight,
    ) -> None:
        """Analyze findings and publish evidence events."""
        supports = sum(1 for f in findings if f.finding_type == "supports")
        refutes = sum(1 for f in findings if f.finding_type == "refutes")
        now = datetime.now(timezone.utc)

        # Publish per-finding events
        for f in findings:
            await self._bus.publish(Event(
                topic="researcher.finding.stored",
                timestamp=now,
                source="researcher",
                payload={
                    "finding_id": f.id,
                    "insight_id": insight.id,
                    "type": f.finding_type,
                },
                correlation_id=generate_id(),
            ))

        # Publish aggregate evidence event
        if refutes > supports:
            topic = "researcher.evidence.contradicts"
        else:
            topic = "researcher.evidence.supports"

        await self._bus.publish(Event(
            topic=topic,
            timestamp=now,
            source="researcher",
            payload={
                "insight_id": insight.id,
                "supports_count": supports,
                "refutes_count": refutes,
                "total_findings": len(findings),
            },
            correlation_id=generate_id(),
        ))
