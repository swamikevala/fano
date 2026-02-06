"""InsightExtractor — extracts atomic insights from thread exchanges."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from shared.config import Config
from shared.logging import get_logger
from shared.models import (
    Confidence,
    Event,
    EventBusInterface,
    Insight,
    InsightStatus,
    LLMClientInterface,
    Project,
    StateStoreInterface,
    Thread,
    content_hash,
    generate_id,
)
from explorer.src.prompts import build_extraction_prompt

log = get_logger("explorer", "insight_extractor")

_INSIGHT_PATTERN = re.compile(
    r"INSIGHT\s+\d+\s*:\s*(.+?)(?:\nCONFIDENCE:\s*(high|medium|low))?(?:\nTAGS:\s*(.+?))?(?=\nINSIGHT\s+\d+\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_CONFIDENCE_MAP = {
    "high": Confidence.HIGH,
    "medium": Confidence.MEDIUM,
    "low": Confidence.LOW,
}


class InsightExtractor:
    """Extracts atomic insights from thread exchanges and deduplicates them."""

    def __init__(
        self,
        store: StateStoreInterface,
        llm_client: LLMClientInterface,
        event_bus: EventBusInterface,
        config: Config,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._bus = event_bus
        self._config = config

    async def extract(self, thread: Thread, project: Project) -> list[Insight]:
        """Extract atomic insights from thread exchanges."""
        exchanges = await self._store.get_exchanges(thread.id)
        if not exchanges:
            return []

        prompt = build_extraction_prompt(project, exchanges)
        response = await self._llm.send("gemini", prompt, module="explorer")

        raw_insights = self._parse_insights(response.text)
        if not raw_insights:
            # Fallback: split on double-newline
            raw_insights = self._fallback_parse(response.text)

        results: list[Insight] = []
        now = datetime.now(timezone.utc)

        for text, confidence, tags in raw_insights:
            text = text.strip()
            if not text:
                continue

            if await self._is_duplicate(text, project.id, thread.id):
                log.info("explorer.insight.duplicate_skipped",
                         thread_id=thread.id, text_preview=text[:50])
                continue

            insight = Insight(
                id=generate_id(),
                project_id=project.id,
                text=text,
                confidence=confidence,
                tags=tags,
                source_thread_id=thread.id,
                extraction_model="gemini",
                status=InsightStatus.EXTRACTED,
                evaluation_scores={},
                dispute_count=0,
                transient_failure_count=0,
                review_record=None,
                blessed_at=None,
                incorporated_at=None,
                incorporated_in_section=None,
                created_at=now,
                updated_at=now,
            )
            await self._store.create_insight(insight)
            await self._bus.publish(Event(
                topic="explorer.insight.extracted",
                timestamp=now, source="explorer.insight_extractor",
                payload={"insight_id": insight.id, "thread_id": thread.id},
                correlation_id=str(uuid.uuid4()),
            ))
            results.append(insight)

        return results

    def _parse_insights(self, text: str) -> list[tuple[str, Confidence, list[str]]]:
        """Parse structured INSIGHT N: format."""
        results = []
        for match in _INSIGHT_PATTERN.finditer(text):
            insight_text = match.group(1).strip()
            conf_str = (match.group(2) or "low").strip().lower()
            tags_str = (match.group(3) or "").strip()
            confidence = _CONFIDENCE_MAP.get(conf_str, Confidence.LOW)
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            results.append((insight_text, confidence, tags))
        return results

    def _fallback_parse(self, text: str) -> list[tuple[str, Confidence, list[str]]]:
        """Fallback: split on double-newline, each block is low-confidence."""
        blocks = text.strip().split("\n\n")
        return [(b.strip(), Confidence.LOW, []) for b in blocks if b.strip()]

    async def _is_duplicate(self, text: str, project_id: str, thread_id: str) -> bool:
        """Check if insight text is a duplicate by content hash."""
        h = content_hash(text)
        # Check thread-level duplicates
        thread_insights = await self._store.get_insights_for_thread(thread_id)
        for existing in thread_insights:
            if content_hash(existing.text) == h:
                return True
        # Check project-level duplicates
        all_insights = await self._store.list_insights(project_id)
        for existing in all_insights:
            if content_hash(existing.text) == h:
                return True
        return False
