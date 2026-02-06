"""ExplorerEngine — main entry point for the Explorer module.

Implements ModuleInterface. Orchestrates: seeds -> threads -> exchanges ->
insight extraction -> consensus review.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from shared.config import Config
from shared.logging import get_logger
from shared.models import (
    ConsensusEngineInterface,
    Event,
    EventBusInterface,
    HealthStatus,
    InsightStatus,
    LLMClientInterface,
    ModuleInterface,
    StateStoreInterface,
)
from explorer.src.insight_extractor import InsightExtractor
from explorer.src.review import ReviewPanel
from explorer.src.seed_manager import SeedManager
from explorer.src.thread_manager import ThreadManager

log = get_logger("explorer", "engine")


class ExplorerEngine(ModuleInterface):
    """Main entry point for the Explorer module."""

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
        self._consensus = consensus
        self._config = config
        self._running = False

        self._seed_mgr = SeedManager(store, event_bus, config)
        self._thread_mgr = ThreadManager(store, llm_client, event_bus, config)
        self._extractor = InsightExtractor(store, llm_client, event_bus, config)
        self._review_panel = ReviewPanel(consensus, store, event_bus)

    @property
    def module_name(self) -> str:
        return "explorer"

    async def initialize(self) -> bool:
        """Subscribe to events and set up internal state."""
        self._bus.subscribe("user.seed.**", self._seed_mgr.handle_user_event)
        self._bus.subscribe("user.insight.endorsed", self._on_insight_endorsed)
        self._bus.subscribe("user.insight.dismissed", self._on_insight_dismissed)
        self._bus.subscribe("user.insight.reconsider", self._on_insight_reconsider)
        self._bus.subscribe("researcher.evidence.supports", self._on_evidence)
        self._bus.subscribe("researcher.evidence.contradicts", self._on_evidence)
        log.info("explorer.engine.initialized")
        return True

    async def start(self) -> None:
        """Begin exploration loop."""
        self._running = True
        while self._running:
            try:
                await self.run_one_cycle()
            except Exception as exc:
                log.error("explorer.engine.cycle_error", error=str(exc))
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Mark module as stopped."""
        self._running = False
        log.info("explorer.engine.stopped")

    async def health_check(self) -> HealthStatus:
        """Return health status with active thread count."""
        project = self._config.project
        if project is None:
            return HealthStatus(module="explorer", healthy=False,
                                message="No project loaded", details={})
        threads = await self._thread_mgr.get_active_threads(project.id)
        return HealthStatus(
            module="explorer", healthy=True,
            message="OK", details={"active_threads": len(threads)},
        )

    async def run_one_cycle(self) -> None:
        """Execute one iteration of the exploration loop."""
        project = self._config.project
        if project is None:
            return

        # 1. Select seed
        seed = await self._seed_mgr.select_next_seed(project.id)
        if seed is None:
            return

        # 2. Create or resume thread
        active_threads = await self._thread_mgr.get_active_threads(project.id)
        thread = None
        for t in active_threads:
            if t.seed_id == seed.id:
                thread = t
                break
        if thread is None:
            thread = await self._thread_mgr.create_thread(seed, project)

        # 3. Run exchanges until chunk ready
        while not await self._thread_mgr.is_chunk_ready(thread):
            await self._thread_mgr.run_exchange(thread, project)
            thread = await self._store.get_thread(thread.id)

        # 4. Extract insights
        insights = await self._extractor.extract(thread, project)

        # 5. Review each insight
        for insight in insights:
            await self._review_panel.review(insight, project)

        # 6. Record exploration and check retirement
        await self._seed_mgr.record_exploration(seed.id)
        should, reason = await self._thread_mgr.should_retire(thread)
        if should:
            await self._thread_mgr.retire_thread(thread, reason)

    # ── Event handlers ───────────────────────────────────────

    async def _on_insight_endorsed(self, event: Event) -> None:
        insight_id = event.payload.get("insight_id")
        if not insight_id:
            return
        insight = await self._store.get_insight(insight_id)
        if insight and insight.source_thread_id:
            await self._thread_mgr.adjust_priority(insight.source_thread_id, +2)

    async def _on_insight_dismissed(self, event: Event) -> None:
        insight_id = event.payload.get("insight_id")
        if not insight_id:
            return
        insight = await self._store.get_insight(insight_id)
        if insight and insight.source_thread_id:
            await self._thread_mgr.adjust_priority(insight.source_thread_id, -2)

    async def _on_insight_reconsider(self, event: Event) -> None:
        insight_id = event.payload.get("insight_id")
        if not insight_id:
            return
        await self._store.update_insight(insight_id, status=InsightStatus.EXTRACTED)

    async def _on_evidence(self, event: Event) -> None:
        thread_id = event.payload.get("thread_id")
        if not thread_id:
            return
        delta = +1 if "supports" in event.topic else -1
        await self._thread_mgr.adjust_priority(thread_id, delta)
