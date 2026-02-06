"""DocumenterEngine — ModuleInterface facade for the documenter.

Runs the work loop: plan -> execute -> render -> snapshot.
Subscribes to events for insight incorporation and annotation handling.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from shared.models import (
    Event,
    HealthStatus,
    InsightStatus,
    ModuleInterface,
)

from documenter.src.annotations import AnnotationHandler
from documenter.src.context import ContextBuilder
from documenter.src.planner import Planner
from documenter.src.processor import Processor
from documenter.src.renderer import Renderer

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import (
        ConsensusEngineInterface,
        EventBusInterface,
        LLMClientInterface,
        StateStoreInterface,
    )

logger = logging.getLogger(__name__)


class DocumenterEngine(ModuleInterface):
    """Facade that orchestrates all documenter components."""

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface,
        config: Config,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._llm = llm_client
        self._consensus = consensus
        self._config = config
        self._running = False
        self._task: asyncio.Task | None = None

        # Components
        self._ctx_builder = ContextBuilder(store, config)
        self._planner = Planner(store, config)
        self._processor = Processor(
            store, llm_client, consensus, event_bus, self._ctx_builder, config,
        )
        self._renderer = Renderer(store, config)
        self._annotation_handler = AnnotationHandler(store, event_bus)

        # Queues for incoming events
        self._insight_queue: asyncio.Queue[str] = asyncio.Queue()
        self._finding_queue: asyncio.Queue[str] = asyncio.Queue()

    @property
    def module_name(self) -> str:
        return "documenter"

    async def initialize(self) -> bool:
        """Subscribe to events. Load document state from store."""
        self._event_bus.subscribe(
            "explorer.insight.attested", self._on_insight_attested,
        )
        self._event_bus.subscribe(
            "user.annotation.created",
            self._annotation_handler.handle_new_annotation,
        )
        self._event_bus.subscribe(
            "researcher.finding.stored", self._on_finding,
        )
        return True

    async def start(self) -> None:
        """Begin the documenter work loop."""
        self._running = True
        self._task = asyncio.create_task(self._work_loop())

    async def stop(self) -> None:
        """Stop the work loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            module="documenter",
            healthy=self._running,
            message="running" if self._running else "stopped",
        )

    async def _work_loop(self) -> None:
        """Main loop: plan -> execute -> render -> snapshot -> sleep."""
        while self._running:
            try:
                project = self._config.project
                if not project:
                    await asyncio.sleep(5)
                    continue

                items = await self._planner.plan_cycle(project.id)
                structural_change = False

                for item in items:
                    if not self._running:
                        break
                    try:
                        changed = await self._execute_item(item, project)
                        structural_change = structural_change or changed
                    except Exception:
                        logger.exception(
                            "documenter.work_item.failed",
                            extra={"type": item.type, "entity_id": item.entity_id},
                        )

                # Render
                doc_dir = self._config.get("documenter.document_dir", "documenter/document")
                await self._renderer.save(project, Path(doc_dir))

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("documenter.loop.error")

            await asyncio.sleep(10)

    async def _execute_item(self, item, project) -> bool:
        """Execute a single work item. Returns True if structure changed."""
        if item.type == "annotation":
            ann = await self._store.get_annotation(item.entity_id)
            if ann:
                await self._processor.address_annotation(ann, project)
                self._planner.record_call("review")
            return False
        elif item.type == "insight":
            ins = await self._store.get_insight(item.entity_id)
            if ins:
                await self._processor.incorporate_insight(ins, project)
                self._planner.record_call("new")
            return True
        elif item.type == "review":
            self._planner.record_call("review")
            return False
        elif item.type == "suggestion":
            ann = await self._store.get_annotation(item.entity_id)
            if ann:
                await self._processor.address_annotation(ann, project)
                self._planner.record_call("review")
            return False
        return False

    async def _on_insight_attested(self, event: Event) -> None:
        """Handler for explorer.insight.attested events."""
        insight_id = event.payload.get("insight_id")
        if insight_id:
            await self._insight_queue.put(insight_id)

    async def _on_finding(self, event: Event) -> None:
        """Handler for researcher.finding.stored events."""
        finding_id = event.payload.get("finding_id")
        if finding_id:
            await self._finding_queue.put(finding_id)
