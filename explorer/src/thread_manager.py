"""ThreadManager — manages thread lifecycle, exchanges, and retirement."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from shared.config import Config
from shared.logging import get_logger
from shared.models import (
    Event,
    EventBusInterface,
    Exchange,
    ExchangeRole,
    InsightStatus,
    LLMClientInterface,
    Project,
    Seed,
    StateStoreInterface,
    Thread,
    ThreadStatus,
    generate_id,
)
from explorer.src.prompts import build_exchange_prompt

log = get_logger("explorer", "thread_manager")

# Role rotation: 1=explorer, 2=critic, 3=synthesizer, 4=explorer, ...
_ROLE_CYCLE = [ExchangeRole.EXPLORER, ExchangeRole.CRITIC, ExchangeRole.SYNTHESIZER]

# Map role to config weight key
_ROLE_WEIGHT_KEY = {
    ExchangeRole.EXPLORER: "exploration",
    ExchangeRole.CRITIC: "critique",
    ExchangeRole.SYNTHESIZER: "synthesis",
}


class ThreadManager:
    """Manages thread lifecycle: creation, exchange execution, retirement."""

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
        self._max_active = int(config.get("explorer.max_active_threads", 3))
        self._min_exchanges = int(config.get("explorer.min_exchanges_for_chunk", 4))
        self._max_exchanges = int(config.get("explorer.max_exchanges_per_thread", 12))
        self._max_idle_hours = int(config.get("explorer.thread_retirement.max_idle_hours", 48))

    async def create_thread(self, seed: Seed, project: Project) -> Thread:
        """Create a new thread for exploring a seed."""
        now = datetime.now(timezone.utc)
        thread = Thread(
            id=generate_id(),
            project_id=project.id,
            seed_id=seed.id,
            status=ThreadStatus.ACTIVE,
            priority=seed.priority,
            exchange_count=0,
            last_completed_sequence=0,
            created_at=now,
            updated_at=now,
            retired_at=None,
            retire_reason=None,
        )
        await self._store.create_thread(thread)
        await self._bus.publish(Event(
            topic="explorer.thread.created",
            timestamp=now, source="explorer.thread_manager",
            payload={"thread_id": thread.id, "seed_id": seed.id},
            correlation_id=str(uuid.uuid4()),
        ))
        return thread

    async def run_exchange(self, thread: Thread, project: Project) -> Exchange:
        """Run one exploration exchange with role rotation."""
        seq = thread.last_completed_sequence + 1
        role = _ROLE_CYCLE[(seq - 1) % 3]
        backend = self._select_backend(role)

        seed = await self._store.get_seed(thread.seed_id)
        exchanges = await self._store.get_exchanges(thread.id)

        prompt = build_exchange_prompt(project, thread, exchanges, role, seed)
        response = await self._llm.send(backend, prompt, module="explorer")

        now = datetime.now(timezone.utc)
        exchange = Exchange(
            id=generate_id(),
            thread_id=thread.id,
            sequence=seq,
            role=role,
            model=backend,
            prompt=prompt,
            response=response.text,
            created_at=now,
        )
        await self._store.create_exchange(exchange)
        await self._store.update_thread(
            thread.id,
            exchange_count=seq,
            last_completed_sequence=seq,
        )
        return exchange

    async def is_chunk_ready(self, thread: Thread) -> bool:
        """True if thread has >= min_exchanges_for_chunk since last extraction."""
        return thread.exchange_count >= self._min_exchanges

    async def should_retire(self, thread: Thread) -> tuple[bool, str | None]:
        """Check retirement conditions."""
        if thread.exchange_count >= self._max_exchanges:
            return True, "max_exchanges"

        idle_cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=self._max_idle_hours)
        if thread.updated_at < idle_cutoff:
            return True, "idle_timeout"

        # Check low quality: all insights from last extraction discarded
        insights = await self._store.get_insights_for_thread(thread.id)
        if insights:
            latest = [i for i in insights if i.status == InsightStatus.DISCARDED]
            if len(latest) == len(insights) and len(insights) > 0:
                return True, "low_quality"

        return False, None

    async def retire_thread(self, thread: Thread, reason: str) -> None:
        """Transition to RETIRED."""
        now = datetime.now(timezone.utc)
        await self._store.update_thread(
            thread.id, status=ThreadStatus.RETIRED,
            retired_at=now, retire_reason=reason,
        )
        await self._bus.publish(Event(
            topic="explorer.thread.retired",
            timestamp=now, source="explorer.thread_manager",
            payload={"thread_id": thread.id, "reason": reason},
            correlation_id=str(uuid.uuid4()),
        ))

    async def adjust_priority(self, thread_id: str, delta: int) -> None:
        """Adjust thread priority by delta (clamped to 1-10)."""
        thread = await self._store.get_thread(thread_id)
        if thread is None:
            return
        new_priority = max(1, min(10, thread.priority + delta))
        await self._store.update_thread(thread_id, priority=new_priority)

    async def get_active_threads(self, project_id: str) -> list[Thread]:
        """Return active threads ordered by priority DESC."""
        threads = await self._store.list_threads(project_id, status=ThreadStatus.ACTIVE)
        return sorted(threads, key=lambda t: t.priority, reverse=True)

    def _select_backend(self, role: ExchangeRole) -> str:
        """Weighted random backend selection for role."""
        key = _ROLE_WEIGHT_KEY.get(role, "exploration")
        weights: dict = self._config.get(f"explorer.model_weights.{key}", {})
        if not weights:
            return "gemini"
        backends = list(weights.keys())
        w = [float(weights[b]) for b in backends]
        return random.choices(backends, weights=w, k=1)[0]
