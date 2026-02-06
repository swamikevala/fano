"""SeedManager — manages seed lifecycle and modification proposals."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from shared.config import Config
from shared.logging import get_logger
from shared.models import (
    Event,
    EventBusInterface,
    Seed,
    SeedModification,
    SeedStatus,
    SeedType,
    StateStoreInterface,
    generate_id,
)

log = get_logger("explorer", "seed_manager")


class SeedManager:
    """Manages seed lifecycle and modification proposals."""

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        config: Config,
    ) -> None:
        self._store = store
        self._bus = event_bus
        self._config = config
        self._max_reexplore = int(
            config.get("explorer.thread_retirement.max_reexplore_count", 3)
        )

    async def select_next_seed(self, project_id: str) -> Seed | None:
        """Select highest-priority active seed below max exploration count."""
        seeds = await self._store.list_seeds(project_id, status=SeedStatus.ACTIVE)
        for seed in seeds:  # already ordered by priority DESC, created_at ASC
            if seed.exploration_count < self._max_reexplore:
                return seed
        return None

    async def record_exploration(self, seed_id: str) -> None:
        """Increment exploration_count. Transition to EXPLORED if at max."""
        seed = await self._store.get_seed(seed_id)
        if seed is None:
            return
        new_count = seed.exploration_count + 1
        updates: dict = {"exploration_count": new_count}
        if new_count >= self._max_reexplore:
            updates["status"] = SeedStatus.EXPLORED
        await self._store.update_seed(seed_id, **updates)

    async def propose_modification(
        self,
        seed: Seed,
        proposed_text: str,
        reasoning: str,
        proposing_thread_id: str,
        agreement_ratio: float,
    ) -> SeedModification:
        """Create a seed modification proposal."""
        now = datetime.now(timezone.utc)
        mod = SeedModification(
            id=generate_id(),
            seed_id=seed.id,
            original_text=seed.text,
            proposed_text=proposed_text,
            reasoning=reasoning,
            proposing_thread_id=proposing_thread_id,
            agreement_ratio=agreement_ratio,
            status="pending",
            child_seed_id=None,
            created_at=now,
            resolved_at=None,
        )

        if agreement_ratio < 0.66:
            mod = SeedModification(
                id=mod.id, seed_id=mod.seed_id,
                original_text=mod.original_text, proposed_text=mod.proposed_text,
                reasoning=mod.reasoning, proposing_thread_id=mod.proposing_thread_id,
                agreement_ratio=mod.agreement_ratio,
                status="rejected", child_seed_id=None,
                created_at=mod.created_at, resolved_at=now,
            )
            await self._store.create_seed_modification(mod)
            return mod

        project = await self._store.get_project(seed.project_id)
        require_approval = project.seed_modification_require_approval if project else True

        await self._store.create_seed_modification(mod)

        if require_approval:
            await self._bus.publish(Event(
                topic="explorer.seed.modification.proposed",
                timestamp=now, source="explorer.seed_manager",
                payload={"modification_id": mod.id, "seed_id": seed.id},
                correlation_id=str(uuid.uuid4()),
            ))
            return mod

        # Auto-approve
        return await self.approve_modification(seed.id, mod)

    async def approve_modification(self, seed_id: str, modification: SeedModification) -> SeedModification:
        """Apply an approved modification: evolve original, create child."""
        now = datetime.now(timezone.utc)
        await self._store.update_seed(seed_id, status=SeedStatus.EVOLVED)

        child = Seed(
            id=generate_id(),
            project_id=(await self._store.get_seed(seed_id)).project_id,
            text=modification.proposed_text,
            type=SeedType.CONJECTURE,
            priority=(await self._store.get_seed(seed_id)).priority,
            tags=[], confidence=None, source="seed_modification",
            notes=modification.reasoning,
            status=SeedStatus.ACTIVE,
            parent_seed_id=seed_id,
            modification_reason=modification.reasoning,
            exploration_count=0,
            created_at=now, updated_at=now,
        )
        await self._store.create_seed(child)

        await self._store.update_seed_modification(
            modification.id, status="approved", child_seed_id=child.id, resolved_at=now,
        )
        updated_mod = await self._store.get_seed_modification(modification.id)
        return updated_mod

    async def handle_user_event(self, event: Event) -> None:
        """Handle user.seed.* events."""
        topic = event.topic
        payload = event.payload

        if topic == "user.seed.created":
            seed_data = payload.get("seed")
            if seed_data:
                await self._store.create_seed(seed_data)
        elif topic == "user.seed.prioritized":
            sid = payload.get("seed_id")
            pri = payload.get("priority")
            if sid and pri is not None:
                await self._store.update_seed(sid, priority=pri)
        elif topic == "user.seed.retired":
            sid = payload.get("seed_id")
            if sid:
                await self._store.update_seed(sid, status=SeedStatus.RETIRED)
        elif topic == "user.seed.modification.approved":
            mid = payload.get("modification_id")
            sid = payload.get("seed_id")
            if mid and sid:
                mod = await self._store.get_seed_modification(mid)
                if mod:
                    await self.approve_modification(sid, mod)
        elif topic == "user.seed.modification.rejected":
            mid = payload.get("modification_id")
            if mid:
                await self._store.update_seed_modification(mid, status="rejected")
