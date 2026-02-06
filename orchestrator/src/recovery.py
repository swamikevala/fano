"""RecoveryManager — crash recovery via event replay.

On startup, reads the last checkpoint timestamp from the store and
replays all events that occurred since then.  Periodic checkpoints
let us minimize replay work on the next restart.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.models import EventBusInterface, StateStoreInterface

logger = logging.getLogger(__name__)

CHECKPOINT_METRIC_NAME = "orchestrator.checkpoint"


class RecoveryManager:
    """Replays missed events after a crash or restart."""

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
    ) -> None:
        self._store = store
        self._event_bus = event_bus

    async def recover(self) -> None:
        """Replay events since the last checkpoint.

        1. Read the last checkpoint timestamp from the store
        2. Replay all events since that timestamp
        3. Record a new checkpoint
        """
        since = await self._get_last_checkpoint()
        if since is not None:
            logger.info(
                "Recovery: replaying events since %s",
                since.isoformat(),
            )
            await self._event_bus.replay(since=since)
        else:
            logger.info("Recovery: no checkpoint found, skipping replay")

        # Record a fresh checkpoint after recovery
        await self.checkpoint()

    async def checkpoint(self) -> None:
        """Record current timestamp as a checkpoint in the store."""
        now = datetime.now(timezone.utc)
        await self._store.record_metric(
            project_id="__system__",
            name=CHECKPOINT_METRIC_NAME,
            value=now.timestamp(),
        )
        logger.info("Recovery: checkpoint recorded at %s", now.isoformat())

    async def _get_last_checkpoint(self) -> datetime | None:
        """Read the most recent checkpoint timestamp from the store."""
        metrics = await self._store.query_metrics(CHECKPOINT_METRIC_NAME)
        if not metrics:
            return None
        # The most recent metric has the latest checkpoint
        latest = metrics[-1]
        ts_value = latest["value"]
        return datetime.fromtimestamp(ts_value, tz=timezone.utc)
