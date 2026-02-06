"""RetryScheduler — exponential-backoff retries for transient failures.

Scans insights with status=transient_failure and either retries them
(if the backoff period has elapsed) or shelves them (if max retries exceeded).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from shared.models import Event, InsightStatus

if TYPE_CHECKING:
    from shared.models import EventBusInterface, StateStoreInterface

logger = logging.getLogger(__name__)


class RetryScheduler:
    """Periodically scans for transient failures and re-queues them."""

    BACKOFF_BASE_SECONDS: int = 300   # 5 minutes
    MAX_BACKOFF_SECONDS: int = 3600   # 1 hour
    MAX_TRANSIENT_RETRIES: int = 5    # After this, mark as SHELVED

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
    ) -> None:
        self._store = store
        self._event_bus = event_bus

    async def scan_and_retry(self, project_id: str) -> None:
        """Scan transient_failure insights, apply exponential backoff, retry or shelve.

        1. Query insights WHERE status='transient_failure'
        2. For each: if transient_failure_count >= MAX_TRANSIENT_RETRIES -> SHELVED
        3. Otherwise: check backoff period. If elapsed -> transition to REVIEWING
        4. Publish retry event for successful retries
        """
        insights = await self._store.list_insights(
            project_id,
            status=InsightStatus.TRANSIENT_FAILURE,
        )

        now = datetime.now(timezone.utc)

        for insight in insights:
            if insight.transient_failure_count >= self.MAX_TRANSIENT_RETRIES:
                # Max retries exceeded — shelve it
                await self._store.update_insight(
                    insight.id,
                    status=InsightStatus.SHELVED,
                )
                logger.info(
                    "Insight %s shelved after %d transient failures",
                    insight.id,
                    insight.transient_failure_count,
                )
                continue

            # Calculate backoff
            backoff_seconds = min(
                self.BACKOFF_BASE_SECONDS * (2 ** insight.transient_failure_count),
                self.MAX_BACKOFF_SECONDS,
            )

            elapsed = (now - insight.updated_at).total_seconds()
            if elapsed < backoff_seconds:
                # Not enough time has elapsed — skip
                logger.debug(
                    "Insight %s: %ds elapsed, need %ds backoff — skipping",
                    insight.id,
                    int(elapsed),
                    backoff_seconds,
                )
                continue

            # Ready to retry — transition back to REVIEWING
            await self._store.update_insight(
                insight.id,
                status=InsightStatus.REVIEWING,
            )

            retry_event = Event(
                topic="orchestrator.insight.retry",
                timestamp=now,
                source="orchestrator.scheduler",
                payload={
                    "insight_id": insight.id,
                    "project_id": project_id,
                    "retry_count": insight.transient_failure_count,
                },
                correlation_id=insight.id,
            )
            await self._event_bus.publish(retry_event)

            logger.info(
                "Insight %s retried (attempt %d)",
                insight.id,
                insight.transient_failure_count + 1,
            )
