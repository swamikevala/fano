"""EventBus — in-process async pub/sub with topic-based routing.

Implements EventBusInterface from shared.models. Topics use dot-separated
hierarchy with wildcard support:
    *   matches exactly one level
    **  matches one or more remaining levels

Events are persisted (to StateStore or in-memory) before delivery.
Handler errors are logged and isolated — one failure never blocks others.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from shared.models import Event, EventBusInterface, EventHandler

if TYPE_CHECKING:
    from shared.models import StateStoreInterface

logger = logging.getLogger(__name__)


def _compile_topic_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a dot-separated topic pattern to a regex.

    Rules:
        *   -> matches exactly one non-dot segment
        **  -> matches one or more segments (must be last or stand-alone segment)
    """
    parts = pattern.split(".")
    regex_parts: list[str] = []
    for part in parts:
        if part == "**":
            # Match one or more dot-separated segments (rest of string)
            regex_parts.append(r"[^.]+(?:\.[^.]+)*")
        elif part == "*":
            regex_parts.append(r"[^.]+")
        else:
            regex_parts.append(re.escape(part))
    return re.compile(r"^" + r"\." .join(regex_parts) + r"$")


class _Subscription:
    """A single subscription: compiled pattern + handler."""

    __slots__ = ("pattern_str", "regex", "handler")

    def __init__(self, pattern_str: str, regex: re.Pattern[str], handler: EventHandler):
        self.pattern_str = pattern_str
        self.regex = regex
        self.handler = handler


class EventBus(EventBusInterface):
    """In-process async EventBus with topic wildcards and optional persistence."""

    def __init__(self, store: StateStoreInterface | None = None) -> None:
        self._store = store
        self._subscriptions: list[_Subscription] = []
        # In-memory event log used when no external store is provided,
        # and also as the replay source when store is None.
        self._events: list[Event] = []

    # ------------------------------------------------------------------ #
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------ #

    def subscribe(self, topic_pattern: str, handler: EventHandler) -> None:
        """Register *handler* to receive events matching *topic_pattern*."""
        regex = _compile_topic_pattern(topic_pattern)
        self._subscriptions.append(_Subscription(topic_pattern, regex, handler))
        logger.debug(
            "EventBus subscription added: pattern=%s",
            topic_pattern,
        )

    def unsubscribe(self, topic_pattern: str, handler: EventHandler) -> None:
        """Remove a previously registered subscription."""
        self._subscriptions = [
            s
            for s in self._subscriptions
            if not (s.pattern_str == topic_pattern and s.handler is handler)
        ]
        logger.debug(
            "EventBus subscription removed: pattern=%s",
            topic_pattern,
        )

    # ------------------------------------------------------------------ #
    # Publish
    # ------------------------------------------------------------------ #

    async def publish(self, event: Event) -> None:
        """Persist event, then deliver to all matching handlers.

        Persist-before-deliver: the event is stored before any handler runs.
        Handler exceptions are caught, logged, and do not propagate.
        """
        # 1. Persist
        if self._store is not None:
            await self._store.persist_event(event)
        self._events.append(event)

        # 2. Deliver
        await self._deliver(event)

    # ------------------------------------------------------------------ #
    # Replay
    # ------------------------------------------------------------------ #

    async def replay(self, since: datetime) -> None:
        """Re-deliver historical events to current subscribers.

        If an external store is configured, events are loaded from there.
        Otherwise the in-memory log is used.
        """
        if self._store is not None:
            events = await self._store.list_events(since=since)
        else:
            events = [e for e in self._events if e.timestamp >= since]

        for event in events:
            await self._deliver(event)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _deliver(self, event: Event) -> None:
        """Dispatch *event* to every matching handler, isolating errors."""
        for sub in self._subscriptions:
            if sub.regex.match(event.topic):
                try:
                    await sub.handler(event)
                except Exception:
                    logger.exception(
                        "EventBus handler error: pattern=%s topic=%s",
                        sub.pattern_str,
                        event.topic,
                    )
