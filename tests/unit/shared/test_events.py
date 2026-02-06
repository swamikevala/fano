"""Tests for shared.events — EventBus."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from shared.events import EventBus
from shared.models import Event, StateStoreInterface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    topic: str,
    payload: dict | None = None,
    ts: datetime | None = None,
) -> Event:
    """Create an Event with sensible defaults."""
    return Event(
        topic=topic,
        timestamp=ts or datetime.now(timezone.utc),
        source="test",
        payload=payload or {},
        correlation_id="corr-001",
    )


class FakeStore:
    """Minimal in-memory store that satisfies persist_event / list_events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def persist_event(self, event: Event) -> None:
        self.events.append(event)

    async def list_events(
        self,
        since: datetime | None = None,
        topic: str | None = None,
    ) -> list[Event]:
        result = self.events
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        if topic is not None:
            result = [e for e in result if e.topic == topic]
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventBus:
    """All tests from Design Spec Section 4.7."""

    # -- test_subscribe_and_publish -----------------------------------------

    async def test_subscribe_and_publish(self) -> None:
        """Handler receives published event."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("explorer.insight.attested", handler)

        event = _make_event("explorer.insight.attested", {"key": "value"})
        await bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    # -- test_wildcard_matching ---------------------------------------------

    async def test_wildcard_matching(self) -> None:
        """explorer.** matches explorer.insight.attested."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("explorer.**", handler)

        await bus.publish(_make_event("explorer.insight.attested"))
        await bus.publish(_make_event("explorer.started"))
        await bus.publish(_make_event("documenter.section.written"))

        assert len(received) == 2
        assert received[0].topic == "explorer.insight.attested"
        assert received[1].topic == "explorer.started"

    # -- test_single_level_wildcard -----------------------------------------

    async def test_single_level_wildcard(self) -> None:
        """* matches exactly one level."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("explorer.*", handler)

        await bus.publish(_make_event("explorer.started"))        # match
        await bus.publish(_make_event("explorer.insight.attested"))  # no match (two levels)
        await bus.publish(_make_event("documenter.started"))      # no match (wrong prefix)

        assert len(received) == 1
        assert received[0].topic == "explorer.started"

    async def test_single_level_wildcard_middle(self) -> None:
        """*.insight.* matches explorer.insight.attested."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("*.insight.*", handler)

        await bus.publish(_make_event("explorer.insight.attested"))  # match
        await bus.publish(_make_event("documenter.insight.saved"))   # match
        await bus.publish(_make_event("explorer.thread.started"))    # no match

        assert len(received) == 2

    # -- test_handler_error_isolation ---------------------------------------

    async def test_handler_error_isolation(self) -> None:
        """Failing handler doesn't prevent other handlers from executing."""
        bus = EventBus()
        received: list[Event] = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("handler exploded")

        async def good_handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.topic", bad_handler)
        bus.subscribe("test.topic", good_handler)

        event = _make_event("test.topic")
        await bus.publish(event)  # should not raise

        assert len(received) == 1
        assert received[0] is event

    # -- test_event_persisted -----------------------------------------------

    async def test_event_persisted(self) -> None:
        """Published event appears in store when store is provided."""
        fake_store = FakeStore()
        bus = EventBus(store=fake_store)

        event = _make_event("test.persist")
        await bus.publish(event)

        assert len(fake_store.events) == 1
        assert fake_store.events[0] is event

    async def test_event_not_persisted_without_store(self) -> None:
        """When no store is provided, events are stored in-memory only."""
        bus = EventBus()

        event = _make_event("test.memory")
        await bus.publish(event)

        # Internal in-memory store should have captured it
        assert len(bus._events) == 1
        assert bus._events[0] is event

    # -- test_replay --------------------------------------------------------

    async def test_replay(self) -> None:
        """Events published before subscriber added are delivered on replay."""
        fake_store = FakeStore()
        bus = EventBus(store=fake_store)

        t0 = datetime.now(timezone.utc)
        event1 = _make_event("explorer.started", ts=t0)
        event2 = _make_event("explorer.insight.attested", ts=t0 + timedelta(seconds=1))
        await bus.publish(event1)
        await bus.publish(event2)

        # Subscribe after publishing
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("explorer.**", handler)

        # Replay since t0 -- both events should be delivered
        await bus.replay(since=t0)

        assert len(received) == 2
        assert received[0].topic == "explorer.started"
        assert received[1].topic == "explorer.insight.attested"

    async def test_replay_without_store(self) -> None:
        """Replay works with in-memory events when no store is provided."""
        bus = EventBus()

        t0 = datetime.now(timezone.utc)
        event = _make_event("test.event", ts=t0)
        await bus.publish(event)

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.**", handler)
        await bus.replay(since=t0)

        assert len(received) == 1

    async def test_replay_respects_since(self) -> None:
        """Replay only delivers events at or after the since timestamp."""
        fake_store = FakeStore()
        bus = EventBus(store=fake_store)

        t0 = datetime.now(timezone.utc)
        old_event = _make_event("test.old", ts=t0 - timedelta(hours=1))
        new_event = _make_event("test.new", ts=t0)
        await bus.publish(old_event)
        await bus.publish(new_event)

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.**", handler)
        await bus.replay(since=t0)

        assert len(received) == 1
        assert received[0].topic == "test.new"

    # -- test_unsubscribe ---------------------------------------------------

    async def test_unsubscribe(self) -> None:
        """Unsubscribed handler no longer receives events."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.topic", handler)

        await bus.publish(_make_event("test.topic"))
        assert len(received) == 1

        bus.unsubscribe("test.topic", handler)

        await bus.publish(_make_event("test.topic"))
        assert len(received) == 1  # no new delivery

    # -- test_multiple_subscribers ------------------------------------------

    async def test_multiple_subscribers(self) -> None:
        """Multiple handlers on same topic all receive event."""
        bus = EventBus()
        received_a: list[Event] = []
        received_b: list[Event] = []

        async def handler_a(event: Event) -> None:
            received_a.append(event)

        async def handler_b(event: Event) -> None:
            received_b.append(event)

        bus.subscribe("test.topic", handler_a)
        bus.subscribe("test.topic", handler_b)

        event = _make_event("test.topic")
        await bus.publish(event)

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0] is event
        assert received_b[0] is event

    # -- test_no_match_no_delivery ------------------------------------------

    async def test_no_match_no_delivery(self) -> None:
        """Non-matching topic doesn't trigger handler."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("explorer.insight.*", handler)

        await bus.publish(_make_event("documenter.section.written"))

        assert len(received) == 0

    # -- test_publish_ordering ----------------------------------------------

    async def test_publish_ordering(self) -> None:
        """Events delivered in publish order."""
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.topic)

        bus.subscribe("order.**", handler)

        for i in range(5):
            await bus.publish(_make_event(f"order.event.{i}"))

        assert received == [f"order.event.{i}" for i in range(5)]

    # -- test_persist_before_deliver ----------------------------------------

    async def test_persist_before_deliver(self) -> None:
        """Event is persisted to store before handlers are called."""
        fake_store = FakeStore()
        bus = EventBus(store=fake_store)
        store_len_at_handler_time: list[int] = []

        async def handler(event: Event) -> None:
            # At handler call time the event should already be in the store
            store_len_at_handler_time.append(len(fake_store.events))

        bus.subscribe("test.order", handler)
        await bus.publish(_make_event("test.order"))

        assert store_len_at_handler_time == [1]

    # -- Edge cases ---------------------------------------------------------

    async def test_double_star_matches_single_level(self) -> None:
        """** also matches a single remaining level."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("user.**", handler)

        await bus.publish(_make_event("user.seed.created"))   # match
        await bus.publish(_make_event("user.login"))           # match (single level)

        assert len(received) == 2

    async def test_exact_topic_no_wildcards(self) -> None:
        """Exact match with no wildcards in pattern."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("a.b.c", handler)

        await bus.publish(_make_event("a.b.c"))   # match
        await bus.publish(_make_event("a.b.c.d")) # no match
        await bus.publish(_make_event("a.b"))      # no match

        assert len(received) == 1
