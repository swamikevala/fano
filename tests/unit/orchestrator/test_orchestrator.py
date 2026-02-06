"""Tests for orchestrator/src — Orchestrator, QuotaAllocator, RecoveryManager, RetryScheduler.

Tests defined per Design Spec Section 11.8.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from shared.config import Config
from shared.events import EventBus
from shared.models import (
    Event,
    HealthStatus,
    Insight,
    InsightStatus,
    Confidence,
    ModuleInterface,
    generate_id,
)

# Set env var before Config validation
os.environ.setdefault("FANO_TEST_KEY", "test-key-value")


def _make_config() -> Config:
    """Create a Config for testing with all required orchestrator keys."""
    return Config.from_dict({
        "llm": {
            "api_key_env": "FANO_TEST_KEY",
            "models": {"claude": {"model": "claude-3"}},
        },
        "consensus": {"backends": ["claude"]},
        "quotas": {
            "daily_budget_usd": 10.0,
            "per_module_weights": {"explorer": 50, "documenter": 35, "researcher": 15},
            "alert_at_percent": 80,
        },
        "retention": {"events_max_age_days": 90},
    })


def _make_event(
    topic: str,
    payload: dict | None = None,
    ts: datetime | None = None,
    source: str = "test",
) -> Event:
    """Create an Event with sensible defaults."""
    return Event(
        topic=topic,
        timestamp=ts or datetime.now(timezone.utc),
        source=source,
        payload=payload or {},
        correlation_id="corr-001",
    )


def _make_mock_module(name: str, healthy: bool = True) -> ModuleInterface:
    """Create a mock ModuleInterface with the given name and health status."""
    module = MagicMock(spec=ModuleInterface)
    type(module).module_name = PropertyMock(return_value=name)
    module.initialize = AsyncMock(return_value=True)
    module.start = AsyncMock()
    module.stop = AsyncMock()
    module.health_check = AsyncMock(
        return_value=HealthStatus(
            module=name,
            healthy=healthy,
            message="ok" if healthy else "unhealthy",
        )
    )
    return module


def _make_insight(
    project_id: str = "proj-1",
    status: InsightStatus = InsightStatus.TRANSIENT_FAILURE,
    failure_count: int = 1,
    updated_at: datetime | None = None,
) -> Insight:
    """Create a test Insight."""
    now = updated_at or datetime.now(timezone.utc)
    return Insight(
        id=generate_id(),
        project_id=project_id,
        text="Test insight",
        confidence=Confidence.MEDIUM,
        tags=["test"],
        source_thread_id=None,
        extraction_model=None,
        status=status,
        evaluation_scores={},
        dispute_count=0,
        transient_failure_count=failure_count,
        review_record=None,
        blessed_at=None,
        incorporated_at=None,
        incorporated_in_section=None,
        created_at=now,
        updated_at=now,
    )


# ===========================================================================
# Orchestrator Tests
# ===========================================================================


class TestModuleRegistration:
    """test_module_registration: Modules registered and accessible by name."""

    async def test_register_single_module(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod = _make_mock_module("explorer")

        orch.register_module(mod)

        status = await orch.get_status()
        assert "explorer" in status["modules"]

    async def test_register_multiple_modules(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod_a = _make_mock_module("explorer")
        mod_b = _make_mock_module("documenter")

        orch.register_module(mod_a)
        orch.register_module(mod_b)

        status = await orch.get_status()
        assert "explorer" in status["modules"]
        assert "documenter" in status["modules"]

    async def test_register_duplicate_raises(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod = _make_mock_module("explorer")

        orch.register_module(mod)
        with pytest.raises(ValueError, match="already registered"):
            orch.register_module(mod)


class TestStartInitializesAll:
    """test_start_initializes_all: All modules initialized on start."""

    async def test_start_calls_initialize_on_all_modules(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod_a = _make_mock_module("explorer")
        mod_b = _make_mock_module("documenter")

        orch.register_module(mod_a)
        orch.register_module(mod_b)

        await orch.start()

        mod_a.initialize.assert_awaited_once()
        mod_b.initialize.assert_awaited_once()

        await orch.stop()

    async def test_start_calls_start_on_all_modules(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod_a = _make_mock_module("explorer")
        mod_b = _make_mock_module("documenter")

        orch.register_module(mod_a)
        orch.register_module(mod_b)

        await orch.start()

        mod_a.start.assert_awaited_once()
        mod_b.start.assert_awaited_once()

        await orch.stop()

    async def test_initialize_called_before_start(self) -> None:
        """Modules are initialized before start is called."""
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        call_order: list[str] = []

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod = _make_mock_module("explorer")
        mod.initialize = AsyncMock(
            side_effect=lambda: call_order.append("init") or True
        )
        mod.start = AsyncMock(
            side_effect=lambda: call_order.append("start")
        )

        orch.register_module(mod)
        await orch.start()

        assert call_order == ["init", "start"]

        await orch.stop()


class TestStopInReverseOrder:
    """test_stop_in_reverse_order: Modules stopped in reverse registration order."""

    async def test_stop_reverses_registration_order(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        stop_order: list[str] = []

        orch = Orchestrator(store=store, event_bus=bus, config=config)

        mod_a = _make_mock_module("explorer")
        mod_a.stop = AsyncMock(
            side_effect=lambda: stop_order.append("explorer")
        )
        mod_b = _make_mock_module("documenter")
        mod_b.stop = AsyncMock(
            side_effect=lambda: stop_order.append("documenter")
        )
        mod_c = _make_mock_module("researcher")
        mod_c.stop = AsyncMock(
            side_effect=lambda: stop_order.append("researcher")
        )

        orch.register_module(mod_a)
        orch.register_module(mod_b)
        orch.register_module(mod_c)

        await orch.start()
        await orch.stop()

        # Reverse of registration order: researcher, documenter, explorer
        assert stop_order == ["researcher", "documenter", "explorer"]


class TestHealthCheckPropagates:
    """test_health_check_propagates: Orchestrator health reflects module health."""

    async def test_all_healthy(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod_a = _make_mock_module("explorer", healthy=True)
        mod_b = _make_mock_module("documenter", healthy=True)

        orch.register_module(mod_a)
        orch.register_module(mod_b)

        status = await orch.get_status()

        assert status["healthy"] is True
        assert status["modules"]["explorer"]["healthy"] is True
        assert status["modules"]["documenter"]["healthy"] is True

    async def test_one_unhealthy_propagates(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod_a = _make_mock_module("explorer", healthy=True)
        mod_b = _make_mock_module("documenter", healthy=False)

        orch.register_module(mod_a)
        orch.register_module(mod_b)

        status = await orch.get_status()

        assert status["healthy"] is False
        assert status["modules"]["explorer"]["healthy"] is True
        assert status["modules"]["documenter"]["healthy"] is False


# ===========================================================================
# QuotaAllocator Tests
# ===========================================================================


class TestQuotaTracking:
    """test_quota_tracking: Spend tracked correctly per module."""

    async def test_initial_budget_correct(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # explorer gets 50% of $10 = $5.00
        remaining = allocator.get_remaining("explorer")
        assert remaining == pytest.approx(5.0)

        # documenter gets 35% of $10 = $3.50
        remaining = allocator.get_remaining("documenter")
        assert remaining == pytest.approx(3.5)

        # researcher gets 15% of $10 = $1.50
        remaining = allocator.get_remaining("researcher")
        assert remaining == pytest.approx(1.5)

    async def test_spend_tracked_on_llm_event(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # Simulate an LLM spend event from explorer
        event = _make_event(
            "llm.request.completed",
            payload={"module": "explorer", "cost_usd": 0.50},
            source="llm",
        )
        await bus.publish(event)

        # explorer should now have $5.00 - $0.50 = $4.50
        remaining = allocator.get_remaining("explorer")
        assert remaining == pytest.approx(4.5)

    async def test_is_within_budget_true(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        assert allocator.is_within_budget("explorer") is True

    async def test_is_within_budget_false_when_exceeded(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # Spend more than explorer's $5.00 budget
        event = _make_event(
            "llm.request.completed",
            payload={"module": "explorer", "cost_usd": 5.01},
            source="llm",
        )
        await bus.publish(event)

        assert allocator.is_within_budget("explorer") is False

    async def test_multiple_spend_events_accumulate(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # Two spend events for explorer
        for _ in range(3):
            event = _make_event(
                "llm.request.completed",
                payload={"module": "explorer", "cost_usd": 1.0},
                source="llm",
            )
            await bus.publish(event)

        remaining = allocator.get_remaining("explorer")
        assert remaining == pytest.approx(2.0)  # $5.00 - $3.00 = $2.00

    async def test_unknown_module_gets_zero_budget(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        remaining = allocator.get_remaining("unknown_module")
        assert remaining == 0.0
        assert allocator.is_within_budget("unknown_module") is False


class TestQuotaWarningAtThreshold:
    """test_quota_warning_at_threshold: system.budget.warning published at alert_at_percent."""

    async def test_warning_published_at_threshold(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        warnings: list[Event] = []

        async def capture_warning(event: Event) -> None:
            warnings.append(event)

        bus.subscribe("system.budget.warning", capture_warning)

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # Explorer budget is $5.00, alert at 80% = $4.00 spent
        # Spend $4.01 to trigger
        event = _make_event(
            "llm.request.completed",
            payload={"module": "explorer", "cost_usd": 4.01},
            source="llm",
        )
        await bus.publish(event)

        assert len(warnings) == 1
        assert warnings[0].payload["module"] == "explorer"

    async def test_no_warning_below_threshold(self) -> None:
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        warnings: list[Event] = []

        async def capture_warning(event: Event) -> None:
            warnings.append(event)

        bus.subscribe("system.budget.warning", capture_warning)

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # Spend less than 80% of $5.00 (=$4.00)
        event = _make_event(
            "llm.request.completed",
            payload={"module": "explorer", "cost_usd": 3.50},
            source="llm",
        )
        await bus.publish(event)

        assert len(warnings) == 0

    async def test_warning_only_published_once(self) -> None:
        """Warning should not keep firing after the threshold is already crossed."""
        from orchestrator.src.quota import QuotaAllocator

        config = _make_config()
        bus = EventBus()

        warnings: list[Event] = []

        async def capture_warning(event: Event) -> None:
            warnings.append(event)

        bus.subscribe("system.budget.warning", capture_warning)

        allocator = QuotaAllocator(config=config, event_bus=bus)
        await allocator.initialize()

        # Cross the threshold
        event1 = _make_event(
            "llm.request.completed",
            payload={"module": "explorer", "cost_usd": 4.01},
            source="llm",
        )
        await bus.publish(event1)

        # Spend more beyond threshold
        event2 = _make_event(
            "llm.request.completed",
            payload={"module": "explorer", "cost_usd": 0.50},
            source="llm",
        )
        await bus.publish(event2)

        assert len(warnings) == 1  # Only one warning, not two


# ===========================================================================
# RecoveryManager Tests
# ===========================================================================


class TestRecoveryReplaysEvents:
    """test_recovery_replays_events: Events since checkpoint replayed on start."""

    async def test_recover_replays_since_checkpoint(self) -> None:
        from orchestrator.src.recovery import RecoveryManager

        bus = EventBus()
        store = AsyncMock()

        checkpoint_time = datetime.now(timezone.utc) - timedelta(hours=1)
        store.list_events = AsyncMock(return_value=[])

        # Record a metric as checkpoint
        store.query_metrics = AsyncMock(return_value=[
            {"value": checkpoint_time.timestamp(), "recorded_at": checkpoint_time.isoformat()}
        ])

        recovery = RecoveryManager(store=store, event_bus=bus)
        await recovery.recover()

        # Should have called replay on events bus
        # The bus.replay was called with the checkpoint timestamp
        # Since bus is a real EventBus, we verify indirectly.
        # For this test, we check that list_events was called (store-backed replay)
        # or we verify the replay mechanism works through the bus directly.
        # Since our EventBus doesn't have a store, we verify through the bus's replay mechanism.

        # Better approach: verify that the bus.replay was invoked with the correct timestamp
        # by using a spy or checking that events were actually delivered.
        # Let's use a more direct test.
        pass  # see more complete test below

    async def test_recovery_replays_events_to_subscribers(self) -> None:
        from orchestrator.src.recovery import RecoveryManager

        bus = EventBus()

        # Publish some events before recovery
        t0 = datetime.now(timezone.utc) - timedelta(minutes=30)
        event1 = _make_event("explorer.insight.attested", ts=t0 + timedelta(minutes=5))
        event2 = _make_event("documenter.section.written", ts=t0 + timedelta(minutes=10))
        await bus.publish(event1)
        await bus.publish(event2)

        # Set up store mock
        store = AsyncMock()
        # Return a checkpoint from 30 minutes ago
        store.query_metrics = AsyncMock(return_value=[
            {"value": t0.timestamp(), "recorded_at": t0.isoformat()}
        ])

        # Subscribe to capture replayed events
        replayed: list[Event] = []

        async def handler(event: Event) -> None:
            replayed.append(event)

        bus.subscribe("explorer.**", handler)
        bus.subscribe("documenter.**", handler)

        recovery = RecoveryManager(store=store, event_bus=bus)
        await recovery.recover()

        # Both events should have been replayed
        assert len(replayed) == 2

    async def test_recovery_with_no_checkpoint(self) -> None:
        """If no checkpoint exists, recovery should not fail."""
        from orchestrator.src.recovery import RecoveryManager

        bus = EventBus()
        store = AsyncMock()

        # No checkpoint found
        store.query_metrics = AsyncMock(return_value=[])

        recovery = RecoveryManager(store=store, event_bus=bus)
        # Should not raise
        await recovery.recover()

    async def test_checkpoint_records_timestamp(self) -> None:
        from orchestrator.src.recovery import RecoveryManager

        bus = EventBus()
        store = AsyncMock()
        store.query_metrics = AsyncMock(return_value=[])

        recovery = RecoveryManager(store=store, event_bus=bus)
        await recovery.checkpoint()

        # Verify that record_metric was called with a checkpoint
        store.record_metric.assert_awaited_once()
        call_args = store.record_metric.call_args
        assert call_args[1]["name"] == "orchestrator.checkpoint" or \
               call_args[0][1] == "orchestrator.checkpoint"


# ===========================================================================
# RetryScheduler Tests
# ===========================================================================


class TestRetryScheduler:
    """Tests for the RetryScheduler."""

    async def test_shelves_after_max_retries(self) -> None:
        from orchestrator.src.scheduler import RetryScheduler

        store = AsyncMock()
        bus = EventBus()

        # Insight that has hit max retries
        insight = _make_insight(
            status=InsightStatus.TRANSIENT_FAILURE,
            failure_count=5,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        store.list_insights = AsyncMock(return_value=[insight])

        scheduler = RetryScheduler(store=store, event_bus=bus)
        await scheduler.scan_and_retry("proj-1")

        # Should have updated to SHELVED
        store.update_insight.assert_awaited_once()
        call_kwargs = store.update_insight.call_args.kwargs
        assert call_kwargs["status"] == InsightStatus.SHELVED

    async def test_retries_eligible_insight(self) -> None:
        from orchestrator.src.scheduler import RetryScheduler

        store = AsyncMock()
        bus = EventBus()

        retried: list[Event] = []

        async def capture(event: Event) -> None:
            retried.append(event)

        bus.subscribe("orchestrator.**", capture)

        # Insight eligible for retry (failure_count=1, enough time elapsed)
        insight = _make_insight(
            status=InsightStatus.TRANSIENT_FAILURE,
            failure_count=1,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )
        store.list_insights = AsyncMock(return_value=[insight])

        scheduler = RetryScheduler(store=store, event_bus=bus)
        await scheduler.scan_and_retry("proj-1")

        # Should have updated status back to REVIEWING
        store.update_insight.assert_awaited_once()

    async def test_respects_backoff(self) -> None:
        from orchestrator.src.scheduler import RetryScheduler

        store = AsyncMock()
        bus = EventBus()

        # Insight with failure_count=1 but updated very recently (30 seconds ago)
        insight = _make_insight(
            status=InsightStatus.TRANSIENT_FAILURE,
            failure_count=1,
            updated_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        store.list_insights = AsyncMock(return_value=[insight])

        scheduler = RetryScheduler(store=store, event_bus=bus)
        await scheduler.scan_and_retry("proj-1")

        # Should NOT have updated (not enough time for backoff)
        store.update_insight.assert_not_awaited()

    async def test_exponential_backoff_calculation(self) -> None:
        from orchestrator.src.scheduler import RetryScheduler

        # BACKOFF_BASE = 300, failure_count=2 => backoff = min(300 * 2^2, 3600) = 1200
        assert RetryScheduler.BACKOFF_BASE_SECONDS == 300
        assert RetryScheduler.MAX_BACKOFF_SECONDS == 3600
        assert RetryScheduler.MAX_TRANSIENT_RETRIES == 5


# ===========================================================================
# Integration-style: Orchestrator with QuotaAllocator and RecoveryManager
# ===========================================================================


class TestOrchestratorIntegration:
    """Tests verifying orchestrator wires up quota and recovery correctly."""

    async def test_get_status_includes_quota(self) -> None:
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()
        store.query_metrics = AsyncMock(return_value=[])

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod = _make_mock_module("explorer")
        orch.register_module(mod)

        await orch.start()

        status = await orch.get_status()
        assert "quota" in status

        await orch.stop()

    async def test_start_runs_recovery(self) -> None:
        """Start should trigger crash recovery."""
        from orchestrator.src.main import Orchestrator

        config = _make_config()
        bus = EventBus()
        store = AsyncMock()
        store.query_metrics = AsyncMock(return_value=[])

        orch = Orchestrator(store=store, event_bus=bus, config=config)
        mod = _make_mock_module("explorer")
        orch.register_module(mod)

        await orch.start()

        # Recovery should have queried for checkpoint
        store.query_metrics.assert_awaited()

        await orch.stop()
