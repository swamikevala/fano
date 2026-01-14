"""
Tests for orchestrator/scheduler.py
"""

import time
import pytest

from orchestrator.models import Task, TaskState
from orchestrator.scheduler import Scheduler, RecentFailureCache, ScoredTask


class TestRecentFailureCache:
    """Tests for RecentFailureCache."""

    def test_add_and_contains(self, failure_cache):
        """Cache should track added keys."""
        assert failure_cache.contains("key1") is False
        failure_cache.add("key1")
        assert failure_cache.contains("key1") is True

    def test_remove(self, failure_cache):
        """Cache should allow key removal."""
        failure_cache.add("key1")
        assert failure_cache.contains("key1") is True
        failure_cache.remove("key1")
        assert failure_cache.contains("key1") is False

    def test_max_size_eviction(self):
        """Cache should evict oldest entries when full."""
        cache = RecentFailureCache(ttl_seconds=3600, max_size=3)
        cache.add("key1")
        cache.add("key2")
        cache.add("key3")
        cache.add("key4")  # Should evict key1

        assert cache.contains("key1") is False
        assert cache.contains("key2") is True
        assert cache.contains("key3") is True
        assert cache.contains("key4") is True

    def test_ttl_expiration(self):
        """Cache should expire entries after TTL."""
        cache = RecentFailureCache(ttl_seconds=0, max_size=100)  # Immediate expiry
        cache.add("key1")
        # Entry should be expired immediately
        time.sleep(0.01)
        assert cache.contains("key1") is False

    def test_update_moves_to_end(self, failure_cache):
        """Adding existing key should move it to end (refresh)."""
        failure_cache.add("key1")
        failure_cache.add("key2")
        failure_cache.add("key1")  # Refresh key1

        # key1 should now be at end
        keys = list(failure_cache._cache.keys())
        assert keys[-1] == "key1"

    def test_clear_expired(self):
        """clear_expired should remove expired entries."""
        cache = RecentFailureCache(ttl_seconds=0, max_size=100)
        cache.add("key1")
        cache.add("key2")
        time.sleep(0.01)

        cache.clear_expired()
        assert len(cache._cache) == 0


class TestScoredTask:
    """Tests for ScoredTask ordering."""

    def test_higher_score_sorts_first(self):
        """Higher scores should sort before lower scores."""
        task1 = Task(id="t1", key="k1", module="explorer",
                    task_type="exploration", priority=50)
        task2 = Task(id="t2", key="k2", module="explorer",
                    task_type="exploration", priority=60)

        scored1 = ScoredTask(score=50, task_id="t1", task=task1)
        scored2 = ScoredTask(score=60, task_id="t2", task=task2)

        # Higher score should be "less than" for min-heap to work as max-heap
        assert scored2 < scored1


class TestScheduler:
    """Tests for Scheduler."""

    @pytest.mark.asyncio
    async def test_submit_new_task(self, scheduler, state_manager):
        """submit should accept new tasks."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            result = scheduler.submit(task)

            assert result is True
            assert task.priority > 0  # Priority was computed
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_submit_duplicate_rejected(self, scheduler, state_manager):
        """submit should reject duplicate keys."""
        await state_manager.startup()
        try:
            task1 = Task(id="t1", key="same-key", module="explorer",
                        task_type="exploration", priority=0)
            task2 = Task(id="t2", key="same-key", module="explorer",
                        task_type="exploration", priority=0)

            assert scheduler.submit(task1) is True
            assert scheduler.submit(task2) is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_submit_failure_cache_rejection(self, scheduler, state_manager):
        """submit should reject keys in failure cache."""
        await state_manager.startup()
        try:
            # Add key to failure cache
            scheduler.failure_cache.add("failed-key")

            task = Task(id="t1", key="failed-key", module="explorer",
                       task_type="exploration", priority=0)
            result = scheduler.submit(task)

            assert result is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_submit_force_retry_bypasses_cache(self, scheduler, state_manager):
        """submit with force_retry should bypass failure cache."""
        await state_manager.startup()
        try:
            scheduler.failure_cache.add("failed-key")

            task = Task(id="t1", key="failed-key", module="explorer",
                       task_type="exploration", priority=0)
            result = scheduler.submit(task, force_retry=True)

            assert result is True
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_next_task_priority_order(self, scheduler, state_manager):
        """get_next_task should return highest priority task."""
        await state_manager.startup()
        try:
            # Submit tasks with different priorities
            low = Task(id="low", key="k-low", module="explorer",
                      task_type="exploration", priority=0)
            high = Task(id="high", key="k-high", module="explorer",
                       task_type="address_comment", priority=0)  # Higher base priority

            scheduler.submit(low)
            scheduler.submit(high)

            next_task = scheduler.get_next_task()
            assert next_task is not None
            assert next_task.id == "high"  # address_comment has higher priority
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_next_task_empty_queue(self, scheduler, state_manager):
        """get_next_task should return None when queue empty."""
        await state_manager.startup()
        try:
            next_task = scheduler.get_next_task()
            assert next_task is None
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_next_task_backend_filter(self, scheduler, state_manager):
        """get_next_task should filter by preferred backend."""
        await state_manager.startup()
        try:
            gemini_task = Task(id="gemini", key="k1", module="explorer",
                              task_type="exploration", priority=0,
                              preferred_backend="gemini")
            chatgpt_task = Task(id="chatgpt", key="k2", module="explorer",
                               task_type="exploration", priority=0,
                               preferred_backend="chatgpt")

            scheduler.submit(gemini_task)
            scheduler.submit(chatgpt_task)

            # Should prefer tasks for requested backend
            next_task = scheduler.get_next_task(backend="chatgpt")
            assert next_task is not None
            assert next_task.preferred_backend == "chatgpt"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_mark_task_completed(self, scheduler, state_manager):
        """mark_task_completed should update state and clear failure cache."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            scheduler.submit(task)

            # Add to failure cache to verify it gets cleared
            scheduler.failure_cache.add("k1")

            scheduler.mark_task_completed("t1", "Success!")

            assert state_manager.get_task("t1").state == TaskState.COMPLETED
            assert scheduler.failure_cache.contains("k1") is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_mark_task_failed(self, scheduler, state_manager):
        """mark_task_failed should update state and add to failure cache."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            scheduler.submit(task)

            scheduler.mark_task_failed("t1", "Connection error")

            assert state_manager.get_task("t1").state == TaskState.FAILED
            assert scheduler.failure_cache.contains("k1") is True
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_mark_task_failed_no_cache(self, scheduler, state_manager):
        """mark_task_failed with add_to_cache=False should not cache."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            scheduler.submit(task)

            scheduler.mark_task_failed("t1", "Temporary error", add_to_cache=False)

            assert scheduler.failure_cache.contains("k1") is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_mark_task_running(self, scheduler, state_manager):
        """mark_task_running should update state."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            scheduler.submit(task)

            scheduler.mark_task_running("t1", "pool-123")

            updated = state_manager.get_task("t1")
            assert updated.state == TaskState.RUNNING
            assert updated.pool_request_id == "pool-123"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_requeue_task(self, scheduler, state_manager):
        """requeue_task should reset task to pending."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            scheduler.submit(task)
            scheduler.mark_task_running("t1")

            result = scheduler.requeue_task("t1")

            assert result is True
            assert state_manager.get_task("t1").state == TaskState.PENDING
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_requeue_task_max_retries(self, scheduler, state_manager):
        """requeue_task should fail when max retries exceeded."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0, max_attempts=2)
            scheduler.submit(task)

            # Exhaust retries
            scheduler.mark_task_running("t1")  # attempt 1
            task.attempts = 2

            result = scheduler.requeue_task("t1")
            assert result is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_priority_computation_base(self, scheduler, state_manager):
        """Priority should include base priority for task type."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            priority = scheduler._compute_priority(task)
            assert priority >= 50  # Base for exploration
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_priority_computation_module_weight(self, scheduler, state_manager):
        """Priority should scale by module weight."""
        await state_manager.startup()
        try:
            scheduler.module_weights["explorer"] = 2.0

            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=0)
            priority = scheduler._compute_priority(task)
            assert priority >= 100  # 50 * 2.0
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_priority_computation_comment_boost(self, scheduler, state_manager):
        """Address comment tasks should get priority boost."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="documenter",
                       task_type="address_comment", priority=0)
            priority = scheduler._compute_priority(task)
            assert priority >= 95  # 70 base + 25 boost
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, scheduler, state_manager):
        """get_queue_stats should return accurate statistics."""
        await state_manager.startup()
        try:
            task1 = Task(id="t1", key="k1", module="explorer",
                        task_type="exploration", priority=0)
            task2 = Task(id="t2", key="k2", module="documenter",
                        task_type="incorporate_insight", priority=0)
            scheduler.submit(task1)
            scheduler.submit(task2)
            scheduler.mark_task_running("t1")

            stats = scheduler.get_queue_stats()

            assert stats["total"] == 2
            assert stats["by_state"]["running"] == 1
            assert stats["by_state"]["pending"] == 1
            assert stats["by_module"]["explorer"] == 1
            assert stats["by_module"]["documenter"] == 1
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_set_module_weights(self, scheduler, state_manager):
        """set_module_weights should update weights."""
        await state_manager.startup()
        try:
            scheduler.set_module_weights({"explorer": 0.5, "documenter": 1.5})

            assert scheduler.module_weights["explorer"] == 0.5
            assert scheduler.module_weights["documenter"] == 1.5
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_update_system_state(self, scheduler, state_manager):
        """update_system_state should update scheduler state."""
        await state_manager.startup()
        try:
            scheduler.update_system_state(blessed_pending=10, comments_pending=5)

            assert scheduler.blessed_insights_pending == 10
            assert scheduler.comments_pending == 5
        finally:
            await state_manager.shutdown()
