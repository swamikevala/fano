"""
Scheduler for the Orchestrator.

Manages the priority queue and task selection based on v4.0 design:
- Submit-If-Absent: Prevents duplicate tasks
- Failure Cache: Prevents infinite retry loops
- Priority Computation: Multi-factor scoring
"""

import heapq
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from shared.logging import get_logger

from .models import Task, TaskState, get_base_priority
from .state import StateManager

log = get_logger("orchestrator", "scheduler")


class RecentFailureCache:
    """
    LRU cache of recently failed task keys.

    Prevents infinite retry loops for deterministic failures
    (e.g., "Context too long").
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: OrderedDict[str, float] = OrderedDict()

    def add(self, key: str):
        """Add a key to the failure cache."""
        # Remove if already exists (to update position)
        if key in self._cache:
            del self._cache[key]

        self._cache[key] = time.time()

        # Evict oldest if over max size
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def contains(self, key: str) -> bool:
        """Check if key is in cache and not expired."""
        if key not in self._cache:
            return False

        # Check TTL
        added_at = self._cache[key]
        if time.time() - added_at > self.ttl_seconds:
            del self._cache[key]
            return False

        return True

    def remove(self, key: str):
        """Remove a key from the cache."""
        if key in self._cache:
            del self._cache[key]

    def clear_expired(self):
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v > self.ttl_seconds]
        for key in expired:
            del self._cache[key]


@dataclass
class ScoredTask:
    """Task with computed priority score for heap ordering."""
    score: float
    task_id: str
    task: Task

    def __lt__(self, other):
        # Higher score = higher priority (negate for min-heap)
        return self.score > other.score


class Scheduler:
    """
    Priority-based task scheduler.

    Implements:
    - Submit-If-Absent for deduplication
    - Failure cache for loop prevention
    - Multi-factor priority scoring
    """

    def __init__(self, state_manager: StateManager, config: dict = None):
        self.state = state_manager
        self.config = config or {}

        # Failure cache
        failure_cache_ttl = self.config.get("failure_cache_ttl", 3600)
        self.failure_cache = RecentFailureCache(ttl_seconds=failure_cache_ttl)

        # Priority weights (can be updated by LLM consultation)
        self.module_weights: dict[str, float] = {
            "explorer": 1.0,
            "documenter": 1.0,
            "researcher": 1.0,
        }

        # System state for priority computation
        self.blessed_insights_pending = 0
        self.comments_pending = 0

    def submit(self, task: Task, force_retry: bool = False) -> bool:
        """
        Submit a task if not already active.

        Returns True if task was accepted, False if rejected.
        """
        # Check failure cache
        if not force_retry and self.failure_cache.contains(task.key):
            log.debug("orchestrator.scheduler.rejected_by_failure_cache",
                     key=task.key, task_type=task.task_type)
            return False

        # Check if key is already active (Submit-If-Absent)
        if self.state.is_key_active(task.key):
            log.debug("orchestrator.scheduler.rejected_duplicate",
                     key=task.key, task_type=task.task_type)
            return False

        # Compute priority score
        task.priority = self._compute_priority(task)

        # Add to state
        if self.state.add_task(task):
            log.info("orchestrator.scheduler.task_submitted",
                    task_id=task.id, key=task.key,
                    task_type=task.task_type, priority=task.priority)
            return True

        return False

    def get_next_task(self, backend: str = None) -> Optional[Task]:
        """
        Get the highest priority pending task.

        Args:
            backend: Optional backend filter (for JIT worker loops)

        Returns:
            Highest priority pending task, or None if queue empty
        """
        pending = self.state.get_pending_tasks()
        if not pending:
            return None

        # Filter by backend preference if specified
        if backend:
            # Prefer tasks that want this backend, or have no preference
            preferred = [t for t in pending
                        if t.preferred_backend == backend or t.preferred_backend is None]
            if preferred:
                pending = preferred

        # Compute current scores and find best
        best_task = None
        best_score = float('-inf')

        for task in pending:
            score = self._compute_priority(task)
            if score > best_score:
                best_score = score
                best_task = task

        return best_task

    def mark_task_failed(self, task_id: str, error: str, add_to_cache: bool = True):
        """
        Mark a task as failed.

        Args:
            task_id: Task ID
            error: Error message
            add_to_cache: If True, add key to failure cache
        """
        task = self.state.get_task(task_id)
        if not task:
            return

        task.mark_failed(error)
        self.state.update_task_state(task_id, TaskState.FAILED)

        if add_to_cache:
            self.failure_cache.add(task.key)
            log.info("orchestrator.scheduler.task_failed_cached",
                    task_id=task_id, key=task.key, error=error)
        else:
            log.info("orchestrator.scheduler.task_failed",
                    task_id=task_id, error=error)

    def mark_task_completed(self, task_id: str, result: Optional[str] = None):
        """Mark a task as completed."""
        task = self.state.get_task(task_id)
        if not task:
            return

        task.mark_completed(result)
        self.state.update_task_state(task_id, TaskState.COMPLETED)

        # Remove from failure cache if it was there
        self.failure_cache.remove(task.key)

        log.info("orchestrator.scheduler.task_completed", task_id=task_id)

    def mark_task_running(self, task_id: str, pool_request_id: Optional[str] = None):
        """Mark a task as running."""
        task = self.state.get_task(task_id)
        if not task:
            return

        task.mark_running(pool_request_id)
        self.state.update_task_state(task_id, TaskState.RUNNING)

        log.info("orchestrator.scheduler.task_running",
                task_id=task_id, pool_request_id=pool_request_id)

    def requeue_task(self, task_id: str):
        """
        Requeue a failed/running task for retry.

        Returns True if task was requeued, False if max retries exceeded.
        """
        task = self.state.get_task(task_id)
        if not task:
            return False

        if not task.can_retry():
            log.warning("orchestrator.scheduler.max_retries_exceeded",
                       task_id=task_id, attempts=task.attempts)
            return False

        task.state = TaskState.PENDING
        task.pool_request_id = None
        task.updated_at = time.time()
        self.state.update_task_state(task_id, TaskState.PENDING)

        log.info("orchestrator.scheduler.task_requeued",
                task_id=task_id, attempts=task.attempts)
        return True

    def update_system_state(self, blessed_pending: int = None, comments_pending: int = None):
        """Update system state used for priority computation."""
        if blessed_pending is not None:
            self.blessed_insights_pending = blessed_pending
        if comments_pending is not None:
            self.comments_pending = comments_pending

    def set_module_weights(self, weights: dict[str, float]):
        """Set module priority weights (from LLM consultation)."""
        self.module_weights.update(weights)
        log.info("orchestrator.scheduler.weights_updated", weights=weights)

    def _compute_priority(self, task: Task) -> int:
        """
        Compute priority score for a task.

        Based on v4.0 design spec priority algorithm.
        """
        # Start with base priority for task type
        score = float(get_base_priority(task.task_type))

        # Factor 1: Module weight
        module_weight = self.module_weights.get(task.module, 1.0)
        score *= module_weight

        # Factor 2: Backlog pressure (for documenter)
        if task.module == "documenter":
            if self.blessed_insights_pending > 20:
                score += 30
            elif self.blessed_insights_pending > 10:
                score += 15

        # Factor 3: Starvation prevention
        # Boost priority based on how long task has been waiting
        if task.state == TaskState.PENDING:
            wait_time = time.time() - task.created_at
            hours_waiting = wait_time / 3600
            score += min(hours_waiting * 2, 20)  # Cap at +20

        # Factor 4: Comment responsiveness
        if task.task_type == "address_comment":
            score += 25

        # Factor 5: Seed priority (for explorer explorations)
        if task.task_type == "exploration":
            seed_priority = task.payload.get("seed_priority", 0)
            score += seed_priority * 10

        # Factor 6: Deep mode penalty (if quota might be exhausted)
        # This is checked by the allocator, not here

        return int(score)

    def get_queue_stats(self) -> dict:
        """Get statistics about the task queue."""
        all_tasks = self.state.get_all_tasks()

        by_state = {}
        by_module = {}
        by_type = {}

        for task in all_tasks:
            # By state
            state_key = task.state.value
            by_state[state_key] = by_state.get(state_key, 0) + 1

            # By module
            by_module[task.module] = by_module.get(task.module, 0) + 1

            # By type
            by_type[task.task_type] = by_type.get(task.task_type, 0) + 1

        return {
            "total": len(all_tasks),
            "by_state": by_state,
            "by_module": by_module,
            "by_type": by_type,
            "failure_cache_size": len(self.failure_cache._cache),
        }
