"""
Tests for orchestrator/state.py
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from orchestrator.models import Task, TaskState
from orchestrator.state import (
    StateManager,
    OrchestratorState,
    WALEntry,
    WALOperation,
)


class TestWALEntry:
    """Tests for WAL entry serialization."""

    def test_wal_entry_to_dict(self):
        """WALEntry should serialize to dict."""
        entry = WALEntry(
            timestamp=1234567890.123,
            operation=WALOperation.TASK_ADDED,
            data={"task_id": "t1", "key": "k1"},
        )
        data = entry.to_dict()
        assert data["timestamp"] == 1234567890.123
        assert data["operation"] == "task_added"
        assert data["data"]["task_id"] == "t1"

    def test_wal_entry_from_dict(self):
        """WALEntry should deserialize from dict."""
        data = {
            "timestamp": 1234567890.123,
            "operation": "task_state_change",
            "data": {"task_id": "t1", "new_state": "running"},
        }
        entry = WALEntry.from_dict(data)
        assert entry.timestamp == 1234567890.123
        assert entry.operation == WALOperation.TASK_STATE_CHANGE
        assert entry.data["task_id"] == "t1"


class TestOrchestratorState:
    """Tests for OrchestratorState dataclass."""

    def test_state_defaults(self):
        """OrchestratorState should have sensible defaults."""
        state = OrchestratorState()
        assert state.tasks == {}
        assert state.active_task_keys == set()
        assert "gemini_deep" in state.quota_usage
        assert state.priority_weights["explorer"] == 40.0

    def test_state_to_dict(self):
        """OrchestratorState should serialize to dict."""
        state = OrchestratorState()
        task = Task(
            id="t1", key="k1", module="explorer",
            task_type="exploration", priority=50,
        )
        state.tasks["t1"] = task
        state.active_task_keys.add("k1")

        data = state.to_dict()
        assert "t1" in data["tasks"]
        assert "k1" in data["active_task_keys"]
        assert data["quota_usage"]["gemini_deep"]["explorer"] == 0

    def test_state_from_dict(self):
        """OrchestratorState should deserialize from dict."""
        data = {
            "tasks": {
                "t1": {
                    "id": "t1", "key": "k1", "module": "explorer",
                    "task_type": "exploration", "priority": 50, "state": "pending",
                }
            },
            "active_task_keys": ["k1"],
            "quota_usage": {"gemini_deep": {"explorer": 5}},
            "quota_reset_date": "2024-01-15",
            "priority_weights": {"explorer": 50.0},
            "started_at": 1000.0,
            "last_checkpoint": 2000.0,
        }
        state = OrchestratorState.from_dict(data)
        assert "t1" in state.tasks
        assert state.tasks["t1"].key == "k1"
        assert "k1" in state.active_task_keys
        assert state.quota_usage["gemini_deep"]["explorer"] == 5


class TestStateManager:
    """Tests for StateManager."""

    @pytest.mark.asyncio
    async def test_startup_creates_directory(self, temp_dir):
        """startup should create checkpoint directory if missing."""
        new_dir = temp_dir / "new_subdir"
        manager = StateManager(checkpoint_dir=str(new_dir))
        await manager.startup()
        try:
            assert new_dir.exists()
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_add_task(self, state_manager):
        """add_task should add task to state."""
        await state_manager.startup()
        try:
            task = Task(
                id="t1", key="k1", module="explorer",
                task_type="exploration", priority=50,
            )
            result = state_manager.add_task(task)

            assert result is True
            assert "t1" in state_manager.state.tasks
            assert "k1" in state_manager.state.active_task_keys
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_add_task_duplicate_key_rejected(self, state_manager):
        """add_task should reject duplicate keys."""
        await state_manager.startup()
        try:
            task1 = Task(
                id="t1", key="same-key", module="explorer",
                task_type="exploration", priority=50,
            )
            task2 = Task(
                id="t2", key="same-key", module="explorer",
                task_type="exploration", priority=50,
            )

            assert state_manager.add_task(task1) is True
            assert state_manager.add_task(task2) is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_update_task_state(self, state_manager):
        """update_task_state should update task and write WAL."""
        await state_manager.startup()
        try:
            task = Task(
                id="t1", key="k1", module="explorer",
                task_type="exploration", priority=50,
            )
            state_manager.add_task(task)

            state_manager.update_task_state("t1", TaskState.RUNNING)
            assert state_manager.state.tasks["t1"].state == TaskState.RUNNING

            # Completing should remove from active keys
            state_manager.update_task_state("t1", TaskState.COMPLETED)
            assert state_manager.state.tasks["t1"].state == TaskState.COMPLETED
            assert "k1" not in state_manager.state.active_task_keys
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, state_manager):
        """get_pending_tasks should return only pending tasks."""
        await state_manager.startup()
        try:
            pending = Task(id="t1", key="k1", module="explorer",
                          task_type="exploration", priority=50)
            running = Task(id="t2", key="k2", module="explorer",
                          task_type="exploration", priority=50)
            running.state = TaskState.RUNNING

            state_manager.add_task(pending)
            state_manager.state.tasks["t2"] = running

            pending_tasks = state_manager.get_pending_tasks()
            assert len(pending_tasks) == 1
            assert pending_tasks[0].id == "t1"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_running_tasks(self, state_manager):
        """get_running_tasks should return only running tasks."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=50)
            state_manager.add_task(task)
            state_manager.update_task_state("t1", TaskState.RUNNING)

            running_tasks = state_manager.get_running_tasks()
            assert len(running_tasks) == 1
            assert running_tasks[0].id == "t1"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_is_key_active(self, state_manager):
        """is_key_active should check active keys set."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=50)

            assert state_manager.is_key_active("k1") is False
            state_manager.add_task(task)
            assert state_manager.is_key_active("k1") is True
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_quota_operations(self, state_manager):
        """Quota update and get should work correctly."""
        await state_manager.startup()
        try:
            assert state_manager.get_quota_usage("gemini_deep", "explorer") == 0

            state_manager.update_quota("gemini_deep", "explorer", 5)
            assert state_manager.get_quota_usage("gemini_deep", "explorer") == 5

            new_val = state_manager.increment_quota("gemini_deep", "explorer")
            assert new_val == 6
            assert state_manager.get_quota_usage("gemini_deep", "explorer") == 6
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self, temp_dir):
        """State should survive checkpoint and restore."""
        # Create state and add tasks
        manager1 = StateManager(checkpoint_dir=str(temp_dir))
        await manager1.startup()

        task = Task(id="t1", key="k1", module="explorer",
                   task_type="exploration", priority=50)
        manager1.add_task(task)
        manager1.update_quota("gemini_deep", "explorer", 7)

        # Force checkpoint
        manager1._save_checkpoint()
        await manager1.shutdown()

        # Create new manager and verify state restored
        manager2 = StateManager(checkpoint_dir=str(temp_dir))
        await manager2.startup()
        try:
            assert "t1" in manager2.state.tasks
            assert manager2.state.tasks["t1"].key == "k1"
            assert manager2.get_quota_usage("gemini_deep", "explorer") == 7
        finally:
            await manager2.shutdown()

    @pytest.mark.asyncio
    async def test_wal_replay(self, temp_dir):
        """WAL entries should be replayed on startup."""
        # Create state, add task, then shutdown without checkpoint
        manager1 = StateManager(checkpoint_dir=str(temp_dir))
        await manager1.startup()

        task = Task(id="t1", key="k1", module="explorer",
                   task_type="exploration", priority=50)
        manager1.add_task(task)

        # Close WAL file but don't checkpoint
        if manager1._wal_file:
            manager1._wal_file.close()
            manager1._wal_file = None
        manager1._running = False

        # New manager should replay WAL
        manager2 = StateManager(checkpoint_dir=str(temp_dir))
        await manager2.startup()
        try:
            assert "t1" in manager2.state.tasks
        finally:
            await manager2.shutdown()

    @pytest.mark.asyncio
    async def test_remove_task(self, state_manager):
        """remove_task should remove task from state."""
        await state_manager.startup()
        try:
            task = Task(id="t1", key="k1", module="explorer",
                       task_type="exploration", priority=50)
            state_manager.add_task(task)
            assert "t1" in state_manager.state.tasks

            state_manager.remove_task("t1")
            assert "t1" not in state_manager.state.tasks
            assert "k1" not in state_manager.state.active_task_keys
        finally:
            await state_manager.shutdown()
