"""
State Manager for the Orchestrator.

Implements Snapshot + Delta WAL persistence as per v4.0 design:
1. WAL (Deltas): Records only changes since last checkpoint
2. Checkpoint: Full in-memory state serialized every 60s
3. Atomic Writes: .tmp -> fsync -> replace

This prevents write amplification (logging 100MB of history for 1KB change).
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any

from shared.logging import get_logger

from .models import Task, TaskState

log = get_logger("orchestrator", "state")


class WALOperation(str, Enum):
    """Types of WAL operations."""
    TASK_ADDED = "task_added"
    TASK_STATE_CHANGE = "task_state_change"
    TASK_UPDATED = "task_updated"
    TASK_REMOVED = "task_removed"
    QUOTA_UPDATED = "quota_updated"
    CONFIG_UPDATED = "config_updated"


@dataclass
class WALEntry:
    """A single WAL entry."""
    timestamp: float
    operation: WALOperation
    data: dict

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WALEntry":
        return cls(
            timestamp=data["timestamp"],
            operation=WALOperation(data["operation"]),
            data=data["data"],
        )


@dataclass
class OrchestratorState:
    """
    Full orchestrator state for checkpointing.
    """
    # All tasks (by ID)
    tasks: dict[str, Task] = field(default_factory=dict)

    # Active task keys for deduplication
    active_task_keys: set[str] = field(default_factory=set)

    # Quota usage tracking
    quota_usage: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "gemini_deep": {"explorer": 0, "documenter": 0},
        "chatgpt_pro": {"explorer": 0, "documenter": 0},
        "claude_extended": {"explorer": 0, "documenter": 0},
    })

    # Last quota reset date (YYYY-MM-DD)
    quota_reset_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    # Scheduler state
    priority_weights: dict[str, float] = field(default_factory=lambda: {
        "explorer": 40.0,
        "documenter": 40.0,
        "researcher": 20.0,
    })

    # Timestamps
    started_at: float = field(default_factory=lambda: time.time())
    last_checkpoint: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict:
        return {
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "active_task_keys": list(self.active_task_keys),
            "quota_usage": self.quota_usage,
            "quota_reset_date": self.quota_reset_date,
            "priority_weights": self.priority_weights,
            "started_at": self.started_at,
            "last_checkpoint": self.last_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrchestratorState":
        tasks = {}
        for task_id, task_data in data.get("tasks", {}).items():
            tasks[task_id] = Task.from_dict(task_data)

        return cls(
            tasks=tasks,
            active_task_keys=set(data.get("active_task_keys", [])),
            quota_usage=data.get("quota_usage", {
                "gemini_deep": {"explorer": 0, "documenter": 0},
                "chatgpt_pro": {"explorer": 0, "documenter": 0},
                "claude_extended": {"explorer": 0, "documenter": 0},
            }),
            quota_reset_date=data.get("quota_reset_date", datetime.now().strftime("%Y-%m-%d")),
            priority_weights=data.get("priority_weights", {
                "explorer": 40.0,
                "documenter": 40.0,
                "researcher": 20.0,
            }),
            started_at=data.get("started_at", time.time()),
            last_checkpoint=data.get("last_checkpoint", time.time()),
        )


class StateManager:
    """
    Manages orchestrator state with WAL + checkpoint persistence.
    """

    def __init__(self, checkpoint_dir: str = "data/orchestrator", checkpoint_interval: int = 60):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.checkpoint_dir / "checkpoint.json"
        self.wal_path = self.checkpoint_dir / "wal.jsonl"

        self.checkpoint_interval = checkpoint_interval
        self._running = False
        self._checkpoint_task: Optional[asyncio.Task] = None

        # In-memory state
        self.state = OrchestratorState()

        # WAL buffer (flushed on each operation)
        self._wal_file: Optional[Any] = None

    async def startup(self):
        """Load state from checkpoint + WAL and start checkpoint loop."""
        # Load from checkpoint
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.state = OrchestratorState.from_dict(data)
                log.info("orchestrator.state.checkpoint_loaded",
                        tasks=len(self.state.tasks),
                        last_checkpoint=self.state.last_checkpoint)
            except Exception as e:
                log.error("orchestrator.state.checkpoint_load_failed", error=str(e))
                self.state = OrchestratorState()

        # Replay WAL
        if self.wal_path.exists():
            try:
                entries_replayed = 0
                with open(self.wal_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = WALEntry.from_dict(json.loads(line))
                        self._apply_wal_entry(entry)
                        entries_replayed += 1

                if entries_replayed > 0:
                    log.info("orchestrator.state.wal_replayed", entries=entries_replayed)

            except Exception as e:
                log.error("orchestrator.state.wal_replay_failed", error=str(e))

        # Check for quota reset (new day)
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.quota_reset_date != today:
            log.info("orchestrator.state.quota_reset",
                    old_date=self.state.quota_reset_date,
                    new_date=today)
            self.state.quota_reset_date = today
            self.state.quota_usage = {
                "gemini_deep": {"explorer": 0, "documenter": 0},
                "chatgpt_pro": {"explorer": 0, "documenter": 0},
                "claude_extended": {"explorer": 0, "documenter": 0},
            }

        # Mark any RUNNING tasks as needing recovery
        for task in self.state.tasks.values():
            if task.state == TaskState.RUNNING:
                log.warning("orchestrator.state.task_needs_recovery",
                           task_id=task.id, task_type=task.task_type)
                # Keep as RUNNING for reconciliation with Pool

        # Open WAL file for appending
        self._wal_file = open(self.wal_path, "a", encoding="utf-8")

        # Start checkpoint loop
        self._running = True
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())

        log.info("orchestrator.state.started",
                tasks=len(self.state.tasks),
                active_keys=len(self.state.active_task_keys))

    async def shutdown(self):
        """Save final checkpoint and stop."""
        self._running = False

        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        # Final checkpoint
        self._save_checkpoint()

        # Close WAL
        if self._wal_file:
            self._wal_file.close()
            self._wal_file = None

        log.info("orchestrator.state.shutdown")

    async def _checkpoint_loop(self):
        """Periodically save full checkpoint."""
        while self._running:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                self._save_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(e, "orchestrator.state.checkpoint_error", {})

    def _save_checkpoint(self):
        """Atomically save full state to checkpoint file."""
        self.state.last_checkpoint = time.time()

        # Write to temp file
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_path, self.checkpoint_path)

            # Clear WAL after successful checkpoint
            if self._wal_file:
                self._wal_file.close()
            # Truncate WAL
            self._wal_file = open(self.wal_path, "w", encoding="utf-8")

            log.debug("orchestrator.state.checkpoint_saved",
                     tasks=len(self.state.tasks))

        except Exception as e:
            log.error("orchestrator.state.checkpoint_save_failed", error=str(e))
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()

    def _write_wal(self, operation: WALOperation, data: dict):
        """Write a WAL entry."""
        if not self._wal_file:
            return

        entry = WALEntry(
            timestamp=time.time(),
            operation=operation,
            data=data,
        )

        try:
            self._wal_file.write(json.dumps(entry.to_dict()) + "\n")
            self._wal_file.flush()
        except Exception as e:
            log.error("orchestrator.state.wal_write_failed", error=str(e))

    def _apply_wal_entry(self, entry: WALEntry):
        """Apply a WAL entry to in-memory state."""
        if entry.operation == WALOperation.TASK_ADDED:
            task = Task.from_dict(entry.data)
            self.state.tasks[task.id] = task
            self.state.active_task_keys.add(task.key)

        elif entry.operation == WALOperation.TASK_STATE_CHANGE:
            task_id = entry.data["task_id"]
            new_state = TaskState(entry.data["new_state"])
            if task_id in self.state.tasks:
                self.state.tasks[task_id].state = new_state
                self.state.tasks[task_id].updated_at = entry.timestamp
                # Remove from active keys if terminal
                if new_state in (TaskState.COMPLETED, TaskState.FAILED):
                    task_key = self.state.tasks[task_id].key
                    self.state.active_task_keys.discard(task_key)

        elif entry.operation == WALOperation.TASK_UPDATED:
            task_id = entry.data["task_id"]
            if task_id in self.state.tasks:
                task = self.state.tasks[task_id]
                for key, value in entry.data.get("updates", {}).items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.updated_at = entry.timestamp

        elif entry.operation == WALOperation.TASK_REMOVED:
            task_id = entry.data["task_id"]
            if task_id in self.state.tasks:
                task_key = self.state.tasks[task_id].key
                del self.state.tasks[task_id]
                self.state.active_task_keys.discard(task_key)

        elif entry.operation == WALOperation.QUOTA_UPDATED:
            quota_type = entry.data["quota_type"]
            module = entry.data["module"]
            value = entry.data["value"]
            if quota_type in self.state.quota_usage:
                self.state.quota_usage[quota_type][module] = value

    # ==================== Public API ====================

    def add_task(self, task: Task) -> bool:
        """
        Add a task if its key is not already active.

        Returns True if task was added, False if duplicate.
        """
        if task.key in self.state.active_task_keys:
            return False

        self.state.tasks[task.id] = task
        self.state.active_task_keys.add(task.key)

        self._write_wal(WALOperation.TASK_ADDED, task.to_dict())

        log.info("orchestrator.state.task_added",
                task_id=task.id, key=task.key, task_type=task.task_type)
        return True

    def update_task_state(self, task_id: str, new_state: TaskState):
        """Update a task's state."""
        if task_id not in self.state.tasks:
            return

        task = self.state.tasks[task_id]
        old_state = task.state
        task.state = new_state
        task.updated_at = time.time()

        # Remove from active keys if terminal
        if new_state in (TaskState.COMPLETED, TaskState.FAILED):
            self.state.active_task_keys.discard(task.key)

        self._write_wal(WALOperation.TASK_STATE_CHANGE, {
            "task_id": task_id,
            "old_state": old_state.value,
            "new_state": new_state.value,
        })

        log.info("orchestrator.state.task_state_changed",
                task_id=task_id, old_state=old_state.value, new_state=new_state.value)

    def update_task(self, task_id: str, **updates):
        """Update task fields."""
        if task_id not in self.state.tasks:
            return

        task = self.state.tasks[task_id]
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = time.time()

        self._write_wal(WALOperation.TASK_UPDATED, {
            "task_id": task_id,
            "updates": updates,
        })

    def remove_task(self, task_id: str):
        """Remove a task."""
        if task_id not in self.state.tasks:
            return

        task = self.state.tasks[task_id]
        del self.state.tasks[task_id]
        self.state.active_task_keys.discard(task.key)

        self._write_wal(WALOperation.TASK_REMOVED, {"task_id": task_id})

        log.info("orchestrator.state.task_removed", task_id=task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.state.tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks."""
        return list(self.state.tasks.values())

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return [t for t in self.state.tasks.values() if t.state == TaskState.PENDING]

    def get_running_tasks(self) -> list[Task]:
        """Get all running tasks."""
        return [t for t in self.state.tasks.values() if t.state == TaskState.RUNNING]

    def is_key_active(self, key: str) -> bool:
        """Check if a task key is active."""
        return key in self.state.active_task_keys

    def update_quota(self, quota_type: str, module: str, value: int):
        """Update quota usage."""
        if quota_type not in self.state.quota_usage:
            self.state.quota_usage[quota_type] = {}

        self.state.quota_usage[quota_type][module] = value

        self._write_wal(WALOperation.QUOTA_UPDATED, {
            "quota_type": quota_type,
            "module": module,
            "value": value,
        })

    def get_quota_usage(self, quota_type: str, module: str) -> int:
        """Get quota usage for a module."""
        return self.state.quota_usage.get(quota_type, {}).get(module, 0)

    def increment_quota(self, quota_type: str, module: str) -> int:
        """Increment quota usage and return new value."""
        current = self.get_quota_usage(quota_type, module)
        new_value = current + 1
        self.update_quota(quota_type, module, new_value)
        return new_value
