"""Request queue management for the Browser Pool Service."""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from heapq import heappush, heappop

from shared.logging import get_logger
from .models import Priority, SendRequest, SendOptions, Backend

log = get_logger("pool", "queue")

# Default location for persisted queue state
DEFAULT_QUEUE_STATE_FILE = Path(__file__).parent.parent / "queue_state.json"


@dataclass(order=True)
class QueuedRequest:
    """A request waiting in the queue."""
    priority_value: int  # Lower = higher priority (for heap)
    timestamp: datetime = field(compare=True)
    request: SendRequest = field(compare=False)
    future: asyncio.Future = field(compare=False)
    request_id: str = field(compare=False, default="")

    @classmethod
    def create(cls, request: SendRequest, future: asyncio.Future, request_id: str):
        """Create a queued request with proper priority ordering."""
        # Map priority to numeric value (lower = higher priority)
        priority_map = {
            Priority.HIGH: 0,
            Priority.NORMAL: 1,
            Priority.LOW: 2,
        }
        return cls(
            priority_value=priority_map[request.options.priority],
            timestamp=datetime.now(),
            request=request,
            future=future,
            request_id=request_id,
        )

    def to_dict(self) -> dict:
        """Serialize to dict for persistence (excludes Future)."""
        return {
            "priority_value": self.priority_value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "request": {
                "backend": self.request.backend.value,
                "prompt": self.request.prompt,
                "thread_id": self.request.thread_id,
                "options": {
                    "deep_mode": self.request.options.deep_mode,
                    "timeout_seconds": self.request.options.timeout_seconds,
                    "priority": self.request.options.priority.value,
                    "new_chat": self.request.options.new_chat,
                },
            },
        }

    @classmethod
    def from_dict(cls, data: dict, future: asyncio.Future) -> "QueuedRequest":
        """Deserialize from dict, with a new Future."""
        options = SendOptions(
            deep_mode=data["request"]["options"]["deep_mode"],
            timeout_seconds=data["request"]["options"]["timeout_seconds"],
            priority=Priority(data["request"]["options"]["priority"]),
            new_chat=data["request"]["options"]["new_chat"],
        )
        request = SendRequest(
            backend=Backend(data["request"]["backend"]),
            prompt=data["request"]["prompt"],
            thread_id=data["request"].get("thread_id"),
            options=options,
        )
        return cls(
            priority_value=data["priority_value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            request=request,
            future=future,
            request_id=data["request_id"],
        )


class RequestQueue:
    """
    Priority queue for a single backend.

    Requests are processed in priority order (HIGH > NORMAL > LOW),
    with FIFO ordering within the same priority level.

    Optionally persists queue to disk for recovery after restarts.
    """

    def __init__(self, backend: str, max_depth: int = 10, persist_callback=None):
        self.backend = backend
        self.max_depth = max_depth
        self._queue: list[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._request_counter = 0
        self._persist_callback = persist_callback  # Called when queue changes

    async def enqueue(self, request: SendRequest) -> asyncio.Future:
        """
        Add a request to the queue.

        Returns a Future that will be resolved with the response.
        Raises if queue is full.
        """
        async with self._lock:
            if len(self._queue) >= self.max_depth:
                raise QueueFullError(f"Queue for {self.backend} is full ({self.max_depth} items)")

            self._request_counter += 1
            request_id = f"{self.backend}-{self._request_counter}"

            future = asyncio.get_event_loop().create_future()
            queued = QueuedRequest.create(request, future, request_id)
            heappush(self._queue, queued)

            log.info(
                "pool.queue.enqueue",
                backend=self.backend,
                request_id=request_id,
                priority=request.options.priority.value,
                queue_depth=len(self._queue),
            )

            # Persist queue after modification
            if self._persist_callback:
                self._persist_callback()

            return future

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get the next request from the queue (highest priority first)."""
        async with self._lock:
            if not self._queue:
                return None
            item = heappop(self._queue)

            # Persist queue after modification
            if self._persist_callback:
                self._persist_callback()

            return item

    def get_all_serialized(self) -> list[dict]:
        """Get all queued items as serialized dicts (for persistence)."""
        return [q.to_dict() for q in self._queue]

    def restore_from_serialized(self, items: list[dict]) -> int:
        """
        Restore queue items from serialized data.

        Creates new Futures for each item since old ones are not serializable.
        Returns number of restored items.
        """
        restored = 0
        for item_data in items:
            # Track counter from ALL items regardless of restoration success
            # This prevents ID collisions even if some items fail to restore
            try:
                request_id = item_data.get("request_id", "")
                counter = int(request_id.split("-")[-1])
                if counter > self._request_counter:
                    self._request_counter = counter
            except (ValueError, IndexError, AttributeError):
                pass

            try:
                future = asyncio.get_event_loop().create_future()
                queued = QueuedRequest.from_dict(item_data, future)
                heappush(self._queue, queued)
                restored += 1

            except Exception as e:
                log.error("pool.queue.restore_item_failed",
                         backend=self.backend,
                         error=str(e))

        if restored > 0:
            log.info("pool.queue.restored",
                    backend=self.backend,
                    count=restored)

        return restored

    @property
    def depth(self) -> int:
        """Current queue depth."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0


class QueueFullError(Exception):
    """Raised when a queue is at capacity."""
    pass


class QueueManager:
    """Manages queues for all backends with optional persistence."""

    def __init__(self, config: dict, state_file: Optional[Path] = None):
        max_depth = config.get("queue", {}).get("max_depth_per_backend", 10)
        self.state_file = state_file or DEFAULT_QUEUE_STATE_FILE

        # Create queues with persistence callback
        self.queues = {
            "gemini": RequestQueue("gemini", max_depth, self._persist),
            "chatgpt": RequestQueue("chatgpt", max_depth, self._persist),
            "claude": RequestQueue("claude", max_depth, self._persist),
        }

    def _persist(self):
        """Save all queue states to disk."""
        try:
            state = {}
            for backend, queue in self.queues.items():
                state[backend] = queue.get_all_serialized()

            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            log.error("pool.queue.persist_failed", error=str(e))

    def restore_pending(self) -> dict[str, int]:
        """
        Restore pending queue items from disk.

        Call this on startup before workers start processing.
        Returns dict mapping backend to count of restored items.
        """
        restored = {}

        if not self.state_file.exists():
            return restored

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            for backend, items in state.items():
                if backend in self.queues and items:
                    count = self.queues[backend].restore_from_serialized(items)
                    if count > 0:
                        restored[backend] = count

            # Clear the state file after successful restore
            if restored:
                log.info("pool.queue.restore_complete",
                        total=sum(restored.values()),
                        by_backend=restored)
                self.state_file.unlink()

        except Exception as e:
            log.error("pool.queue.restore_failed", error=str(e))

        return restored

    def get_queue(self, backend: str) -> RequestQueue:
        """Get the queue for a backend."""
        return self.queues.get(backend)

    def get_depths(self) -> dict[str, int]:
        """Get queue depths for all backends."""
        return {name: q.depth for name, q in self.queues.items()}
