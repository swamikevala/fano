"""Request queue management for the Browser Pool Service."""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from heapq import heappush, heappop

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger
from .models import Priority, SendRequest

log = get_logger("pool", "queue")


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


class RequestQueue:
    """
    Priority queue for a single backend.

    Requests are processed in priority order (HIGH > NORMAL > LOW),
    with FIFO ordering within the same priority level.
    """

    def __init__(self, backend: str, max_depth: int = 10):
        self.backend = backend
        self.max_depth = max_depth
        self._queue: list[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._request_counter = 0

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
            return future

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get the next request from the queue (highest priority first)."""
        async with self._lock:
            if not self._queue:
                return None
            return heappop(self._queue)

    async def peek(self) -> Optional[QueuedRequest]:
        """Look at the next request without removing it."""
        async with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

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
    """Manages queues for all backends."""

    def __init__(self, config: dict):
        max_depth = config.get("queue", {}).get("max_depth_per_backend", 10)
        self.queues = {
            "gemini": RequestQueue("gemini", max_depth),
            "chatgpt": RequestQueue("chatgpt", max_depth),
            "claude": RequestQueue("claude", max_depth),
        }

    def get_queue(self, backend: str) -> RequestQueue:
        """Get the queue for a backend."""
        return self.queues.get(backend)

    def get_depths(self) -> dict[str, int]:
        """Get queue depths for all backends."""
        return {name: q.depth for name, q in self.queues.items()}
