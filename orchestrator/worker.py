"""
JIT Worker Loops for the Orchestrator.

Each backend has a dedicated worker loop that:
1. Polls for available work (highest priority pending task)
2. Checks if backend is free
3. Submits task via Pool's submit_immediate endpoint
4. Waits for completion and updates task state

Based on v4.0 design spec Section 3.3: JIT Model.
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass

import httpx

from shared.logging import get_logger

from .models import Task, TaskState
from .scheduler import Scheduler
from .allocator import QuotaAllocator

log = get_logger("orchestrator", "worker")


@dataclass
class WorkerConfig:
    """Configuration for a JIT worker."""
    backend: str
    pool_base_url: str = "http://localhost:8765"
    poll_interval: float = 2.0  # Seconds between checks when idle
    busy_check_interval: float = 5.0  # Seconds between busy checks
    request_timeout: float = 600.0  # 10 minutes max for LLM response
    max_consecutive_errors: int = 5


class JITWorker:
    """
    Just-In-Time worker for a single backend.

    Continuously polls for work and submits to Pool when backend is free.
    """

    def __init__(
        self,
        config: WorkerConfig,
        scheduler: Scheduler,
        allocator: QuotaAllocator,
        prompt_builder: Callable[[Task], Awaitable[dict]] = None,
        result_handler: Callable[[Task, dict], Awaitable[None]] = None,
    ):
        self.config = config
        self.scheduler = scheduler
        self.allocator = allocator
        self.prompt_builder = prompt_builder or self._default_prompt_builder
        self.result_handler = result_handler or self._default_result_handler

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_task: Optional[Task] = None
        self._consecutive_errors = 0
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start the worker loop."""
        if self._running:
            return

        self._running = True
        self._client = httpx.AsyncClient(timeout=self.config.request_timeout)
        self._task = asyncio.create_task(self._run_loop())

        log.info("orchestrator.worker.started", backend=self.config.backend)

    async def stop(self):
        """Stop the worker loop gracefully."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        log.info("orchestrator.worker.stopped", backend=self.config.backend)

    async def _run_loop(self):
        """Main worker loop."""
        while self._running:
            try:
                # Check for excessive errors
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    log.error("orchestrator.worker.too_many_errors",
                             backend=self.config.backend,
                             count=self._consecutive_errors)
                    await asyncio.sleep(60)  # Back off for 1 minute
                    self._consecutive_errors = 0
                    continue

                # Get next task for this backend
                task = self.scheduler.get_next_task(backend=self.config.backend)
                if not task:
                    await asyncio.sleep(self.config.poll_interval)
                    continue

                # Check if backend is free
                if not await self._is_backend_free():
                    await asyncio.sleep(self.config.busy_check_interval)
                    continue

                # Check quota if deep mode required
                if task.requires_deep_mode:
                    quota_type = self._get_quota_type()
                    result = self.allocator.can_allocate(quota_type, task.module)
                    if not result.granted:
                        log.debug("orchestrator.worker.quota_blocked",
                                 backend=self.config.backend,
                                 task_id=task.id,
                                 reason=result.message)
                        # Skip this task, try next
                        await asyncio.sleep(self.config.poll_interval)
                        continue

                # Execute the task
                await self._execute_task(task)
                self._consecutive_errors = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_errors += 1
                log.exception(e, "orchestrator.worker.loop_error", {
                    "backend": self.config.backend,
                    "consecutive_errors": self._consecutive_errors,
                })
                await asyncio.sleep(self.config.poll_interval)

    async def _is_backend_free(self) -> bool:
        """Check if the backend is available for work."""
        try:
            url = f"{self.config.pool_base_url}/backend/{self.config.backend}/busy"
            response = await self._client.get(url)

            if response.status_code == 200:
                data = response.json()
                return not data.get("busy", True)
            elif response.status_code == 404:
                # Backend not found - might not be started
                return False

            return False

        except Exception as e:
            log.warning("orchestrator.worker.busy_check_failed",
                       backend=self.config.backend, error=str(e))
            return False

    async def _execute_task(self, task: Task):
        """Execute a single task via Pool."""
        self._current_task = task

        try:
            # Mark task as running
            self.scheduler.mark_task_running(
                task.id,
                pool_request_id=f"{task.id}:{task.attempts}"
            )

            # Build prompt
            prompt_data = await self.prompt_builder(task)

            # Allocate quota if deep mode
            if task.requires_deep_mode:
                quota_type = self._get_quota_type()
                self.allocator.allocate(quota_type, task.module)

            # Build request payload
            payload = {
                "backend": self.config.backend,
                "prompt": prompt_data.get("prompt", ""),
                "idempotency_token": f"{task.id}:{task.attempts}",
                "deep_mode": task.requires_deep_mode,
            }

            # Add thread navigation if continuing conversation
            if task.conversation:
                if task.conversation.external_thread_id:
                    payload["thread_id"] = task.conversation.external_thread_id
                if task.conversation.thread_title:
                    payload["thread_title"] = task.conversation.thread_title

            # Add images if present
            if "images" in prompt_data:
                payload["images"] = prompt_data["images"]

            # Submit to Pool
            log.info("orchestrator.worker.submitting",
                    backend=self.config.backend,
                    task_id=task.id,
                    task_type=task.task_type,
                    deep_mode=task.requires_deep_mode)

            url = f"{self.config.pool_base_url}/submit_immediate"
            response = await self._client.post(url, json=payload)

            if response.status_code != 200:
                error_msg = f"Pool returned {response.status_code}: {response.text}"
                self.scheduler.mark_task_failed(task.id, error_msg)
                log.error("orchestrator.worker.submit_failed",
                         backend=self.config.backend,
                         task_id=task.id,
                         status=response.status_code)
                return

            result = response.json()

            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                # Check if it's a retryable error
                if "busy" in error_msg.lower():
                    # Backend became busy, requeue
                    self.scheduler.requeue_task(task.id)
                else:
                    self.scheduler.mark_task_failed(task.id, error_msg)
                return

            # Update conversation state
            if result.get("thread_url") or result.get("thread_title"):
                task.update_conversation(
                    thread_url=result.get("thread_url"),
                    thread_title=result.get("thread_title"),
                    backend=self.config.backend
                )

            # Handle result via callback
            await self.result_handler(task, result)

            # Mark completed (handler may have already done this)
            if task.state == TaskState.RUNNING:
                self.scheduler.mark_task_completed(task.id, result.get("response"))

            log.info("orchestrator.worker.task_completed",
                    backend=self.config.backend,
                    task_id=task.id,
                    task_type=task.task_type)

        except asyncio.TimeoutError:
            log.error("orchestrator.worker.timeout",
                     backend=self.config.backend,
                     task_id=task.id)
            # Requeue for retry
            if not self.scheduler.requeue_task(task.id):
                self.scheduler.mark_task_failed(task.id, "Timeout after max retries")

        except Exception as e:
            log.exception(e, "orchestrator.worker.execution_error", {
                "backend": self.config.backend,
                "task_id": task.id,
            })
            self.scheduler.mark_task_failed(task.id, str(e))

        finally:
            self._current_task = None

    def _get_quota_type(self) -> str:
        """Map backend to quota type."""
        quota_map = {
            "gemini": "gemini_deep",
            "chatgpt": "chatgpt_pro",
            "claude": "claude_extended",
        }
        return quota_map.get(self.config.backend, f"{self.config.backend}_deep")

    async def _default_prompt_builder(self, task: Task) -> dict:
        """Default prompt builder - just uses payload prompt."""
        return {
            "prompt": task.payload.get("prompt", ""),
            "images": task.payload.get("images", []),
        }

    async def _default_result_handler(self, task: Task, result: dict):
        """Default result handler - stores response in task."""
        task.result = result.get("response")


class WorkerPool:
    """
    Manages multiple JIT workers, one per backend.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        allocator: QuotaAllocator,
        pool_base_url: str = "http://localhost:8765",
        backends: list[str] = None,
        prompt_builder: Callable[[Task], Awaitable[dict]] = None,
        result_handler: Callable[[Task, dict], Awaitable[None]] = None,
    ):
        self.scheduler = scheduler
        self.allocator = allocator
        self.pool_base_url = pool_base_url
        self.backends = backends or ["gemini", "chatgpt", "claude"]
        self.prompt_builder = prompt_builder
        self.result_handler = result_handler

        self.workers: dict[str, JITWorker] = {}

    async def start(self):
        """Start all workers."""
        for backend in self.backends:
            config = WorkerConfig(
                backend=backend,
                pool_base_url=self.pool_base_url,
            )
            worker = JITWorker(
                config=config,
                scheduler=self.scheduler,
                allocator=self.allocator,
                prompt_builder=self.prompt_builder,
                result_handler=self.result_handler,
            )
            self.workers[backend] = worker
            await worker.start()

        log.info("orchestrator.worker_pool.started",
                backends=self.backends)

    async def stop(self):
        """Stop all workers."""
        for worker in self.workers.values():
            await worker.stop()

        self.workers.clear()
        log.info("orchestrator.worker_pool.stopped")

    def get_worker(self, backend: str) -> Optional[JITWorker]:
        """Get worker for a specific backend."""
        return self.workers.get(backend)

    def get_status(self) -> dict:
        """Get status of all workers."""
        return {
            backend: {
                "running": worker._running,
                "current_task": worker._current_task.id if worker._current_task else None,
                "consecutive_errors": worker._consecutive_errors,
            }
            for backend, worker in self.workers.items()
        }
