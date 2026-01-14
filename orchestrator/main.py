"""
Main Orchestrator Entry Point.

Ties together all orchestrator components:
- StateManager: WAL + checkpoint persistence
- Scheduler: Priority queue, deduplication, failure cache
- QuotaAllocator: Per-module quota budgets
- WorkerPool: JIT worker loops for each backend
- ModuleRegistry: Registered module adapters

Based on v4.0 design specification.
"""

import asyncio
import signal
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

from .models import Task, TaskState, ConversationState
from .state import StateManager
from .scheduler import Scheduler
from .allocator import QuotaAllocator
from .worker import WorkerPool, WorkerConfig
from .adapters import ModuleInterface, ModuleRegistry, PromptContext, TaskResult

log = get_logger("orchestrator", "main")


class Orchestrator:
    """
    Unified orchestrator for Explorer and Documenter modules.

    Provides:
    - Task submission and prioritization
    - Quota management for deep mode resources
    - Automatic task execution via JIT workers
    - State persistence with crash recovery
    - Module adapter integration
    """

    def __init__(
        self,
        data_dir: str = "data/orchestrator",
        pool_base_url: str = "http://localhost:8765",
        backends: list[str] = None,
        config: dict = None,
    ):
        self.data_dir = Path(data_dir)
        self.pool_base_url = pool_base_url
        self.backends = backends or ["gemini", "chatgpt", "claude"]
        self.config = config or {}

        # Initialize components
        self.state = StateManager(
            checkpoint_dir=str(self.data_dir),
            checkpoint_interval=self.config.get("checkpoint_interval", 60),
        )

        self.scheduler = Scheduler(
            state_manager=self.state,
            config=self.config.get("scheduler", {}),
        )

        self.allocator = QuotaAllocator(
            state_manager=self.state,
        )

        # Module registry for adapters
        self.registry = ModuleRegistry()

        self.workers: Optional[WorkerPool] = None
        self._work_poll_task: Optional[asyncio.Task] = None
        self._system_state_task: Optional[asyncio.Task] = None

        # Configuration
        self._work_poll_interval = self.config.get("work_poll_interval", 5.0)
        self._system_state_interval = self.config.get("system_state_interval", 30.0)

        self._running = False

    def register_module(self, module: ModuleInterface):
        """
        Register a module adapter.

        Args:
            module: Module implementing ModuleInterface
        """
        self.registry.register(module)

    async def start(self):
        """Start the orchestrator and all workers."""
        if self._running:
            return

        log.info("orchestrator.starting",
                data_dir=str(self.data_dir),
                backends=self.backends)

        # Start state manager (loads checkpoint + WAL)
        await self.state.startup()

        # Initialize all registered modules
        if not await self.registry.initialize_all():
            log.error("orchestrator.module_init_failed")
            raise RuntimeError("Failed to initialize modules")

        # Recover any tasks that were running before shutdown
        await self._recover_running_tasks()

        # Start worker pool
        self.workers = WorkerPool(
            scheduler=self.scheduler,
            allocator=self.allocator,
            pool_base_url=self.pool_base_url,
            backends=self.backends,
            prompt_builder=self._build_prompt_for_task,
            result_handler=self._handle_task_result,
        )
        await self.workers.start()

        # Start work polling loop
        self._work_poll_task = asyncio.create_task(self._work_poll_loop())

        # Start system state update loop
        self._system_state_task = asyncio.create_task(self._system_state_loop())

        self._running = True
        log.info("orchestrator.started",
                modules=len(self.registry._modules))

    async def stop(self):
        """Stop the orchestrator gracefully."""
        if not self._running:
            return

        log.info("orchestrator.stopping")
        self._running = False

        # Stop polling loops
        if self._work_poll_task:
            self._work_poll_task.cancel()
            try:
                await self._work_poll_task
            except asyncio.CancelledError:
                pass

        if self._system_state_task:
            self._system_state_task.cancel()
            try:
                await self._system_state_task
            except asyncio.CancelledError:
                pass

        # Stop workers
        if self.workers:
            await self.workers.stop()
            self.workers = None

        # Shutdown modules
        await self.registry.shutdown_all()

        # Save final state
        await self.state.shutdown()

        log.info("orchestrator.stopped")

    async def _recover_running_tasks(self):
        """Recover tasks that were running before shutdown."""
        running_tasks = self.state.get_running_tasks()

        for task in running_tasks:
            log.warning("orchestrator.recovering_task",
                       task_id=task.id,
                       task_type=task.task_type,
                       module=task.module)

            # Notify module of failed task
            module = self.registry.get_module(task.module)
            if module:
                await module.on_task_failed(task, "Task was running during shutdown")

            # Requeue if possible
            if task.can_retry():
                self.scheduler.requeue_task(task.id)
            else:
                self.scheduler.mark_task_failed(
                    task.id,
                    "Task was running during shutdown, max retries exceeded"
                )

    async def _work_poll_loop(self):
        """Periodically poll modules for pending work."""
        while self._running:
            try:
                await self._poll_modules_for_work()
                await asyncio.sleep(self._work_poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(e, "orchestrator.work_poll_error", {})
                await asyncio.sleep(self._work_poll_interval)

    async def _poll_modules_for_work(self):
        """Poll all modules for pending work and submit tasks."""
        for module in self.registry.get_all_modules():
            try:
                work_items = await module.get_pending_work()

                for item in work_items:
                    # Submit each work item as a task
                    task = self.submit_task(
                        module=module.module_name,
                        task_type=item["task_type"],
                        key=item["key"],
                        payload=item.get("payload", {}),
                        requires_deep_mode=item.get("requires_deep_mode", False),
                        preferred_backend=item.get("preferred_backend"),
                    )

                    if task:
                        log.debug("orchestrator.work_submitted",
                                 module=module.module_name,
                                 task_type=item["task_type"],
                                 task_id=task.id)

            except Exception as e:
                log.exception(e, "orchestrator.module_poll_error", {
                    "module": module.module_name,
                })

    async def _system_state_loop(self):
        """Periodically update system state from modules."""
        while self._running:
            try:
                await self._update_system_state()
                await asyncio.sleep(self._system_state_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(e, "orchestrator.system_state_error", {})
                await asyncio.sleep(self._system_state_interval)

    async def _update_system_state(self):
        """Collect system state from all modules."""
        blessed_pending = 0
        comments_pending = 0

        for module in self.registry.get_all_modules():
            try:
                state = await module.get_system_state()
                blessed_pending += state.get("blessed_insights_pending", 0)
                comments_pending += state.get("comments_pending", 0)
            except Exception as e:
                log.warning("orchestrator.module_state_error",
                           module=module.module_name, error=str(e))

        self.scheduler.update_system_state(
            blessed_pending=blessed_pending,
            comments_pending=comments_pending,
        )

    async def _build_prompt_for_task(self, task: Task) -> dict:
        """Build prompt for a task via its module adapter."""
        module = self.registry.get_module(task.module)
        if not module:
            log.warning("orchestrator.module_not_found", module=task.module)
            return {
                "prompt": task.payload.get("prompt", ""),
                "images": task.payload.get("images", []),
            }

        try:
            # Notify module task is starting
            await module.on_task_started(task)

            # Build prompt via adapter
            context = await module.build_prompt(task)

            return {
                "prompt": context.prompt,
                "images": context.images,
                "system_context": context.system_context,
            }

        except Exception as e:
            log.exception(e, "orchestrator.prompt_build_error", {
                "task_id": task.id,
                "module": task.module,
            })
            return {
                "prompt": task.payload.get("prompt", ""),
                "images": [],
            }

    async def _handle_task_result(self, task: Task, result: dict):
        """Handle task result via its module adapter."""
        module = self.registry.get_module(task.module)
        if not module:
            log.warning("orchestrator.module_not_found", module=task.module)
            return

        try:
            # Convert to TaskResult
            task_result = TaskResult(
                success=result.get("success", True),
                response=result.get("response"),
                error=result.get("error"),
                thread_url=result.get("thread_url"),
                thread_title=result.get("thread_title"),
                deep_mode_used=result.get("deep_mode_used", False),
            )

            # Dispatch to module
            handled = await module.handle_result(task, task_result)

            if not handled:
                log.warning("orchestrator.result_not_handled",
                           task_id=task.id, module=task.module)

        except Exception as e:
            log.exception(e, "orchestrator.result_handle_error", {
                "task_id": task.id,
                "module": task.module,
            })
            await module.on_task_failed(task, str(e))

    # ==================== Public API ====================

    def submit_task(
        self,
        module: str,
        task_type: str,
        key: str,
        payload: dict = None,
        requires_deep_mode: bool = False,
        preferred_backend: str = None,
        max_attempts: int = 3,
    ) -> Optional[Task]:
        """
        Submit a new task to the orchestrator.

        Args:
            module: Module name ("explorer" or "documenter")
            task_type: Type of task (e.g., "exploration", "incorporate_insight")
            key: Deduplication key (tasks with same key are deduplicated)
            payload: Task-specific data
            requires_deep_mode: Whether task needs deep mode backend
            preferred_backend: Preferred LLM backend
            max_attempts: Maximum retry attempts

        Returns:
            Task if submitted, None if rejected (duplicate or in failure cache)
        """
        import uuid

        task = Task(
            id=str(uuid.uuid4()),
            key=key,
            module=module,
            task_type=task_type,
            priority=0,  # Will be computed by scheduler
            payload=payload or {},
            requires_deep_mode=requires_deep_mode,
            preferred_backend=preferred_backend,
            max_attempts=max_attempts,
        )

        if self.scheduler.submit(task):
            return task
        return None

    def submit_continuation(
        self,
        task: Task,
        new_payload: dict = None,
    ) -> Optional[Task]:
        """
        Submit a continuation of an existing task (same thread).

        Preserves conversation state for thread navigation.
        """
        import uuid

        continuation = Task(
            id=str(uuid.uuid4()),
            key=f"{task.key}:turn{task.conversation.turn_count + 1 if task.conversation else 1}",
            module=task.module,
            task_type=task.task_type,
            priority=0,
            payload=new_payload or task.payload,
            conversation=task.conversation,  # Preserve thread info
            requires_deep_mode=task.requires_deep_mode,
            preferred_backend=task.preferred_backend,
            max_attempts=task.max_attempts,
        )

        if self.scheduler.submit(continuation):
            return continuation
        return None

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.state.get_task(task_id)

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        return len(self.state.get_pending_tasks())

    def get_running_count(self) -> int:
        """Get count of running tasks."""
        return len(self.state.get_running_tasks())

    def get_status(self) -> dict:
        """Get full orchestrator status."""
        queue_stats = self.scheduler.get_queue_stats()
        quota_status = self.allocator.get_status()
        worker_status = self.workers.get_status() if self.workers else {}

        return {
            "running": self._running,
            "queue": queue_stats,
            "quotas": quota_status,
            "workers": worker_status,
            "modules": list(self.registry._modules.keys()),
            "data_dir": str(self.data_dir),
            "backends": self.backends,
        }

    def update_module_weights(self, weights: dict[str, float]):
        """Update priority weights for modules (from LLM consultation)."""
        self.scheduler.set_module_weights(weights)

    def clear_failure_cache(self, key: str = None):
        """Clear failure cache for a specific key or all keys."""
        if key:
            self.scheduler.failure_cache.remove(key)
        else:
            self.scheduler.failure_cache._cache.clear()

        log.info("orchestrator.failure_cache_cleared", key=key or "all")


async def run_unified_orchestrator(
    data_dir: str = "data/orchestrator",
    pool_url: str = "http://localhost:8765",
    backends: list[str] = None,
    enable_explorer: bool = True,
    enable_documenter: bool = True,
):
    """
    Run the unified orchestrator with Explorer and Documenter modules.

    This is the main entry point for the new orchestration system.
    """
    from explorer.src.adapter import ExplorerAdapter
    from documenter.adapter import DocumenterAdapter

    orchestrator = Orchestrator(
        data_dir=data_dir,
        pool_base_url=pool_url,
        backends=backends or ["gemini", "chatgpt", "claude"],
    )

    # Register modules
    if enable_explorer:
        explorer = ExplorerAdapter()
        orchestrator.register_module(explorer)
        log.info("orchestrator.explorer_registered")

    if enable_documenter:
        documenter = DocumenterAdapter()
        orchestrator.register_module(documenter)
        log.info("orchestrator.documenter_registered")

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        log.info("orchestrator.signal_received")
        asyncio.create_task(orchestrator.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await orchestrator.start()

        # Keep running until stopped
        while orchestrator._running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        log.info("orchestrator.keyboard_interrupt")
    finally:
        await orchestrator.stop()


async def main():
    """Entry point for running orchestrator standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Fano Unified Orchestrator")
    parser.add_argument("--data-dir", default="data/orchestrator",
                       help="Directory for state persistence")
    parser.add_argument("--pool-url", default="http://localhost:8765",
                       help="Pool service URL")
    parser.add_argument("--backends", nargs="+", default=["gemini", "chatgpt", "claude"],
                       help="LLM backends to use")
    parser.add_argument("--no-explorer", action="store_true",
                       help="Disable Explorer module")
    parser.add_argument("--no-documenter", action="store_true",
                       help="Disable Documenter module")
    args = parser.parse_args()

    await run_unified_orchestrator(
        data_dir=args.data_dir,
        pool_url=args.pool_url,
        backends=args.backends,
        enable_explorer=not args.no_explorer,
        enable_documenter=not args.no_documenter,
    )


if __name__ == "__main__":
    asyncio.run(main())
