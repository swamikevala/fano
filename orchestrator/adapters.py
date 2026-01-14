"""
Module Adapters for the Orchestrator.

Defines the ModuleInterface that Explorer and Documenter must implement
to integrate with the unified orchestrator.

Based on v4.0 design specification Phase 3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from shared.logging import get_logger

from .models import Task, TaskState

log = get_logger("orchestrator", "adapters")


class TaskType(str, Enum):
    """All supported task types across modules."""
    # Explorer tasks
    EXPLORATION = "exploration"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    REVIEW = "review"

    # Documenter tasks (legacy v2)
    ADDRESS_COMMENT = "address_comment"
    INCORPORATE_INSIGHT = "incorporate_insight"
    REVIEW_SECTION = "review_section"
    DRAFT_SECTION = "draft_section"
    DRAFT_PREREQUISITE = "draft_prerequisite"

    # Documenter tasks (v3 multi-phase)
    INCORPORATE_INSIGHT_PLAN = "incorporate_insight_plan"      # Architect decision
    INCORPORATE_INSIGHT_DRAFT = "incorporate_insight_draft"    # Mason drafting
    INCORPORATE_INSIGHT_REVIEW = "incorporate_insight_review"  # Consensus Board
    GENERATE_DIAGRAM = "generate_diagram"                      # Illustrator
    GLOBAL_REFACTOR = "global_refactor"                        # Architect reorganization

    # Shared
    CONSENSUS = "consensus"


@dataclass
class PromptContext:
    """Context for building a prompt."""
    prompt: str
    images: list[dict] = field(default_factory=list)
    system_context: Optional[str] = None
    thread_context: Optional[str] = None
    requires_deep_mode: bool = False
    preferred_backend: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from executing a task."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    thread_url: Optional[str] = None
    thread_title: Optional[str] = None
    deep_mode_used: bool = False
    metadata: dict = field(default_factory=dict)

    # For multi-turn tasks
    needs_continuation: bool = False
    continuation_payload: Optional[dict] = None


class ModuleInterface(ABC):
    """
    Abstract interface for module adapters.

    Each module (Explorer, Documenter) must implement this interface
    to integrate with the unified orchestrator.
    """

    @property
    @abstractmethod
    def module_name(self) -> str:
        """Return the module name (e.g., 'explorer', 'documenter')."""
        pass

    @property
    @abstractmethod
    def supported_task_types(self) -> list[str]:
        """Return list of task types this module handles."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the module.

        Called once when orchestrator starts.
        Returns True if initialization successful.
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Cleanup the module.

        Called when orchestrator stops.
        """
        pass

    @abstractmethod
    async def get_pending_work(self) -> list[dict]:
        """
        Get list of pending work items from the module.

        Returns list of work items, each containing:
        - task_type: str
        - key: str (deduplication key)
        - payload: dict (task-specific data)
        - priority: int (optional, will be computed if not provided)
        - requires_deep_mode: bool (optional)
        - preferred_backend: str (optional)
        """
        pass

    @abstractmethod
    async def build_prompt(self, task: Task) -> PromptContext:
        """
        Build the prompt for a task.

        Args:
            task: The task to build prompt for

        Returns:
            PromptContext with prompt text and metadata
        """
        pass

    @abstractmethod
    async def handle_result(self, task: Task, result: TaskResult) -> bool:
        """
        Handle the result of task execution.

        Args:
            task: The completed task
            result: The execution result

        Returns:
            True if result was handled successfully
        """
        pass

    @abstractmethod
    async def on_task_failed(self, task: Task, error: str):
        """
        Handle task failure.

        Called when a task fails after all retries.

        Args:
            task: The failed task
            error: Error message
        """
        pass

    # Optional hooks

    async def on_task_started(self, task: Task):
        """Called when a task starts executing."""
        pass

    async def on_quota_exhausted(self, task_type: str, backend: str):
        """Called when quota is exhausted for a task type."""
        pass

    async def get_system_state(self) -> dict:
        """
        Get current system state for priority computation.

        Returns dict with keys like:
        - blessed_insights_pending: int
        - comments_pending: int
        - active_threads: int
        """
        return {}


class ModuleRegistry:
    """
    Registry of module adapters.

    Used by the orchestrator to dispatch tasks to the appropriate module.
    """

    def __init__(self):
        self._modules: dict[str, ModuleInterface] = {}
        self._task_type_to_module: dict[str, str] = {}

    def register(self, module: ModuleInterface):
        """Register a module adapter."""
        name = module.module_name
        self._modules[name] = module

        for task_type in module.supported_task_types:
            self._task_type_to_module[task_type] = name

        log.info("orchestrator.registry.module_registered",
                module=name,
                task_types=module.supported_task_types)

    def get_module(self, name: str) -> Optional[ModuleInterface]:
        """Get module by name."""
        return self._modules.get(name)

    def get_module_for_task(self, task_type: str) -> Optional[ModuleInterface]:
        """Get module that handles a task type."""
        module_name = self._task_type_to_module.get(task_type)
        if module_name:
            return self._modules.get(module_name)
        return None

    def get_all_modules(self) -> list[ModuleInterface]:
        """Get all registered modules."""
        return list(self._modules.values())

    async def initialize_all(self) -> bool:
        """Initialize all modules."""
        for name, module in self._modules.items():
            try:
                if not await module.initialize():
                    log.error("orchestrator.registry.module_init_failed", module=name)
                    return False
            except Exception as e:
                log.exception(e, "orchestrator.registry.module_init_error", {"module": name})
                return False

        log.info("orchestrator.registry.all_modules_initialized",
                count=len(self._modules))
        return True

    async def shutdown_all(self):
        """Shutdown all modules."""
        for name, module in self._modules.items():
            try:
                await module.shutdown()
            except Exception as e:
                log.exception(e, "orchestrator.registry.module_shutdown_error", {"module": name})

        log.info("orchestrator.registry.all_modules_shutdown")


# Helper for running legacy sync code in async context
async def run_in_executor(func, *args, **kwargs):
    """
    Run a synchronous function in a thread pool executor.

    Useful for adapting legacy synchronous code.
    """
    import asyncio
    from functools import partial

    loop = asyncio.get_running_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(None, func, *args)
