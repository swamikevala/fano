"""
Fano Orchestrator - Unified task coordination for Explorer and Documenter modules.

This module implements the v4.0 orchestration design:
- JIT (Just-In-Time) submission to Pool service
- Priority-based task scheduling
- Quota rationing for deep mode
- WAL + checkpoint state persistence
- Server-side thread navigation for conversation continuity
"""

from .models import Task, TaskState, ConversationState
from .scheduler import Scheduler
from .allocator import QuotaAllocator, QuotaBudget, AllocationResult
from .state import StateManager, OrchestratorState
from .worker import JITWorker, WorkerPool, WorkerConfig
from .main import Orchestrator
from .adapters import (
    ModuleInterface,
    ModuleRegistry,
    PromptContext,
    TaskResult,
    TaskType,
    run_in_executor,
)
from .legacy import (
    LegacyExplorerOrchestrator,
    LegacyDocumenterOrchestrator,
)

__all__ = [
    # Models
    "Task",
    "TaskState",
    "ConversationState",
    # State
    "StateManager",
    "OrchestratorState",
    # Scheduler
    "Scheduler",
    # Allocator
    "QuotaAllocator",
    "QuotaBudget",
    "AllocationResult",
    # Workers
    "JITWorker",
    "WorkerPool",
    "WorkerConfig",
    # Main
    "Orchestrator",
    # Adapters
    "ModuleInterface",
    "ModuleRegistry",
    "PromptContext",
    "TaskResult",
    "TaskType",
    "run_in_executor",
    # Legacy Wrappers
    "LegacyExplorerOrchestrator",
    "LegacyDocumenterOrchestrator",
]
