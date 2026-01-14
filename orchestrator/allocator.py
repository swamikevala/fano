"""
Quota Allocator for the Orchestrator.

Manages per-module budgets for scarce resources (deep mode backends)
based on v4.0 design spec Section 5.3.

Default Daily Budgets:
- Gemini Deep: Explorer 10, Documenter 5, Buffer 5 (total 20)
- ChatGPT Pro: Explorer 8, Documenter 4, Buffer 3 (total 15)
- Claude Extended: Explorer 6, Documenter 4, Buffer 2 (total 12)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from shared.logging import get_logger

from .state import StateManager

log = get_logger("orchestrator", "allocator")


@dataclass
class QuotaBudget:
    """Per-module budget for a quota type."""
    explorer: int = 0
    documenter: int = 0
    buffer: int = 0  # Shared buffer for overflow

    @property
    def total(self) -> int:
        return self.explorer + self.documenter + self.buffer


# Default daily budgets per backend
DEFAULT_BUDGETS = {
    "gemini_deep": QuotaBudget(explorer=10, documenter=5, buffer=5),
    "chatgpt_pro": QuotaBudget(explorer=8, documenter=4, buffer=3),
    "claude_extended": QuotaBudget(explorer=6, documenter=4, buffer=2),
}


@dataclass
class AllocationResult:
    """Result of a quota allocation request."""
    granted: bool
    quota_type: str
    module: str
    used_buffer: bool = False
    current_usage: int = 0
    budget_limit: int = 0
    message: Optional[str] = None


class QuotaAllocator:
    """
    Manages quota allocation for scarce resources.

    Implements the v4.0 quota rationing strategy:
    1. Each module has a dedicated budget
    2. Buffer pool for overflow when module budget exhausted
    3. Daily reset at midnight
    """

    def __init__(self, state_manager: StateManager, budgets: dict = None):
        self.state = state_manager
        self.budgets = budgets or DEFAULT_BUDGETS.copy()

    def can_allocate(self, quota_type: str, module: str) -> AllocationResult:
        """
        Check if quota is available for the module.

        Does NOT consume the quota - use allocate() to actually consume.
        """
        if quota_type not in self.budgets:
            # Non-rationed resource, always allow
            return AllocationResult(
                granted=True,
                quota_type=quota_type,
                module=module,
                message="Non-rationed resource"
            )

        budget = self.budgets[quota_type]
        module_budget = getattr(budget, module, 0)
        current_usage = self.state.get_quota_usage(quota_type, module)

        # Check module budget
        if current_usage < module_budget:
            return AllocationResult(
                granted=True,
                quota_type=quota_type,
                module=module,
                current_usage=current_usage,
                budget_limit=module_budget,
            )

        # Module budget exhausted, check buffer
        buffer_usage = self.state.get_quota_usage(quota_type, "buffer")
        if buffer_usage < budget.buffer:
            return AllocationResult(
                granted=True,
                quota_type=quota_type,
                module=module,
                used_buffer=True,
                current_usage=buffer_usage,
                budget_limit=budget.buffer,
                message="Using shared buffer"
            )

        # Both exhausted
        return AllocationResult(
            granted=False,
            quota_type=quota_type,
            module=module,
            current_usage=current_usage,
            budget_limit=module_budget,
            message=f"Quota exhausted for {module} ({current_usage}/{module_budget}) and buffer ({buffer_usage}/{budget.buffer})"
        )

    def allocate(self, quota_type: str, module: str) -> AllocationResult:
        """
        Allocate one unit of quota for the module.

        Returns AllocationResult with granted=True if successful.
        """
        result = self.can_allocate(quota_type, module)

        if not result.granted:
            log.warning("orchestrator.allocator.quota_denied",
                       quota_type=quota_type, module=module,
                       message=result.message)
            return result

        # Consume the quota
        if result.used_buffer:
            new_usage = self.state.increment_quota(quota_type, "buffer")
            log.info("orchestrator.allocator.buffer_used",
                    quota_type=quota_type, module=module,
                    buffer_usage=new_usage, buffer_limit=self.budgets[quota_type].buffer)
        else:
            new_usage = self.state.increment_quota(quota_type, module)
            budget = self.budgets[quota_type]
            module_budget = getattr(budget, module, 0)
            log.info("orchestrator.allocator.quota_allocated",
                    quota_type=quota_type, module=module,
                    usage=new_usage, limit=module_budget)

        result.current_usage = new_usage
        return result

    def get_remaining(self, quota_type: str, module: str) -> int:
        """
        Get remaining quota for a module (including potential buffer).
        """
        if quota_type not in self.budgets:
            return float('inf')  # Non-rationed

        budget = self.budgets[quota_type]
        module_budget = getattr(budget, module, 0)
        current_usage = self.state.get_quota_usage(quota_type, module)

        # Module remaining
        module_remaining = max(0, module_budget - current_usage)

        # Buffer remaining
        buffer_usage = self.state.get_quota_usage(quota_type, "buffer")
        buffer_remaining = max(0, budget.buffer - buffer_usage)

        return module_remaining + buffer_remaining

    def get_status(self, quota_type: str = None) -> dict:
        """
        Get quota status for all or specific quota types.
        """
        types_to_check = [quota_type] if quota_type else list(self.budgets.keys())
        status = {}

        for qt in types_to_check:
            if qt not in self.budgets:
                continue

            budget = self.budgets[qt]
            status[qt] = {
                "explorer": {
                    "used": self.state.get_quota_usage(qt, "explorer"),
                    "budget": budget.explorer,
                    "remaining": max(0, budget.explorer - self.state.get_quota_usage(qt, "explorer")),
                },
                "documenter": {
                    "used": self.state.get_quota_usage(qt, "documenter"),
                    "budget": budget.documenter,
                    "remaining": max(0, budget.documenter - self.state.get_quota_usage(qt, "documenter")),
                },
                "buffer": {
                    "used": self.state.get_quota_usage(qt, "buffer"),
                    "budget": budget.buffer,
                    "remaining": max(0, budget.buffer - self.state.get_quota_usage(qt, "buffer")),
                },
                "total_remaining": self.get_remaining(qt, "explorer") + self.get_remaining(qt, "documenter"),
            }

        return status

    def set_budget(self, quota_type: str, explorer: int = None,
                   documenter: int = None, buffer: int = None):
        """
        Update budget limits for a quota type.

        Can be called by LLM consultation to rebalance budgets.
        """
        if quota_type not in self.budgets:
            self.budgets[quota_type] = QuotaBudget()

        budget = self.budgets[quota_type]
        if explorer is not None:
            budget.explorer = explorer
        if documenter is not None:
            budget.documenter = documenter
        if buffer is not None:
            budget.buffer = buffer

        log.info("orchestrator.allocator.budget_updated",
                quota_type=quota_type,
                explorer=budget.explorer,
                documenter=budget.documenter,
                buffer=budget.buffer)

    def should_use_deep_mode(self, module: str, task_type: str,
                              preferred_backend: str = "gemini_deep") -> tuple[bool, str]:
        """
        Determine if a task should use deep mode.

        Returns (should_use, backend_or_reason).
        """
        # Map task types to deep mode preference
        deep_mode_tasks = {
            # Explorer
            "exploration": True,  # Always prefer deep for exploration
            "synthesis": True,    # Synthesis benefits from deep
            "review": False,      # Review can use standard
            "critique": False,

            # Documenter
            "incorporate_insight": True,  # Insight incorporation needs depth
            "draft_section": True,        # Section drafting needs depth
            "review_section": False,
            "address_comment": False,     # Quick response needed
        }

        wants_deep = deep_mode_tasks.get(task_type, False)
        if not wants_deep:
            return False, "Task type doesn't require deep mode"

        # Check quota availability
        result = self.can_allocate(preferred_backend, module)
        if result.granted:
            return True, preferred_backend

        # Try alternative backends
        for backend in self.budgets.keys():
            if backend != preferred_backend:
                alt_result = self.can_allocate(backend, module)
                if alt_result.granted:
                    return True, backend

        # No deep mode available
        return False, f"Deep mode quota exhausted: {result.message}"
