"""QuotaAllocator — per-module LLM budget tracking and alerting.

Subscribes to llm.request.completed events, tracks spend per module,
and publishes system.budget.warning when a module crosses its alert threshold.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from shared.models import Event

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import EventBusInterface

logger = logging.getLogger(__name__)


class QuotaAllocator:
    """Tracks per-module LLM spend against a configured daily budget."""

    def __init__(self, config: Config, event_bus: EventBusInterface) -> None:
        self._config = config
        self._event_bus = event_bus

        daily_budget: float = config.get("quotas.daily_budget_usd", 0.0)
        weights: dict[str, int] = config.get("quotas.per_module_weights", {})
        alert_pct: float = config.get("quotas.alert_at_percent", 80)

        total_weight = sum(weights.values()) or 1
        self._budgets: dict[str, float] = {
            mod: daily_budget * (w / total_weight)
            for mod, w in weights.items()
        }
        self._alert_threshold: float = alert_pct / 100.0
        self._spend: dict[str, float] = {mod: 0.0 for mod in weights}
        self._warned: set[str] = set()

    async def initialize(self) -> None:
        """Subscribe to LLM completion events to track spend."""
        self._event_bus.subscribe("llm.request.completed", self._on_llm_request)

    def get_remaining(self, module: str) -> float:
        """Return remaining budget in USD for this module today."""
        budget = self._budgets.get(module, 0.0)
        spent = self._spend.get(module, 0.0)
        return max(budget - spent, 0.0)

    def is_within_budget(self, module: str) -> bool:
        """True if module hasn't exceeded its daily allocation."""
        if module not in self._budgets:
            return False
        return self.get_remaining(module) > 0.0

    def get_all_remaining(self) -> dict[str, float]:
        """Return remaining budget for all tracked modules."""
        return {mod: self.get_remaining(mod) for mod in self._budgets}

    def get_all_spend(self) -> dict[str, float]:
        """Return total spend for all tracked modules."""
        return dict(self._spend)

    async def _on_llm_request(self, event: Event) -> None:
        """Track spend per module. Publish system.budget.warning at alert threshold."""
        module = event.payload.get("module")
        cost = event.payload.get("cost_usd", 0.0)

        if module is None or cost <= 0:
            return

        if module not in self._spend:
            self._spend[module] = 0.0

        self._spend[module] += cost

        # Check if we've crossed the alert threshold for this module
        budget = self._budgets.get(module, 0.0)
        if budget <= 0:
            return

        spent_fraction = self._spend[module] / budget
        if spent_fraction >= self._alert_threshold and module not in self._warned:
            self._warned.add(module)
            warning_event = Event(
                topic="system.budget.warning",
                timestamp=datetime.now(timezone.utc),
                source="orchestrator.quota",
                payload={
                    "module": module,
                    "spent_usd": self._spend[module],
                    "budget_usd": budget,
                    "percent_used": round(spent_fraction * 100, 1),
                },
                correlation_id=event.correlation_id,
            )
            await self._event_bus.publish(warning_event)
