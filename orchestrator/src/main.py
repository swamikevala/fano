"""Orchestrator — facade for coordinating all engine modules.

Manages module lifecycle (register, start, stop), delegates to
QuotaAllocator for budget tracking and RecoveryManager for crash recovery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orchestrator.src.quota import QuotaAllocator
from orchestrator.src.recovery import RecoveryManager

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import EventBusInterface, ModuleInterface, StateStoreInterface

logger = logging.getLogger(__name__)


class Orchestrator:
    """Top-level coordinator for all registered engine modules."""

    def __init__(
        self,
        store: StateStoreInterface,
        event_bus: EventBusInterface,
        config: Config,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._config = config
        self._modules: list[ModuleInterface] = []
        self._module_names: set[str] = set()
        self._quota = QuotaAllocator(config=config, event_bus=event_bus)
        self._recovery = RecoveryManager(store=store, event_bus=event_bus)
        self._started = False

    def register_module(self, module: ModuleInterface) -> None:
        """Register an engine module. Called during setup.

        Raises ValueError if a module with the same name is already registered.
        """
        name = module.module_name
        if name in self._module_names:
            raise ValueError(f"Module '{name}' already registered")
        self._modules.append(module)
        self._module_names.add(name)
        logger.info("Module registered: %s", name)

    async def start(self) -> None:
        """Start the orchestrator and all registered modules.

        1. Initialize all registered modules (dependency order)
        2. Run crash recovery (replay missed events)
        3. Start all modules
        4. Begin health check loop
        5. Begin quota tracking
        """
        # 1. Initialize all modules
        for module in self._modules:
            logger.info("Initializing module: %s", module.module_name)
            await module.initialize()

        # 2. Run crash recovery
        await self._recovery.recover()

        # 3. Start all modules
        for module in self._modules:
            logger.info("Starting module: %s", module.module_name)
            await module.start()

        # 4 & 5. Initialize quota tracking (subscribes to events)
        await self._quota.initialize()

        self._started = True
        logger.info("Orchestrator started with %d modules", len(self._modules))

    async def stop(self) -> None:
        """Stop all modules in reverse registration order. Save state."""
        for module in reversed(self._modules):
            logger.info("Stopping module: %s", module.module_name)
            await module.stop()

        # Record a checkpoint before shutting down
        await self._recovery.checkpoint()

        self._started = False
        logger.info("Orchestrator stopped")

    async def get_status(self) -> dict:
        """Return health status of all modules, quota usage, event stats."""
        modules_status: dict[str, dict] = {}
        all_healthy = True

        for module in self._modules:
            health = await module.health_check()
            modules_status[module.module_name] = {
                "healthy": health.healthy,
                "message": health.message,
                "details": health.details,
            }
            if not health.healthy:
                all_healthy = False

        return {
            "healthy": all_healthy,
            "modules": modules_status,
            "quota": {
                "remaining": self._quota.get_all_remaining(),
                "spend": self._quota.get_all_spend(),
            },
            "started": self._started,
        }
