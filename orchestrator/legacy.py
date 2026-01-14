"""
Legacy Orchestrator Wrappers.

Provides backward-compatible interfaces for code that expects the
old Explorer or Documenter orchestrators.

These wrappers delegate to the unified orchestrator while maintaining
the original API signatures.
"""

import asyncio
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

from .main import Orchestrator
from .adapters import ModuleInterface

log = get_logger("orchestrator", "legacy")


class LegacyExplorerOrchestrator:
    """
    Backward-compatible wrapper for Explorer orchestrator.

    Provides the same interface as explorer/src/orchestrator.py Orchestrator
    but delegates to the unified orchestrator.

    Usage:
        # Instead of:
        # from explorer.src.orchestrator import Orchestrator
        # orchestrator = Orchestrator()
        # await orchestrator.run()

        # Use:
        from orchestrator.legacy import LegacyExplorerOrchestrator
        orchestrator = LegacyExplorerOrchestrator()
        await orchestrator.run()
    """

    def __init__(self):
        self._orchestrator: Optional[Orchestrator] = None
        self._adapter = None
        self.running = False

    async def run(self, process_backlog_first: bool = True):
        """
        Main exploration loop.

        Args:
            process_backlog_first: If True, process unextracted threads first
        """
        from explorer.src.adapter import ExplorerAdapter

        log.info("legacy.explorer.starting")

        # Create unified orchestrator with only Explorer
        self._orchestrator = Orchestrator(
            data_dir="data/orchestrator",
            pool_base_url="http://localhost:8765",
        )

        # Register Explorer adapter
        self._adapter = ExplorerAdapter()
        self._orchestrator.register_module(self._adapter)

        self.running = True

        try:
            await self._orchestrator.start()

            # Process backlog if requested
            if process_backlog_first:
                log.info("legacy.explorer.processing_backlog")
                # Backlog processing is handled by the adapter's get_pending_work

            # Keep running until stopped
            while self.running and self._orchestrator._running:
                await asyncio.sleep(1)

        finally:
            await self.cleanup()

    def stop(self):
        """Signal the orchestrator to stop."""
        self.running = False
        if self._orchestrator:
            asyncio.create_task(self._orchestrator.stop())
        log.info("legacy.explorer.stop_requested")

    async def cleanup(self):
        """Clean up resources."""
        if self._orchestrator:
            await self._orchestrator.stop()
            self._orchestrator = None
        log.info("legacy.explorer.cleanup_complete")

    async def process_backlog(self):
        """Process backlog of unextracted threads."""
        # This is now handled automatically by the adapter
        log.info("legacy.explorer.backlog_delegated")


class LegacyDocumenterOrchestrator:
    """
    Backward-compatible wrapper for Documenter orchestrator.

    Provides the same interface as documenter/main.py Orchestrator
    but delegates to the unified orchestrator.

    Usage:
        # Instead of:
        # from documenter.main import Orchestrator
        # orchestrator = Orchestrator()
        # await orchestrator.run()

        # Use:
        from orchestrator.legacy import LegacyDocumenterOrchestrator
        orchestrator = LegacyDocumenterOrchestrator()
        await orchestrator.run()
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._orchestrator: Optional[Orchestrator] = None
        self._adapter = None

    async def run(self):
        """Main loop - grow the document."""
        from documenter.adapter import DocumenterAdapter

        log.info("legacy.documenter.starting")

        # Create unified orchestrator with only Documenter
        self._orchestrator = Orchestrator(
            data_dir="data/orchestrator",
            pool_base_url="http://localhost:8765",
        )

        # Register Documenter adapter
        self._adapter = DocumenterAdapter(config_path=self.config_path)
        self._orchestrator.register_module(self._adapter)

        try:
            await self._orchestrator.start()

            # Keep running until stopped or exhausted
            while self._orchestrator._running:
                # Check if adapter's session is exhausted
                if self._adapter.session and self._adapter.session.exhausted:
                    log.info("legacy.documenter.session_exhausted")
                    break
                await asyncio.sleep(1)

        finally:
            if self._orchestrator:
                await self._orchestrator.stop()

        log.info("legacy.documenter.completed")


# Convenience aliases
ExplorerOrchestrator = LegacyExplorerOrchestrator
DocumenterOrchestrator = LegacyDocumenterOrchestrator
Documenter = LegacyDocumenterOrchestrator


async def run_explorer_legacy():
    """Run Explorer using legacy wrapper."""
    orchestrator = LegacyExplorerOrchestrator()
    await orchestrator.run()


async def run_documenter_legacy():
    """Run Documenter using legacy wrapper."""
    orchestrator = LegacyDocumenterOrchestrator()
    await orchestrator.run()
