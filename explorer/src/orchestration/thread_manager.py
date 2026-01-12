"""
Thread Manager - Handles exploration thread operations.

This module centralizes:
- Thread loading and selection
- Thread spawning from seeds
- Seed discovery and prioritization
- Thread context building
"""

import logging
from typing import Optional

from explorer.src.models import ExplorationThread, ThreadStatus, AxiomStore
from explorer.src.storage import ExplorerPaths

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages exploration thread lifecycle.

    Handles:
    - Loading active threads
    - Selecting threads for work
    - Spawning new threads from seeds
    - Finding new/unexplored seeds
    """

    def __init__(self, config: dict, paths: ExplorerPaths, axioms: AxiomStore):
        """
        Initialize thread manager.

        Args:
            config: Orchestration config section
            paths: ExplorerPaths instance
            axioms: AxiomStore for seed access
        """
        self.config = config
        self.paths = paths
        self.axioms = axioms

    def load_active_threads(self) -> list[ExplorationThread]:
        """
        Load all active exploration threads.

        Returns:
            List of active threads, limited by max_active_threads config.
        """
        threads = []

        if self.paths.explorations_dir.exists():
            for filepath in self.paths.explorations_dir.glob("*.json"):
                try:
                    thread = ExplorationThread.load(filepath)
                    if thread.status == ThreadStatus.ACTIVE:
                        threads.append(thread)
                except Exception as e:
                    logger.warning(f"Could not load thread {filepath}: {e}")

        return threads[: self.config.get("max_active_threads", 5)]

    def select_thread(self) -> Optional[ExplorationThread]:
        """
        Select an active thread to work on.

        Prioritizes threads that need exploration or critique.

        Returns:
            Selected thread, or None if no threads available.
        """
        threads = self.load_active_threads()

        if not threads:
            return None

        # Prioritize threads that need work
        for thread in threads:
            if thread.needs_exploration or thread.needs_critique:
                return thread

        return threads[0] if threads else None

    def load_thread_by_id(self, thread_id: str) -> Optional[ExplorationThread]:
        """
        Load a thread by its ID (short or full).

        Args:
            thread_id: Thread ID to find (can be partial match)

        Returns:
            ExplorationThread if found, None otherwise.
        """
        if not self.paths.explorations_dir.exists():
            return None

        # Try to find matching thread file
        for filepath in self.paths.explorations_dir.glob("*.json"):
            if thread_id in filepath.stem:
                try:
                    return ExplorationThread.load(filepath)
                except Exception:
                    pass
        return None

    def spawn_new_thread(self) -> ExplorationThread:
        """
        Create a new exploration thread based on all seed aphorisms.

        Returns:
            Newly created thread.
        """
        seeds = self.axioms.get_seed_aphorisms()
        seed_ids = [s.id for s in seeds] if seeds else []
        topic = self._generate_topic(seeds)

        thread = ExplorationThread.create_new(
            topic=topic,
            seed_axioms=seed_ids,
            target_numbers=[],
        )

        thread.save(self.paths.data_dir)
        return thread

    def spawn_thread_for_seeds(self, seeds: list) -> ExplorationThread:
        """
        Create a new exploration thread for specific seeds.

        Args:
            seeds: List of seed aphorisms to explore.

        Returns:
            Newly created thread.
        """
        seed_ids = [s.id for s in seeds]
        topic = self._generate_topic(seeds)

        thread = ExplorationThread.create_new(
            topic=topic,
            seed_axioms=seed_ids,
            target_numbers=[],
        )

        thread.save(self.paths.data_dir)
        return thread

    async def check_and_spawn_for_new_seeds(self) -> Optional[ExplorationThread]:
        """
        Check if there are new seeds that haven't been explored yet.
        If so, spawn a new thread prioritizing those seeds.

        Returns:
            Newly spawned thread if new seeds found, None otherwise.
        """
        logger.info("[seeds] Checking for new/unexplored seeds...")

        # Get all current seeds
        all_seeds = self.axioms.get_seed_aphorisms()
        if not all_seeds:
            logger.info("[seeds] No seeds found")
            return None

        all_seed_ids = {s.id for s in all_seeds}

        # Find which seeds have already been explored (in any thread)
        explored_seed_ids = set()
        if self.paths.explorations_dir.exists():
            for filepath in self.paths.explorations_dir.glob("*.json"):
                try:
                    thread = ExplorationThread.load(filepath)
                    explored_seed_ids.update(thread.seed_axioms or [])
                except Exception:
                    pass

        # Find new seeds
        new_seed_ids = all_seed_ids - explored_seed_ids

        if not new_seed_ids:
            logger.info(f"[seeds] All {len(all_seed_ids)} seeds have been explored")
            return None

        new_seeds = [s for s in all_seeds if s.id in new_seed_ids]
        logger.info(f"[seeds] Found {len(new_seeds)} NEW seeds to explore:")
        for seed in new_seeds:
            logger.info(f"[seeds]   - {seed.id}: {seed.text[:60]}...")

        # Spawn a new thread specifically for these seeds
        thread = self.spawn_thread_for_seeds(new_seeds)
        logger.info(f"[seeds] Spawned new thread {thread.id} for new seeds")

        return thread

    def get_context_for_seeds(self, seed_ids: list[str]) -> str:
        """
        Get exploration context for specific seed IDs only.

        Args:
            seed_ids: List of seed IDs to include in context.

        Returns:
            Formatted context string for prompts.
        """
        all_seeds = self.axioms.get_seed_aphorisms()
        filtered_seeds = [s for s in all_seeds if s.id in seed_ids]

        if not filtered_seeds:
            return self.axioms.get_context_for_exploration()

        lines = []
        lines.append("=== SEED APHORISMS ===")
        lines.append("These are the foundational conjectures to explore, verify, and build upon:\n")

        for seed in filtered_seeds:
            confidence_marker = {"high": "⚡", "medium": "?", "low": "○"}.get(
                seed.confidence, "?"
            )
            lines.append(f"{confidence_marker} {seed.text}")
            if seed.tags:
                lines.append(f"   [Tags: {', '.join(seed.tags)}]")
            if seed.notes:
                lines.append(f"   Note: {seed.notes}")
        lines.append("")

        return "\n".join(lines)

    def _generate_topic(self, seeds: list) -> str:
        """
        Generate a topic description from seed aphorisms.

        Args:
            seeds: List of seed aphorisms.

        Returns:
            Generated topic string.
        """
        if not seeds:
            return "Open exploration of seed aphorisms and their mathematical structures"

        # Collect all unique tags from seeds
        all_tags = set()
        for seed in seeds:
            all_tags.update(seed.tags)

        if all_tags:
            tag_list = ", ".join(sorted(all_tags)[:5])  # Top 5 tags
            return f"Exploring connections between: {tag_list}"

        # Fallback: use first seed's text (truncated)
        first_seed = seeds[0].text[:80]
        return f"Exploring: {first_seed}..."
