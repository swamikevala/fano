"""
Thread Manager - Handles exploration thread operations.

This module centralizes:
- Thread loading and selection
- Thread spawning from seeds
- Seed discovery and prioritization
- Thread context building
"""

from typing import Optional, Set

from shared.logging import get_logger

from explorer.src.models import ExplorationThread, ThreadStatus, AxiomStore
from explorer.src.storage import ExplorerPaths

log = get_logger("explorer", "orchestration.threads")


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
            List of active threads sorted by priority (highest first),
            limited by max_active_threads config.
        """
        threads = []

        if self.paths.explorations_dir.exists():
            for filepath in self.paths.explorations_dir.glob("*.json"):
                try:
                    thread = ExplorationThread.load(filepath)
                    if thread.status == ThreadStatus.ACTIVE:
                        threads.append(thread)
                except Exception as e:
                    log.warning(f"Could not load thread {filepath}: {e}")

        # Sort by priority (highest first) before limiting
        threads.sort(key=lambda t: getattr(t, 'priority', 0), reverse=True)

        return threads[: self.config.get("max_active_threads", 5)]

    def select_thread(
        self,
        exclude_ids: Optional[Set[str]] = None,
        exclude_seeds: Optional[Set[str]] = None,
    ) -> Optional[ExplorationThread]:
        """
        Select an active thread to work on, or return None if a higher-priority
        unexplored seed should be spawned instead.

        Compares priorities across:
        - Active threads needing work
        - Unexplored seeds waiting to be started

        Args:
            exclude_ids: Set of thread IDs to exclude from selection.
                        Used when assigning multiple threads to different models.
            exclude_seeds: Set of seed IDs to exclude from priority comparison.
                          Used when seeds are already selected for spawning this cycle.

        Returns:
            Selected thread, or None if no threads available or if a
            higher-priority unexplored seed exists (triggers spawning).
        """
        threads = self.load_active_threads()

        # Filter out excluded threads
        if exclude_ids:
            threads = [t for t in threads if t.id not in exclude_ids]

        # Sort by priority (highest first), then by needs_work status
        if threads:
            threads.sort(
                key=lambda t: (
                    getattr(t, 'priority', 0),  # Higher priority first
                    t.needs_exploration or t.needs_critique,  # Needs work second
                ),
                reverse=True
            )

        # Find highest priority thread that needs work
        best_thread = None
        best_thread_priority = -1
        for thread in threads:
            if thread.needs_exploration or thread.needs_critique:
                best_thread = thread
                best_thread_priority = getattr(thread, 'priority', 0)
                break

        # Check if there's a higher-priority unexplored seed
        next_seed = self.select_next_seed(exclude_seeds=exclude_seeds)
        if next_seed:
            seed_priority = getattr(next_seed, 'priority', 0)
            if seed_priority > best_thread_priority:
                log.info(
                    "thread_manager.seed_priority_higher",
                    seed_id=next_seed.id,
                    seed_priority=seed_priority,
                    thread_priority=best_thread_priority,
                )
                # Return None to trigger spawning the higher-priority seed
                return None

        return best_thread

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

    def spawn_new_thread(self, exclude_seeds: Optional[Set[str]] = None) -> ExplorationThread:
        """
        Create a new exploration thread focused on a single seed.

        The thread will focus on ONE question or conjecture, with axioms
        always included as foundational context.

        Args:
            exclude_seeds: Set of seed IDs to exclude from selection
                          (used when spawning multiple threads in parallel)

        Returns:
            Newly created thread.
        """
        # Select the single seed to focus on
        focus_seed = self.select_next_seed(exclude_seeds=exclude_seeds)

        # Get axiom IDs (always included as foundational context)
        axioms = self.axioms.get_axioms()
        axiom_ids = [a.id for a in axioms]

        if focus_seed is None:
            # No seeds to explore - create a general exploration thread
            log.warning("[seeds] No seeds available, creating general thread")
            thread = ExplorationThread.create_new(
                topic="Open mathematical exploration",
                seed_axioms=axiom_ids,
                target_numbers=[],
            )
        else:
            # Create focused thread
            topic = self._generate_topic([focus_seed])

            thread = ExplorationThread.create_new(
                topic=topic,
                seed_axioms=axiom_ids,  # Only axioms as foundational context
                target_numbers=[],
            )

            # Set the focused exploration fields
            if focus_seed.type == "question":
                thread.primary_question_id = focus_seed.id
            else:
                thread.related_conjecture_ids = [focus_seed.id]
            thread.priority = focus_seed.priority

            log.info(
                f"[seeds] Created focused thread {thread.id} for {focus_seed.type}: {focus_seed.id}"
            )

        thread.save(self.paths.data_dir)
        return thread

    async def check_and_spawn_for_new_seeds(self) -> Optional[ExplorationThread]:
        """
        Check if there are new seeds that haven't been explored yet.
        If so, spawn a focused thread for the highest-priority unexplored seed.

        Returns:
            Newly spawned thread if new seeds found, None otherwise.
        """
        log.info("[seeds] Checking for new/unexplored seeds...")

        # Get questions and conjectures (not axioms)
        questions = self.axioms.get_questions()
        conjectures = self.axioms.get_conjectures()

        if not questions and not conjectures:
            log.info("[seeds] No seeds found")
            return None

        explored = self.get_explored_seed_ids()
        unexplored_count = sum(1 for q in questions if q.id not in explored)
        unexplored_count += sum(1 for c in conjectures if c.id not in explored)

        if unexplored_count == 0:
            log.info(f"[seeds] All {len(questions) + len(conjectures)} seeds have been explored")
            return None

        log.info(f"[seeds] Found {unexplored_count} unexplored seeds")

        # Spawn a focused thread for the next unexplored seed
        thread = self.spawn_new_thread()
        return thread

    def get_explored_seed_ids(self) -> set[str]:
        """
        Get IDs of seeds that have already been explored in existing threads.

        Checks both primary_question_id and related_conjecture_ids fields
        to find seeds that have dedicated exploration threads.

        Returns:
            Set of seed IDs that have been explored.
        """
        explored = set()

        if not self.paths.explorations_dir.exists():
            return explored

        for filepath in self.paths.explorations_dir.glob("*.json"):
            try:
                thread = ExplorationThread.load(filepath)
                # Collect from focused exploration fields
                if thread.primary_question_id:
                    explored.add(thread.primary_question_id)
                if thread.related_conjecture_ids:
                    explored.update(thread.related_conjecture_ids)
            except Exception:
                pass

        return explored

    def select_next_seed(self, exclude_seeds: Optional[Set[str]] = None):
        """
        Select the next unexplored seed to focus on.

        Priority order:
        1. Unexplored questions (by priority, highest first)
        2. Unexplored conjectures (by priority, highest first)
        3. If all explored, return highest-priority seed for re-exploration

        Args:
            exclude_seeds: Set of seed IDs to exclude from selection
                          (used when spawning multiple threads in parallel)

        Returns:
            SeedAphorism to focus on, or None if no seeds exist.
        """
        exclude_seeds = exclude_seeds or set()

        # Get questions and conjectures (not axioms - those are always context)
        questions = self.axioms.get_questions()
        conjectures = self.axioms.get_conjectures()

        if not questions and not conjectures:
            return None

        explored = self.get_explored_seed_ids()

        # Try unexplored questions first (already sorted by priority)
        for q in questions:
            if q.id not in explored and q.id not in exclude_seeds:
                log.info(f"[seeds] Selected unexplored question: {q.id} (P{q.priority})")
                return q

        # Then unexplored conjectures
        for c in conjectures:
            if c.id not in explored and c.id not in exclude_seeds:
                log.info(f"[seeds] Selected unexplored conjecture: {c.id} (P{c.priority})")
                return c

        # All explored - return highest priority for re-exploration (excluding already selected)
        all_seeds = questions + conjectures
        all_seeds = [s for s in all_seeds if s.id not in exclude_seeds]
        all_seeds.sort(key=lambda s: s.priority, reverse=True)
        if all_seeds:
            log.info(f"[seeds] All seeds explored, re-exploring: {all_seeds[0].id}")
            return all_seeds[0]

        return None

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

    def get_focused_context(self, thread: ExplorationThread) -> str:
        """
        Build exploration context for a single-seed focused thread.

        Includes:
        - Axioms as foundational facts (from thread.seed_axioms)
        - The single focus seed (from primary_question_id or related_conjecture_ids)

        Args:
            thread: The exploration thread with focused seed info.

        Returns:
            Formatted context string optimized for single-seed exploration.
        """
        lines = []

        # 1. AXIOMS - Foundational facts (always included)
        axioms = self.axioms.get_axioms()
        if axioms:
            lines.append("=== FOUNDATIONAL AXIOMS ===")
            lines.append("These are established facts. Take them as given:\n")
            for axiom in axioms:
                lines.append(f"• {axiom.text}")
                if axiom.notes:
                    lines.append(f"   Note: {axiom.notes}")
            lines.append("")

        # 2. FOCUS SEED - The single seed to explore deeply
        focus_seed = None
        seed_type = None

        if thread.primary_question_id:
            focus_seed = self.axioms.get_seed_by_id(thread.primary_question_id)
            seed_type = "QUESTION"
        elif thread.related_conjecture_ids:
            focus_seed = self.axioms.get_seed_by_id(thread.related_conjecture_ids[0])
            seed_type = "CONJECTURE"

        if focus_seed:
            lines.append(f"=== YOUR FOCUS: {seed_type} ===")
            lines.append("Explore this single seed DEEPLY. Do not spread thin across many topics.\n")

            if seed_type == "QUESTION":
                lines.append(f"❓ {focus_seed.text}")
            else:
                confidence_marker = {"high": "⚡", "medium": "?", "low": "○"}.get(
                    focus_seed.confidence, "?"
                )
                lines.append(f"{confidence_marker} {focus_seed.text}")

            if focus_seed.tags:
                lines.append(f"   [Tags: {', '.join(focus_seed.tags)}]")
            if focus_seed.notes:
                lines.append(f"   Context: {focus_seed.notes}")
            lines.append("")

            lines.append("Your task:")
            if seed_type == "QUESTION":
                lines.append("- Investigate this question thoroughly")
                lines.append("- Use the axioms above as your foundation")
                lines.append("- Develop a clear, well-reasoned answer")
            else:
                lines.append("- Verify or refute this conjecture")
                lines.append("- Find the deeper structure that explains WHY")
                lines.append("- Use the axioms above as your foundation")
            lines.append("")
        else:
            # Fallback if no focus seed found
            lines.append("=== EXPLORATION ===")
            lines.append("Explore mathematical structures based on the axioms above.")
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
