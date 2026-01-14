"""
Orchestrator - The Brain of Fano Explorer

The orchestrator is responsible for:
- Deciding which thread to work on next
- Formulating prompts for exploration and critique
- Judging when a thread is mature enough to synthesize
- Creating chunks from mature threads
- Managing rate limits and model selection

This is a lightweight coordinator that delegates to specialized modules:
- LLMManager: LLM connection and communication
- ThreadManager: Thread loading, selection, and spawning
- ExplorationEngine: Exploration and critique operations
- SynthesisEngine: Chunk synthesis from threads
- InsightProcessor: Insight extraction and automated review
- BlessedStore: Blessed insights management
"""

import asyncio
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from shared.logging import get_logger

from explorer.src.models import AxiomStore, ThreadStatus
from explorer.src.storage.db import Database
from explorer.src.storage import ExplorerPaths
from explorer.src.chunking import (
    AtomicExtractor,
    PanelExtractor,
    get_dedup_checker,
)
from explorer.src.review_panel import AutomatedReviewer
from explorer.src.augmentation import get_augmenter

# Import orchestration modules
from explorer.src.orchestration import (
    LLMManager,
    ThreadManager,
    ExplorationEngine,
    SynthesisEngine,
    InsightProcessor,
    BlessedStore,
)

log = get_logger("explorer", "orchestrator")


# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


class Orchestrator:
    """
    The brain of the exploration system.

    A lightweight coordinator that delegates to specialized modules
    for each concern (LLM management, threading, exploration, etc.).
    """

    def __init__(self):
        # Initialize paths
        self.paths = ExplorerPaths(Path(__file__).parent.parent / "data")

        # Core data stores
        self.db = Database(self.paths.database_file)
        self.axioms = AxiomStore(self.paths.data_dir)

        # Configuration
        self.config = CONFIG
        self.orchestration_config = CONFIG["orchestration"]
        self.running = False

        # Consecutive failure tracking for exponential backoff
        self._consecutive_failures = 0
        self._max_backoff = 300  # Max 5 minutes between retries

        # Initialize LLM manager
        self.llm_manager = LLMManager(CONFIG, self.paths)

        # Initialize thread manager
        self.thread_manager = ThreadManager(
            self.orchestration_config,
            self.paths,
            self.axioms,
        )

        # Initialize exploration engine
        self.exploration_engine = ExplorationEngine(
            CONFIG,
            self.paths,
            self.axioms,
            self.thread_manager.get_context_for_seeds,
            self.thread_manager.get_focused_context,
        )

        # Initialize synthesis engine
        self.synthesis_engine = SynthesisEngine(CONFIG, self.paths)

        # These will be initialized after LLM connection
        self.reviewer: Optional[AutomatedReviewer] = None
        self.blessed_store: Optional[BlessedStore] = None
        self.insight_processor: Optional[InsightProcessor] = None

        # Extractors (needed before full initialization)
        self.extractor = AtomicExtractor(data_dir=self.paths.data_dir, config=CONFIG)
        self.panel_extractor = PanelExtractor(data_dir=self.paths.data_dir, config=CONFIG)

    async def run(self, process_backlog_first: bool = True):
        """
        Main exploration loop.

        Args:
            process_backlog_first: If True, process any unextracted threads before
                                   starting the main exploration loop.
        """
        self.running = True
        log.info("Starting exploration loop")
        review_host = self.config.get("review_server", {}).get("host", "127.0.0.1")
        review_port = self.config.get("review_server", {}).get("port", 8765)
        log.info(f"Review UI available at: http://{review_host}:{review_port}")

        # Connect to LLMs and initialize components
        await self._connect_and_initialize()

        try:
            # Check for any recovered responses from pool restart
            await self.llm_manager.check_recovered_responses(
                self.thread_manager.load_thread_by_id
            )

            # Check for new seeds and prioritize them
            await self.thread_manager.check_and_spawn_for_new_seeds()

            # Process backlog of unextracted threads first
            if process_backlog_first:
                await self.process_backlog()

            while self.running:
                await self._exploration_cycle()
                await asyncio.sleep(self.orchestration_config["poll_interval"])
        finally:
            await self.llm_manager.disconnect()

    def stop(self):
        """Signal the orchestrator to stop."""
        self.running = False
        log.info("Stop signal received")

    async def cleanup(self):
        """Clean up resources (called after interrupt)."""
        log.info("Cleaning up...")
        await self.llm_manager.disconnect()

    async def _connect_and_initialize(self):
        """Connect to LLMs and initialize dependent components."""
        # Connect to LLMs
        await self.llm_manager.connect()

        # Initialize reviewer if enabled
        if CONFIG.get("review_panel", {}).get("enabled", False):
            try:
                self.reviewer = AutomatedReviewer(
                    gemini_browser=self.llm_manager.gemini,
                    chatgpt_browser=self.llm_manager.chatgpt,
                    config=CONFIG,
                    data_dir=self.paths.data_dir,
                )
                log.info("Initialized automated review panel")

                # Initialize augmenter
                augmenter = None
                if CONFIG.get("augmentation", {}).get("enabled", False):
                    augmenter = get_augmenter(
                        claude_reviewer=self.reviewer.claude_reviewer,
                        config=CONFIG,
                        data_dir=self.paths.data_dir,
                    )
                    if augmenter:
                        log.info("Initialized augmenter for blessed insights")

                # Initialize deduplication checker
                dedup_checker = None
                if CONFIG.get("deduplication", {}).get("enabled", True):
                    dedup_checker = get_dedup_checker(
                        claude_reviewer=self.reviewer.claude_reviewer,
                        config=CONFIG,
                    )

                # Initialize blessed store
                self.blessed_store = BlessedStore(
                    CONFIG,
                    self.paths,
                    self.axioms,
                    augmenter=augmenter,
                    dedup_checker=dedup_checker,
                )

                # Load existing blessed insights into dedup checker
                if dedup_checker:
                    count = self.blessed_store.load_blessed_into_dedup()
                    log.info(f"Initialized deduplication checker with {count} known insights")

                # Initialize insight processor
                self.insight_processor = InsightProcessor(
                    CONFIG,
                    self.paths,
                    self.extractor,
                    self.panel_extractor,
                    reviewer=self.reviewer,
                    dedup_checker=dedup_checker,
                )

            except Exception as e:
                log.warning(f"Could not initialize review panel: {e}")
                self.reviewer = None

        # Create minimal blessed store if reviewer not available
        if self.blessed_store is None:
            self.blessed_store = BlessedStore(CONFIG, self.paths, self.axioms)

        # Create minimal insight processor if reviewer not available
        if self.insight_processor is None:
            self.insight_processor = InsightProcessor(
                CONFIG,
                self.paths,
                self.extractor,
                self.panel_extractor,
            )

    async def process_backlog(self):
        """
        Process all threads that haven't had atomic extraction run yet.

        Scans for threads where:
        - Status is CHUNK_READY or ARCHIVED (synthesis completed)
        - chunks_extracted is False or missing

        For each such thread, runs atomic extraction and review panel.
        """
        log.info("[backlog] Scanning for unprocessed threads...")

        if not self.paths.explorations_dir.exists():
            log.info("[backlog] No explorations directory found")
            return

        # Find all threads needing extraction
        from explorer.src.models import ExplorationThread

        unprocessed = []
        for filepath in self.paths.explorations_dir.glob("*.json"):
            try:
                thread = ExplorationThread.load(filepath)

                # Check if thread is ready for extraction but hasn't been processed
                if thread.status in (ThreadStatus.CHUNK_READY, ThreadStatus.ARCHIVED):
                    if not getattr(thread, 'chunks_extracted', False):
                        unprocessed.append(thread)

            except Exception as e:
                log.warning(f"[backlog] Could not load {filepath}: {e}")

        if not unprocessed:
            log.info("[backlog] No unprocessed threads found")
            return

        log.info(f"[backlog] Found {len(unprocessed)} threads to process")

        # Get an available model for extraction
        model_name, model = self.llm_manager.get_backlog_model()
        if not model:
            log.warning("[backlog] No model available for extraction")
            return

        # In-progress marker file to track what we're working on (for crash recovery)
        progress_file = self.paths.data_dir / ".backlog_in_progress"

        # Process each thread
        for i, thread in enumerate(unprocessed, 1):
            log.info(f"[backlog] Processing {i}/{len(unprocessed)}: {thread.id}")

            # Write in-progress marker (for crash recovery debugging)
            try:
                progress_file.write_text(f"{thread.id}\n{i}/{len(unprocessed)}")
            except Exception:
                pass  # Non-critical, just for debugging

            try:
                await self.insight_processor.extract_and_review(
                    thread,
                    model_name,
                    model,
                    self.llm_manager,
                    self.blessed_store,
                )
            except Exception as e:
                log.error(f"[backlog] Failed to process {thread.id}: {e}")

            # Small delay between threads to avoid rate limits
            if i < len(unprocessed):
                await asyncio.sleep(5)

        # Clear in-progress marker when done
        try:
            if progress_file.exists():
                progress_file.unlink()
        except Exception:
            pass

        log.info(f"[backlog] Completed processing {len(unprocessed)} threads")

    async def _exploration_cycle(self):
        """
        Single cycle of the exploration loop.

        Runs work in parallel across all available models, assigning
        each model to a different thread for maximum throughput.
        """
        # Ensure we're connected to pool (reconnect if needed)
        await self.llm_manager.ensure_connected()

        available_models = self.llm_manager.get_available_models()
        log.info(
            "exploration.cycle.start",
            available_models=list(available_models.keys()),
            gemini_connected=self.llm_manager.gemini is not None,
        )

        if not available_models:
            log.info("exploration.cycle.all_rate_limited")
            await asyncio.sleep(self.orchestration_config["backoff_base"])
            return

        # Collect work items: (thread, model_name, model, task_type)
        work_items = []
        assigned_thread_ids = set()
        selected_seed_ids = set()  # Track seeds selected this cycle to avoid duplicates

        for model_name, model in available_models.items():
            # Select a thread not already assigned to another model
            thread = self.thread_manager.select_thread(
                exclude_ids=assigned_thread_ids,
                exclude_seeds=selected_seed_ids,
            )

            if thread is None:
                # Spawn a new thread if none available, excluding already-selected seeds
                thread = self.thread_manager.spawn_new_thread(exclude_seeds=selected_seed_ids)
                log.info(
                    "exploration.thread.spawned",
                    thread_id=thread.id,
                    topic=thread.topic[:60],
                )
                # Track the seed used by this thread to prevent duplicate selection
                if thread.primary_question_id:
                    selected_seed_ids.add(thread.primary_question_id)
                if thread.related_conjecture_ids:
                    selected_seed_ids.update(thread.related_conjecture_ids)

            assigned_thread_ids.add(thread.id)

            # Determine what work this thread needs
            if thread.needs_exploration:
                work_items.append((thread, model_name, model, "exploration"))
            elif thread.needs_critique:
                work_items.append((thread, model_name, model, "critique"))
            else:
                log.info(
                    "exploration.thread.no_work_needed",
                    thread_id=thread.id,
                    model=model_name,
                )

        if not work_items:
            log.info("exploration.cycle.no_work")
            return

        log.info(
            "exploration.cycle.parallel_work",
            work_count=len(work_items),
            assignments=[(w[1], w[0].id[:8], w[3]) for w in work_items],
        )

        # Run all work in parallel
        tasks = []
        for thread, model_name, model, task_type in work_items:
            if task_type == "exploration":
                task = self._do_exploration_and_save(thread, model_name, model)
            else:
                task = self._do_critique_and_save(thread, model_name, model)
            tasks.append(task)

        # Wait for all parallel work to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count failures and implement exponential backoff
        failure_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failure_count += 1
                thread, model_name, _, task_type = work_items[i]
                log.error(
                    "exploration.parallel_task.failed",
                    thread_id=thread.id,
                    model=model_name,
                    task_type=task_type,
                    error=str(result),
                )

        # Track consecutive failures for backoff
        if failure_count == len(results) and len(results) > 0:
            # All tasks failed - increment consecutive failure counter
            self._consecutive_failures += 1
            backoff = min(
                self.orchestration_config.get("backoff_base", 5) * (2 ** self._consecutive_failures),
                self._max_backoff,
            )
            log.warning(
                "exploration.cycle.all_failed",
                consecutive_failures=self._consecutive_failures,
                backoff_seconds=backoff,
            )
            await asyncio.sleep(backoff)
        else:
            # At least one task succeeded - reset failure counter
            self._consecutive_failures = 0

    async def _do_exploration_and_save(self, thread, model_name, model):
        """Run exploration on a thread and save state."""
        try:
            await self.exploration_engine.do_exploration(
                thread, model_name, model, self.llm_manager
            )

            # Check if thread is ready for synthesis
            if self.synthesis_engine.is_chunk_ready(thread):
                await self.synthesis_engine.synthesize_chunk(
                    thread,
                    self.llm_manager,
                    extract_and_review_fn=self._extract_and_review_wrapper,
                )

            # Save thread state
            thread.save(self.paths.data_dir)
        except Exception as e:
            log.error(
                "exploration.do_exploration.failed",
                thread_id=thread.id,
                model=model_name,
                error=str(e),
            )
            raise

    async def _do_critique_and_save(self, thread, model_name, model):
        """Run critique on a thread and save state."""
        try:
            await self.exploration_engine.do_critique(
                thread, model_name, model, self.llm_manager
            )

            # Check if thread is ready for synthesis
            if self.synthesis_engine.is_chunk_ready(thread):
                await self.synthesis_engine.synthesize_chunk(
                    thread,
                    self.llm_manager,
                    extract_and_review_fn=self._extract_and_review_wrapper,
                )

            # Save thread state
            thread.save(self.paths.data_dir)
        except Exception as e:
            log.error(
                "exploration.do_critique.failed",
                thread_id=thread.id,
                model=model_name,
                error=str(e),
            )
            raise

    async def _extract_and_review_wrapper(self, thread, model_name, model):
        """Wrapper for insight processor extract_and_review."""
        await self.insight_processor.extract_and_review(
            thread,
            model_name,
            model,
            self.llm_manager,
            self.blessed_store,
        )
