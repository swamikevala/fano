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

        # Process each thread
        for i, thread in enumerate(unprocessed, 1):
            log.info(f"[backlog] Processing {i}/{len(unprocessed)}: {thread.id}")

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

        log.info(f"[backlog] Completed processing {len(unprocessed)} threads")

    async def _exploration_cycle(self):
        """Single cycle of the exploration loop."""
        available_models = self.llm_manager.get_available_models()
        log.info(
            f"Available models: {list(available_models.keys())} "
            f"(gemini={self.llm_manager.gemini is not None})"
        )

        if not available_models:
            log.info("All models rate-limited, waiting...")
            await asyncio.sleep(self.orchestration_config["backoff_base"])
            return

        # Get or create a thread to work on
        thread = self.thread_manager.select_thread()

        if thread is None:
            # Spawn a new thread
            thread = self.thread_manager.spawn_new_thread()
            log.info(f"Spawned new thread: {thread.topic[:60]}...")

        # Perform work on the thread
        if thread.needs_exploration:
            model_name = self.llm_manager.select_model_for_task("exploration", available_models)
            if model_name:
                model = available_models[model_name]
                await self.exploration_engine.do_exploration(
                    thread, model_name, model, self.llm_manager
                )
        elif thread.needs_critique:
            # Use first available model (critique engine does its own selection)
            model_name = list(available_models.keys())[0]
            model = available_models[model_name]
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

    async def _extract_and_review_wrapper(self, thread, model_name, model):
        """Wrapper for insight processor extract_and_review."""
        await self.insight_processor.extract_and_review(
            thread,
            model_name,
            model,
            self.llm_manager,
            self.blessed_store,
        )
