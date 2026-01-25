"""
Explorer Module Adapter for the Unified Orchestrator.

Adapts the existing Explorer components to work with the new
orchestration system while preserving all existing functionality.

Based on v4.0 design specification Phase 3.
"""

import asyncio
from pathlib import Path
from typing import Optional, Any
import yaml

from shared.logging import get_logger

from orchestrator.adapters import (
    ModuleInterface,
    PromptContext,
    TaskResult,
    TaskType,
    run_in_executor,
)
from orchestrator.models import Task

from explorer.src.models import (
    AxiomStore,
    ExplorationThread,
    ThreadStatus,
    ExchangeRole,
)
from explorer.src.storage.db import Database
from explorer.src.storage import ExplorerPaths
from explorer.src.chunking import AtomicExtractor, PanelExtractor, get_dedup_checker
from explorer.src.review_panel import AutomatedReviewer
from explorer.src.augmentation import get_augmenter

from explorer.src.orchestration import (
    LLMManager,
    ThreadManager,
    ExplorationEngine,
    SynthesisEngine,
    InsightProcessor,
    BlessedStore,
)

log = get_logger("explorer", "adapter")


# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


class ExplorerAdapter(ModuleInterface):
    """
    Adapter that wraps Explorer functionality for the unified orchestrator.

    Maps orchestrator tasks to Explorer operations:
    - exploration -> ExplorationEngine.do_exploration()
    - critique -> ExplorationEngine.do_critique()
    - synthesis -> SynthesisEngine.synthesize_chunk()
    - review -> InsightProcessor.extract_and_review()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the Explorer adapter.

        Args:
            data_dir: Optional custom data directory
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.paths = ExplorerPaths(self.data_dir)

        # Core data stores
        self.db = Database(self.paths.database_file)
        self.axioms = AxiomStore(self.paths.data_dir)

        # Configuration
        self.config = CONFIG
        self.orchestration_config = CONFIG["orchestration"]

        # Managers (initialized in initialize())
        self.llm_manager: Optional[LLMManager] = None
        self.thread_manager: Optional[ThreadManager] = None
        self.exploration_engine: Optional[ExplorationEngine] = None
        self.synthesis_engine: Optional[SynthesisEngine] = None
        self.insight_processor: Optional[InsightProcessor] = None
        self.blessed_store: Optional[BlessedStore] = None
        self.reviewer: Optional[AutomatedReviewer] = None

        # Extractors
        self.extractor = AtomicExtractor(data_dir=self.paths.data_dir, config=CONFIG)
        self.panel_extractor = PanelExtractor(data_dir=self.paths.data_dir, config=CONFIG)

        # Track active threads being worked on
        self._active_threads: dict[str, ExplorationThread] = {}

        self._initialized = False

    @property
    def module_name(self) -> str:
        return "explorer"

    @property
    def supported_task_types(self) -> list[str]:
        return [
            TaskType.EXPLORATION.value,
            TaskType.CRITIQUE.value,
            TaskType.SYNTHESIS.value,
            TaskType.REVIEW.value,
        ]

    async def initialize(self) -> bool:
        """Initialize Explorer components."""
        if self._initialized:
            return True

        try:
            log.info("explorer.adapter.initializing")

            # Initialize LLM manager (but don't connect - orchestrator handles LLM)
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

            # Initialize blessed store (minimal, without reviewer)
            self.blessed_store = BlessedStore(CONFIG, self.paths, self.axioms)

            # Initialize insight processor (minimal, without reviewer)
            self.insight_processor = InsightProcessor(
                CONFIG,
                self.paths,
                self.extractor,
                self.panel_extractor,
            )

            # Check for new seeds
            await self.thread_manager.check_and_spawn_for_new_seeds()

            self._initialized = True
            log.info("explorer.adapter.initialized")
            return True

        except Exception as e:
            log.exception(e, "explorer.adapter.init_failed", {})
            return False

    async def shutdown(self):
        """Cleanup Explorer resources."""
        log.info("explorer.adapter.shutting_down")

        # Save any active threads
        for thread in self._active_threads.values():
            try:
                thread.save(self.paths.data_dir)
            except Exception as e:
                log.error("explorer.adapter.thread_save_failed",
                         thread_id=thread.id, error=str(e))

        self._active_threads.clear()
        self._initialized = False

        log.info("explorer.adapter.shutdown_complete")

    async def get_pending_work(self) -> list[dict]:
        """
        Get list of pending work from Explorer.

        Returns work items for threads that need exploration, critique, or synthesis.
        """
        if not self._initialized:
            return []

        work_items = []

        # Check for new seeds first
        await self.thread_manager.check_and_spawn_for_new_seeds()

        # Get all active threads
        threads = self._get_active_threads()

        for thread in threads:
            # Check synthesis readiness first (higher priority)
            if self.synthesis_engine.is_chunk_ready(thread):
                work_items.append({
                    "task_type": TaskType.SYNTHESIS.value,
                    "key": f"synthesis:{thread.id}",
                    "payload": {
                        "thread_id": thread.id,
                    },
                    "requires_deep_mode": True,  # Synthesis always uses deep mode
                    "priority": 60,
                })

            # Then check for exploration needs
            elif thread.needs_exploration:
                # Get seed priority for exploration
                seed_priority = self._get_seed_priority(thread)
                work_items.append({
                    "task_type": TaskType.EXPLORATION.value,
                    "key": f"exploration:{thread.id}",
                    "payload": {
                        "thread_id": thread.id,
                        "seed_priority": seed_priority,
                    },
                    "requires_deep_mode": self._should_use_deep_mode(thread, "exploration"),
                    "priority": 50 + seed_priority * 10,
                })

            # Then check for critique needs
            elif thread.needs_critique:
                work_items.append({
                    "task_type": TaskType.CRITIQUE.value,
                    "key": f"critique:{thread.id}",
                    "payload": {
                        "thread_id": thread.id,
                    },
                    "requires_deep_mode": self._should_use_deep_mode(thread, "critique"),
                    "priority": 45,
                    "preferred_backend": "chatgpt",  # Critique prefers ChatGPT
                })

        # Check for threads needing review (extraction)
        unreviewed = self._get_unreviewed_threads()
        for thread in unreviewed:
            work_items.append({
                "task_type": TaskType.REVIEW.value,
                "key": f"review:{thread.id}",
                "payload": {
                    "thread_id": thread.id,
                },
                "requires_deep_mode": False,
                "priority": 55,
            })

        log.debug("explorer.adapter.pending_work",
                 count=len(work_items),
                 types=[w["task_type"] for w in work_items])

        return work_items

    async def build_prompt(self, task: Task) -> PromptContext:
        """Build prompt for an Explorer task."""
        thread = self._get_thread(task.payload.get("thread_id"))
        if not thread:
            raise ValueError(f"Thread not found: {task.payload.get('thread_id')}")

        task_type = task.task_type

        if task_type == TaskType.EXPLORATION.value:
            prompt, images = self.exploration_engine._build_exploration_prompt(thread)
            return PromptContext(
                prompt=prompt,
                images=[img.to_dict() for img in images] if images else [],
                requires_deep_mode=self._should_use_deep_mode(thread, "exploration"),
                metadata={"thread_id": thread.id},
            )

        elif task_type == TaskType.CRITIQUE.value:
            prompt = self.exploration_engine._build_critique_prompt(thread)
            images = self.exploration_engine._get_thread_images(thread)
            return PromptContext(
                prompt=prompt,
                images=[img.to_dict() for img in images] if images else [],
                requires_deep_mode=self._should_use_deep_mode(thread, "critique"),
                preferred_backend="chatgpt",
                metadata={"thread_id": thread.id},
            )

        elif task_type == TaskType.SYNTHESIS.value:
            prompt = self.synthesis_engine._build_synthesis_prompt(thread)
            return PromptContext(
                prompt=prompt,
                requires_deep_mode=True,
                metadata={"thread_id": thread.id},
            )

        elif task_type == TaskType.REVIEW.value:
            # Review uses existing extraction logic
            return PromptContext(
                prompt="",  # Review is handled differently
                metadata={"thread_id": thread.id, "is_review": True},
            )

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def handle_result(self, task: Task, result: TaskResult) -> bool:
        """Handle the result of a task execution."""
        thread = self._get_thread(task.payload.get("thread_id"))
        if not thread:
            log.error("explorer.adapter.thread_not_found",
                     thread_id=task.payload.get("thread_id"))
            return False

        task_type = task.task_type

        try:
            if task_type == TaskType.EXPLORATION.value:
                return await self._handle_exploration_result(thread, task, result)

            elif task_type == TaskType.CRITIQUE.value:
                return await self._handle_critique_result(thread, task, result)

            elif task_type == TaskType.SYNTHESIS.value:
                return await self._handle_synthesis_result(thread, task, result)

            elif task_type == TaskType.REVIEW.value:
                return await self._handle_review_result(thread, task, result)

            else:
                log.warning("explorer.adapter.unknown_task_type", task_type=task_type)
                return False

        except Exception as e:
            log.exception(e, "explorer.adapter.handle_result_error", {
                "task_id": task.id,
                "task_type": task_type,
            })
            return False

    async def on_task_failed(self, task: Task, error: str):
        """Handle task failure."""
        log.error("explorer.adapter.task_failed",
                 task_id=task.id,
                 task_type=task.task_type,
                 thread_id=task.payload.get("thread_id"),
                 error=error)

        # Save thread state to preserve any partial progress
        thread = self._get_thread(task.payload.get("thread_id"))
        if thread:
            thread.save(self.paths.data_dir)

    async def get_system_state(self) -> dict:
        """Get Explorer's current system state for priority computation."""
        threads = self._get_active_threads()

        return {
            "active_threads": len(threads),
            "threads_needing_exploration": sum(1 for t in threads if t.needs_exploration),
            "threads_needing_critique": sum(1 for t in threads if t.needs_critique),
            "threads_ready_for_synthesis": sum(
                1 for t in threads if self.synthesis_engine.is_chunk_ready(t)
            ),
            "blessed_insights_count": len(self.blessed_store.get_blessed_insights()) if self.blessed_store else 0,
        }

    # ==================== Private Helper Methods ====================

    def _get_active_threads(self) -> list[ExplorationThread]:
        """Get all active exploration threads."""
        threads = []

        if not self.paths.explorations_dir.exists():
            return threads

        for filepath in self.paths.explorations_dir.glob("*.json"):
            try:
                thread = ExplorationThread.load(filepath)
                if thread.status == ThreadStatus.ACTIVE:
                    threads.append(thread)
            except Exception as e:
                log.warning("explorer.adapter.thread_load_failed",
                           path=str(filepath), error=str(e))

        return threads

    def _get_unreviewed_threads(self) -> list[ExplorationThread]:
        """Get threads that need extraction/review."""
        threads = []

        if not self.paths.explorations_dir.exists():
            return threads

        for filepath in self.paths.explorations_dir.glob("*.json"):
            try:
                thread = ExplorationThread.load(filepath)
                if thread.status in (ThreadStatus.CHUNK_READY, ThreadStatus.ARCHIVED):
                    if not getattr(thread, 'chunks_extracted', False):
                        threads.append(thread)
            except Exception:
                pass

        return threads

    def _get_thread(self, thread_id: str) -> Optional[ExplorationThread]:
        """Get a thread by ID, using cache if available."""
        if thread_id in self._active_threads:
            return self._active_threads[thread_id]

        # Load from disk
        filepath = self.paths.explorations_dir / f"{thread_id}.json"
        if filepath.exists():
            thread = ExplorationThread.load(filepath)
            self._active_threads[thread_id] = thread
            return thread

        return None

    def _get_seed_priority(self, thread: ExplorationThread) -> int:
        """Get priority based on seed configuration."""
        if thread.primary_question_id:
            seed = self.axioms.get_seed_by_id(thread.primary_question_id)
            if seed and hasattr(seed, 'priority'):
                return seed.priority
        return 0

    def _should_use_deep_mode(self, thread: ExplorationThread, phase: str) -> bool:
        """Determine if deep mode should be used for this thread/phase."""
        # Check exchange count threshold
        min_exchanges = self.config.get("deep_modes", {}).get("min_exchanges_for_deep", 4)
        if len(thread.exchanges) < min_exchanges:
            return False

        # Check profundity signals
        if hasattr(thread, 'profundity_signals') and thread.profundity_signals:
            return True

        return False

    async def _handle_exploration_result(
        self,
        thread: ExplorationThread,
        task: Task,
        result: TaskResult
    ) -> bool:
        """Handle exploration result."""
        if not result.success or not result.response:
            return False

        # Extract structured response
        clean_response = self.exploration_engine._extract_structured_response(
            result.response,
            ["NEW_INSIGHTS", "CONNECTIONS", "QUESTIONS"]
        )

        # Add exchange to thread
        thread.add_exchange(
            role=ExchangeRole.EXPLORER,
            model=task.preferred_backend or "unknown",
            prompt=task.payload.get("prompt", ""),
            response=clean_response,
            deep_mode_used=result.deep_mode_used,
        )

        # Save thread
        thread.save(self.paths.data_dir)

        log.info("explorer.adapter.exploration_complete",
                thread_id=thread.id,
                response_len=len(clean_response),
                deep_mode=result.deep_mode_used)

        return True

    async def _handle_critique_result(
        self,
        thread: ExplorationThread,
        task: Task,
        result: TaskResult
    ) -> bool:
        """Handle critique result."""
        if not result.success or not result.response:
            return False

        # Extract structured response
        clean_response = self.exploration_engine._extract_structured_response(
            result.response,
            ["CRITICAL_ISSUES", "PROMISING_DIRECTIONS", "PROBING_QUESTIONS"]
        )

        # Add exchange to thread
        thread.add_exchange(
            role=ExchangeRole.CRITIC,
            model=task.preferred_backend or "chatgpt",
            prompt=task.payload.get("prompt", ""),
            response=clean_response,
            deep_mode_used=result.deep_mode_used,
        )

        # Save thread
        thread.save(self.paths.data_dir)

        log.info("explorer.adapter.critique_complete",
                thread_id=thread.id,
                response_len=len(clean_response),
                deep_mode=result.deep_mode_used)

        return True

    async def _handle_synthesis_result(
        self,
        thread: ExplorationThread,
        task: Task,
        result: TaskResult
    ) -> bool:
        """Handle synthesis result."""
        if not result.success or not result.response:
            return False

        # Parse synthesis response to create chunk
        chunk_data = self.synthesis_engine._parse_synthesis_response(result.response)
        if not chunk_data:
            log.warning("explorer.adapter.synthesis_parse_failed", thread_id=thread.id)
            return False

        # Create and save chunk
        chunk = self.synthesis_engine._create_chunk(thread, chunk_data)

        # Mark thread as chunk ready
        thread.status = ThreadStatus.CHUNK_READY
        thread.save(self.paths.data_dir)

        log.info("explorer.adapter.synthesis_complete",
                thread_id=thread.id,
                chunk_id=chunk.id if chunk else None)

        return True

    async def _handle_review_result(
        self,
        thread: ExplorationThread,
        task: Task,
        result: TaskResult
    ) -> bool:
        """Handle review/extraction result."""
        # Review is a multi-step process handled by InsightProcessor
        # For now, mark as needing the full extraction pipeline
        log.info("explorer.adapter.review_triggered", thread_id=thread.id)

        # The actual review will be done through the existing pipeline
        # This is a placeholder for future integration
        thread.chunks_extracted = True
        thread.save(self.paths.data_dir)

        return True
