"""
Session management for the documenter.

Handles initialization, cleanup, configuration loading, and
component setup for a documenter session.
"""

from datetime import time
from pathlib import Path
from typing import Optional

import yaml

from shared.logging import get_logger
from shared.deduplication import (
    DeduplicationChecker,
    ContentItem,
    ContentType,
    load_dedup_config,
)

from llm.src.client import LLMClient
from llm.src.consensus import ConsensusReviewer

from .document import Document, Section
from .concepts import ConceptTracker
from .tasks import TaskBuilder
from .opportunities import OpportunityFinder
from .review import ReviewManager
from .snapshots import SnapshotManager
from .annotations import AnnotationManager

log = get_logger("documenter", "session")

# Project root for resolving relative paths
FANO_ROOT = Path(__file__).resolve().parent.parent


class SessionManager:
    """
    Manages documenter session lifecycle.

    Handles:
    - Configuration loading
    - Component initialization
    - Resource cleanup
    - Deduplication setup
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            config_path: Path to config file (default: config.yaml in project root)
        """
        self.config = self._load_config(config_path)

        # Core components (initialized later)
        self.document: Optional[Document] = None
        self.concept_tracker: Optional[ConceptTracker] = None
        self.task_builder: Optional[TaskBuilder] = None
        self.llm_client: Optional[LLMClient] = None
        self.consensus: Optional[ConsensusReviewer] = None
        self.opportunity_finder: Optional[OpportunityFinder] = None
        self.review_manager: Optional[ReviewManager] = None
        self.snapshot_manager: Optional[SnapshotManager] = None
        self.annotation_manager: Optional[AnnotationManager] = None
        self.dedup_checker: Optional[DeduplicationChecker] = None

        # Configuration values
        self._extract_config_values()

        # Session state
        self.consensus_calls = 0
        self.exhausted = False

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from file."""
        if config_path is None:
            config_path = FANO_ROOT / "config.yaml"

        if config_path.exists():
            try:
                return yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("session.config.load_error", path=str(config_path), error=str(e))

        return {}

    def _extract_config_values(self):
        """Extract configuration values from config dict."""
        doc_config = self.config.get("documenter", {})

        # Document paths
        self.doc_path = Path(doc_config.get("document", {}).get("path", "document/main.md"))
        self.archive_dir = Path(doc_config.get("document", {}).get("archive_dir", "document/archive"))

        # Snapshot time
        snapshot_time_str = doc_config.get("document", {}).get("snapshot_time", "00:00")
        hour, minute = map(int, snapshot_time_str.split(":"))
        self.snapshot_time = time(hour, minute)

        # Blessed insights directory
        blessed_dir = Path(doc_config.get("inputs", {}).get("blessed_insights_dir", "blessed_insights"))
        if not blessed_dir.is_absolute():
            blessed_dir = FANO_ROOT / blessed_dir
        self.blessed_dir = blessed_dir

        # Guidance file
        guidance_path = doc_config.get("inputs", {}).get("guidance_file", "document/guidance.md")
        if not Path(guidance_path).is_absolute():
            guidance_path = FANO_ROOT / guidance_path
        self.guidance_path = Path(guidance_path)

        # Limits and thresholds
        self.max_context_tokens = doc_config.get("context", {}).get("max_tokens", 8000)
        self.max_disputes = doc_config.get("termination", {}).get("max_consecutive_disputes", 3)
        self.max_consensus_calls = doc_config.get("termination", {}).get("max_consensus_calls_per_session", 100)
        self.max_age_days = doc_config.get("review", {}).get("max_age_days", 7)
        self.review_allocation = doc_config.get("work_allocation", {}).get("review_existing", 30)

        # LLM settings
        self.use_deep_mode = self.config.get("llm", {}).get("consensus", {}).get("use_deep_mode", False)

    def _load_guidance(self) -> str:
        """Load optional guidance file."""
        if self.guidance_path.exists():
            try:
                text = self.guidance_path.read_text(encoding="utf-8")
                log.info("session.guidance.loaded", path=str(self.guidance_path))
                return text
            except Exception as e:
                log.warning("session.guidance.load_error", error=str(e))
        return ""

    async def initialize(self) -> str:
        """
        Initialize all components.

        Returns:
            Guidance text loaded from file
        """
        log.info("session.initializing")

        # Load document
        self.document = Document(self.doc_path)
        if not self.document.load():
            log.warning("session.document.creating_seed")
            self._create_seed_document()
            self.document.load()

        # Initialize components
        self.concept_tracker = ConceptTracker(self.document)
        self.annotation_manager = AnnotationManager(self.document.path)
        self.task_builder = TaskBuilder(max_context_tokens=self.max_context_tokens)

        # LLM client with pool URL from config
        pool_config = self.config.get("llm", {}).get("pool", {})
        pool_host = pool_config.get("host", "127.0.0.1")
        pool_port = pool_config.get("port", 9000)
        pool_url = f"http://{pool_host}:{pool_port}"
        self.llm_client = LLMClient(pool_url=pool_url)
        self.consensus = ConsensusReviewer(self.llm_client)

        # Verify we have enough backends for consensus
        available_backends = await self.llm_client.get_available_backends()
        if len(available_backends) < 2:
            log.error(
                "session.insufficient_backends",
                available=available_backends,
                required=2,
            )
            await self.llm_client.close()
            raise RuntimeError(
                f"Documenter requires at least 2 LLM backends for consensus, but only {len(available_backends)} available: {available_backends}. "
                f"Either start the pool service (for gemini/chatgpt) or set API keys (ANTHROPIC_API_KEY, OPENROUTER_API_KEY)."
            )
        log.info("session.backends_available", backends=available_backends)

        self.opportunity_finder = OpportunityFinder(
            self.document,
            self.concept_tracker,
            self.blessed_dir,
            self.max_disputes,
        )

        self.review_manager = ReviewManager(
            self.document,
            self.max_age_days,
            self.review_allocation,
        )

        self.snapshot_manager = SnapshotManager(
            self.document,
            self.archive_dir,
            self.snapshot_time,
        )

        # Initialize deduplication
        await self._initialize_dedup()

        # Load guidance
        guidance_text = self._load_guidance()

        log.info(
            "session.initialized",
            sections=len(self.document.sections),
            concepts=len(self.concept_tracker.get_established_concepts()),
            pending_opportunities=self.opportunity_finder.get_pending_count(),
            dedup_known_items=self.dedup_checker.known_count,
            use_deep_mode=self.use_deep_mode,
        )

        return guidance_text

    async def _initialize_dedup(self):
        """Initialize deduplication checker."""
        self._dedup_config = load_dedup_config()

        self.dedup_checker = DeduplicationChecker(
            llm_callback=self._dedup_llm_callback,
            use_signature_check=self._dedup_config.get("use_signature_check", True),
            use_heuristic_check=self._dedup_config.get("use_heuristic_check", False),
            use_llm_check=self._dedup_config.get("use_llm_check", True),
            use_batch_llm=self._dedup_config.get("use_batch_llm", True),
            batch_size=self._dedup_config.get("batch_size", 20),
            stats_log_interval=self._dedup_config.get("stats_log_interval", 50),
        )

        # Load existing document sections into dedup checker
        for section in self.document.sections:
            self.dedup_checker.add_content(ContentItem(
                id=section.id,
                text=section.content,
                content_type=ContentType.SECTION,
            ))

    async def _dedup_llm_callback(self, prompt: str) -> str:
        """LLM callback for deduplication checks."""
        model = self._dedup_config.get("model", "claude-sonnet-4-20250514")
        timeout = self._dedup_config.get("llm_timeout", 60)

        response = await self.llm_client.send(
            "claude",
            prompt,
            model=model,
            timeout_seconds=timeout,
        )
        if response.success:
            return response.text
        raise RuntimeError(f"LLM call failed: {response.error}")

    def _create_seed_document(self):
        """Create the seed document if it doesn't exist."""
        seed_content = '''# The Principles of Creation

1. The first manifestation is always in the form of three.
2. The fundamental design of the Cosmos is common. From that design it has evolved into such complex possibilities.
3. It is the perfection of geometry which is holding the existence together.
4. The only way to articulate the profound intelligence of Creation is in complex geometrical forms.

This document explores the mathematical structure that emerges from
this principle, and its appearance across traditions.

We seek what is natural, elegant, beautiful, interesting, and
inevitable — structure that must exist, not structure we impose.

'''
        self.document.path.parent.mkdir(parents=True, exist_ok=True)
        self.document.path.write_text(seed_content, encoding="utf-8")
        log.info("session.seed_created", path=str(self.document.path))

    async def cleanup(self):
        """Clean up resources."""
        if self.llm_client:
            await self.llm_client.close()
            log.info("session.cleanup.complete")

    def check_budget(self) -> bool:
        """
        Check if consensus call budget is exhausted.

        Returns:
            True if budget exhausted, False otherwise
        """
        if self.consensus_calls >= self.max_consensus_calls:
            log.info(
                "session.budget_exhausted",
                consensus_calls=self.consensus_calls,
            )
            self.exhausted = True
            return True
        return False

    def increment_consensus_calls(self):
        """Increment the consensus call counter."""
        self.consensus_calls += 1

    def log_summary(self):
        """Log session summary."""
        review_stats = self.review_manager.get_review_stats()

        log.info(
            "session.summary",
            sections=len(self.document.sections),
            concepts=len(self.concept_tracker.get_established_concepts()),
            consensus_calls=self.consensus_calls,
            pending_opportunities=self.opportunity_finder.get_pending_count(),
            **review_stats,
        )
