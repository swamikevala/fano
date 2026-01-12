"""
Insight Processor - Handles insight extraction and review.

This module centralizes:
- Insight extraction from threads
- Deduplication
- Automated review loop
- Review outcome handling
"""

import asyncio
import logging
from typing import Optional, Any

from explorer.src.browser import GeminiQuotaExhausted
from explorer.src.models import ExplorationThread
from explorer.src.chunking import (
    AtomicExtractor,
    AtomicInsight,
    InsightStatus,
    PanelExtractor,
    DeduplicationChecker,
)
from explorer.src.review_panel import AutomatedReviewer
from explorer.src.storage import ExplorerPaths

logger = logging.getLogger(__name__)


class InsightProcessor:
    """
    Handles insight extraction and automated review.

    Responsible for:
    - Extracting atomic insights from threads
    - Deduplicating against known insights
    - Running priority-based automated review
    - Handling review outcomes
    """

    def __init__(
        self,
        config: dict,
        paths: ExplorerPaths,
        extractor: AtomicExtractor,
        panel_extractor: PanelExtractor,
        reviewer: Optional[AutomatedReviewer] = None,
        dedup_checker: Optional[DeduplicationChecker] = None,
    ):
        """
        Initialize insight processor.

        Args:
            config: Full configuration dict
            paths: ExplorerPaths instance
            extractor: AtomicExtractor for single-LLM extraction
            panel_extractor: PanelExtractor for multi-LLM extraction
            reviewer: Optional AutomatedReviewer
            dedup_checker: Optional DeduplicationChecker
        """
        self.config = config
        self.paths = paths
        self.extractor = extractor
        self.panel_extractor = panel_extractor
        self.reviewer = reviewer
        self.dedup_checker = dedup_checker

        # Use panel extraction by default (can be overridden in config)
        chunking_config = config.get("chunking", {})
        self.use_panel_extraction = chunking_config.get("use_panel_extraction", True)

    async def extract_and_review(
        self,
        thread: ExplorationThread,
        model_name: str,
        model: Any,
        llm_manager,
        blessed_store,
    ) -> None:
        """
        Extract atomic insights from a thread and run automated review.

        Args:
            thread: Thread to process
            model_name: Model name for fallback extraction
            model: Model instance for fallback extraction
            llm_manager: LLMManager for LLM access
            blessed_store: BlessedStore for blessed insights
        """
        if thread.chunks_extracted:
            logger.info(f"[{thread.id}] Already extracted, skipping")
            return

        # Check if insights already exist for this thread (from previous interrupted run)
        existing_insights = self._get_existing_insights_for_thread(thread.id)
        if existing_insights:
            logger.info(
                f"[{thread.id}] Found {len(existing_insights)} existing insights, skipping extraction"
            )
            review_config = self.config.get("review_panel", {})
            if self.reviewer and review_config.get("enabled", False):
                unreviewed = [
                    i for i in existing_insights if not getattr(i, "reviewed_at", None)
                ]
                if unreviewed:
                    logger.info(
                        f"[{thread.id}] Running review on {len(unreviewed)} unreviewed insights"
                    )
                    blessed_summary = blessed_store.get_blessed_summary()
                    await self._review_insights(unreviewed, blessed_summary, blessed_store)
            thread.chunks_extracted = True
            thread.extraction_note = f"Resumed with {len(existing_insights)} existing insights"
            thread.save(self.paths.data_dir)
            return

        try:
            insights = await self._extract_insights(
                thread, model_name, model, llm_manager, blessed_store
            )

            if not insights:
                thread.chunks_extracted = True
                thread.save(self.paths.data_dir)
                return

            await self._save_and_review_insights(thread, insights, blessed_store)

        except Exception as e:
            logger.error(f"[{thread.id}] Extraction failed: {e}")
            thread.extraction_note = f"Extraction error: {str(e)}"
            await asyncio.to_thread(thread.save, self.paths.data_dir)

    async def _extract_insights(
        self,
        thread: ExplorationThread,
        model_name: str,
        model: Any,
        llm_manager,
        blessed_store,
    ) -> Optional[list[AtomicInsight]]:
        """
        Extract and deduplicate insights from a thread.

        Returns:
            List of unique insights, or None if extraction failed/empty.
        """
        # Get Claude reviewer for extraction
        claude_reviewer = None
        if self.reviewer:
            claude_reviewer = self.reviewer.claude_reviewer

        # Get blessed insights for dependency context
        blessed_insights = blessed_store.get_blessed_insights()

        # Choose extraction method based on config
        if self.use_panel_extraction:
            logger.info(f"[{thread.id}] Starting PANEL extraction (all 3 LLMs)")
            insights = await self.panel_extractor.extract_from_thread(
                thread=thread,
                gemini=llm_manager.gemini,
                chatgpt=llm_manager.chatgpt,
                claude_reviewer=claude_reviewer,
                blessed_insights=blessed_insights,
            )
        else:
            author = (
                "Claude Opus"
                if claude_reviewer and claude_reviewer.is_available()
                else model_name
            )
            logger.info(f"[{thread.id}] Starting single-LLM extraction with {author}")
            insights = await self.extractor.extract_from_thread(
                thread=thread,
                claude_reviewer=claude_reviewer,
                fallback_model=model,
                fallback_model_name=model_name,
                blessed_insights=blessed_insights,
            )

        if not insights:
            # Check if extraction_note indicates an error
            if thread.extraction_note and "failed" in thread.extraction_note.lower():
                logger.warning(
                    f"[{thread.id}] Extraction failed, will retry: {thread.extraction_note}"
                )
                return None
            logger.info(f"[{thread.id}] No insights extracted")
            thread.extraction_note = "No insights met extraction criteria"
            return None

        logger.info(f"[{thread.id}] Extracted {len(insights)} atomic insights")

        # Deduplicate insights against known blessed insights
        if self.dedup_checker:
            unique_insights = []
            duplicate_count = 0

            for insight in insights:
                is_dup, dup_of_id = await self.dedup_checker.is_duplicate(
                    new_text=insight.insight,
                    new_id=insight.id,
                )
                if is_dup:
                    logger.info(
                        f"[{thread.id}] Skipping duplicate insight [{insight.id}] (similar to {dup_of_id})"
                    )
                    duplicate_count += 1
                else:
                    unique_insights.append(insight)
                    self.dedup_checker.add_known_insight(insight.id, insight.insight)

            if duplicate_count > 0:
                logger.info(f"[{thread.id}] Filtered out {duplicate_count} duplicate insights")

            insights = unique_insights

        if not insights:
            logger.info(f"[{thread.id}] All insights were duplicates")
            thread.extraction_note = "All insights were duplicates of existing ones"
            return None

        return insights

    async def _save_and_review_insights(
        self,
        thread: ExplorationThread,
        insights: list[AtomicInsight],
        blessed_store,
    ) -> None:
        """Save insights to disk and run automated review if enabled."""
        self.paths.chunks_dir.mkdir(parents=True, exist_ok=True)

        for insight in insights:
            await asyncio.to_thread(insight.save, self.paths.chunks_dir)

        # Run automated review if enabled
        review_config = self.config.get("review_panel", {})
        if self.reviewer and review_config.get("enabled", False):
            blessed_summary = blessed_store.get_blessed_summary()
            await self._review_insights(insights, blessed_summary, blessed_store)

        # Mark thread as extracted
        thread.chunks_extracted = True
        await asyncio.to_thread(thread.save, self.paths.data_dir)
        logger.info(f"[{thread.id}] Extraction and review complete")

    async def _review_insights(
        self,
        insights: list[AtomicInsight],
        blessed_summary: str,
        blessed_store,
    ) -> None:
        """
        Run automated review on extracted insights using priority-based selection.

        Always processes the highest priority item next. If a higher priority
        item appears (via UI), pauses the current review and switches to it.
        """
        logger.info(f"Starting priority-based review of {len(insights)} insights")

        # Process insights by priority until none remain
        processed_ids = set()
        max_iterations = len(insights) * 3  # Safety limit

        for iteration in range(max_iterations):
            # Get the highest priority pending insight
            insight_id, priority, rounds_completed = self.reviewer.get_highest_priority_pending()

            if not insight_id:
                logger.info("No more pending insights to review")
                break

            if insight_id in processed_ids:
                # Already tried this one, may be stuck - skip for now
                logger.warning(f"[{insight_id}] Already processed, skipping to avoid loop")
                break

            # Load the insight
            insight = self._load_insight_by_id(insight_id)
            if not insight:
                logger.warning(f"[{insight_id}] Could not load insight, skipping")
                processed_ids.add(insight_id)
                continue

            try:
                logger.info(
                    f"Reviewing [{insight.id}] priority={priority} (rounds_completed={rounds_completed})"
                )

                review = await self.reviewer.review_insight(
                    chunk_id=insight.id,
                    insight_text=insight.insight,
                    confidence=insight.confidence,
                    tags=insight.tags,
                    dependencies=insight.depends_on,
                    blessed_axioms_summary=blessed_summary,
                    priority=priority,
                    check_priority_switches=True,
                )

                # Check if review was paused for higher priority item
                if review.is_paused:
                    logger.info(
                        f"[{insight.id}] Paused for higher priority item {review.paused_for_id}"
                    )
                    continue

                # Review completed - mark as processed
                processed_ids.add(insight.id)

                # Handle the review outcome
                await self._handle_review_outcome(insight, review, blessed_store)

            except GeminiQuotaExhausted as e:
                self._handle_quota_exhausted(insight, e)
                break

            except Exception as e:
                logger.error(f"[{insight.id}] Review failed: {e}")
                insight.status = InsightStatus.PENDING
                await asyncio.to_thread(insight.save, self.paths.chunks_dir)
                processed_ids.add(insight.id)

    async def _handle_review_outcome(
        self,
        insight: AtomicInsight,
        review,
        blessed_store,
    ) -> None:
        """Handle the outcome of an insight review."""
        # Get outcome action
        action = self.reviewer.get_outcome_action(review)
        logger.info(f"[{insight.id}] Review outcome: {review.final_rating} ({action})")

        # Apply rating to insight
        insight.rating = review.final_rating
        insight.is_disputed = review.is_disputed
        insight.reviewed_at = review.reviewed_at

        # If the insight was modified during deliberation, update the text
        if review.final_insight_text and review.final_insight_text != insight.insight:
            logger.info(f"[{insight.id}] Using modified insight text from deliberation")
            insight.insight = review.final_insight_text

        # Update status based on rating
        if review.final_rating == "⚡":
            insight.status = InsightStatus.BLESSED
            # Build review summary for augmentation context
            review_summary = self._build_review_summary(review)
            # Add to blessed insights and augment
            await blessed_store.bless_insight(insight, review_summary)
        elif review.final_rating == "✗":
            insight.status = InsightStatus.REJECTED
        else:
            insight.status = InsightStatus.INTERESTING

        # Save updated insight
        await asyncio.to_thread(insight.save, self.paths.chunks_dir)

    def _handle_quota_exhausted(self, insight: AtomicInsight, error: GeminiQuotaExhausted) -> None:
        """Handle Gemini Deep Think quota exhaustion."""
        logger.error(f"[{insight.id}] Gemini Deep Think quota exhausted")
        print(f"\n{'='*60}")
        print("GEMINI DEEP THINK QUOTA EXHAUSTED")
        print(f"{'='*60}")
        print(f"\nDeep Think mode is unavailable until: {error.resume_time}")
        print("\nReviews require Gemini Deep Think for quality analysis.")
        print("Progress has been saved and can be resumed later.")
        print(f"\nPlease wait until {error.resume_time} before running reviews again.")
        print(f"{'='*60}\n")

        # Save the insight in its current state (pending)
        insight.status = InsightStatus.PENDING
        asyncio.create_task(asyncio.to_thread(insight.save, self.paths.chunks_dir))

    def _load_insight_by_id(self, insight_id: str) -> Optional[AtomicInsight]:
        """Load an insight by ID from any status directory."""
        for status in ["pending", "blessed", "interesting", "rejected"]:
            path = self.paths.insights_dir / status / f"{insight_id}.json"
            if path.exists():
                return AtomicInsight.load(path)
        return None

    def _get_existing_insights_for_thread(self, thread_id: str) -> list[AtomicInsight]:
        """
        Find any insights that were already extracted for this thread.
        Checks pending, reviewing, and blessed directories.
        """
        existing = []

        # Check all subdirectories where insights might be
        subdirs = ["pending", "reviewing", "insights/blessed", "insights/rejected"]

        for subdir in subdirs:
            search_dir = self.paths.chunks_dir / subdir
            if not search_dir.exists():
                continue

            for filepath in search_dir.glob("*.json"):
                try:
                    insight = AtomicInsight.load(filepath)
                    if insight.source_thread_id == thread_id:
                        existing.append(insight)
                except Exception:
                    pass

        return existing

    def _build_review_summary(self, review) -> str:
        """
        Build a summary of reviewer findings for augmentation context.

        Extracts key points that reviewers found compelling, which helps
        the augmenter understand what makes this insight worth illustrating.
        """
        parts = []

        if not review.rounds:
            return ""

        # Get final round (most relevant reasoning)
        final_round = review.rounds[-1]

        for llm, resp in final_round.responses.items():
            if resp.rating == "⚡":  # Only include endorsements
                parts.append(f"{llm.title()} found compelling:")
                if resp.naturalness_assessment:
                    parts.append(f"  - Naturalness: {resp.naturalness_assessment}")
                if resp.structural_analysis:
                    parts.append(f"  - Structure: {resp.structural_analysis}")
                if resp.reasoning:
                    parts.append(f"  - Reasoning: {resp.reasoning}")

        # Note if there were mind changes
        if review.mind_changes:
            for mc in review.mind_changes:
                if mc.to_rating == "⚡":
                    parts.append(f"{mc.llm.title()} changed to bless because: {mc.reason}")

        return "\n".join(parts)
