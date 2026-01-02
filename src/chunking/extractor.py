"""
Atomic insight extractor.

Extracts multiple atomic insights (1-3 sentences each) from exploration threads.
Claude Opus is the primary extraction author (best at precise articulation).
Falls back to browser LLMs if Claude is unavailable.
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .models import AtomicInsight, InsightStatus
from .prompts import build_extraction_prompt, format_blessed_summary, parse_extraction_response
from .dependencies import resolve_dependencies

if TYPE_CHECKING:
    from review_panel.claude_api import ClaudeReviewer

logger = logging.getLogger(__name__)


class AtomicExtractor:
    """
    Extracts atomic insights from exploration threads.

    Each thread can produce multiple atomic insights. The extractor:
    1. Builds context from the thread and blessed axioms
    2. Sends extraction prompt to an LLM
    3. Parses the response into AtomicInsight objects
    4. Resolves dependencies to blessed insight IDs
    5. Saves insights and marks thread as extracted
    """

    def __init__(self, data_dir: Path, config: dict = None):
        """
        Initialize the extractor.

        Args:
            data_dir: Base data directory
            config: Chunking configuration
        """
        self.data_dir = data_dir
        self.config = config or {}
        self.insights_dir = data_dir / "insights"
        self.insights_dir.mkdir(parents=True, exist_ok=True)

    async def extract_from_thread(
        self,
        thread,
        claude_reviewer: "ClaudeReviewer" = None,
        fallback_model=None,
        fallback_model_name: str = None,
        blessed_insights: list = None,
    ) -> list[AtomicInsight]:
        """
        Extract atomic insights from an exploration thread.

        Claude Opus is the primary author (best at precise articulation).
        Falls back to browser LLMs if Claude is unavailable.

        Args:
            thread: ExplorationThread to extract from
            claude_reviewer: ClaudeReviewer instance (preferred)
            fallback_model: Browser LLM interface (ChatGPT or Gemini) if Claude unavailable
            fallback_model_name: Name of fallback model ("chatgpt" or "gemini")
            blessed_insights: List of existing blessed insights for dependencies

        Returns:
            List of extracted AtomicInsight objects
        """
        # Skip if already extracted
        if thread.chunks_extracted:
            logger.info(f"[extractor] Thread {thread.id} already extracted, skipping")
            return []

        logger.info(f"[extractor] Extracting insights from thread {thread.id}")

        # Build context
        thread_context = thread.get_context_for_prompt(max_exchanges=20)
        blessed_summary = format_blessed_summary(blessed_insights or [])

        # Get config options
        chunking_config = self.config.get("chunking", {})
        max_insights = chunking_config.get("max_insights_per_thread", 10)
        min_confidence = chunking_config.get("min_confidence_to_keep", "low")

        # Build and send prompt
        prompt = build_extraction_prompt(
            thread_context=thread_context,
            blessed_chunks_summary=blessed_summary,
            max_insights=max_insights,
        )

        # Determine which model to use (Claude preferred)
        extraction_model = "claude"
        try:
            if claude_reviewer and claude_reviewer.is_available():
                logger.info(f"[extractor] Using Claude Opus for extraction")
                response = await claude_reviewer.send_message(prompt, extended_thinking=False)
            elif fallback_model and fallback_model_name:
                logger.info(f"[extractor] Claude unavailable, using {fallback_model_name} as fallback")
                extraction_model = fallback_model_name
                if fallback_model_name == "chatgpt":
                    response = await fallback_model.send_message(prompt, use_pro_mode=False, use_thinking_mode=True)
                else:
                    response = await fallback_model.send_message(prompt, use_deep_think=False)
            else:
                raise ValueError("No extraction model available (Claude or fallback)")

            logger.info(f"[extractor] Got response ({len(response)} chars)")

        except Exception as e:
            logger.error(f"[extractor] Failed to get extraction response: {e}")
            thread.extraction_note = f"Extraction failed: {e}"
            return []

        # Parse response
        parsed_insights = parse_extraction_response(response)

        if not parsed_insights:
            logger.info(f"[extractor] No insights extracted from thread {thread.id}")
            thread.chunks_extracted = True
            thread.extraction_note = "No insights met quality bar"
            return []

        logger.info(f"[extractor] Parsed {len(parsed_insights)} raw insights")

        # Filter by confidence
        confidence_order = ["high", "medium", "low"]
        min_idx = confidence_order.index(min_confidence)
        filtered = [
            p for p in parsed_insights
            if confidence_order.index(p.get("confidence", "low")) <= min_idx
        ]

        logger.info(f"[extractor] {len(filtered)} insights after confidence filter")

        # Create AtomicInsight objects
        insights = []
        for parsed in filtered[:max_insights]:
            insight = AtomicInsight.create(
                insight=parsed["insight"],
                confidence=parsed["confidence"],
                tags=parsed["tags"],
                source_thread_id=thread.id,
                extraction_model=extraction_model,
                depends_on=parsed.get("depends_on", []),
                pending_dependencies=parsed.get("pending_dependencies", []),
            )
            insights.append(insight)

        # Resolve dependencies
        if blessed_insights:
            deps_config = self.config.get("dependencies", {})
            threshold = deps_config.get("semantic_match_threshold", 0.5)
            for insight in insights:
                if insight.pending_dependencies:
                    resolved, still_pending = resolve_dependencies(
                        insight.pending_dependencies,
                        blessed_insights,
                        threshold=threshold,
                    )
                    insight.depends_on.extend(resolved)
                    insight.pending_dependencies = still_pending

        # Save insights
        for insight in insights:
            insight.save(self.data_dir)
            logger.info(f"[extractor] Saved insight {insight.id}: {insight.insight[:50]}...")

        # Mark thread as extracted
        thread.chunks_extracted = True
        thread.extraction_note = f"Extracted {len(insights)} insights"

        logger.info(f"[extractor] Completed extraction: {len(insights)} insights from thread {thread.id}")

        return insights

    def load_pending_insights(self) -> list[AtomicInsight]:
        """Load all pending insights awaiting review."""
        pending_dir = self.insights_dir / "pending"
        if not pending_dir.exists():
            return []

        insights = []
        for json_path in pending_dir.glob("*.json"):
            try:
                insights.append(AtomicInsight.load(json_path))
            except Exception as e:
                logger.warning(f"[extractor] Could not load {json_path}: {e}")

        return insights

    def load_blessed_insights(self) -> list[AtomicInsight]:
        """Load all blessed insights."""
        blessed_dir = self.insights_dir / "blessed"
        if not blessed_dir.exists():
            return []

        insights = []
        for json_path in blessed_dir.glob("*.json"):
            try:
                insights.append(AtomicInsight.load(json_path))
            except Exception as e:
                logger.warning(f"[extractor] Could not load {json_path}: {e}")

        return insights

    def get_blessed_ids(self) -> set[str]:
        """Get set of all blessed insight IDs."""
        return {i.id for i in self.load_blessed_insights()}


def get_extractor(data_dir: Path, config: dict = None) -> AtomicExtractor:
    """Factory function to create an AtomicExtractor."""
    return AtomicExtractor(data_dir, config)
