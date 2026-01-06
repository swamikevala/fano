"""
Panel-based insight extraction.

Instead of a single LLM extracting insights, all 3 LLMs propose independently,
then Claude consolidates. This captures diverse perspectives and ensures
cross-domain bridges aren't missed by a single conservative extractor.

Flow:
1. Send extraction prompt to Gemini, ChatGPT, Claude in parallel
2. Each proposes their list of interesting insights
3. Claude Opus consolidates: merges similar, keeps best articulations
4. Final list goes to review panel
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .models import AtomicInsight, InsightStatus
from .prompts import (
    build_panel_extraction_prompt,
    build_consolidation_prompt,
    parse_panel_extraction_response,
    parse_consolidation_response,
    format_blessed_summary,
)
from .dependencies import resolve_dependencies

if TYPE_CHECKING:
    from review_panel.claude_api import ClaudeReviewer

logger = logging.getLogger(__name__)


class PanelExtractor:
    """
    Extracts insights using a panel of 3 LLMs.

    Each LLM proposes insights independently, emphasizing cross-domain
    bridges and intuitive connections. Claude then consolidates the
    proposals into a final list.
    """

    def __init__(self, data_dir: Path, config: dict = None):
        """
        Initialize the panel extractor.

        Args:
            data_dir: Base data directory
            config: Configuration dict
        """
        self.data_dir = data_dir
        self.config = config or {}
        self.insights_dir = data_dir / "chunks"
        self.insights_dir.mkdir(parents=True, exist_ok=True)

    async def extract_from_thread(
        self,
        thread,
        gemini=None,
        chatgpt=None,
        claude_reviewer: "ClaudeReviewer" = None,
        blessed_insights: list = None,
    ) -> list[AtomicInsight]:
        """
        Extract insights using panel of 3 LLMs.

        Args:
            thread: ExplorationThread to extract from
            gemini: GeminiInterface instance
            chatgpt: ChatGPTInterface instance
            claude_reviewer: ClaudeReviewer instance
            blessed_insights: List of existing blessed insights

        Returns:
            List of extracted AtomicInsight objects
        """
        if thread.chunks_extracted:
            logger.info(f"[panel-extractor] Thread {thread.id} already extracted")
            return []

        logger.info(f"[panel-extractor] Starting panel extraction for thread {thread.id}")

        # Build context
        thread_context = thread.get_context_for_prompt(max_exchanges=20)
        blessed_summary = format_blessed_summary(blessed_insights or [])

        # Get config
        chunking_config = self.config.get("chunking", {})
        max_insights_per_llm = chunking_config.get("max_insights_per_thread", 15)
        max_final = chunking_config.get("max_final_insights", 10)

        # Build extraction prompt
        prompt = build_panel_extraction_prompt(
            thread_context=thread_context,
            blessed_chunks_summary=blessed_summary,
            max_insights=max_insights_per_llm,
        )

        # Phase 1: Parallel extraction from all 3 LLMs
        logger.info("[panel-extractor] Phase 1: Parallel extraction from all LLMs")

        gemini_response = ""
        chatgpt_response = ""
        claude_response = ""

        # Create tasks for parallel execution
        tasks = []

        if gemini:
            tasks.append(("gemini", self._extract_from_gemini(gemini, prompt)))
        if chatgpt:
            tasks.append(("chatgpt", self._extract_from_chatgpt(chatgpt, prompt)))
        if claude_reviewer and claude_reviewer.is_available():
            tasks.append(("claude", self._extract_from_claude(claude_reviewer, prompt)))

        if not tasks:
            logger.error("[panel-extractor] No LLMs available for extraction")
            return []

        # Run in parallel
        results = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )

        # Collect responses
        for i, (llm_name, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                logger.warning(f"[panel-extractor] {llm_name} failed: {result}")
                continue

            if llm_name == "gemini":
                gemini_response = result
                logger.info(f"[panel-extractor] Gemini proposed {len(parse_panel_extraction_response(result))} insights")
            elif llm_name == "chatgpt":
                chatgpt_response = result
                logger.info(f"[panel-extractor] ChatGPT proposed {len(parse_panel_extraction_response(result))} insights")
            elif llm_name == "claude":
                claude_response = result
                logger.info(f"[panel-extractor] Claude proposed {len(parse_panel_extraction_response(result))} insights")

        # Check we have at least one response
        if not any([gemini_response, chatgpt_response, claude_response]):
            logger.error("[panel-extractor] All LLMs failed to respond")
            thread.extraction_note = "Panel extraction failed: no LLM responses"
            return []

        # Phase 2: Consolidation by Claude
        logger.info("[panel-extractor] Phase 2: Consolidation")

        consolidation_prompt = build_consolidation_prompt(
            gemini_proposals=gemini_response or "(no response)",
            chatgpt_proposals=chatgpt_response or "(no response)",
            claude_proposals=claude_response or "(no response)",
            blessed_chunks_summary=blessed_summary,
            max_final=max_final,
        )

        try:
            if claude_reviewer and claude_reviewer.is_available():
                consolidated = await claude_reviewer.send_message(
                    consolidation_prompt,
                    extended_thinking=False
                )
            elif chatgpt:
                # Fallback to ChatGPT for consolidation
                await chatgpt.start_new_chat()
                consolidated = await chatgpt.send_message(
                    consolidation_prompt,
                    use_pro_mode=False,
                    use_thinking_mode=True
                )
            else:
                logger.error("[panel-extractor] No LLM available for consolidation")
                return []

            logger.info(f"[panel-extractor] Got consolidation response ({len(consolidated)} chars)")

        except Exception as e:
            logger.error(f"[panel-extractor] Consolidation failed: {e}")
            thread.extraction_note = f"Consolidation failed: {e}"
            return []

        # Parse consolidated insights
        parsed_insights = parse_consolidation_response(consolidated)

        if not parsed_insights:
            logger.info(f"[panel-extractor] No insights after consolidation")
            thread.chunks_extracted = True
            thread.extraction_note = "Panel extraction yielded no insights"
            return []

        logger.info(f"[panel-extractor] Consolidated to {len(parsed_insights)} insights")

        # Create AtomicInsight objects
        insights = []
        for parsed in parsed_insights[:max_final]:
            insight = AtomicInsight.create(
                insight=parsed["insight"],
                confidence=parsed.get("confidence", "medium"),
                tags=parsed.get("tags", []),
                source_thread_id=thread.id,
                extraction_model=f"panel ({parsed.get('proposed_by', 'unknown')})",
                depends_on=parsed.get("depends_on", []),
                pending_dependencies=[],
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
            logger.info(f"[panel-extractor] Saved: {insight.id}: {insight.insight[:60]}...")

        # Mark thread as extracted
        thread.chunks_extracted = True
        thread.extraction_note = f"Panel extracted {len(insights)} insights"

        logger.info(f"[panel-extractor] Complete: {len(insights)} insights from thread {thread.id}")

        return insights

    async def _extract_from_gemini(self, gemini, prompt: str) -> str:
        """Get extraction proposals from Gemini."""
        try:
            await gemini.start_new_chat()
            response = await gemini.send_message(prompt, use_deep_think=False)
            return response
        except Exception as e:
            logger.warning(f"[panel-extractor] Gemini extraction failed: {e}")
            raise

    async def _extract_from_chatgpt(self, chatgpt, prompt: str) -> str:
        """Get extraction proposals from ChatGPT."""
        try:
            await chatgpt.start_new_chat()
            response = await chatgpt.send_message(
                prompt,
                use_pro_mode=False,
                use_thinking_mode=True
            )
            return response
        except Exception as e:
            logger.warning(f"[panel-extractor] ChatGPT extraction failed: {e}")
            raise

    async def _extract_from_claude(self, claude_reviewer, prompt: str) -> str:
        """Get extraction proposals from Claude."""
        try:
            # Small delay to avoid network contention with browser LLMs
            await asyncio.sleep(2)
            response = await claude_reviewer.send_message(
                prompt,
                extended_thinking=False
            )
            return response
        except Exception as e:
            logger.warning(f"[panel-extractor] Claude extraction failed: {e}")
            raise


def get_panel_extractor(data_dir: Path, config: dict = None) -> PanelExtractor:
    """Factory function to create a PanelExtractor."""
    return PanelExtractor(data_dir, config)
