"""
Synthesis Engine - Handles chunk synthesis from exploration threads.

This module centralizes:
- Chunk readiness evaluation
- Profundity scoring
- Chunk synthesis
- Synthesis response parsing
"""

import re
from datetime import datetime

from shared.logging import get_logger

from explorer.src.models import ExplorationThread, ExchangeRole, ThreadStatus, Chunk
from explorer.src.storage import ExplorerPaths

log = get_logger("explorer", "orchestration.synthesis")


class SynthesisEngine:
    """
    Handles synthesis of exploration threads into reviewable chunks.

    Responsible for:
    - Determining when a thread is ready for synthesis
    - Calculating profundity scores
    - Building synthesis prompts
    - Creating chunks from synthesized content
    """

    def __init__(self, config: dict, paths: ExplorerPaths):
        """
        Initialize synthesis engine.

        Args:
            config: Full configuration dict (needs 'orchestration' and 'synthesis' sections)
            paths: ExplorerPaths instance
        """
        self.config = config
        self.orchestration_config = config.get("orchestration", {})
        self.synthesis_config = config.get("synthesis", {})
        self.paths = paths

    def is_chunk_ready(self, thread: ExplorationThread) -> bool:
        """
        Determine if a thread is ready to be synthesized into a chunk.

        Args:
            thread: Thread to evaluate

        Returns:
            True if thread is ready for synthesis.

        Logic:
            - Minimum 4 exchanges required
            - Force synthesis at 12+ exchanges
            - Allow synthesis at 8+ exchanges even with fewer critiques (prevents stuck threads)
            - Otherwise require 2 critiques OR high profundity score
        """
        # Minimum exchanges
        min_exchanges = self.orchestration_config.get("min_exchanges_for_chunk", 4)
        if thread.exchange_count < min_exchanges:
            return False

        # Maximum exchanges (force synthesis)
        max_exchanges = self.orchestration_config.get("max_exchanges_per_thread", 12)
        if thread.exchange_count >= max_exchanges:
            return True

        # High exchange count threshold - allow synthesis even with fewer critiques
        # This prevents threads from being stuck forever if critique models are unavailable
        high_exchange_threshold = self.orchestration_config.get("high_exchange_threshold", 8)
        if thread.exchange_count >= high_exchange_threshold:
            return True

        # Count critique rounds
        critique_count = sum(1 for e in thread.exchanges if e.role == ExchangeRole.CRITIC)
        min_critiques = self.synthesis_config.get("min_critiques", 2)

        # Check for profundity signals in recent exchanges
        recent_text = " ".join(e.response for e in thread.exchanges[-4:])
        profundity_score = self.calculate_profundity(recent_text)

        # Ready if enough critiques OR high profundity
        if critique_count >= min_critiques:
            return True
        if profundity_score > 0.6:
            return True

        return False

    def calculate_profundity(self, text: str) -> float:
        """
        Calculate a profundity score based on signal words.

        Args:
            text: Text to analyze

        Returns:
            Score between 0 and 1.
        """
        text_lower = text.lower()

        profundity_signals = self.synthesis_config.get(
            "profundity_signals",
            ["inevitable", "natural", "discovered", "emerges", "falls out", "couldn't be otherwise"],
        )
        doubt_signals = self.synthesis_config.get(
            "doubt_signals",
            ["arbitrary", "forced", "contrived", "coincidence", "ad hoc"],
        )

        prof_count = sum(1 for sig in profundity_signals if sig in text_lower)
        doubt_count = sum(1 for sig in doubt_signals if sig in text_lower)

        # Normalize
        score = (prof_count - doubt_count) / max(len(profundity_signals), 1)
        return max(0, min(1, (score + 1) / 2))  # Scale to 0-1

    async def synthesize_chunk(
        self,
        thread: ExplorationThread,
        llm_manager,
        extract_and_review_fn=None,
    ) -> None:
        """
        Synthesize a thread into a reviewable chunk.

        Args:
            thread: Thread to synthesize
            llm_manager: LLMManager for sending messages
            extract_and_review_fn: Optional function for atomic extraction after synthesis
        """
        # Use weighted selection for synthesis (balanced)
        available_models = llm_manager.get_available_models(check_rate_limits=False)
        model_name = llm_manager.select_model_for_task("synthesis", available_models)
        if not model_name:
            log.warning("No model available for synthesis")
            return

        model = available_models[model_name]

        log.info(f"Synthesizing chunk from thread [{thread.id}] with {model_name}")

        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(thread)

        try:
            response, deep_mode_used = await llm_manager.send_message(
                model_name=model_name,
                model=model,
                prompt=synthesis_prompt,
                thread=thread,
                task_type="synthesis",
            )

            mode_str = " [DEEP]" if deep_mode_used else ""
            if not deep_mode_used and self.synthesis_config.get("prefer_deep_mode", True):
                log.warning("Deep mode requested but not available")

            # Parse the synthesis response
            title, summary, content = self._parse_synthesis(response, thread)

            # Calculate profundity
            profundity = self.calculate_profundity(content)
            critique_count = sum(1 for e in thread.exchanges if e.role == ExchangeRole.CRITIC)

            # Create chunk
            chunk = Chunk.create_from_thread(
                thread_id=thread.id,
                title=title,
                content=content,
                summary=summary,
                target_numbers=thread.target_numbers,
                profundity_score=profundity,
                critique_rounds=critique_count,
            )

            # Save chunk
            chunk.save(self.paths.data_dir)

            # Mark thread as chunk-ready
            thread.status = ThreadStatus.CHUNK_READY
            thread.save(self.paths.data_dir)

            log.info(f"Created chunk [{chunk.id}]: {chunk.title}{mode_str}")

            # Atomic extraction and review (if enabled and function provided)
            chunking_config = self.config.get("chunking", {})
            if chunking_config.get("mode") == "atomic" and extract_and_review_fn:
                await extract_and_review_fn(thread, model_name, model)

        except Exception as e:
            log.error(f"Synthesis failed: {e}")

    def _build_synthesis_prompt(self, thread: ExplorationThread) -> str:
        """Build prompt for chunk synthesis."""
        context = thread.get_context_for_prompt(max_exchanges=10)
        date_prefix = datetime.now().strftime("[FANO %m-%d]")

        return f"""{date_prefix} You are a mathematical writer. Synthesize the following exploration into a clear, well-structured write-up.

{context}

Write a synthesis that:
1. Opens with a clear TITLE (single line)
2. Follows with a one-sentence SUMMARY
3. Then provides the full CONTENT explaining the mathematical structure discovered

The write-up should:
- Be clear and readable
- Preserve the mathematical rigor
- Highlight what feels INEVITABLE vs what remains speculative
- Explicitly show how the structure explains the target numbers

Format:
TITLE: [Your title here]
SUMMARY: [One sentence summary]
CONTENT:
[Full write-up in markdown]"""

    def _parse_synthesis(
        self, response: str, thread: ExplorationThread
    ) -> tuple[str, str, str]:
        """
        Parse synthesis response into title, summary, content.

        Args:
            response: Raw LLM response
            thread: Thread for fallback values

        Returns:
            Tuple of (title, summary, content).
        """
        # Try to extract structured parts
        title_match = re.search(r"TITLE:\s*(.+?)(?:\n|SUMMARY:)", response, re.DOTALL)
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\n|CONTENT:)", response, re.DOTALL)
        content_match = re.search(r"CONTENT:\s*(.+)", response, re.DOTALL)

        title = title_match.group(1).strip() if title_match else f"Exploration {thread.id}"
        summary = (
            summary_match.group(1).strip()
            if summary_match
            else "Mathematical exploration synthesis"
        )
        content = content_match.group(1).strip() if content_match else response

        return title, summary, content
