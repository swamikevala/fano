"""
Exploration Engine - Handles exploration and critique operations.

This module centralizes:
- Exploration step execution
- Critique step execution
- Prompt building for both phases
- Response extraction and cleaning
"""

import re
from datetime import datetime
from typing import Any

from shared.logging import get_logger

from explorer.src.models import ExplorationThread, ExchangeRole, AxiomStore
from explorer.src.storage import ExplorerPaths

log = get_logger("explorer", "orchestration.exploration")


class ExplorationEngine:
    """
    Handles exploration and critique operations on threads.

    Responsible for:
    - Building prompts for exploration and critique
    - Executing LLM calls via the LLM manager
    - Extracting structured responses
    - Adding exchanges to threads
    """

    def __init__(
        self,
        config: dict,
        paths: ExplorerPaths,
        axioms: AxiomStore,
        get_context_for_seeds_fn,
        get_focused_context_fn=None,
    ):
        """
        Initialize exploration engine.

        Args:
            config: Full configuration dict
            paths: ExplorerPaths instance
            axioms: AxiomStore for context
            get_context_for_seeds_fn: Function to get context for specific seeds
            get_focused_context_fn: Function to get focused context for single-seed threads
        """
        self.config = config
        self.paths = paths
        self.axioms = axioms
        self.get_context_for_seeds = get_context_for_seeds_fn
        self.get_focused_context = get_focused_context_fn

    async def do_exploration(
        self,
        thread: ExplorationThread,
        model_name: str,
        model: Any,
        llm_manager,
    ) -> None:
        """
        Perform an exploration step.

        Args:
            thread: Thread to explore
            model_name: Name of the model to use
            model: Model adapter instance
            llm_manager: LLMManager for sending messages
        """
        prompt, images = self._build_exploration_prompt(thread)

        image_note = f" with {len(images)} image(s)" if images else ""
        log.info(f"Exploring [{thread.id}] with {model_name}{image_note}")

        try:
            response, deep_mode_used = await llm_manager.send_message(
                model_name=model_name,
                model=model,
                prompt=prompt,
                thread=thread,
                task_type="exploration",
                images=images,
            )

            # Extract only structured sections, stripping preamble and recaps
            clean_response = self._extract_structured_response(
                response, ["NEW_INSIGHTS", "CONNECTIONS", "QUESTIONS"]
            )

            thread.add_exchange(
                role=ExchangeRole.EXPLORER,
                model=model_name,
                prompt=prompt,
                response=clean_response,
                deep_mode_used=deep_mode_used,
            )

            mode_str = " [DEEP]" if deep_mode_used else ""
            log.info(
                f"Exploration complete, {len(response)} -> {len(clean_response)} chars{mode_str}"
            )

        except Exception as e:
            log.error(f"Exploration failed: {e}")

    async def do_critique(
        self,
        thread: ExplorationThread,
        model_name: str,
        model: Any,
        llm_manager,
    ) -> None:
        """
        Perform a critique step.

        Args:
            thread: Thread to critique
            model_name: Starting model name (may be changed by weighted selection)
            model: Model adapter instance
            llm_manager: LLMManager for sending messages
        """
        prompt = self._build_critique_prompt(thread)

        # Use weighted selection for critique (prefers ChatGPT)
        available_models = llm_manager.get_available_models(check_rate_limits=False)
        selected = llm_manager.select_model_for_task("critique", available_models)
        if selected:
            critique_model_name = selected
            critique_model = available_models[selected]
        else:
            # Fallback to original model
            critique_model_name, critique_model = model_name, model

        log.info(f"Critiquing [{thread.id}] with {critique_model_name}")

        try:
            response, deep_mode_used = await llm_manager.send_message(
                model_name=critique_model_name,
                model=critique_model,
                prompt=prompt,
                thread=thread,
                task_type="critique",
            )

            # Extract only structured sections, stripping preamble and recaps
            clean_response = self._extract_structured_response(
                response, ["CRITICAL_ISSUES", "PROMISING_DIRECTIONS", "PROBING_QUESTIONS"]
            )

            thread.add_exchange(
                role=ExchangeRole.CRITIC,
                model=critique_model_name,
                prompt=prompt,
                response=clean_response,
                deep_mode_used=deep_mode_used,
            )

            mode_str = " [DEEP]" if deep_mode_used else ""
            log.info(
                f"Critique complete, {len(response)} -> {len(clean_response)} chars{mode_str}"
            )

        except Exception as e:
            log.error(f"Critique failed: {e}")

    def _build_exploration_prompt(self, thread: ExplorationThread) -> tuple[str, list]:
        """
        Build the prompt for exploration based on seed aphorisms.

        Returns:
            tuple of (prompt_text, image_attachments)
        """
        from llm.src.models import ImageAttachment

        images = []

        # Check if this is a focused single-seed thread
        is_focused = thread.primary_question_id or thread.related_conjecture_ids

        if is_focused and self.get_focused_context:
            # Use focused context for single-seed exploration
            context = self.get_focused_context(thread)
            # Get images for focused seeds
            if thread.primary_question_id:
                seed = self.axioms.get_seed_by_id(thread.primary_question_id)
                if seed:
                    for path in self.axioms.get_seed_images(seed.id):
                        images.append(ImageAttachment.from_file(str(path)))
            for cid in (thread.related_conjecture_ids or []):
                seed = self.axioms.get_seed_by_id(cid)
                if seed:
                    for path in self.axioms.get_seed_images(seed.id):
                        images.append(ImageAttachment.from_file(str(path)))
        elif thread.seed_axioms:
            # Legacy: use context for all seeds in the thread
            context = self.get_context_for_seeds(thread.seed_axioms)
            # Get images for these specific seeds
            for seed_id in thread.seed_axioms:
                for path in self.axioms.get_seed_images(seed_id):
                    images.append(ImageAttachment.from_file(str(path)))
        else:
            # Use full context with all seed images
            context, image_paths = self.axioms.get_context_with_images()
            for path in image_paths:
                images.append(ImageAttachment.from_file(str(path)))

        date_prefix = datetime.now().strftime("[FANO %m-%d]")

        prompt_parts = [
            f"{date_prefix} You are exploring and developing the following seed aphorisms.",
            "",
            "Your goal is to:",
            "1. Verify or refine each conjecture through rigorous mathematical analysis",
            "2. Find deeper structures that explain WHY these connections exist",
            "3. Discover new connections that feel NATURAL and INEVITABLE (not forced)",
            "",
            "The criterion for a good direction: does it feel DISCOVERED rather than INVENTED?",
            "",
            context,
        ]

        if thread.exchanges:
            prompt_parts.append("=== PREVIOUS EXPLORATION ===")
            prompt_parts.append(thread.get_context_for_prompt())
            prompt_parts.append("")
            prompt_parts.append("Build on this exploration. Go deeper. Find the structure.")
        else:
            prompt_parts.append("Begin your exploration. Examine these seed aphorisms mathematically.")
            prompt_parts.append("What structures and patterns emerge? What can you verify or develop?")

        # Add math formatting instructions
        math_formatting = self.config.get("exploration", {}).get("math_formatting", "")
        if math_formatting:
            prompt_parts.extend(["", "=== MATH FORMATTING ===", math_formatting.strip()])

        # Add anti-bloat instructions with structured format
        prompt_parts.extend(
            [
                "",
                "=== RESPONSE FORMAT ===",
                "IMPORTANT: Do not summarize or recap previous discussion.",
                "State only NEW insights. Assume full context is available.",
                "Skip any preamble - go straight to the sections.",
                "",
                "You MUST structure your response EXACTLY as:",
                "",
                "[NEW_INSIGHTS]",
                "- (bullet points of genuinely new ideas or mathematical structures)",
                "",
                "[CONNECTIONS]",
                "- (new connections discovered between domains)",
                "",
                "[QUESTIONS]",
                "- (open questions worth exploring next)",
            ]
        )

        return "\n".join(prompt_parts), images

    def _build_critique_prompt(self, thread: ExplorationThread) -> str:
        """Build the prompt for critique."""
        context = thread.get_context_for_prompt()
        date_prefix = datetime.now().strftime("[FANO %m-%d]")
        math_formatting = self.config.get("exploration", {}).get("math_formatting", "").strip()

        prompt = f"""{date_prefix} You are a rigorous mathematical critic. Review the following exploration:

{context}

Your task:
1. Identify any FORCING or AD HOC assumptions
2. Point out where the structure feels ARBITRARY rather than INEVITABLE
3. Ask probing questions that would deepen the exploration
4. Note what's working well and feels NATURAL
5. Suggest alternative angles if something feels stuck

The core question: Does this feel DISCOVERED or INVENTED?

Signs of discovery (good):
- "It couldn't be any other way"
- "This explains WHY, not just THAT"
- "Multiple independent paths led here"
- "The numbers fall out naturally"

Signs of invention (bad):
- "We need to assume this for it to work"
- "It matches, but we don't know why"
- "This was chosen to make the numbers fit"
- "It feels clever but arbitrary"

Be constructive but rigorous. The goal is truth, not validation.
Push toward depth, not toward any particular application.

=== MATH FORMATTING ===
{math_formatting}

=== RESPONSE FORMAT ===
IMPORTANT: Do not summarize or recap previous discussion.
State only NEW critiques and questions. Assume full context is available.

Structure your response as:
[CRITICAL_ISSUES]
(Genuine problems with the reasoning - be specific)

[PROMISING_DIRECTIONS]
(What's working and worth pursuing deeper)

[PROBING_QUESTIONS]
(Questions that would test or deepen the ideas)"""

        return prompt

    def _extract_structured_response(self, response: str, sections: list[str]) -> str:
        """
        Extract only the structured sections from a response, stripping preamble.

        Args:
            response: Full LLM response text
            sections: List of section names to extract (e.g. ['NEW_INSIGHTS', 'CONNECTIONS'])

        Returns:
            Cleaned response with only the structured sections
        """
        # Build pattern to find all sections
        section_pattern = r"\[(" + "|".join(sections) + r")\]"

        # Find the first section marker - everything before it is preamble
        first_match = re.search(section_pattern, response)
        if not first_match:
            # No structured sections found, return original (but log warning)
            log.warning("No structured sections found in response, keeping original")
            return response

        # Extract from first section onwards
        structured_part = response[first_match.start() :]

        # Clean up: remove any trailing content after the last section's content
        # by keeping only content up to any obvious sign-off phrases
        signoff_patterns = [
            r"\n\nLet me know",
            r"\n\nWould you like",
            r"\n\nShall I",
            r"\n\nI hope this",
            r"\n\nIn summary,",
            r"\n\nTo summarize,",
            r"\n\nOverall,",
        ]
        for pattern in signoff_patterns:
            match = re.search(pattern, structured_part, re.IGNORECASE)
            if match:
                structured_part = structured_part[: match.start()]

        return structured_part.strip()
