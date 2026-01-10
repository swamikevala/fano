"""
Consensus Reviewer - Multi-LLM validation for reliable answers.

Uses multiple LLMs to review and validate insights/claims,
reaching consensus through structured deliberation.
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

from .client import LLMClient
from .models import LLMResponse, ConsensusResult, ReviewResponse

log = get_logger("llm", "consensus")


class ConsensusReviewer:
    """
    Multi-LLM consensus reviewer for validating insights.

    Orchestrates multiple rounds of review across available LLMs
    to reach reliable consensus on whether an insight should be
    blessed (accepted), rejected, or needs more development.

    Usage:
        client = LLMClient()
        reviewer = ConsensusReviewer(client)

        result = await reviewer.review(
            text="The number 84 appears in yoga as 84 asanas...",
            tags=["yoga", "numbers"],
        )

        if result.final_rating == "bless":
            print("Insight validated!")
    """

    def __init__(
        self,
        client: LLMClient,
        config: Optional[dict] = None,
    ):
        """
        Initialize the reviewer.

        Args:
            client: LLMClient for sending requests
            config: Optional configuration dict
        """
        self.client = client
        self.config = config or {}

    async def review(
        self,
        text: str,
        *,
        tags: Optional[list[str]] = None,
        context: str = "",
        confidence: str = "medium",
        dependencies: Optional[list[str]] = None,
        use_deep_mode: bool = True,
    ) -> ConsensusResult:
        """
        Run multi-LLM consensus review on a piece of text.

        Process:
        1. Round 1: Independent parallel review (standard modes)
        2. Round 2: Deep analysis with Round 1 responses visible
        3. Round 3: Structured deliberation if still split

        Args:
            text: The insight/claim to review
            tags: Tags for context
            context: Additional context (e.g., blessed axioms)
            confidence: Confidence level from extraction
            dependencies: Dependencies on other insights
            use_deep_mode: Whether to use deep/pro modes in Round 2

        Returns:
            ConsensusResult with final rating and review history
        """
        import time
        start_time = time.time()

        tags = tags or []
        dependencies = dependencies or []

        # Get available backends
        available = await self.client.get_available_backends()
        if len(available) < 2:
            return ConsensusResult(
                success=False,
                final_rating="uncertain",
                is_unanimous=False,
                is_disputed=True,
                rounds=[{"error": f"Need at least 2 backends, only {len(available)} available"}],
            )

        log.info(
            "llm.consensus.review_start",
            backends=available,
            text_length=len(text),
            tags=tags,
            use_deep_mode=use_deep_mode,
        )

        # Round 1: Independent parallel review
        round1_responses = await self._run_round1(
            text, tags, context, confidence, dependencies, available
        )

        # Check for early exit (unanimous)
        ratings = [r.rating for r in round1_responses.values()]
        log.info(
            "llm.consensus.round_complete",
            round=1,
            ratings={k: v.rating for k, v in round1_responses.items()},
            unanimous=len(set(ratings)) == 1,
        )
        if len(set(ratings)) == 1:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.review_complete",
                final_rating=ratings[0],
                rounds_needed=1,
                is_unanimous=True,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusResult(
                success=True,
                final_rating=ratings[0],
                is_unanimous=True,
                is_disputed=False,
                rounds=[{"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}}],
                review_duration_seconds=elapsed,
            )

        # Round 2: Deep analysis with Round 1 context
        round2_responses = await self._run_round2(
            text, round1_responses, context, available, use_deep_mode
        )

        # Check for resolution
        ratings = [r.rating for r in round2_responses.values()]
        # Track mind changes
        for backend in round1_responses:
            if backend in round2_responses:
                r1_rating = round1_responses[backend].rating
                r2_rating = round2_responses[backend].rating
                if r1_rating != r2_rating:
                    log.info(
                        "llm.consensus.mind_change",
                        llm=backend,
                        from_rating=r1_rating,
                        to_rating=r2_rating,
                    )
        log.info(
            "llm.consensus.round_complete",
            round=2,
            ratings={k: v.rating for k, v in round2_responses.items()},
            unanimous=len(set(ratings)) == 1,
        )
        if len(set(ratings)) == 1:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.review_complete",
                final_rating=ratings[0],
                rounds_needed=2,
                is_unanimous=True,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusResult(
                success=True,
                final_rating=ratings[0],
                is_unanimous=True,
                is_disputed=False,
                rounds=[
                    {"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}},
                    {"round": 2, "responses": {k: v.to_dict() for k, v in round2_responses.items()}},
                ],
                review_duration_seconds=time.time() - start_time,
            )

        # Round 3: Deliberation if still split
        final_rating, is_disputed = await self._run_round3(
            text, round2_responses, context, available
        )

        elapsed = time.time() - start_time
        log.info(
            "llm.consensus.review_complete",
            final_rating=final_rating,
            rounds_needed=3,
            is_unanimous=not is_disputed,
            is_disputed=is_disputed,
            duration_ms=round(elapsed * 1000, 2),
        )
        return ConsensusResult(
            success=True,
            final_rating=final_rating,
            is_unanimous=not is_disputed,
            is_disputed=is_disputed,
            rounds=[
                {"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}},
                {"round": 2, "responses": {k: v.to_dict() for k, v in round2_responses.items()}},
                {"round": 3, "final_rating": final_rating, "disputed": is_disputed},
            ],
            review_duration_seconds=elapsed,
        )

    async def _run_round1(
        self,
        text: str,
        tags: list[str],
        context: str,
        confidence: str,
        dependencies: list[str],
        backends: list[str],
    ) -> dict[str, ReviewResponse]:
        """Run Round 1: Independent parallel review."""
        prompt = self._build_round1_prompt(text, tags, context, confidence, dependencies)

        # Send to all backends in parallel
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=False)

        # Parse responses
        parsed = {}
        for backend, response in responses.items():
            if response.success:
                parsed[backend] = self._parse_review_response(backend, response.text, "standard")
            else:
                # Create error response
                parsed[backend] = ReviewResponse(
                    llm=backend,
                    mode="standard",
                    rating="uncertain",
                    reasoning=f"Error: {response.error} - {response.message}",
                    confidence="low",
                )

        return parsed

    async def _run_round2(
        self,
        text: str,
        round1: dict[str, ReviewResponse],
        context: str,
        backends: list[str],
        use_deep_mode: bool,
    ) -> dict[str, ReviewResponse]:
        """Run Round 2: Deep analysis with Round 1 context."""
        prompt = self._build_round2_prompt(text, round1, context)

        # Send to all backends in parallel with deep mode
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

        # Parse responses
        parsed = {}
        for backend, response in responses.items():
            mode = "deep" if use_deep_mode and response.deep_mode_used else "standard"
            if response.success:
                parsed[backend] = self._parse_review_response(backend, response.text, mode)
            else:
                parsed[backend] = ReviewResponse(
                    llm=backend,
                    mode=mode,
                    rating="uncertain",
                    reasoning=f"Error: {response.error} - {response.message}",
                    confidence="low",
                )

        return parsed

    async def _run_round3(
        self,
        text: str,
        round2: dict[str, ReviewResponse],
        context: str,
        backends: list[str],
    ) -> tuple[str, bool]:
        """Run Round 3: Deliberation to reach final decision."""
        # Get majority rating from Round 2
        ratings = [r.rating for r in round2.values()]
        rating_counts = {}
        for r in ratings:
            rating_counts[r] = rating_counts.get(r, 0) + 1

        # Find majority (2 out of 3)
        majority_rating = None
        for rating, count in rating_counts.items():
            if count >= 2:
                majority_rating = rating
                break

        if majority_rating:
            # Majority exists
            return majority_rating, False

        # No majority - use "uncertain" as default
        return "uncertain", True

    def _build_round1_prompt(
        self,
        text: str,
        tags: list[str],
        context: str,
        confidence: str,
        dependencies: list[str],
    ) -> str:
        """Build prompt for Round 1 independent review."""
        tags_str = ", ".join(tags) if tags else "none"
        deps_str = ", ".join(dependencies) if dependencies else "none"

        return f"""You are reviewing a mathematical/philosophical insight for validity.

INSIGHT TO REVIEW:
{text}

METADATA:
- Tags: {tags_str}
- Confidence: {confidence}
- Dependencies: {deps_str}

CONTEXT:
{context}

REVIEW CRITERIA:
1. Mathematical Verification: Are any numerical claims correct?
2. Structural Analysis: Is this a deep connection or superficial pattern?
3. Naturalness: Does this feel DISCOVERED (inevitable) or INVENTED (forced)?

RATE THIS INSIGHT:
- "bless" (⚡): Profound, verified, inevitable - should become an axiom
- "uncertain" (?): Interesting but needs more development
- "reject" (✗): Flawed, superficial, or unfalsifiable

FORMAT YOUR RESPONSE AS:
RATING: [bless/uncertain/reject]
MATHEMATICAL_VERIFICATION: [your analysis]
STRUCTURAL_ANALYSIS: [your analysis]
NATURALNESS: [your assessment]
REASONING: [2-4 sentences justifying your rating]
CONFIDENCE: [high/medium/low]
"""

    def _build_round2_prompt(
        self,
        text: str,
        round1: dict[str, ReviewResponse],
        context: str,
    ) -> str:
        """Build prompt for Round 2 deep analysis."""
        # Summarize Round 1 responses
        r1_summary = []
        for llm, resp in round1.items():
            r1_summary.append(f"{llm.upper()} rated {resp.rating}:")
            r1_summary.append(f"  Reasoning: {resp.reasoning}")
            if resp.mathematical_verification:
                r1_summary.append(f"  Math: {resp.mathematical_verification}")

        r1_text = "\n".join(r1_summary)

        return f"""DEEP ANALYSIS - Round 2

You previously reviewed this insight. Now consider the other reviewers' perspectives.

INSIGHT:
{text}

ROUND 1 REVIEWS:
{r1_text}

CONTEXT:
{context}

INSTRUCTIONS:
1. Consider what the other reviewers pointed out
2. Did they notice something you missed?
3. Has your assessment changed?

Provide your updated rating and reasoning.

FORMAT YOUR RESPONSE AS:
RATING: [bless/uncertain/reject]
NEW_INFORMATION: [what did others point out that you missed?]
CHANGED_MIND: [yes/no]
REASONING: [updated justification]
CONFIDENCE: [high/medium/low]
"""

    def _parse_review_response(
        self,
        llm: str,
        text: str,
        mode: str,
    ) -> ReviewResponse:
        """Parse LLM response into ReviewResponse."""
        import re

        # Default values
        rating = "uncertain"
        reasoning = text[:500]
        confidence = "medium"
        math_verification = ""
        structural_analysis = ""
        naturalness = ""

        # Try to extract structured fields
        rating_match = re.search(r'RATING:\s*(\w+)', text, re.IGNORECASE)
        if rating_match:
            r = rating_match.group(1).lower()
            if "bless" in r or "⚡" in r:
                rating = "bless"
            elif "reject" in r or "✗" in r:
                rating = "reject"
            else:
                rating = "uncertain"

        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:500]

        confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', text, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).lower()

        math_match = re.search(r'MATHEMATICAL_VERIFICATION:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if math_match:
            math_verification = math_match.group(1).strip()[:300]

        struct_match = re.search(r'STRUCTURAL_ANALYSIS:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if struct_match:
            structural_analysis = struct_match.group(1).strip()[:300]

        natural_match = re.search(r'NATURALNESS:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if natural_match:
            naturalness = natural_match.group(1).strip()[:300]

        return ReviewResponse(
            llm=llm,
            mode=mode,
            rating=rating,
            reasoning=reasoning,
            confidence=confidence,
            mathematical_verification=math_verification,
            structural_analysis=structural_analysis,
            naturalness_assessment=naturalness,
        )

    async def quick_check(
        self,
        text: str,
        *,
        context: str = "",
    ) -> tuple[str, str]:
        """
        Quick single-LLM check (no consensus, just one opinion).

        Useful for fast validation when consensus isn't needed.

        Args:
            text: The text to check
            context: Optional context

        Returns:
            Tuple of (rating, reasoning)
        """
        available = await self.client.get_available_backends()
        if not available:
            return "uncertain", "No backends available"

        # Prefer Claude for quick checks (API is faster)
        backend = "claude" if "claude" in available else available[0]

        prompt = f"""Quick review of this insight:

{text}

Context: {context}

Rate as: bless (valid), uncertain (needs work), or reject (flawed)
Give a one-sentence reason.

RATING:
REASON:"""

        response = await self.client.send(backend, prompt, timeout_seconds=60)

        if not response.success:
            return "uncertain", f"Error: {response.message}"

        # Parse simple response
        import re
        text_lower = response.text.lower()

        if "bless" in text_lower or "valid" in text_lower:
            rating = "bless"
        elif "reject" in text_lower or "flawed" in text_lower:
            rating = "reject"
        else:
            rating = "uncertain"

        reason_match = re.search(r'REASON:\s*(.+)', response.text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else response.text[:200]

        return rating, reason
