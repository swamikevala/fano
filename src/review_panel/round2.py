"""
Round 2: Deep Analysis

Each model sees all Round 1 responses and re-evaluates with deep thinking modes:
- Gemini: Deep Think mode
- ChatGPT: Pro mode (or o1 if available)
- Claude: Extended Thinking

If unanimous after this round, we're done. Otherwise proceed to Round 3.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .models import ReviewResponse, ReviewRound, MindChange, VerificationResult
from .prompts import build_round2_prompt, parse_round2_response
from .claude_api import ClaudeReviewer

# Import quota exception for special handling
try:
    from ..browser.gemini import GeminiQuotaExhausted
except ImportError:
    GeminiQuotaExhausted = None

logger = logging.getLogger(__name__)


async def run_round2(
    chunk_insight: str,
    blessed_axioms_summary: str,
    round1: ReviewRound,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
    config: dict,
    math_verification: Optional[VerificationResult] = None,
    accepted_modification: str = "",
    modification_source: str = "",
) -> tuple[ReviewRound, list[MindChange]]:
    """
    Run Round 2: Deep analysis with all Round 1 responses visible.

    Args:
        chunk_insight: The insight text being reviewed (may be modified from Round 1)
        blessed_axioms_summary: Summary of blessed axioms
        round1: The completed Round 1
        gemini_browser: GeminiBrowser instance
        chatgpt_browser: ChatGPTBrowser instance
        claude_reviewer: ClaudeReviewer instance
        config: Review panel configuration
        math_verification: Optional DeepSeek verification result
        accepted_modification: If a modification was accepted after Round 1
        modification_source: Which LLM proposed the accepted modification

    Returns:
        Tuple of (ReviewRound, list of MindChanges)
    """
    logger.info("[round2] Starting deep analysis")
    if accepted_modification:
        logger.info(f"[round2] Reviewing MODIFIED insight (proposed by {modification_source})")

    # Extract Round 1 responses for the prompt
    gemini_r1 = round1.responses.get("gemini", _empty_response("gemini"))
    chatgpt_r1 = round1.responses.get("chatgpt", _empty_response("chatgpt"))
    claude_r1 = round1.responses.get("claude", _empty_response("claude"))

    # Run all deep analyses in parallel
    tasks = []

    # Gemini Deep Think
    if gemini_browser:
        prompt = build_round2_prompt(
            chunk_insight=chunk_insight,
            blessed_axioms_summary=blessed_axioms_summary,
            gemini_response=gemini_r1.to_dict() if hasattr(gemini_r1, 'to_dict') else gemini_r1,
            chatgpt_response=chatgpt_r1.to_dict() if hasattr(chatgpt_r1, 'to_dict') else chatgpt_r1,
            claude_response=claude_r1.to_dict() if hasattr(claude_r1, 'to_dict') else claude_r1,
            this_llm="gemini",
            this_llm_round1_rating=gemini_r1.rating if hasattr(gemini_r1, 'rating') else gemini_r1.get("rating", "?"),
            math_verification=math_verification,
            accepted_modification=accepted_modification,
            modification_source=modification_source,
        )
        tasks.append(("gemini", _deep_review_gemini(gemini_browser, prompt)))
    else:
        logger.warning("[round2] Gemini browser not available")

    # ChatGPT Pro/o1
    if chatgpt_browser:
        prompt = build_round2_prompt(
            chunk_insight=chunk_insight,
            blessed_axioms_summary=blessed_axioms_summary,
            gemini_response=gemini_r1.to_dict() if hasattr(gemini_r1, 'to_dict') else gemini_r1,
            chatgpt_response=chatgpt_r1.to_dict() if hasattr(chatgpt_r1, 'to_dict') else chatgpt_r1,
            claude_response=claude_r1.to_dict() if hasattr(claude_r1, 'to_dict') else claude_r1,
            this_llm="chatgpt",
            this_llm_round1_rating=chatgpt_r1.rating if hasattr(chatgpt_r1, 'rating') else chatgpt_r1.get("rating", "?"),
            math_verification=math_verification,
            accepted_modification=accepted_modification,
            modification_source=modification_source,
        )
        tasks.append(("chatgpt", _deep_review_chatgpt(chatgpt_browser, prompt)))
    else:
        logger.warning("[round2] ChatGPT browser not available")

    # Claude Extended Thinking
    if claude_reviewer and claude_reviewer.is_available():
        prompt = build_round2_prompt(
            chunk_insight=chunk_insight,
            blessed_axioms_summary=blessed_axioms_summary,
            gemini_response=gemini_r1.to_dict() if hasattr(gemini_r1, 'to_dict') else gemini_r1,
            chatgpt_response=chatgpt_r1.to_dict() if hasattr(chatgpt_r1, 'to_dict') else chatgpt_r1,
            claude_response=claude_r1.to_dict() if hasattr(claude_r1, 'to_dict') else claude_r1,
            this_llm="claude",
            this_llm_round1_rating=claude_r1.rating if hasattr(claude_r1, 'rating') else claude_r1.get("rating", "?"),
            math_verification=math_verification,
            accepted_modification=accepted_modification,
            modification_source=modification_source,
        )
        tasks.append(("claude", _deep_review_claude(claude_reviewer, prompt)))
    else:
        logger.warning("[round2] Claude API not available")

    if not tasks:
        raise RuntimeError("No review models available for Round 2")

    # Execute all reviews in parallel
    task_names = [t[0] for t in tasks]
    task_coros = [t[1] for t in tasks]

    logger.info(f"[round2] Running deep analysis: {task_names}")
    results = await asyncio.gather(*task_coros, return_exceptions=True)

    # Collect responses and track mind changes
    responses = {}
    mind_changes = []
    quota_exhausted_exception = None  # Track quota exhaustion for later re-raise

    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            # Check for Gemini quota exhaustion - special handling
            if GeminiQuotaExhausted and isinstance(result, GeminiQuotaExhausted):
                logger.error(f"[round2] Gemini Deep Think quota exhausted: {result}")
                quota_exhausted_exception = result
                # Carry forward Round 1 response so we have something
                if name in round1.responses:
                    responses[name] = round1.responses[name]
                else:
                    responses[name] = ReviewResponse(
                        llm=name,
                        mode="deep",
                        rating="?",
                        mathematical_verification="Gemini Deep Think quota exhausted",
                        structural_analysis="Gemini Deep Think quota exhausted",
                        naturalness_assessment="Gemini Deep Think quota exhausted",
                        reasoning=f"Quota exhausted: {str(result)}",
                        confidence="low",
                    )
            else:
                logger.error(f"[round2] {name} failed: {result}")
                # Carry forward Round 1 response on failure
                if name in round1.responses:
                    responses[name] = round1.responses[name]
                else:
                    responses[name] = ReviewResponse(
                        llm=name,
                        mode="deep",
                        rating="?",
                        mathematical_verification="Deep review failed",
                        structural_analysis="Deep review failed",
                        naturalness_assessment="Deep review failed",
                        reasoning=f"Error during deep review: {str(result)}",
                        confidence="low",
                    )
        else:
            responses[name] = result

            # Check for mind change
            if name in round1.responses:
                r1_rating = round1.responses[name].rating
                r2_rating = result.rating

                if r1_rating != r2_rating:
                    logger.info(f"[round2] {name} changed mind: {r1_rating} -> {r2_rating}")
                    mind_changes.append(MindChange(
                        llm=name,
                        round_number=2,
                        from_rating=r1_rating,
                        to_rating=r2_rating,
                        reason=result.new_information or result.reasoning or "No reason given",
                    ))

    # Determine outcome
    ratings = [r.rating for r in responses.values()]
    unique_ratings = set(ratings)

    # Check for ABANDON votes (unanimous = early exit)
    abandon_count = ratings.count("ABANDON")
    if abandon_count == len(ratings) and abandon_count >= 2:
        outcome = "abandoned"
        logger.info(f"[round2] ABANDONED unanimously ({abandon_count} votes)")
    elif len(unique_ratings) == 1:
        outcome = "unanimous"
        logger.info(f"[round2] Unanimous after deep analysis: {ratings[0]}")
    else:
        outcome = "split"
        logger.info(f"[round2] Still split after deep analysis: {ratings}")

    # Log any modifications proposed
    modifications = [(name, r.proposed_modification) for name, r in responses.items()
                     if r.proposed_modification]
    if modifications:
        for name, mod in modifications:
            logger.info(f"[round2] {name} proposed modification: {mod[:100]}...")

    review_round = ReviewRound(
        round_number=2,
        mode="deep",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    # If quota was exhausted, re-raise AFTER creating the round
    # so the caller can save partial progress before handling the error
    if quota_exhausted_exception:
        logger.warning(f"[round2] Re-raising quota exhaustion after collecting other LLM results")
        # Attach the round to the exception so caller can save it
        quota_exhausted_exception.partial_round = review_round
        quota_exhausted_exception.mind_changes = mind_changes
        raise quota_exhausted_exception

    return review_round, mind_changes


def _empty_response(llm: str) -> dict:
    """Create an empty response dict for missing reviewers."""
    return {
        "rating": "?",
        "mathematical_verification": "Not available",
        "structural_analysis": "Not available",
        "naturalness_assessment": "Not available",
        "reasoning": "Reviewer not available",
    }


def _build_deep_response(llm: str, mode: str, parsed: dict) -> ReviewResponse:
    """Build a ReviewResponse from parsed deep review data."""
    return ReviewResponse(
        llm=llm,
        mode=mode,
        rating=parsed["rating"],
        mathematical_verification=parsed.get("mathematical_verification", ""),
        structural_analysis=parsed.get("structural_analysis", ""),
        naturalness_assessment=parsed.get("naturalness_assessment", ""),
        reasoning=parsed["reasoning"],
        confidence=parsed["confidence"],
        proposed_modification=parsed.get("proposed_modification") or None,
        modification_rationale=parsed.get("modification_rationale") or None,
        new_information=parsed.get("new_information"),
        changed_mind=parsed.get("changed_mind"),
    )


async def _deep_review_gemini(gemini_browser, prompt: str) -> ReviewResponse:
    """Get deep review from Gemini using Deep Think mode."""
    logger.info("[round2] Sending to Gemini (Deep Think mode)")

    try:
        # Start fresh chat first to ensure clean state
        await gemini_browser.start_new_chat()

        # Enable Deep Think mode (this will handle confirmation and setup new chat)
        await gemini_browser.enable_deep_think()

        # Send the prompt and get response
        response_text = await gemini_browser.send_message(prompt)

        # Parse the response
        parsed = parse_round2_response(response_text)

        return _build_deep_response("gemini", "deep_think", parsed)

    except Exception as e:
        logger.error(f"[round2] Gemini Deep Think error: {e}")
        raise


async def _deep_review_chatgpt(chatgpt_browser, prompt: str) -> ReviewResponse:
    """Get deep review from ChatGPT using Pro/o1 mode."""
    logger.info("[round2] Sending to ChatGPT (Pro mode)")

    try:
        # Start fresh chat first to ensure clean state
        await chatgpt_browser.start_new_chat()

        # Try to enable Pro mode if available
        try:
            await chatgpt_browser.enable_pro_mode()
        except Exception as e:
            logger.warning(f"[round2] Could not enable Pro mode: {e}")

        # Send the prompt and get response
        response_text = await chatgpt_browser.send_message(prompt)

        # Parse the response
        parsed = parse_round2_response(response_text)

        return _build_deep_response("chatgpt", "pro", parsed)

    except Exception as e:
        logger.error(f"[round2] ChatGPT Pro error: {e}")
        raise


async def _deep_review_claude(claude_reviewer: ClaudeReviewer, prompt: str) -> ReviewResponse:
    """Get deep review from Claude using Extended Thinking."""
    logger.info("[round2] Sending to Claude (Extended Thinking)")

    try:
        # Use extended thinking for deep analysis
        response_text = await claude_reviewer.send_message(
            prompt,
            extended_thinking=True,
        )

        # Parse the response
        parsed = parse_round2_response(response_text)

        return _build_deep_response("claude", "extended_thinking", parsed)

    except Exception as e:
        logger.error(f"[round2] Claude Extended Thinking error: {e}")
        raise
