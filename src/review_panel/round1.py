"""
Round 1: Independent Review

All three models review in parallel using standard modes.
If unanimous, we're done. Otherwise proceed to Round 2.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .models import ReviewResponse, ReviewRound
from .prompts import build_round1_prompt, parse_round1_response
from .claude_api import ClaudeReviewer

logger = logging.getLogger(__name__)


async def run_round1(
    chunk_insight: str,
    confidence: str,
    tags: list[str],
    dependencies: list[str],
    blessed_axioms_summary: str,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
    config: dict,
) -> ReviewRound:
    """
    Run Round 1: Independent parallel review by all three models.

    Args:
        chunk_insight: The insight text to review
        confidence: Confidence level from extraction
        tags: Tags assigned to the insight
        dependencies: Dependencies listed for the insight
        blessed_axioms_summary: Summary of blessed axioms for context
        gemini_browser: GeminiBrowser instance
        chatgpt_browser: ChatGPTBrowser instance
        claude_reviewer: ClaudeReviewer instance (or None if unavailable)
        config: Review panel configuration

    Returns:
        ReviewRound with all responses
    """
    logger.info("[round1] Starting independent review")

    # Build the prompt
    prompt = build_round1_prompt(
        chunk_insight=chunk_insight,
        confidence=confidence,
        tags=tags,
        dependencies=dependencies,
        blessed_axioms_summary=blessed_axioms_summary,
    )

    # Run all reviews in parallel
    tasks = []

    # Gemini review
    if gemini_browser:
        tasks.append(("gemini", _review_with_gemini(gemini_browser, prompt)))
    else:
        logger.warning("[round1] Gemini browser not available")

    # ChatGPT review
    if chatgpt_browser:
        tasks.append(("chatgpt", _review_with_chatgpt(chatgpt_browser, prompt)))
    else:
        logger.warning("[round1] ChatGPT browser not available")

    # Claude review
    if claude_reviewer and claude_reviewer.is_available():
        tasks.append(("claude", _review_with_claude(claude_reviewer, prompt)))
    else:
        logger.warning("[round1] Claude API not available")

    if not tasks:
        raise RuntimeError("No review models available for Round 1")

    # Execute all reviews in parallel
    task_names = [t[0] for t in tasks]
    task_coros = [t[1] for t in tasks]

    logger.info(f"[round1] Running parallel reviews: {task_names}")
    results = await asyncio.gather(*task_coros, return_exceptions=True)

    # Collect responses
    responses = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.error(f"[round1] {name} failed: {result}")
            # Create a failure response
            responses[name] = ReviewResponse(
                llm=name,
                mode="standard",
                rating="?",
                mathematical_verification="Review failed",
                structural_analysis="Review failed",
                naturalness_assessment="Review failed",
                reasoning=f"Error during review: {str(result)}",
                confidence="low",
            )
        else:
            responses[name] = result

    # Determine outcome
    ratings = [r.rating for r in responses.values()]
    unique_ratings = set(ratings)

    # Check for ABANDON votes (unanimous = early exit)
    abandon_count = ratings.count("ABANDON")
    if abandon_count == len(ratings) and abandon_count >= 2:
        outcome = "abandoned"
        logger.info(f"[round1] ABANDONED unanimously ({abandon_count} votes)")
    elif len(unique_ratings) == 1:
        outcome = "unanimous"
        logger.info(f"[round1] Unanimous: {ratings[0]}")
    else:
        outcome = "split"
        logger.info(f"[round1] Split: {ratings}")

    # Log any modifications proposed
    modifications = [(name, r.proposed_modification) for name, r in responses.items()
                     if r.proposed_modification]
    if modifications:
        for name, mod in modifications:
            logger.info(f"[round1] {name} proposed modification: {mod[:100]}...")

    return ReviewRound(
        round_number=1,
        mode="standard",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )


def _build_response(llm: str, mode: str, parsed: dict) -> ReviewResponse:
    """Build a ReviewResponse from parsed data."""
    return ReviewResponse(
        llm=llm,
        mode=mode,
        rating=parsed["rating"],
        mathematical_verification=parsed["mathematical_verification"],
        structural_analysis=parsed["structural_analysis"],
        naturalness_assessment=parsed["naturalness_assessment"],
        reasoning=parsed["reasoning"],
        confidence=parsed["confidence"],
        proposed_modification=parsed.get("proposed_modification") or None,
        modification_rationale=parsed.get("modification_rationale") or None,
    )


async def _review_with_gemini(gemini_browser, prompt: str) -> ReviewResponse:
    """Get review from Gemini using browser automation."""
    logger.info("[round1] Sending to Gemini (standard mode)")

    try:
        # Start fresh chat to reset any Deep Think state and clear old content
        await gemini_browser.start_new_chat()

        # Send the prompt and get response
        response_text = await gemini_browser.send_message(prompt)

        # Parse the response
        parsed = parse_round1_response(response_text)

        return _build_response("gemini", "standard", parsed)

    except Exception as e:
        logger.error(f"[round1] Gemini error: {e}")
        raise


async def _review_with_chatgpt(chatgpt_browser, prompt: str) -> ReviewResponse:
    """Get review from ChatGPT using browser automation."""
    logger.info("[round1] Sending to ChatGPT (Thinking mode)")

    try:
        # Start fresh chat to clear old content
        await chatgpt_browser.start_new_chat()

        # Send the prompt with Thinking mode enabled (not Pro - save Pro for Round 2)
        response_text = await chatgpt_browser.send_message(prompt, use_thinking_mode=True)

        # Parse the response
        parsed = parse_round1_response(response_text)

        return _build_response("chatgpt", "thinking", parsed)

    except Exception as e:
        logger.error(f"[round1] ChatGPT error: {e}")
        raise


async def _review_with_claude(claude_reviewer: ClaudeReviewer, prompt: str) -> ReviewResponse:
    """Get review from Claude using API."""
    logger.info("[round1] Sending to Claude (standard mode)")

    try:
        # Send the prompt (no extended thinking for Round 1)
        response_text = await claude_reviewer.send_message(
            prompt,
            extended_thinking=False,
        )

        # Parse the response
        parsed = parse_round1_response(response_text)

        return _build_response("claude", "standard", parsed)

    except Exception as e:
        logger.error(f"[round1] Claude error: {e}")
        raise
