"""
Round 1: Independent Review

All three models review in parallel using standard modes.
If unanimous, we're done. Otherwise proceed to Round 2.
"""

import asyncio
from datetime import datetime
from typing import Optional

from shared.logging import get_logger

from .models import ReviewResponse, ReviewRound
from .prompts import build_round1_prompt, parse_round1_response
from .claude_api import ClaudeReviewer
from .llm_executor import LLMExecutor, create_executors

log = get_logger("explorer", "review_panel.round1")


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
    log.info("round1.started")

    # Build the prompt
    prompt = build_round1_prompt(
        chunk_insight=chunk_insight,
        confidence=confidence,
        tags=tags,
        dependencies=dependencies,
        blessed_axioms_summary=blessed_axioms_summary,
    )

    # Create executors for available LLMs
    executors = create_executors(gemini_browser, chatgpt_browser, claude_reviewer)

    if not executors:
        raise RuntimeError("No review models available for Round 1")

    # Determine thinking modes for Round 1
    # Gemini: standard, ChatGPT: thinking, Claude: standard
    thinking_modes = {
        "gemini": "standard",
        "chatgpt": "thinking",
        "claude": "standard",
    }

    # Run all reviews in parallel
    log.info("round1.parallel_reviews", llms=list(executors.keys()))

    tasks = [
        _review_with_executor(executor, prompt, thinking_modes.get(name, "standard"))
        for name, executor in executors.items()
    ]
    task_names = list(executors.keys())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect responses
    responses = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            log.error("round1.review_failed", llm=name, error=str(result))
            responses[name] = _create_failure_response(name, result)
        else:
            responses[name] = result

    # Determine outcome
    ratings = [r.rating for r in responses.values()]
    unique_ratings = set(ratings)

    # Check for ABANDON votes (unanimous = early exit)
    abandon_count = ratings.count("ABANDON")
    if abandon_count == len(ratings) and abandon_count >= 2:
        outcome = "abandoned"
        log.info("round1.abandoned", abandon_count=abandon_count)
    elif len(unique_ratings) == 1:
        outcome = "unanimous"
        log.info("round1.unanimous", rating=ratings[0])
    else:
        outcome = "split"
        log.info("round1.split", ratings=ratings)

    # Log any modifications proposed
    modifications = [(name, r.proposed_modification) for name, r in responses.items()
                     if r.proposed_modification]
    if modifications:
        for name, mod in modifications:
            log.info("round1.modification_proposed", llm=name, preview=mod[:100])

    return ReviewRound(
        round_number=1,
        mode="standard",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )


async def _review_with_executor(
    executor: LLMExecutor,
    prompt: str,
    thinking_mode: str,
) -> ReviewResponse:
    """
    Get review from an LLM using its executor.

    Args:
        executor: The LLM executor to use
        prompt: The review prompt
        thinking_mode: The thinking mode to use

    Returns:
        ReviewResponse from the LLM
    """
    log.info("round1.sending", llm=executor.name, thinking_mode=thinking_mode)

    try:
        response_text = await executor.send(prompt, thinking_mode=thinking_mode)
        parsed = parse_round1_response(response_text)
        return _build_response(executor.name, thinking_mode, parsed)

    except Exception as e:
        log.error("round1.error", llm=executor.name, error=str(e))
        raise


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


def _create_failure_response(llm: str, error: Exception) -> ReviewResponse:
    """Create a failure response for an LLM that errored."""
    return ReviewResponse(
        llm=llm,
        mode="standard",
        rating="?",
        mathematical_verification="Review failed",
        structural_analysis="Review failed",
        naturalness_assessment="Review failed",
        reasoning=f"Error during review: {str(error)}",
        confidence="low",
    )
