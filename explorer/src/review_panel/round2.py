"""
Round 2: Deep Analysis

Each model sees all Round 1 responses and re-evaluates with deep thinking modes:
- Gemini: Deep Think mode
- ChatGPT: Pro mode (or o1 if available)
- Claude: Extended Thinking

If unanimous after this round, we're done. Otherwise proceed to Round 3.
"""

import asyncio
from datetime import datetime
from typing import Optional

from shared.logging import get_logger

from .models import ReviewResponse, ReviewRound, MindChange, VerificationResult
from .prompts import build_round2_prompt, parse_round2_response
from .claude_api import ClaudeReviewer
from .llm_executor import LLMExecutor, create_executors

# Import quota exception for special handling
try:
    from ..browser.gemini import GeminiQuotaExhausted
except ImportError:
    GeminiQuotaExhausted = None

log = get_logger("explorer", "review_panel.round2")


def _to_dict(obj) -> dict:
    """Convert response object to dict, handling both objects and dicts."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj


def _get_rating(obj) -> str:
    """Get rating from response object or dict."""
    if hasattr(obj, 'rating'):
        return obj.rating
    return obj.get("rating", "?")


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
    log.info("round2.started")
    if accepted_modification:
        log.info("round2.reviewing_modified", source=modification_source)

    # Extract Round 1 responses for the prompt
    r1_responses = {
        "gemini": round1.responses.get("gemini", _empty_response("gemini")),
        "chatgpt": round1.responses.get("chatgpt", _empty_response("chatgpt")),
        "claude": round1.responses.get("claude", _empty_response("claude")),
    }

    # Create executors for available LLMs
    executors = create_executors(gemini_browser, chatgpt_browser, claude_reviewer)

    if not executors:
        raise RuntimeError("No review models available for Round 2")

    # Build prompts for each LLM (they see different "this_llm" values)
    prompts = {}
    for llm_name in executors.keys():
        prompts[llm_name] = build_round2_prompt(
            chunk_insight=chunk_insight,
            blessed_axioms_summary=blessed_axioms_summary,
            gemini_response=_to_dict(r1_responses["gemini"]),
            chatgpt_response=_to_dict(r1_responses["chatgpt"]),
            claude_response=_to_dict(r1_responses["claude"]),
            this_llm=llm_name,
            this_llm_round1_rating=_get_rating(r1_responses[llm_name]),
            math_verification=math_verification,
            accepted_modification=accepted_modification,
            modification_source=modification_source,
        )

    # Run all deep analyses in parallel with "deep" thinking mode
    log.info("round2.parallel_reviews", llms=list(executors.keys()))

    tasks = [
        _deep_review_with_executor(executors[name], prompts[name])
        for name in executors.keys()
    ]
    task_names = list(executors.keys())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect responses and track mind changes
    responses = {}
    mind_changes = []
    quota_exhausted_exception = None

    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            # Check for Gemini quota exhaustion - special handling
            if GeminiQuotaExhausted and isinstance(result, GeminiQuotaExhausted):
                log.error("round2.quota_exhausted", llm=name, error=str(result))
                quota_exhausted_exception = result
            else:
                log.error("round2.review_failed", llm=name, error=str(result))

            # Carry forward Round 1 response or create failure
            responses[name] = _carry_forward_or_fail(name, round1, result)
        else:
            responses[name] = result

            # Check for mind change
            if name in round1.responses:
                r1_rating = round1.responses[name].rating
                r2_rating = result.rating

                if r1_rating != r2_rating:
                    log.info("round2.mind_changed", llm=name, from_rating=r1_rating, to_rating=r2_rating)
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

    # Check for ABANDON votes
    abandon_count = ratings.count("ABANDON")
    if abandon_count == len(ratings) and abandon_count >= 2:
        outcome = "abandoned"
        log.info("round2.abandoned", abandon_count=abandon_count)
    elif len(unique_ratings) == 1:
        outcome = "unanimous"
        log.info("round2.unanimous", rating=ratings[0])
    else:
        outcome = "split"
        log.info("round2.split", ratings=ratings)

    # Log any modifications proposed
    modifications = [(name, r.proposed_modification) for name, r in responses.items()
                     if r.proposed_modification]
    if modifications:
        for name, mod in modifications:
            log.info("round2.modification_proposed", llm=name, preview=mod[:100])

    review_round = ReviewRound(
        round_number=2,
        mode="deep",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    # If quota was exhausted, re-raise AFTER creating the round
    if quota_exhausted_exception:
        log.warning("round2.reraise_quota_exhaustion")
        quota_exhausted_exception.partial_round = review_round
        quota_exhausted_exception.mind_changes = mind_changes
        raise quota_exhausted_exception

    return review_round, mind_changes


async def _deep_review_with_executor(
    executor: LLMExecutor,
    prompt: str,
) -> ReviewResponse:
    """
    Get deep review from an LLM using its executor.

    Args:
        executor: The LLM executor to use
        prompt: The review prompt

    Returns:
        ReviewResponse from the LLM
    """
    log.info("round2.sending", llm=executor.name, thinking_mode="deep")

    try:
        response_text = await executor.send(prompt, thinking_mode="deep")
        parsed = parse_round2_response(response_text)
        return _build_deep_response(executor.name, parsed)

    except Exception as e:
        log.error("round2.error", llm=executor.name, error=str(e))
        raise


def _empty_response(llm: str) -> dict:
    """Create an empty response dict for missing reviewers."""
    return {
        "rating": "?",
        "mathematical_verification": "Not available",
        "structural_analysis": "Not available",
        "naturalness_assessment": "Not available",
        "reasoning": "Reviewer not available",
    }


def _build_deep_response(llm: str, parsed: dict) -> ReviewResponse:
    """Build a ReviewResponse from parsed deep review data."""
    return ReviewResponse(
        llm=llm,
        mode="deep",
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


def _carry_forward_or_fail(llm: str, round1: ReviewRound, error: Exception) -> ReviewResponse:
    """Carry forward Round 1 response or create failure response."""
    if llm in round1.responses:
        return round1.responses[llm]

    return ReviewResponse(
        llm=llm,
        mode="deep",
        rating="?",
        mathematical_verification="Deep review failed",
        structural_analysis="Deep review failed",
        naturalness_assessment="Deep review failed",
        reasoning=f"Error during deep review: {str(error)}",
        confidence="low",
    )
