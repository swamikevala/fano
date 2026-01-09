"""
Refinement Round Logic (DEPRECATED)

NOTE: This module is deprecated. The new review flow allows LLMs to propose
modifications directly during any round (Round 1, 2, or 3). The modification
consensus logic in reviewer.py handles accepting and tracking these changes.

This file is kept for backward compatibility but is no longer used by the
main review flow.

Old approach (deprecated):
- Handles the refinement of insights when Round 1 reviews are mixed
- Claude Opus writes the refinement
- All three LLMs review the refined version

New approach (active):
- Any LLM can propose PROPOSED_MODIFICATION during review
- Modifications are evaluated for consensus between rounds
- ABANDON vote allows early exit for unsalvageable chunks
"""

import logging
from datetime import datetime
from typing import Optional

from .models import ReviewRound, ReviewResponse, RefinementRecord
from .claude_api import ClaudeReviewer
from chunking.prompts import (
    build_refinement_prompt,
    build_post_refinement_review_prompt,
    parse_refinement_response,
    parse_post_refinement_review,
)

logger = logging.getLogger(__name__)


async def run_refinement_round(
    original_insight: str,
    confidence: str,
    tags: list[str],
    dependencies: list[str],
    round1: ReviewRound,
    claude_reviewer: ClaudeReviewer,
    current_version: int = 1,
) -> tuple[Optional[RefinementRecord], str]:
    """
    Run a refinement round using Claude Opus.

    Args:
        original_insight: The original insight text
        confidence: Confidence level from extraction
        tags: Tags for the insight
        dependencies: Dependency IDs
        round1: The Round 1 review results
        claude_reviewer: Claude API client
        current_version: Current version number of the insight

    Returns:
        Tuple of (RefinementRecord, refined_insight_text)
        Returns (None, original_insight) if refinement fails
    """
    if not claude_reviewer or not claude_reviewer.is_available():
        logger.warning("[refinement] Claude not available for refinement")
        return None, original_insight

    logger.info(f"[refinement] Starting refinement round (version {current_version} → {current_version + 1})")

    # Extract critiques from Round 1
    gemini_resp = round1.responses.get("gemini")
    chatgpt_resp = round1.responses.get("chatgpt")
    claude_resp = round1.responses.get("claude")

    # Build refinement prompt
    prompt = build_refinement_prompt(
        original_insight=original_insight,
        confidence=confidence,
        tags=tags,
        dependencies=dependencies,
        gemini_rating=gemini_resp.rating if gemini_resp else "N/A",
        gemini_reasoning=_format_critique(gemini_resp) if gemini_resp else "Not available",
        chatgpt_rating=chatgpt_resp.rating if chatgpt_resp else "N/A",
        chatgpt_reasoning=_format_critique(chatgpt_resp) if chatgpt_resp else "Not available",
        claude_rating=claude_resp.rating if claude_resp else "N/A",
        claude_reasoning=_format_critique(claude_resp) if claude_resp else "Not available",
    )

    try:
        # Send to Claude Opus for refinement
        response = await claude_reviewer.send_message(prompt, extended_thinking=False)
        logger.info(f"[refinement] Got response ({len(response)} chars)")

        # Parse the response
        parsed = parse_refinement_response(response)

        if not parsed.get("refined_insight"):
            logger.warning("[refinement] No refined insight in response")
            return None, original_insight

        refined_insight = parsed["refined_insight"]

        # Create refinement record
        refinement = RefinementRecord(
            from_version=current_version,
            to_version=current_version + 1,
            original_insight=original_insight,
            refined_insight=refined_insight,
            changes_made=parsed.get("changes_made", []),
            addressed_critiques=parsed.get("addressed_critiques", []),
            unresolved_issues=parsed.get("unresolved_issues", []),
            refinement_confidence=parsed.get("refinement_confidence", "medium"),
            triggered_by_ratings=round1.get_ratings(),
            timestamp=datetime.now(),
        )

        logger.info(f"[refinement] Created refinement: {len(refinement.changes_made)} changes, "
                   f"confidence={refinement.refinement_confidence}")

        return refinement, refined_insight

    except Exception as e:
        logger.error(f"[refinement] Failed: {e}")
        return None, original_insight


async def run_post_refinement_review(
    original_insight: str,
    refined_insight: str,
    refinement: RefinementRecord,
    round1: ReviewRound,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: ClaudeReviewer,
    config: dict,
) -> tuple[ReviewRound, list]:
    """
    Run a review round on the refined insight.

    All 3 reviewers see both original and refined versions.

    Args:
        original_insight: Original insight text
        refined_insight: Refined insight text
        refinement: The refinement record
        round1: Round 1 review results (for original ratings/critiques)
        gemini_browser: Gemini browser interface
        chatgpt_browser: ChatGPT browser interface
        claude_reviewer: Claude API client
        config: Review panel config

    Returns:
        Tuple of (ReviewRound, list of mind changes)
    """
    from .round2 import _parse_review_response

    logger.info("[refinement] Starting post-refinement review")

    responses = {}
    mind_changes = []

    # Format changes made and summary of critiques
    changes_made = "\n".join(f"- {c}" for c in refinement.changes_made)
    critiques_summary = _format_critiques_summary(round1)

    # Review with each available model
    reviewers = []
    if gemini_browser:
        reviewers.append(("gemini", gemini_browser, "gemini"))
    if chatgpt_browser:
        reviewers.append(("chatgpt", chatgpt_browser, "chatgpt"))
    if claude_reviewer and claude_reviewer.is_available():
        reviewers.append(("claude", claude_reviewer, "claude"))

    for llm_name, model, model_key in reviewers:
        try:
            # Get this reviewer's original response
            original_resp = round1.responses.get(llm_name)
            original_rating = original_resp.rating if original_resp else "N/A"
            original_critique = _format_critique(original_resp) if original_resp else "Not available"

            # Build post-refinement review prompt
            prompt = build_post_refinement_review_prompt(
                original_insight=original_insight,
                refined_insight=refined_insight,
                changes_made=changes_made,
                summary_of_critiques=critiques_summary,
                reviewer_original_rating=original_rating,
                reviewer_original_critique=original_critique,
            )

            # Send to model
            if llm_name == "claude":
                raw_response = await model.send_message(prompt, extended_thinking=False)
            elif llm_name == "chatgpt":
                await model.start_new_chat()
                raw_response = await model.send_message(prompt, use_pro_mode=False, use_thinking_mode=True)
            else:  # gemini
                await model.start_new_chat()
                raw_response = await model.send_message(prompt, use_deep_think=False)

            # Parse response
            parsed = parse_post_refinement_review(raw_response)

            # Create ReviewResponse
            response = ReviewResponse(
                llm=llm_name,
                mode="post_refinement",
                rating=parsed["rating"],
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=parsed["reasoning"],
                confidence="medium",
                new_information=parsed.get("new_issues", ""),
                changed_mind=(parsed["rating"] != original_rating) if original_resp else None,
                previous_rating=original_rating if original_resp else None,
            )
            responses[llm_name] = response

            logger.info(f"[refinement] {llm_name}: {parsed['rating']} "
                       f"(issues addressed: {parsed['issues_addressed']})")

            # Track mind changes
            if original_resp and parsed["rating"] != original_rating:
                from .models import MindChange
                mind_changes.append(MindChange(
                    llm=llm_name,
                    round_number=2,  # Post-refinement is essentially Round 2
                    from_rating=original_rating,
                    to_rating=parsed["rating"],
                    reason=f"Refinement addressed: {parsed.get('issues_addressed', 'unknown')}",
                ))

        except Exception as e:
            logger.error(f"[refinement] {llm_name} review failed: {e}")

    # Determine outcome
    if not responses:
        outcome = "failed"
    elif len(set(r.rating for r in responses.values())) == 1:
        outcome = "unanimous"
    else:
        outcome = "split"

    review_round = ReviewRound(
        round_number=2,  # Post-refinement review is Round 2
        mode="post_refinement",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    return review_round, mind_changes


def _format_critique(response: ReviewResponse) -> str:
    """Format a reviewer's critique for the refinement prompt."""
    parts = []

    if response.mathematical_verification:
        parts.append(f"Math: {response.mathematical_verification}")
    if response.structural_analysis:
        parts.append(f"Structure: {response.structural_analysis}")
    if response.naturalness_assessment:
        parts.append(f"Naturalness: {response.naturalness_assessment}")
    if response.reasoning:
        parts.append(f"Reasoning: {response.reasoning}")

    return "\n".join(parts) if parts else response.reasoning or "No detailed critique"


def _format_critiques_summary(round1: ReviewRound) -> str:
    """Format a summary of all critiques from Round 1."""
    lines = []

    for llm_name, resp in round1.responses.items():
        lines.append(f"{llm_name.upper()} ({resp.rating}):")
        lines.append(f"  {resp.reasoning[:200]}..." if len(resp.reasoning) > 200 else f"  {resp.reasoning}")
        lines.append("")

    return "\n".join(lines)
