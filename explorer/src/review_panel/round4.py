"""
Round 4: Modification Focus

For disputed insights that went through all 3 rounds without consensus.
This round shows the full deliberation history and focuses specifically
on finding a modification that could resolve the disagreement.

Phase 1: All LLMs see full history and propose modifications
Phase 2: If a modification is accepted, all LLMs vote on the modified insight
"""

import asyncio
from datetime import datetime
from typing import Optional, Tuple

from shared.logging import get_logger

from .models import ReviewResponse, ReviewRound, RefinementRecord
from .prompts import (
    build_round4_modification_prompt,
    build_round4_final_vote_prompt,
    build_round_summary,
    parse_round4_modification_response,
    parse_round4_final_vote_response,
)
from .claude_api import ClaudeReviewer
from .llm_executor import LLMExecutor, create_executors

log = get_logger("explorer", "review_panel.round4")


async def run_round4(
    chunk_insight: str,
    review_rounds: list[ReviewRound],
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
    config: dict,
) -> Tuple[ReviewRound, Optional[str], Optional[RefinementRecord], bool]:
    """
    Run Round 4: Modification-focused review for disputed insights.

    Args:
        chunk_insight: The insight text being reviewed
        review_rounds: List of previous ReviewRounds (1, 2, 3)
        gemini_browser: GeminiBrowser instance
        chatgpt_browser: ChatGPTBrowser instance
        claude_reviewer: ClaudeReviewer instance
        config: Review panel configuration

    Returns:
        Tuple of (ReviewRound, modified_insight, refinement_record, is_intractable)
        - modified_insight: The new insight text if modification was accepted, else None
        - refinement_record: Record of the modification if one was accepted, else None
        - is_intractable: True if no LLM could propose a valid fix
    """
    log.info("round4.started")

    # Create executors for available LLMs
    executors = create_executors(gemini_browser, chatgpt_browser, claude_reviewer)

    if not executors:
        raise RuntimeError("No LLMs available for Round 4")

    # Build summaries of previous rounds
    round1 = next((r for r in review_rounds if r.round_number == 1), None)
    round2 = next((r for r in review_rounds if r.round_number == 2), None)
    round3 = next((r for r in review_rounds if r.round_number == 3), None)

    round1_summary = build_round_summary(round1.responses, 1) if round1 else "Not available"
    round2_summary = build_round_summary(round2.responses, 2) if round2 else "Not available"
    round3_summary = build_round_summary(round3.responses, 3) if round3 else "Not available"

    # Get final ratings from Round 3 (or Round 2 if no Round 3)
    final_round = round3 or round2 or round1
    final_ratings = {llm: r.rating for llm, r in final_round.responses.items()} if final_round else {}

    # Phase 1: Get modification proposals from all LLMs
    log.info("round4.phase1_proposals", llms=list(executors.keys()))

    prompt = build_round4_modification_prompt(
        chunk_insight=chunk_insight,
        round1_summary=round1_summary,
        round2_summary=round2_summary,
        round3_summary=round3_summary,
        final_ratings=final_ratings,
    )

    # Run all proposals in parallel with deep thinking
    tasks = [
        (name, _get_modification_proposal(executor, prompt))
        for name, executor in executors.items()
    ]

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    # Collect proposals
    proposals = {}
    for (name, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            log.error("round4.proposal_failed", llm=name, error=str(result))
        else:
            proposals[name] = result
            if result.get("can_be_fixed"):
                log.info("round4.proposal_fix", llm=name, expected_rating=result.get('expected_rating'))
            else:
                log.info("round4.proposal_no_fix", llm=name)

    # Select best modification using priority logic
    best_mod, mod_source, mod_rationale = _select_best_modification(proposals)

    if not best_mod:
        # No valid modification proposed - mark as intractable
        log.info("round4.intractable")

        # Build response round with diagnoses
        responses = {}
        for name, proposal in proposals.items():
            responses[name] = ReviewResponse(
                llm=name,
                mode="modification_focus",
                rating="?",
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=proposal.get("diagnosis", "No diagnosis"),
                confidence="high",
                proposed_modification=None,
                modification_rationale=proposal.get("modification_rationale", ""),
            )

        review_round = ReviewRound(
            round_number=4,
            mode="modification_focus",
            responses=responses,
            outcome="intractable",
            timestamp=datetime.now(),
        )

        return review_round, None, None, True

    # Phase 2: Get final votes on the modified insight
    log.info("round4.phase2_votes", modification_source=mod_source)

    vote_prompt = build_round4_final_vote_prompt(
        original_insight=chunk_insight,
        modified_insight=best_mod,
        modification_source=mod_source,
        modification_rationale=mod_rationale,
        round3_summary=round3_summary,
    )

    # Run all votes in parallel with standard mode
    vote_tasks = [
        (name, _get_final_vote(executor, vote_prompt))
        for name, executor in executors.items()
    ]

    vote_results = await asyncio.gather(*[t[1] for t in vote_tasks], return_exceptions=True)

    # Build responses
    responses = {}
    for (name, _), result in zip(vote_tasks, vote_results):
        if isinstance(result, Exception):
            log.error("round4.vote_failed", llm=name, error=str(result))
            responses[name] = ReviewResponse(
                llm=name,
                mode="final_vote",
                rating="?",
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=f"Vote failed: {result}",
                confidence="low",
            )
        else:
            responses[name] = ReviewResponse(
                llm=name,
                mode="final_vote",
                rating=result["rating"],
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=result["reasoning"],
                confidence=result["confidence"],
            )
            log.info("round4.vote", llm=name, rating=result['rating'],
                     resolves=result.get('resolves_dispute', '?'))

    # Determine outcome
    ratings = [r.rating for r in responses.values()]
    unique_ratings = set(ratings)

    if len(unique_ratings) == 1:
        outcome = "resolved"
        log.info("round4.resolved", rating=ratings[0])
    else:
        rating_counts = {r: ratings.count(r) for r in unique_ratings}
        max_count = max(rating_counts.values())
        if max_count >= 2:
            outcome = "majority"
            majority_rating = [r for r, c in rating_counts.items() if c == max_count][0]
            log.info("round4.majority", rating=majority_rating, count=max_count, total=len(ratings))
        else:
            outcome = "still_disputed"
            log.info("round4.still_disputed", ratings=ratings)

    # Create refinement record
    refinement_record = RefinementRecord(
        from_version=1,
        to_version=2,
        original_insight=chunk_insight,
        refined_insight=best_mod,
        changes_made=[mod_rationale] if mod_rationale else ["Modification proposed in Round 4"],
        addressed_critiques=["Disputed insight modified to resolve disagreement"],
        unresolved_issues=[],
        refinement_confidence="medium",
        triggered_by_ratings=final_ratings,
        timestamp=datetime.now(),
        proposer=mod_source,
        round_proposed=4,
    )

    review_round = ReviewRound(
        round_number=4,
        mode="modification_focus",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    return review_round, best_mod, refinement_record, False


def _select_best_modification(proposals: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Select the best modification from proposals.

    Priority: Gemini Deep Think > ChatGPT Pro > Claude
    Only accept if can_be_fixed is True and expected_rating is good.

    Returns:
        Tuple of (modification, source, rationale) or (None, None, None)
    """
    priority = ["gemini", "chatgpt", "claude"]

    # First pass: prefer modifications with expected_rating of Profound
    for llm in priority:
        if llm in proposals:
            p = proposals[llm]
            if p.get("can_be_fixed") and p.get("proposed_modification"):
                if p.get("expected_rating") == "⚡":
                    log.info("round4.accept_profound", llm=llm)
                    return p["proposed_modification"], llm, p.get("modification_rationale", "")

    # Second pass: accept any valid modification
    for llm in priority:
        if llm in proposals:
            p = proposals[llm]
            if p.get("can_be_fixed") and p.get("proposed_modification"):
                log.info("round4.accept_modification", llm=llm, expected=p.get('expected_rating'))
                return p["proposed_modification"], llm, p.get("modification_rationale", "")

    return None, None, None


async def _get_modification_proposal(executor: LLMExecutor, prompt: str) -> dict:
    """Get modification proposal from an LLM using its executor."""
    log.info("round4.getting_proposal", llm=executor.name)

    try:
        response_text = await executor.send(prompt, thinking_mode="deep")
        return parse_round4_modification_response(response_text)
    except Exception as e:
        log.error("round4.proposal_error", llm=executor.name, error=str(e))
        raise


async def _get_final_vote(executor: LLMExecutor, prompt: str) -> dict:
    """Get final vote from an LLM using its executor."""
    log.info("round4.getting_vote", llm=executor.name)

    try:
        # Use standard mode for simple vote
        response_text = await executor.send(prompt, thinking_mode="standard")
        return parse_round4_final_vote_response(response_text)
    except Exception as e:
        log.error("round4.vote_error", llm=executor.name, error=str(e))
        raise
