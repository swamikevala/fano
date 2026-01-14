"""
The Consensus Board - 3-LLM Review Panel.

The Consensus Board reviews every content change with three perspectives:
- The Mathematician: Checks rigor and LaTeX validity
- The Pedagogue: Checks clarity and narrative flow
- The Editor: Checks formatting, style, and consistency

Decisions require 2/3 agreement. Max 3 rounds before escalation.

Backend: Multi-backend (Claude, Gemini, ChatGPT)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from shared.logging import get_logger
from llm.src.client import LLMClient

from .mason import DraftedContent

log = get_logger("documenter", "consensus_board")


class ReviewDecision(Enum):
    """Possible review decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"


@dataclass
class ReviewerVote:
    """A single reviewer's vote."""
    role: str
    decision: ReviewDecision
    feedback: str
    issues: list[str] = field(default_factory=list)


@dataclass
class BoardReview:
    """Result of Consensus Board review."""
    approved: bool
    votes: list[ReviewerVote]
    round_number: int
    consolidated_feedback: str = ""
    needs_revision: bool = False
    escalate_to_human: bool = False


# Role definitions with their prompts
REVIEWER_ROLES = {
    "mathematician": {
        "name": "The Mathematician",
        "backend": "claude",
        "focus": """You are a rigorous mathematician reviewing content for a mathematical textbook.

Your focus:
1. MATHEMATICAL CORRECTNESS: Are all statements mathematically true?
2. PROOF VALIDITY: Are arguments logically sound and complete?
3. LATEX CORRECTNESS: Is all mathematical notation valid and rendered correctly?
4. PRECISION: Are terms defined before use? Are claims properly qualified?

Do NOT concern yourself with:
- Writing style or accessibility (that's the Pedagogue's job)
- Formatting conventions (that's the Editor's job)

Be strict but fair. Mathematical errors are unacceptable.""",
    },
    "pedagogue": {
        "name": "The Pedagogue",
        "backend": "gemini",
        "focus": """You are an experienced mathematics educator reviewing content for clarity.

Your focus:
1. COMPREHENSIBILITY: Can a motivated learner follow the exposition?
2. NARRATIVE FLOW: Does the content build naturally from previous material?
3. MOTIVATION: Are concepts motivated before being defined?
4. EXAMPLES: Are there enough examples to illustrate abstract concepts?

Do NOT concern yourself with:
- Mathematical proof details (that's the Mathematician's job)
- Formatting rules (that's the Editor's job)

Consider: Would a graduate student find this accessible?""",
    },
    "editor": {
        "name": "The Editor",
        "backend": "chatgpt",
        "focus": """You are a technical editor reviewing content for a mathematical publication.

Your focus:
1. FORMATTING: Is markdown/LaTeX formatting consistent and correct?
2. STYLE: Is the writing style appropriate for a textbook?
3. CONSISTENCY: Does notation match the established style guide?
4. STRUCTURE: Are sections and subsections properly organized?

Do NOT concern yourself with:
- Mathematical correctness (that's the Mathematician's job)
- Pedagogical clarity (that's the Pedagogue's job)

Ensure professional publication quality.""",
    },
}


class ConsensusBoard:
    """
    The Consensus Board - reviews all content changes.

    Uses three specialized reviewers to ensure quality:
    - Mathematical rigor
    - Pedagogical clarity
    - Editorial polish

    Requires 2/3 agreement to approve content.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        min_agreement: int = 2,
        max_rounds: int = 3,
    ):
        """
        Initialize the Consensus Board.

        Args:
            llm_client: LLM client for backend communication
            min_agreement: Minimum votes needed for decision (default 2/3)
            max_rounds: Maximum revision rounds before escalation
        """
        self.llm_client = llm_client
        self.min_agreement = min_agreement
        self.max_rounds = max_rounds

    async def review(
        self,
        draft: DraftedContent,
        document_context: str = "",
    ) -> BoardReview:
        """
        Review drafted content with the Consensus Board.

        This is Phase 3 of the incorporate_insight workflow.

        Args:
            draft: The drafted content from Mason
            document_context: Optional context about the document

        Returns:
            BoardReview with decision and feedback
        """
        round_number = 1
        current_draft = draft

        while round_number <= self.max_rounds:
            log.info(
                "consensus_board.review_started",
                round=round_number,
                title=current_draft.title,
            )

            # Collect votes from all reviewers
            votes = await self._collect_votes(current_draft, document_context)

            # Tally results
            approve_count = sum(1 for v in votes if v.decision == ReviewDecision.APPROVE)
            reject_count = sum(1 for v in votes if v.decision == ReviewDecision.REJECT)
            revise_count = sum(1 for v in votes if v.decision == ReviewDecision.REVISE)

            log.info(
                "consensus_board.votes_collected",
                approve=approve_count,
                reject=reject_count,
                revise=revise_count,
            )

            # Decision logic
            if approve_count >= self.min_agreement:
                # Approved!
                return BoardReview(
                    approved=True,
                    votes=votes,
                    round_number=round_number,
                    consolidated_feedback=self._consolidate_feedback(votes),
                )

            if reject_count >= self.min_agreement:
                # Rejected - no revision possible
                return BoardReview(
                    approved=False,
                    votes=votes,
                    round_number=round_number,
                    consolidated_feedback=self._consolidate_feedback(votes),
                    needs_revision=False,
                )

            # Mixed result - needs revision if not final round
            if round_number < self.max_rounds:
                return BoardReview(
                    approved=False,
                    votes=votes,
                    round_number=round_number,
                    consolidated_feedback=self._consolidate_feedback(votes),
                    needs_revision=True,
                )

            round_number += 1

        # Exceeded max rounds - escalate to human
        log.warning(
            "consensus_board.max_rounds_exceeded",
            title=draft.title,
        )
        return BoardReview(
            approved=False,
            votes=votes,
            round_number=self.max_rounds,
            consolidated_feedback=self._consolidate_feedback(votes),
            escalate_to_human=True,
        )

    async def _collect_votes(
        self,
        draft: DraftedContent,
        document_context: str,
    ) -> list[ReviewerVote]:
        """Collect votes from all reviewers."""
        votes = []

        for role_id, role_config in REVIEWER_ROLES.items():
            vote = await self._get_reviewer_vote(
                role_id=role_id,
                role_config=role_config,
                draft=draft,
                document_context=document_context,
            )
            votes.append(vote)

        return votes

    async def _get_reviewer_vote(
        self,
        role_id: str,
        role_config: dict,
        draft: DraftedContent,
        document_context: str,
    ) -> ReviewerVote:
        """Get a single reviewer's vote."""
        prompt = f"""{role_config['focus']}

## Document Context
{document_context if document_context else 'New content for the document.'}

## Content to Review

Title: {draft.title}

Summary: {draft.summary}

Content:
{draft.content}

Concepts Defined: {', '.join(draft.concepts_defined)}
Concepts Required: {', '.join(draft.concepts_required)}

## Review Task

Review this content from your perspective ({role_config['name']}).

Provide your decision in this EXACT format:

DECISION: [APPROVE/REJECT/REVISE]

ISSUES:
- [List each specific issue, one per line]
- [Or "None" if approving]

FEEDBACK:
[Your overall assessment and any suggestions]
"""

        try:
            response = await self.llm_client.send(
                backend=role_config['backend'],
                prompt=prompt,
                timeout_seconds=60,
            )

            if response.success:
                return self._parse_vote(role_id, response.text)
            else:
                # Backend failed - abstain with warning
                log.warning(
                    "consensus_board.reviewer_failed",
                    role=role_id,
                    error=response.error,
                )
                return ReviewerVote(
                    role=role_id,
                    decision=ReviewDecision.REVISE,
                    feedback=f"Review failed: {response.error}",
                    issues=["Unable to complete review"],
                )

        except Exception as e:
            log.error(
                "consensus_board.reviewer_error",
                role=role_id,
                error=str(e),
            )
            return ReviewerVote(
                role=role_id,
                decision=ReviewDecision.REVISE,
                feedback=f"Review error: {str(e)}",
                issues=["Review error occurred"],
            )

    def _parse_vote(self, role_id: str, response_text: str) -> ReviewerVote:
        """Parse reviewer response into a vote."""
        text_upper = response_text.upper()

        # Parse decision
        decision = ReviewDecision.REVISE  # Default to revise
        if "DECISION: APPROVE" in text_upper or "DECISION:APPROVE" in text_upper:
            decision = ReviewDecision.APPROVE
        elif "DECISION: REJECT" in text_upper or "DECISION:REJECT" in text_upper:
            decision = ReviewDecision.REJECT

        # Parse issues
        issues = []
        issues_match = re.search(
            r'ISSUES:\s*\n(.*?)(?=\nFEEDBACK:|$)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if issues_match:
            issues_text = issues_match.group(1)
            for line in issues_text.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    issue = line[1:].strip()
                    if issue.lower() not in ('none', 'n/a', ''):
                        issues.append(issue)

        # Parse feedback
        feedback = ""
        feedback_match = re.search(
            r'FEEDBACK:\s*\n?(.*?)$',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()

        return ReviewerVote(
            role=role_id,
            decision=decision,
            feedback=feedback,
            issues=issues,
        )

    def _consolidate_feedback(self, votes: list[ReviewerVote]) -> str:
        """Consolidate feedback from all reviewers."""
        parts = []

        for vote in votes:
            role_name = REVIEWER_ROLES[vote.role]["name"]
            parts.append(f"## {role_name} ({vote.decision.value.upper()})")

            if vote.issues:
                parts.append("Issues:")
                for issue in vote.issues:
                    parts.append(f"  - {issue}")

            if vote.feedback:
                parts.append(f"Feedback: {vote.feedback}")

            parts.append("")

        return "\n".join(parts)

    async def quick_review(
        self,
        content: str,
        review_type: str = "general",
    ) -> tuple[bool, str]:
        """
        Quick review for simpler content changes.

        Uses single-LLM review for non-critical changes like:
        - Comment responses
        - Minor corrections
        - Formatting fixes

        Args:
            content: Content to review
            review_type: Type of review (general, correction, formatting)

        Returns:
            (approved, feedback)
        """
        prompt = f"""Quick review this {review_type} change:

{content}

Is this change acceptable? Reply:
APPROVED: [yes/no]
REASON: [brief explanation]
"""

        response = await self.llm_client.send(
            backend="claude",
            prompt=prompt,
            timeout_seconds=30,
        )

        if not response.success:
            return False, f"Review failed: {response.error}"

        approved = "APPROVED: YES" in response.text.upper()
        reason_match = re.search(r'REASON:\s*(.+)', response.text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ""

        return approved, reason
