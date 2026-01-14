"""
Comment and review handling for the documenter.

Handles:
- Addressing author comments in the document
- Reviewing existing sections for quality/updates
"""

import re
from typing import TYPE_CHECKING

from shared.logging import get_logger

if TYPE_CHECKING:
    from .session import SessionManager
    from .document import Section

log = get_logger("documenter", "comments")


class CommentHandler:
    """
    Handles author comments and section reviews.

    Comments are markers in the document where the author wants
    specific attention or changes. Reviews check existing sections
    against new knowledge.
    """

    def __init__(self, session: "SessionManager"):
        """
        Initialize comment handler.

        Args:
            session: The session manager with all components
        """
        self.session = session

    async def address_comment(self, comment_text: str, line_num: int):
        """
        Address an author comment in the document.

        Args:
            comment_text: The text of the comment
            line_num: Line number where the comment appears
        """
        s = self.session

        log.info(
            "comment.addressing",
            comment=comment_text[:100],
            line=line_num,
        )

        # Find the section containing the comment
        section_text = self._get_section_around_line(line_num)

        # Build consensus task
        context, task, response_format = s.task_builder.build_address_comment(
            section_text,
            comment_text,
        )
        context = s.task_builder.truncate_context(context)

        # Run consensus
        result = await s.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
        )
        s.increment_consensus_calls()

        if result.converged and "REVISED_SECTION:" in result.outcome:
            # Extract revised section
            match = re.search(r'REVISED_SECTION:\s*(.+)', result.outcome, re.DOTALL | re.IGNORECASE)
            if match and match.group(1).strip().upper() != "UNCHANGED":
                revised = match.group(1).strip()
                # Update document (simplified - just remove the comment marker)
                old_comment = f"<!-- COMMENT: {comment_text} -->"
                s.document.content = s.document.content.replace(old_comment, "", 1)
                s.document.save(f"Resolved comment: {comment_text[:50]}")
                log.info("comment.resolved", comment=comment_text[:50])
        else:
            # Mark as attempted with timestamp so it can be retried later
            # after new concepts are established
            from datetime import datetime
            timestamp = datetime.now().isoformat()[:10]  # Just date part
            old_comment = f"<!-- COMMENT: {comment_text} -->"
            new_comment = f"<!-- COMMENT: {comment_text} (attempted: {timestamp}) -->"
            s.document.content = s.document.content.replace(old_comment, new_comment, 1)
            s.document.save(f"Attempted comment: {comment_text[:50]}")
            log.warning("comment.unresolved", comment=comment_text[:50])

    def should_retry_comment(self, comment_marker: str) -> bool:
        """
        Check if an attempted comment should be retried.

        Comments are retried if:
        1. They were attempted more than 1 session ago (time-based retry)
        2. OR new concepts have been established since the attempt

        Args:
            comment_marker: The full comment marker including (attempted: date)

        Returns:
            True if the comment should be retried
        """
        import re
        from datetime import datetime, timedelta

        # Extract the attempt date
        match = re.search(r'\(attempted:\s*(\d{4}-\d{2}-\d{2})\)', comment_marker)
        if not match:
            return True  # No date, retry it

        attempt_date_str = match.group(1)
        try:
            attempt_date = datetime.fromisoformat(attempt_date_str)
            # Retry if more than 7 days old
            if datetime.now() - attempt_date > timedelta(days=7):
                return True
        except ValueError:
            return True  # Invalid date, retry it

        return False

    def get_retriable_comments(self) -> list[tuple[str, int]]:
        """
        Get list of previously attempted comments that should be retried.

        Returns:
            List of (comment_text, line_number) tuples for comments to retry.
        """
        import re
        s = self.session
        retriable = []

        lines = s.document.content.split('\n')
        for i, line in enumerate(lines):
            if '<!-- COMMENT:' in line and '(attempted:' in line:
                if self.should_retry_comment(line):
                    # Extract the comment text
                    match = re.search(r'<!-- COMMENT:\s*(.+?)\s*\(attempted:', line)
                    if match:
                        comment_text = match.group(1).strip()
                        retriable.append((comment_text, i + 1))

        return retriable

    def _get_section_around_line(self, line_num: int, context_lines: int = 20) -> str:
        """
        Get document content around a line number.

        Args:
            line_num: The line number to center on
            context_lines: Number of lines before/after to include

        Returns:
            Content string around the specified line
        """
        s = self.session
        lines = s.document.content.split('\n')
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines)
        return '\n'.join(lines[start:end])

    async def review_section(self, section: "Section"):
        """
        Review an existing section for quality and updates.

        Args:
            section: The section to review
        """
        s = self.session

        log.info(
            "review.starting",
            section_id=section.id,
        )

        # Get concepts at creation and since
        concepts_at_creation = s.review_manager.get_concepts_at_creation(section)
        concepts_since = s.review_manager.get_concepts_since_creation(section)

        # Build consensus task
        context, task, response_format = s.task_builder.build_review_section(
            section.content,
            section.created.strftime("%Y-%m-%d"),
            concepts_at_creation,
            concepts_since,
        )
        context = s.task_builder.truncate_context(context)

        # Run consensus
        result = await s.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
        )
        s.increment_consensus_calls()

        if result.converged:
            # Check decision
            if "DECISION: CONFIRMED" in result.outcome.upper():
                s.document.mark_section_reviewed(section.id)
                s.document.save(f"Reviewed section {section.id}: confirmed")
                log.info("review.confirmed", section_id=section.id)
            elif "DECISION: REVISE" in result.outcome.upper():
                # Extract revision notes and apply
                # For now, just mark as reviewed
                s.document.mark_section_reviewed(section.id)
                s.document.save(f"Reviewed section {section.id}: revision suggested")
                log.info("review.revision_suggested", section_id=section.id)
        else:
            log.warning("review.no_consensus", section_id=section.id)
