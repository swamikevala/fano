"""
Opportunity processing for the documenter.

Handles the pipeline for evaluating and incorporating insights:
- Duplicate detection
- Mathematical evaluation
- Content drafting
- Exposition evaluation
- Document addition
"""

import re
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from shared.logging import get_logger
from shared.deduplication import ContentItem, ContentType

from .document import Section
from .concepts import ConceptTracker
from .formatting import fix_math_if_needed

if TYPE_CHECKING:
    from .session import SessionManager
    from .opportunities import Opportunity

log = get_logger("documenter", "opportunity_processor")


class OpportunityProcessor:
    """
    Processes opportunities through the evaluation pipeline.

    Pipeline stages:
    1. Duplicate detection
    2. Math evaluation (consensus)
    3. Content drafting (consensus)
    4. Exposition evaluation (consensus)
    5. Document addition
    """

    def __init__(self, session: "SessionManager"):
        """
        Initialize opportunity processor.

        Args:
            session: The session manager with all components
        """
        self.session = session

    async def process_opportunity(self, opportunity: "Opportunity"):
        """
        Process an opportunity through the full evaluation pipeline.

        Args:
            opportunity: The opportunity to process
        """
        s = self.session  # Shorthand

        log.info(
            "opportunity.processing",
            type=opportunity.type,
            insight_id=opportunity.insight_id,
        )

        # Step 0: Check for duplicates with existing document content
        if s.dedup_checker:
            dedup_result = await s.dedup_checker.check_duplicate(
                opportunity.text,
                item_id=opportunity.insight_id or "unknown",
                content_type=ContentType.INSIGHT,
            )
            if dedup_result.is_duplicate:
                log.info(
                    "opportunity.duplicate_detected",
                    insight_id=opportunity.insight_id,
                    duplicate_of=dedup_result.duplicate_of,
                    method=dedup_result.check_method,
                    reason=dedup_result.reason,
                )
                # Mark as incorporated (skip it, don't dispute)
                s.opportunity_finder.mark_incorporated(opportunity)
                return

        # Step 1: Evaluate mathematics
        math_approved = await self._evaluate_math(opportunity)
        if not math_approved:
            s.opportunity_finder.mark_disputed(opportunity)
            return

        # Step 2: Draft content
        draft = await self._draft_content(opportunity)
        if not draft:
            s.opportunity_finder.mark_disputed(opportunity)
            return

        # Step 3: Evaluate exposition
        exposition_approved = await self._evaluate_exposition(draft)
        if not exposition_approved:
            s.opportunity_finder.mark_disputed(opportunity)
            return

        # Step 4: Add to document
        await self._add_to_document(opportunity, draft)

    async def _evaluate_math(self, opportunity: "Opportunity") -> bool:
        """Evaluate if the mathematics should be included."""
        s = self.session

        context, task, response_format = s.task_builder.build_math_evaluation(
            s.document.get_summary(),
            opportunity.text,
        )
        context = s.task_builder.truncate_context(context)

        result = await s.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
        )
        s.increment_consensus_calls()

        # Check for errors first
        if not result.success:
            log.error(
                "math.evaluation_error",
                insight_id=opportunity.insight_id,
                outcome=result.outcome,
                rounds=result.rounds,
            )
            return False

        if result.converged and "DECISION: INCLUDE" in result.outcome.upper():
            log.info("math.approved", insight_id=opportunity.insight_id)
            return True
        else:
            log.info(
                "math.rejected",
                insight_id=opportunity.insight_id,
                converged=result.converged,
                confidence=result.confidence,
            )
            return False

    async def _draft_content(self, opportunity: "Opportunity") -> Optional[str]:
        """Draft content for the opportunity."""
        s = self.session

        # Get preceding section
        preceding = ""
        if s.document.sections:
            preceding = s.document.sections[-1].content

        context, task, response_format = s.task_builder.build_draft_content(
            s.document.get_summary(),
            preceding,
            list(s.concept_tracker.get_established_concepts()),
            opportunity.text[:100],  # Topic
            opportunity.text,  # Full insight
        )

        # Add user annotations context
        annotations_context = self._get_annotations_context()
        if annotations_context:
            context += annotations_context

        context = s.task_builder.truncate_context(context)

        result = await s.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
            select_best=True,  # Let LLMs vote on best draft
        )
        s.increment_consensus_calls()

        # Check for errors first
        if not result.success:
            log.error(
                "draft.error",
                insight_id=opportunity.insight_id,
                outcome=result.outcome,
            )
            return None

        log.info(
            "draft.selected",
            insight_id=opportunity.insight_id,
            selection_stats=result.selection_stats,
        )

        if "DRAFT:" in result.outcome.upper():
            # Extract draft
            match = re.search(r'DRAFT:\s*(.+?)(?=ESTABLISHES:|$)', result.outcome, re.DOTALL | re.IGNORECASE)
            if match:
                log.info("draft.created", insight_id=opportunity.insight_id)
                return result.outcome  # Return full outcome for parsing

        log.warning("draft.failed", insight_id=opportunity.insight_id)
        return None

    async def _evaluate_exposition(self, draft: str) -> bool:
        """Evaluate if the exposition is acceptable."""
        s = self.session

        # Extract just the draft text
        match = re.search(r'DRAFT:\s*(.+?)(?=ESTABLISHES:|$)', draft, re.DOTALL | re.IGNORECASE)
        draft_text = match.group(1).strip() if match else draft

        # Get preceding section
        preceding = ""
        if s.document.sections:
            preceding = s.document.sections[-1].content

        context, task, response_format = s.task_builder.build_exposition_evaluation(
            s.document.get_summary(),
            preceding,
            list(s.concept_tracker.get_established_concepts()),
            draft_text,
        )
        context = s.task_builder.truncate_context(context)

        result = await s.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
        )
        s.increment_consensus_calls()

        # Check for errors first
        if not result.success:
            log.error("exposition.error", outcome=result.outcome)
            return False

        if result.converged and "DECISION: APPROVE" in result.outcome.upper():
            log.info("exposition.approved")
            return True
        else:
            log.info(
                "exposition.not_approved",
                converged=result.converged,
                confidence=result.confidence,
            )
            return False

    async def _add_to_document(self, opportunity: "Opportunity", draft: str):
        """Add approved content to the document."""
        s = self.session

        # Extract draft text
        match = re.search(r'DRAFT:\s*(.+?)(?=ESTABLISHES:|$)', draft, re.DOTALL | re.IGNORECASE)
        content = match.group(1).strip() if match else ""

        # Fix math formatting if needed
        content = await self._fix_math_formatting(content)

        # Extract concepts
        establishes = []
        requires = []

        est_match = re.search(r'ESTABLISHES:\s*(.+?)(?=REQUIRES:|DIAGRAM|$)', draft, re.IGNORECASE)
        if est_match:
            establishes = [c.strip() for c in est_match.group(1).split(',') if c.strip()]

        req_match = re.search(r'REQUIRES:\s*(.+?)(?=DIAGRAM|$)', draft, re.IGNORECASE)
        if req_match:
            requires = [c.strip() for c in req_match.group(1).split(',') if c.strip()]

        # Create section
        section = Section(
            id=s.document.generate_next_section_id(),
            content=content,
            created=datetime.now(),
            status="provisional",
            establishes=establishes,
            requires=requires,
        )

        # Add to document
        s.document.append_section(section, content)
        concepts_str = ", ".join(establishes) if establishes else "new content"
        s.document.save(f"Incorporated insight {opportunity.insight_id}: {concepts_str}")

        # Register concepts
        s.concept_tracker.register_section(section)

        # Add to dedup checker for future duplicate detection
        if s.dedup_checker:
            s.dedup_checker.add_content(ContentItem(
                id=section.id,
                text=content,
                content_type=ContentType.SECTION,
            ))

        # Mark opportunity as incorporated
        s.opportunity_finder.mark_incorporated(opportunity)

        log.info(
            "content.added",
            section_id=section.id,
            establishes=establishes,
            insight_id=opportunity.insight_id,
        )

    async def write_prerequisite(self, content: str, establishes: Optional[list[str]] = None):
        """
        Write prerequisite/foundational content to the document.

        This content was already generated through consensus in WorkPlanner,
        so we trust it and write directly. No additional validation needed.

        Args:
            content: The content to write
            establishes: List of concepts this content establishes (for tracking)
        """
        s = self.session
        establishes = establishes or []

        # Check for duplicates before writing
        if s.dedup_checker:
            dedup_result = await s.dedup_checker.check_duplicate(
                content,
                item_id=f"prereq-pending",
                content_type=ContentType.PREREQUISITE,
            )
            if dedup_result.is_duplicate:
                log.info(
                    "prerequisite.duplicate_detected",
                    duplicate_of=dedup_result.duplicate_of,
                    method=dedup_result.check_method,
                    reason=dedup_result.reason,
                )
                return  # Skip this prerequisite, it's already covered

        # Fix math formatting if needed
        content = await self._fix_math_formatting(content)

        log.info(
            "prerequisite.writing",
            content_length=len(content),
            establishes=establishes,
        )

        # Create a section for the prerequisite content
        section = Section(
            id=f"prereq-{uuid.uuid4().hex[:8]}",
            content=content,
            created=datetime.now(),
            status="provisional",
            establishes=establishes,
            requires=[],
        )

        # Add to document
        s.document.append_section(section, content)
        concepts_str = ", ".join(establishes) if establishes else "foundational content"
        s.document.save(f"Added prerequisite: {concepts_str}")

        # Refresh concept tracker by recreating it
        s.concept_tracker = ConceptTracker(s.document)

        # Add to dedup checker for future duplicate detection
        if s.dedup_checker:
            s.dedup_checker.add_content(ContentItem(
                id=section.id,
                text=content,
                content_type=ContentType.PREREQUISITE,
            ))

        log.info(
            "prerequisite.written",
            content_length=len(content),
            new_concepts=establishes,
        )

    def _get_annotations_context(self) -> str:
        """
        Get user annotations formatted for LLM context.

        Returns formatted string with comments and protected regions,
        or empty string if no annotations.
        """
        s = self.session

        if not s.annotation_manager:
            return ""

        # Reload annotations to get latest
        s.annotation_manager.load()

        # Use the built-in formatter
        context = s.annotation_manager.format_for_llm()

        if context:
            # Count annotations by type
            comments = sum(1 for a in s.annotation_manager.annotations.values() if a.type == "comment")
            protected = sum(1 for a in s.annotation_manager.annotations.values() if a.type == "protected")
            log.info(
                "annotations.included",
                comments=comments,
                protected=protected,
            )

        return context

    async def _fix_math_formatting(self, content: str) -> str:
        """
        Fix math delimiter issues in content if needed.

        Uses regex detection first, only calls LLM if issues found.
        """
        s = self.session

        if not s.llm_client:
            return content

        return await fix_math_if_needed(content, s.llm_client)
