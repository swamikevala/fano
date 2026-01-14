"""
Documenter Module Adapter for the Unified Orchestrator.

Adapts the existing Documenter components to work with the new
orchestration system while preserving all existing functionality.

Based on v4.0 design specification Phase 3.
"""

import asyncio
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

from orchestrator.adapters import (
    ModuleInterface,
    PromptContext,
    TaskResult,
    TaskType,
)
from orchestrator.models import Task

from documenter.session import SessionManager
from documenter.planning import WorkPlanner
from documenter.opportunity_processor import OpportunityProcessor
from documenter.comments import CommentHandler
from documenter.tasks import TaskBuilder

log = get_logger("documenter", "adapter")


class DocumenterAdapter(ModuleInterface):
    """
    Adapter that wraps Documenter functionality for the unified orchestrator.

    Maps orchestrator tasks to Documenter operations:
    - address_comment -> CommentHandler.address_comment()
    - incorporate_insight -> OpportunityProcessor.process_opportunity()
    - review_section -> CommentHandler.review_section()
    - draft_section -> OpportunityProcessor.write_prerequisite()
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the Documenter adapter.

        Args:
            config_path: Optional custom config path
        """
        self.config_path = config_path

        # Session and handlers (initialized in initialize())
        self.session: Optional[SessionManager] = None
        self.planner: Optional[WorkPlanner] = None
        self.processor: Optional[OpportunityProcessor] = None
        self.comment_handler: Optional[CommentHandler] = None
        self.task_builder: Optional[TaskBuilder] = None

        # Cached work items
        self._pending_comments: list[tuple[str, int]] = []
        self._pending_opportunities: list[dict] = []
        self._pending_reviews: list = []

        self._initialized = False
        self._guidance_text = ""

    @property
    def module_name(self) -> str:
        return "documenter"

    @property
    def supported_task_types(self) -> list[str]:
        return [
            TaskType.ADDRESS_COMMENT.value,
            TaskType.INCORPORATE_INSIGHT.value,
            TaskType.REVIEW_SECTION.value,
            TaskType.DRAFT_SECTION.value,
            TaskType.DRAFT_PREREQUISITE.value,
        ]

    async def initialize(self) -> bool:
        """Initialize Documenter components."""
        if self._initialized:
            return True

        try:
            log.info("documenter.adapter.initializing")

            # Initialize session
            self.session = SessionManager(self.config_path)
            self._guidance_text = await self.session.initialize()

            # Initialize handlers
            self.planner = WorkPlanner(self.session, self._guidance_text)
            self.processor = OpportunityProcessor(self.session)
            self.comment_handler = CommentHandler(self.session)

            # Task builder
            max_tokens = self.session.config.get("context", {}).get("max_tokens", 8000)
            self.task_builder = TaskBuilder(max_context_tokens=max_tokens)

            self._initialized = True
            log.info("documenter.adapter.initialized")
            return True

        except Exception as e:
            log.exception(e, "documenter.adapter.init_failed", {})
            return False

    async def shutdown(self):
        """Cleanup Documenter resources."""
        log.info("documenter.adapter.shutting_down")

        if self.session:
            try:
                # Final snapshot
                self.session.snapshot_manager.create_snapshot()
                await self.session.cleanup()
            except Exception as e:
                log.error("documenter.adapter.cleanup_error", error=str(e))

        self._initialized = False
        log.info("documenter.adapter.shutdown_complete")

    async def get_pending_work(self) -> list[dict]:
        """
        Get list of pending work from Documenter.

        Returns work items for comments, reviews, and insight incorporation.
        """
        if not self._initialized:
            return []

        work_items = []

        # Check for author comments (highest priority)
        comments = self.session.document.find_unresolved_comments()
        for comment_text, line_num in comments:
            work_items.append({
                "task_type": TaskType.ADDRESS_COMMENT.value,
                "key": f"comment:{line_num}:{hash(comment_text) % 10000}",
                "payload": {
                    "comment_text": comment_text,
                    "line_num": line_num,
                },
                "requires_deep_mode": False,
                "priority": 70,  # Highest priority
            })

        # Check for section reviews
        if self.session.review_manager.should_do_review():
            section = self.session.review_manager.select_section_for_review()
            if section:
                work_items.append({
                    "task_type": TaskType.REVIEW_SECTION.value,
                    "key": f"review:{section.id}",
                    "payload": {
                        "section_id": section.id,
                        "section_content": section.content,
                    },
                    "requires_deep_mode": False,
                    "priority": 40,
                })

        # Check for new insights to incorporate
        opportunities = await self._get_unincorporated_opportunities()
        for opp in opportunities:
            # Check prerequisites
            prereqs_met = self._check_prerequisites(opp)
            if prereqs_met:
                work_items.append({
                    "task_type": TaskType.INCORPORATE_INSIGHT.value,
                    "key": f"insight:{opp.get('insight_id', opp.get('source_file', 'unknown'))}",
                    "payload": {
                        "opportunity": opp,
                    },
                    "requires_deep_mode": True,  # Insight incorporation uses deep mode
                    "priority": 55,
                })

        log.debug("documenter.adapter.pending_work",
                 count=len(work_items),
                 comments=len(comments),
                 opportunities=len(opportunities))

        return work_items

    async def build_prompt(self, task: Task) -> PromptContext:
        """Build prompt for a Documenter task."""
        task_type = task.task_type

        # Get document context
        doc_summary = self.session.document.get_summary()
        established = list(self.session.concept_tracker.get_established())

        if task_type == TaskType.ADDRESS_COMMENT.value:
            comment_text = task.payload.get("comment_text", "")
            line_num = task.payload.get("line_num", 0)

            # Get surrounding context
            context = self._get_surrounding_context(line_num)

            context_str, task_str, format_str = self.task_builder.build_address_comment(
                document_summary=doc_summary,
                comment_text=comment_text,
                surrounding_context=context,
            )

            return PromptContext(
                prompt=f"{context_str}\n\nTASK:\n{task_str}\n\nRESPOND IN THIS FORMAT:\n{format_str}",
                requires_deep_mode=False,
                metadata={"line_num": line_num},
            )

        elif task_type == TaskType.INCORPORATE_INSIGHT.value:
            opp = task.payload.get("opportunity", {})

            # Get preceding section for flow
            preceding = self._get_preceding_section()

            context_str, task_str, format_str = self.task_builder.build_draft_content(
                document_summary=doc_summary,
                preceding_section=preceding,
                established_concepts=established,
                topic=opp.get("text", "")[:200],
                source_insight=opp.get("text", ""),
            )

            return PromptContext(
                prompt=f"{context_str}\n\nTASK:\n{task_str}\n\nRESPOND IN THIS FORMAT:\n{format_str}",
                requires_deep_mode=True,
                metadata={"insight_id": opp.get("insight_id")},
            )

        elif task_type == TaskType.REVIEW_SECTION.value:
            section_id = task.payload.get("section_id", "")
            section_content = task.payload.get("section_content", "")

            context_str, task_str, format_str = self.task_builder.build_review_section(
                document_summary=doc_summary,
                established_concepts=established,
                section_to_review=section_content,
            )

            return PromptContext(
                prompt=f"{context_str}\n\nTASK:\n{task_str}\n\nRESPOND IN THIS FORMAT:\n{format_str}",
                requires_deep_mode=False,
                metadata={"section_id": section_id},
            )

        elif task_type == TaskType.DRAFT_PREREQUISITE.value:
            content = task.payload.get("content", "")
            establishes = task.payload.get("establishes", [])

            preceding = self._get_preceding_section()

            context_str, task_str, format_str = self.task_builder.build_draft_content(
                document_summary=doc_summary,
                preceding_section=preceding,
                established_concepts=established,
                topic=f"Prerequisite establishing: {', '.join(establishes)}",
                source_insight=content,
            )

            return PromptContext(
                prompt=f"{context_str}\n\nTASK:\n{task_str}\n\nRESPOND IN THIS FORMAT:\n{format_str}",
                requires_deep_mode=True,
                metadata={"establishes": establishes},
            )

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def handle_result(self, task: Task, result: TaskResult) -> bool:
        """Handle the result of a task execution."""
        if not result.success or not result.response:
            return False

        task_type = task.task_type

        try:
            if task_type == TaskType.ADDRESS_COMMENT.value:
                return await self._handle_comment_result(task, result)

            elif task_type == TaskType.INCORPORATE_INSIGHT.value:
                return await self._handle_insight_result(task, result)

            elif task_type == TaskType.REVIEW_SECTION.value:
                return await self._handle_review_result(task, result)

            elif task_type == TaskType.DRAFT_PREREQUISITE.value:
                return await self._handle_prerequisite_result(task, result)

            else:
                log.warning("documenter.adapter.unknown_task_type", task_type=task_type)
                return False

        except Exception as e:
            log.exception(e, "documenter.adapter.handle_result_error", {
                "task_id": task.id,
                "task_type": task_type,
            })
            return False

    async def on_task_failed(self, task: Task, error: str):
        """Handle task failure."""
        log.error("documenter.adapter.task_failed",
                 task_id=task.id,
                 task_type=task.task_type,
                 error=error)

        # Mark opportunity as disputed if it's an insight task
        if task.task_type == TaskType.INCORPORATE_INSIGHT.value:
            opp = task.payload.get("opportunity", {})
            if opp and self.session.opportunity_finder:
                self.session.opportunity_finder.mark_disputed(opp)

    async def get_system_state(self) -> dict:
        """Get Documenter's current system state for priority computation."""
        comments = self.session.document.find_unresolved_comments() if self.session else []

        # Count blessed insights
        blessed_count = 0
        if self.session and self.session.opportunity_finder:
            opportunities = self.session.opportunity_finder._gather_all_opportunities()
            blessed_count = len(opportunities)

        return {
            "comments_pending": len(comments),
            "blessed_insights_pending": blessed_count,
            "consensus_calls_remaining": (
                self.session.config.get("termination", {}).get("max_consensus_calls_per_session", 100)
                - self.session.consensus_calls
            ) if self.session else 0,
        }

    # ==================== Private Helper Methods ====================

    async def _get_unincorporated_opportunities(self) -> list[dict]:
        """Get all unincorporated opportunities."""
        if not self.session or not self.session.opportunity_finder:
            return []

        opportunities = self.session.opportunity_finder._gather_all_opportunities()
        return [
            {
                "type": opp.type,
                "source_file": opp.source_file,
                "insight_id": opp.insight_id,
                "text": opp.text,
                "requires": opp.requires,
                "priority": opp.priority,
            }
            for opp in opportunities
        ]

    def _check_prerequisites(self, opportunity: dict) -> bool:
        """Check if all prerequisites for an opportunity are met."""
        if not self.session:
            return False

        requires = opportunity.get("requires", [])
        if not requires:
            return True

        established = self.session.concept_tracker.get_established()
        return all(req in established for req in requires)

    def _get_surrounding_context(self, line_num: int, context_lines: int = 10) -> str:
        """Get text surrounding a line number."""
        if not self.session:
            return ""

        try:
            content = self.session.document.content
            lines = content.split('\n')
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines)
            return '\n'.join(lines[start:end])
        except Exception:
            return ""

    def _get_preceding_section(self) -> str:
        """Get the last section of the document for context."""
        if not self.session:
            return ""

        sections = self.session.document.sections
        if sections:
            return sections[-1].content[:500]
        return ""

    async def _handle_comment_result(self, task: Task, result: TaskResult) -> bool:
        """Handle comment resolution result."""
        response = result.response

        # Parse response for REVISED_SECTION
        if "REVISED_SECTION:" in response:
            revised = response.split("REVISED_SECTION:")[1].strip()
            if revised and revised.upper() != "UNCHANGED":
                # Apply the revision
                line_num = task.payload.get("line_num", 0)
                # Update document at line_num with revised content
                log.info("documenter.adapter.comment_resolved",
                        line_num=line_num,
                        revised_len=len(revised))
                self.session.document.save("Addressed author comment")
                return True

        log.info("documenter.adapter.comment_unchanged",
                line_num=task.payload.get("line_num", 0))
        return True

    async def _handle_insight_result(self, task: Task, result: TaskResult) -> bool:
        """Handle insight incorporation result."""
        response = result.response

        # Parse draft response
        if "DRAFT:" in response:
            draft = response.split("DRAFT:")[1]
            if "ESTABLISHES:" in draft:
                draft = draft.split("ESTABLISHES:")[0]
            draft = draft.strip()

            # Extract metadata
            establishes = []
            requires = []

            if "ESTABLISHES:" in response:
                est_section = response.split("ESTABLISHES:")[1]
                if "REQUIRES:" in est_section:
                    est_section = est_section.split("REQUIRES:")[0]
                establishes = [c.strip() for c in est_section.strip().split(",") if c.strip()]

            if "REQUIRES:" in response:
                req_section = response.split("REQUIRES:")[1]
                if "DIAGRAM_NEEDED:" in req_section:
                    req_section = req_section.split("DIAGRAM_NEEDED:")[0]
                requires = [c.strip() for c in req_section.strip().split(",") if c.strip()]

            # Add section to document
            section_id = self.session.document.generate_next_section_id()
            self.session.document.append_section(
                section_id=section_id,
                content=draft,
                establishes=establishes,
                requires=requires,
            )

            # Update concept tracker
            for concept in establishes:
                self.session.concept_tracker.add_established(concept)

            # Mark opportunity as incorporated
            opp = task.payload.get("opportunity", {})
            if opp and self.session.opportunity_finder:
                self.session.opportunity_finder.mark_incorporated(opp)

            self.session.document.save("Incorporated blessed insight")

            log.info("documenter.adapter.insight_incorporated",
                    section_id=section_id,
                    establishes=establishes)
            return True

        return False

    async def _handle_review_result(self, task: Task, result: TaskResult) -> bool:
        """Handle section review result."""
        response = result.response
        section_id = task.payload.get("section_id", "")

        if "DECISION:" in response:
            decision_line = response.split("DECISION:")[1].split("\n")[0].strip()

            if "CONFIRMED" in decision_line.upper():
                # Mark section as reviewed
                self.session.document.mark_section_reviewed(section_id)
                log.info("documenter.adapter.section_confirmed", section_id=section_id)
                return True

            elif "REVISE" in decision_line.upper():
                # Extract revision notes for future work
                if "REVISION_NOTES:" in response:
                    notes = response.split("REVISION_NOTES:")[1].strip()
                    log.info("documenter.adapter.section_needs_revision",
                            section_id=section_id,
                            notes=notes[:100])
                return True

        return False

    async def _handle_prerequisite_result(self, task: Task, result: TaskResult) -> bool:
        """Handle prerequisite drafting result."""
        response = result.response

        if "DRAFT:" in response:
            draft = response.split("DRAFT:")[1]
            if "ESTABLISHES:" in draft:
                draft = draft.split("ESTABLISHES:")[0]
            draft = draft.strip()

            establishes = task.payload.get("establishes", [])

            # Add section to document
            section_id = self.session.document.generate_next_section_id()
            self.session.document.append_section(
                section_id=section_id,
                content=draft,
                establishes=establishes,
                requires=[],
            )

            # Update concept tracker
            for concept in establishes:
                self.session.concept_tracker.add_established(concept)

            self.session.document.save("Added prerequisite content")

            log.info("documenter.adapter.prerequisite_added",
                    section_id=section_id,
                    establishes=establishes)
            return True

        return False
