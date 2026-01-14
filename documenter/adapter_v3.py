"""
Documenter Module Adapter v3.0 for the "Living Textbook" Architecture.

Adapts the new multi-phase Editorial Board agents to work with
the orchestration system:

Agents:
- Architect: Placement decisions (uses Deep Mode)
- Mason: Content drafting
- Consensus Board: 3-LLM review
- Illustrator: Diagram generation

Multi-Phase Task Flow:
1. incorporate_insight_plan → PlacementPlan (Architect, P:60)
2. incorporate_insight_draft → DraftedContent (Mason, P:58)
3. incorporate_insight_review → APPROVE/REJECT (Board, P:56)
"""

import asyncio
from datetime import datetime
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

from llm.src.client import LLMClient

from .repository import DocumentRepository, SectionContent, PlacementPlan
from .architect import Architect, InsightAnalysis
from .mason import Mason, DraftedContent
from .consensus_board import ConsensusBoard, BoardReview
from .illustrator import Illustrator

log = get_logger("documenter", "adapter_v3")


# Project root
FANO_ROOT = Path(__file__).resolve().parent.parent


class DocumenterAdapterV3(ModuleInterface):
    """
    Adapter v3.0 - Editorial Board Architecture.

    Maps orchestrator tasks to multi-phase agent workflows:
    - Plan → Draft → Review for insights
    - Direct execution for comments and diagrams
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the Documenter v3 adapter.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self._initialized = False

        # Core components
        self.repository: Optional[DocumentRepository] = None
        self.llm_client: Optional[LLMClient] = None

        # Agents
        self.architect: Optional[Architect] = None
        self.mason: Optional[Mason] = None
        self.consensus_board: Optional[ConsensusBoard] = None
        self.illustrator: Optional[Illustrator] = None

        # In-flight state for multi-phase tasks
        self._pending_plans: dict[str, InsightAnalysis] = {}
        self._pending_drafts: dict[str, DraftedContent] = {}

    @property
    def module_name(self) -> str:
        return "documenter"

    @property
    def supported_task_types(self) -> list[str]:
        return [
            # v3 multi-phase tasks
            TaskType.INCORPORATE_INSIGHT_PLAN.value,
            TaskType.INCORPORATE_INSIGHT_DRAFT.value,
            TaskType.INCORPORATE_INSIGHT_REVIEW.value,
            TaskType.GENERATE_DIAGRAM.value,
            TaskType.GLOBAL_REFACTOR.value,
            # Legacy tasks still supported
            TaskType.ADDRESS_COMMENT.value,
        ]

    async def initialize(self) -> bool:
        """Initialize v3 Documenter components."""
        if self._initialized:
            return True

        try:
            log.info("documenter.adapter_v3.initializing")

            # Get configuration
            doc_config = self.config.get("documenter", {})
            base_path = Path(doc_config.get("data_architecture", {}).get(
                "base_path", "data/document"
            ))

            if not base_path.is_absolute():
                base_path = FANO_ROOT / base_path

            # Initialize repository
            self.repository = DocumentRepository(base_path)
            self.repository.initialize()

            # Initialize LLM client
            pool_config = self.config.get("llm", {}).get("pool", {})
            pool_host = pool_config.get("host", "127.0.0.1")
            pool_port = pool_config.get("port", 9000)
            pool_url = f"http://{pool_host}:{pool_port}"
            self.llm_client = LLMClient(pool_url=pool_url)

            # Initialize agents
            self.architect = Architect(
                repository=self.repository,
                llm_client=self.llm_client,
                thread_id=self.repository.architect_thread_id,
            )

            self.mason = Mason(
                repository=self.repository,
                llm_client=self.llm_client,
            )

            board_config = doc_config.get("agents", {}).get("consensus_board", {})
            self.consensus_board = ConsensusBoard(
                llm_client=self.llm_client,
                min_agreement=board_config.get("min_agreement", 2),
                max_rounds=board_config.get("max_rounds", 3),
            )

            illustrator_config = doc_config.get("agents", {}).get("illustrator", {})
            self.illustrator = Illustrator(
                repository=self.repository,
                llm_client=self.llm_client,
                sandbox_timeout=illustrator_config.get("sandbox_timeout_seconds", 30),
            )

            self._initialized = True
            log.info(
                "documenter.adapter_v3.initialized",
                base_path=str(base_path),
            )
            return True

        except Exception as e:
            log.exception(e, "documenter.adapter_v3.init_failed", {})
            return False

    async def shutdown(self):
        """Cleanup resources."""
        log.info("documenter.adapter_v3.shutting_down")

        if self.llm_client:
            await self.llm_client.close()

        # Compile book on shutdown
        if self.repository:
            self.repository.compile_book()

        self._initialized = False
        log.info("documenter.adapter_v3.shutdown_complete")

    async def get_pending_work(self) -> list[dict]:
        """
        Get list of pending work from the document system.

        Returns work items for all phases of the Editorial Board workflow.
        """
        if not self._initialized:
            return []

        work_items = []

        # Check for blessed insights needing incorporation
        # (Would connect to blessed_insights directory in full implementation)
        # For now, check if there are pending plans or drafts to continue

        # Add pending draft reviews
        for insight_id, draft in self._pending_drafts.items():
            work_items.append({
                "task_type": TaskType.INCORPORATE_INSIGHT_REVIEW.value,
                "key": f"doc:review:{insight_id}",
                "payload": {
                    "insight_id": insight_id,
                    "draft": draft.__dict__,
                },
                "requires_deep_mode": False,
                "priority": 56,
            })

        # Check for diagram requests in recently added sections
        structure = self.repository.load_structure()
        for chapter in structure["chapters"]:
            for section in chapter["sections"]:
                if section.get("status") == "provisional":
                    content = self.repository.get_section_content(section["id"])
                    if content:
                        diagram_requests = self.illustrator.extract_requests(content.content)
                        for desc in diagram_requests:
                            work_items.append({
                                "task_type": TaskType.GENERATE_DIAGRAM.value,
                                "key": f"doc:diagram:{section['id']}:{hash(desc) % 10000}",
                                "payload": {
                                    "section_id": section["id"],
                                    "description": desc,
                                },
                                "requires_deep_mode": False,
                                "priority": 40,
                            })

        # Check if refactor is needed
        if self.repository.needs_refactor():
            work_items.append({
                "task_type": TaskType.GLOBAL_REFACTOR.value,
                "key": f"doc:refactor:{datetime.now().strftime('%Y%m%d')}",
                "payload": {},
                "requires_deep_mode": True,
                "priority": 35,
            })

        log.debug(
            "documenter.adapter_v3.pending_work",
            count=len(work_items),
        )

        return work_items

    async def build_prompt(self, task: Task) -> PromptContext:
        """Build prompt for a task."""
        task_type = task.task_type

        if task_type == TaskType.INCORPORATE_INSIGHT_PLAN.value:
            insight_text = task.payload.get("insight_text", "")
            insight_id = task.payload.get("insight_id", "unknown")

            # Architect handles its own prompt building
            # Return minimal context - actual work done in handle_result
            return PromptContext(
                prompt=f"Plan incorporation of insight {insight_id}",
                requires_deep_mode=True,
                preferred_backend="gemini",
                metadata={"insight_id": insight_id},
            )

        elif task_type == TaskType.INCORPORATE_INSIGHT_DRAFT.value:
            insight_id = task.payload.get("insight_id", "")
            plan = self._pending_plans.get(insight_id)

            if not plan:
                return PromptContext(
                    prompt="Error: No placement plan found",
                    metadata={"error": "missing_plan"},
                )

            return PromptContext(
                prompt=f"Draft content for placement: {plan.placement.mode}",
                requires_deep_mode=False,
                preferred_backend="claude",
                metadata={"insight_id": insight_id},
            )

        elif task_type == TaskType.INCORPORATE_INSIGHT_REVIEW.value:
            insight_id = task.payload.get("insight_id", "")
            draft = self._pending_drafts.get(insight_id)

            if not draft:
                return PromptContext(
                    prompt="Error: No draft found for review",
                    metadata={"error": "missing_draft"},
                )

            return PromptContext(
                prompt=f"Review draft: {draft.title}",
                requires_deep_mode=False,
                metadata={"insight_id": insight_id},
            )

        elif task_type == TaskType.GENERATE_DIAGRAM.value:
            description = task.payload.get("description", "")
            return PromptContext(
                prompt=f"Generate diagram: {description}",
                requires_deep_mode=False,
                metadata={"section_id": task.payload.get("section_id")},
            )

        elif task_type == TaskType.GLOBAL_REFACTOR.value:
            return PromptContext(
                prompt="Analyze document structure for reorganization",
                requires_deep_mode=True,
                preferred_backend="gemini",
            )

        elif task_type == TaskType.ADDRESS_COMMENT.value:
            comment_text = task.payload.get("comment_text", "")
            return PromptContext(
                prompt=f"Address comment: {comment_text[:100]}...",
                requires_deep_mode=False,
            )

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def handle_result(self, task: Task, result: TaskResult) -> bool:
        """Handle the result of a task execution."""
        task_type = task.task_type

        try:
            if task_type == TaskType.INCORPORATE_INSIGHT_PLAN.value:
                return await self._handle_plan_result(task, result)

            elif task_type == TaskType.INCORPORATE_INSIGHT_DRAFT.value:
                return await self._handle_draft_result(task, result)

            elif task_type == TaskType.INCORPORATE_INSIGHT_REVIEW.value:
                return await self._handle_review_result(task, result)

            elif task_type == TaskType.GENERATE_DIAGRAM.value:
                return await self._handle_diagram_result(task, result)

            elif task_type == TaskType.GLOBAL_REFACTOR.value:
                return await self._handle_refactor_result(task, result)

            elif task_type == TaskType.ADDRESS_COMMENT.value:
                return await self._handle_comment_result(task, result)

            else:
                log.warning("documenter.adapter_v3.unknown_task_type", task_type=task_type)
                return False

        except Exception as e:
            log.exception(e, "documenter.adapter_v3.handle_result_error", {
                "task_id": task.id,
                "task_type": task_type,
            })
            return False

    async def on_task_failed(self, task: Task, error: str):
        """Handle task failure."""
        log.error(
            "documenter.adapter_v3.task_failed",
            task_id=task.id,
            task_type=task.task_type,
            error=error,
        )

        # Clean up in-flight state
        insight_id = task.payload.get("insight_id", "")
        if insight_id:
            self._pending_plans.pop(insight_id, None)
            self._pending_drafts.pop(insight_id, None)

    async def get_system_state(self) -> dict:
        """Get current system state."""
        structure = self.repository.load_structure() if self.repository else {}
        definitions = self.repository.load_definitions() if self.repository else {}

        section_count = sum(
            len(ch.get("sections", [])) for ch in structure.get("chapters", [])
        )

        return {
            "sections_count": section_count,
            "concepts_defined": len(definitions),
            "pending_plans": len(self._pending_plans),
            "pending_drafts": len(self._pending_drafts),
        }

    # ==================== Phase Handlers ====================

    async def _handle_plan_result(self, task: Task, result: TaskResult) -> bool:
        """Handle Architect's placement plan."""
        insight_id = task.payload.get("insight_id", "")
        insight_text = task.payload.get("insight_text", "")
        insight_concepts = task.payload.get("concepts", [])

        # Run Architect analysis
        analysis = await self.architect.analyze_insight(
            insight_text=insight_text,
            insight_id=insight_id,
            insight_concepts=insight_concepts,
        )

        if not analysis.can_incorporate:
            # Check if it's a conflict
            if analysis.conflict_with_section:
                # Create seed question for Explorer
                question = await self.architect.create_seed_question(
                    conflict_description=analysis.conflict_description,
                    section_id=analysis.conflict_with_section,
                    insight_text=insight_text,
                )
                log.info(
                    "documenter.adapter_v3.conflict_detected",
                    insight_id=insight_id,
                    section=analysis.conflict_with_section,
                    seed_question=question[:100],
                )
                # TODO: Submit seed question to Explorer
                return False

            elif analysis.missing_prerequisites:
                log.info(
                    "documenter.adapter_v3.prerequisites_missing",
                    insight_id=insight_id,
                    missing=analysis.missing_prerequisites,
                )
                return False

        # Store plan for draft phase
        self._pending_plans[insight_id] = analysis

        log.info(
            "documenter.adapter_v3.plan_complete",
            insight_id=insight_id,
            mode=analysis.placement.mode,
            target=analysis.placement.target_section_id or analysis.placement.target_chapter_id,
        )

        # Signal that draft task should be created
        result.needs_continuation = True
        result.continuation_payload = {
            "task_type": TaskType.INCORPORATE_INSIGHT_DRAFT.value,
            "insight_id": insight_id,
            "insight_text": insight_text,
            "placement": analysis.placement.__dict__,
        }

        return True

    async def _handle_draft_result(self, task: Task, result: TaskResult) -> bool:
        """Handle Mason's drafted content."""
        insight_id = task.payload.get("insight_id", "")
        insight_text = task.payload.get("insight_text", "")
        placement_dict = task.payload.get("placement", {})

        # Get plan
        plan = self._pending_plans.get(insight_id)
        if not plan:
            # Reconstruct from payload
            placement = PlacementPlan(**placement_dict)
        else:
            placement = plan.placement

        # Run Mason drafting
        draft = await self.mason.draft_content(
            insight_text=insight_text,
            insight_id=insight_id,
            placement=placement,
        )

        if not draft:
            log.warning("documenter.adapter_v3.draft_failed", insight_id=insight_id)
            return False

        # Store draft for review phase
        self._pending_drafts[insight_id] = draft

        # Clean up plan
        self._pending_plans.pop(insight_id, None)

        log.info(
            "documenter.adapter_v3.draft_complete",
            insight_id=insight_id,
            title=draft.title,
            content_length=len(draft.content),
        )

        # Signal that review task should be created
        result.needs_continuation = True
        result.continuation_payload = {
            "task_type": TaskType.INCORPORATE_INSIGHT_REVIEW.value,
            "insight_id": insight_id,
        }

        return True

    async def _handle_review_result(self, task: Task, result: TaskResult) -> bool:
        """Handle Consensus Board review."""
        insight_id = task.payload.get("insight_id", "")

        # Get draft
        draft = self._pending_drafts.get(insight_id)
        if not draft:
            log.error("documenter.adapter_v3.review_no_draft", insight_id=insight_id)
            return False

        # Run Consensus Board review
        review = await self.consensus_board.review(draft)

        if review.approved:
            # Save to repository
            await self._save_approved_draft(draft)

            # Clean up
            self._pending_drafts.pop(insight_id, None)

            log.info(
                "documenter.adapter_v3.content_approved",
                insight_id=insight_id,
                title=draft.title,
            )
            return True

        elif review.needs_revision:
            # Get Mason to revise
            revised_draft = await self.mason.revise_draft(
                draft=draft,
                feedback=review.consolidated_feedback,
            )

            if revised_draft:
                self._pending_drafts[insight_id] = revised_draft
                # Will be reviewed again next round
                result.needs_continuation = True
                result.continuation_payload = {
                    "task_type": TaskType.INCORPORATE_INSIGHT_REVIEW.value,
                    "insight_id": insight_id,
                }

            return True

        elif review.escalate_to_human:
            log.warning(
                "documenter.adapter_v3.human_escalation",
                insight_id=insight_id,
                feedback=review.consolidated_feedback[:200],
            )
            # Leave draft pending for human review
            return False

        else:
            # Rejected
            log.info(
                "documenter.adapter_v3.content_rejected",
                insight_id=insight_id,
                feedback=review.consolidated_feedback[:200],
            )
            self._pending_drafts.pop(insight_id, None)
            return False

    async def _save_approved_draft(self, draft: DraftedContent):
        """Save approved draft to repository."""
        placement = draft.placement

        # Process diagrams if any
        if draft.diagram_requests:
            processed_content, _ = await self.illustrator.process_diagram_requests(
                draft.content
            )
            draft.content = processed_content

        if placement.mode == "INSERT":
            # Create new section
            section = SectionContent(
                id=self.repository.generate_section_id(placement.target_chapter_id),
                title=draft.title,
                content=draft.content,
                summary=draft.summary,
                concepts_defined=draft.concepts_defined,
                concepts_required=draft.concepts_required,
                chapter_id=placement.target_chapter_id,
            )
            self.repository.add_section(placement.target_chapter_id, section)

        elif placement.mode in ("MERGE", "APPEND"):
            # Update existing section
            existing = self.repository.get_section_content(placement.target_section_id)
            if existing:
                if placement.mode == "MERGE":
                    existing.content = draft.content
                else:  # APPEND
                    existing.content += f"\n\n{draft.content}"

                existing.summary = draft.summary
                existing.concepts_defined = list(set(
                    existing.concepts_defined + draft.concepts_defined
                ))
                existing.last_modified = datetime.now()

                self.repository.update_section(existing)

        # Recompile book
        self.repository.compile_book()

    async def _handle_diagram_result(self, task: Task, result: TaskResult) -> bool:
        """Handle diagram generation."""
        section_id = task.payload.get("section_id", "")
        description = task.payload.get("description", "")

        # Generate diagram
        diagram_path = await self.illustrator.generate_diagram(description)

        if diagram_path:
            # Update section content to embed diagram
            section = self.repository.get_section_content(section_id)
            if section:
                # Replace tag with image link
                tag = f"<<DIAGRAM_REQUEST: {description}>>"
                relative_path = f"assets/{diagram_path.name}"
                image_link = f"![{description}]({relative_path})"
                section.content = section.content.replace(tag, image_link, 1)
                self.repository.update_section(section)

            log.info(
                "documenter.adapter_v3.diagram_generated",
                section_id=section_id,
                path=str(diagram_path),
            )
            return True

        return False

    async def _handle_refactor_result(self, task: Task, result: TaskResult) -> bool:
        """Handle global refactor suggestions."""
        suggestions = await self.architect.suggest_chapter_reorganization()

        if not suggestions:
            log.info("documenter.adapter_v3.no_refactor_needed")
            return True

        log.info(
            "documenter.adapter_v3.refactor_suggestions",
            count=len(suggestions),
        )

        # TODO: Execute reorganization or queue for human review
        # For now, just log the suggestions
        for i, suggestion in enumerate(suggestions):
            log.info(
                "documenter.adapter_v3.refactor_suggestion",
                index=i,
                type=suggestion.get("type"),
                sections=suggestion.get("sections"),
                rationale=suggestion.get("rationale", "")[:100],
            )

        return True

    async def _handle_comment_result(self, task: Task, result: TaskResult) -> bool:
        """Handle comment addressing (uses quick review)."""
        comment_text = task.payload.get("comment_text", "")
        section_id = task.payload.get("section_id", "")

        # Use quick review for simple comment handling
        approved, feedback = await self.consensus_board.quick_review(
            content=result.response or "",
            review_type="correction",
        )

        if approved:
            log.info(
                "documenter.adapter_v3.comment_addressed",
                section_id=section_id,
            )

        return approved
