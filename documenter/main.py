"""
Documenter main module - entry point and main loop.
"""

import asyncio
import re
import sys
import uuid
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import yaml

# Add parent paths for imports
FANO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FANO_ROOT))

from shared.logging import get_logger, correlation_context
from shared.deduplication import (
    DeduplicationChecker,
    ContentItem,
    ContentType,
    load_dedup_config,
)

from llm.src.client import LLMClient
from llm.src.consensus import ConsensusReviewer

from .document import Document, Section
from .concepts import ConceptTracker
from .tasks import TaskBuilder
from .opportunities import OpportunityFinder, Opportunity
from .review import ReviewManager
from .snapshots import SnapshotManager
from .annotations import AnnotationManager
from .formatting import fix_math_if_needed

log = get_logger("documenter", "main")


class Documenter:
    """
    Main documenter class - orchestrates document growth.

    Creates and maintains a living mathematical document by:
    - Incorporating blessed insights from Explorer
    - Validating all additions through LLM consensus
    - Reviewing and improving existing content
    - Creating daily snapshots
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize documenter.

        Args:
            config_path: Path to config file (default: config.yaml in project root)
        """
        self.config = self._load_config(config_path)

        # Document management
        doc_path = Path(self.config.get("documenter", {}).get("document", {}).get("path", "document/main.md"))
        self.document = Document(doc_path)

        # Concepts
        self.concept_tracker: Optional[ConceptTracker] = None

        # Task building
        max_tokens = self.config.get("documenter", {}).get("context", {}).get("max_tokens", 8000)
        self.task_builder = TaskBuilder(max_context_tokens=max_tokens)

        # LLM client and consensus
        self.llm_client: Optional[LLMClient] = None
        self.consensus: Optional[ConsensusReviewer] = None

        # Opportunities
        blessed_dir = Path(self.config.get("documenter", {}).get("inputs", {}).get("blessed_insights_dir", "blessed_insights"))
        # Resolve relative paths to FANO_ROOT
        if not blessed_dir.is_absolute():
            blessed_dir = FANO_ROOT / blessed_dir
        max_disputes = self.config.get("documenter", {}).get("termination", {}).get("max_consecutive_disputes", 3)
        self.opportunity_finder: Optional[OpportunityFinder] = None
        self.blessed_dir = blessed_dir
        self.max_disputes = max_disputes

        # Review
        max_age = self.config.get("documenter", {}).get("review", {}).get("max_age_days", 7)
        work_alloc = self.config.get("documenter", {}).get("work_allocation", {}).get("review_existing", 30)
        self.review_manager: Optional[ReviewManager] = None
        self.max_age_days = max_age
        self.review_allocation = work_alloc

        # Snapshots
        archive_dir = Path(self.config.get("documenter", {}).get("document", {}).get("archive_dir", "document/archive"))
        snapshot_time_str = self.config.get("documenter", {}).get("document", {}).get("snapshot_time", "00:00")
        hour, minute = map(int, snapshot_time_str.split(":"))
        self.snapshot_manager: Optional[SnapshotManager] = None
        self.archive_dir = archive_dir
        self.snapshot_time = time(hour, minute)

        # Termination
        self.max_consensus_calls = self.config.get("documenter", {}).get("termination", {}).get("max_consensus_calls_per_session", 100)
        self.consensus_calls = 0
        self.exhausted = False

        # LLM consensus settings
        self.use_deep_mode = self.config.get("llm", {}).get("consensus", {}).get("use_deep_mode", False)

        # Guidance file (optional author direction)
        guidance_path = self.config.get("documenter", {}).get("inputs", {}).get("guidance_file", "document/guidance.md")
        if not Path(guidance_path).is_absolute():
            guidance_path = FANO_ROOT / guidance_path
        self.guidance_path = Path(guidance_path)
        self.guidance_text = self._load_guidance()

        # Annotations (user comments and protected regions)
        self.annotation_manager: Optional[AnnotationManager] = None

        # Deduplication checker
        self.dedup_checker: Optional[DeduplicationChecker] = None

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from file."""
        if config_path is None:
            config_path = FANO_ROOT / "config.yaml"

        if config_path.exists():
            try:
                return yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("documenter.config.load_error", path=str(config_path), error=str(e))

        return {}

    def _load_guidance(self) -> str:
        """Load optional guidance file."""
        if self.guidance_path.exists():
            try:
                text = self.guidance_path.read_text(encoding="utf-8")
                log.info("documenter.guidance.loaded", path=str(self.guidance_path))
                return text
            except Exception as e:
                log.warning("documenter.guidance.load_error", error=str(e))
        return ""

    async def initialize(self):
        """Initialize all components."""
        log.info("documenter.initializing")

        # Load document
        if not self.document.load():
            log.warning("documenter.document.creating_seed")
            self._create_seed_document()
            self.document.load()

        # Initialize components
        self.concept_tracker = ConceptTracker(self.document)
        self.annotation_manager = AnnotationManager(self.document.path)

        # LLM client with pool URL from config
        pool_config = self.config.get("llm", {}).get("pool", {})
        pool_host = pool_config.get("host", "127.0.0.1")
        pool_port = pool_config.get("port", 9000)
        pool_url = f"http://{pool_host}:{pool_port}"
        self.llm_client = LLMClient(pool_url=pool_url)
        self.consensus = ConsensusReviewer(self.llm_client)

        # Verify we have enough backends for consensus
        available_backends = await self.llm_client.get_available_backends()
        if len(available_backends) < 2:
            log.error(
                "documenter.insufficient_backends",
                available=available_backends,
                required=2,
            )
            # Clean up before raising
            await self.llm_client.close()
            raise RuntimeError(
                f"Documenter requires at least 2 LLM backends for consensus, but only {len(available_backends)} available: {available_backends}. "
                f"Either start the pool service (for gemini/chatgpt) or set API keys (ANTHROPIC_API_KEY, OPENROUTER_API_KEY)."
            )
        log.info("documenter.backends_available", backends=available_backends)

        self.opportunity_finder = OpportunityFinder(
            self.document,
            self.concept_tracker,
            self.blessed_dir,
            self.max_disputes,
        )

        self.review_manager = ReviewManager(
            self.document,
            self.max_age_days,
            self.review_allocation,
        )

        self.snapshot_manager = SnapshotManager(
            self.document,
            self.archive_dir,
            self.snapshot_time,
        )

        # Load deduplication config from centralized config.yaml
        self._dedup_config = load_dedup_config()

        # Initialize deduplication checker with LLM callback
        # Uses cheap LLM (configured model) for dedup to preserve expensive quota
        self.dedup_checker = DeduplicationChecker(
            llm_callback=self._dedup_llm_callback,
            use_signature_check=self._dedup_config.get("use_signature_check", True),
            use_heuristic_check=self._dedup_config.get("use_heuristic_check", False),
            use_llm_check=self._dedup_config.get("use_llm_check", True),
            use_batch_llm=self._dedup_config.get("use_batch_llm", True),
            batch_size=self._dedup_config.get("batch_size", 20),
            stats_log_interval=self._dedup_config.get("stats_log_interval", 50),
        )

        # Load existing document sections into dedup checker
        for section in self.document.sections:
            self.dedup_checker.add_content(ContentItem(
                id=section.id,
                text=section.content,
                content_type=ContentType.SECTION,
            ))

        log.info(
            "documenter.initialized",
            sections=len(self.document.sections),
            concepts=len(self.concept_tracker.get_established_concepts()),
            pending_opportunities=self.opportunity_finder.get_pending_count(),
            dedup_known_items=self.dedup_checker.known_count,
            use_deep_mode=self.use_deep_mode,
        )

    def _create_seed_document(self):
        """Create the seed document if it doesn't exist."""
        seed_content = '''# The Principles of Creation

1. The first manifestation is always in the form of three.
2. The fundamental design of the Cosmos is common. From that design it has evolved into such complex possibilities.
3. It is the perfection of geometry which is holding the existence together.
4. The only way to articulate the profound intelligence of Creation is in complex geometrical forms.

This document explores the mathematical structure that emerges from
this principle, and its appearance across traditions.

We seek what is natural, elegant, beautiful, interesting, and
inevitable — structure that must exist, not structure we impose.

'''
        self.document.path.parent.mkdir(parents=True, exist_ok=True)
        self.document.path.write_text(seed_content, encoding="utf-8")
        log.info("documenter.seed_created", path=str(self.document.path))

    async def cleanup(self):
        """Clean up resources."""
        if self.llm_client:
            await self.llm_client.close()
            log.info("documenter.cleanup.complete")

    async def _dedup_llm_callback(self, prompt: str) -> str:
        """
        LLM callback for deduplication checks.

        Uses configured model (cheap/fast) to preserve expensive quota for exploration.
        """
        model = self._dedup_config.get("model", "claude-sonnet-4-20250514")
        timeout = self._dedup_config.get("llm_timeout", 60)

        response = await self.llm_client.send(
            "claude",
            prompt,
            model=model,
            timeout_seconds=timeout,
        )
        if response.success:
            return response.text
        raise RuntimeError(f"LLM call failed: {response.error}")

    async def _plan_next_work(self) -> Optional[dict]:
        """
        Plan what to work on next using batch insight selection.

        Shows all available insights to LLMs and asks them to either:
        1. Select an insight that's ready to incorporate
        2. Identify prerequisites that need to be written first

        Returns:
            dict with "type" ("insight" or "prerequisite") and content,
            or None if no work to do.
        """
        # Get all available opportunities
        all_opportunities = self.opportunity_finder._gather_all_opportunities()
        if not all_opportunities:
            return None

        # Log what we're showing to LLMs for debugging
        established = list(self.concept_tracker.get_established_concepts())
        recent_section = self.document.sections[-1] if self.document.sections else None
        log.info(
            "documenter.planning.started",
            opportunity_count=len(all_opportunities),
            established_concepts=established,
            recent_section_id=recent_section.id if recent_section else None,
            recent_section_establishes=recent_section.establishes if recent_section else [],
        )

        # Build insight summaries for LLM
        insight_summaries = []
        for i, opp in enumerate(all_opportunities):
            summary = f"{i+1}. [{opp.insight_id}] {opp.text[:200]}..."
            if opp.requires:
                summary += f"\n   Depends on: {', '.join(opp.requires)}"
            insight_summaries.append(summary)

        insights_text = "\n\n".join(insight_summaries)

        # Build context with guidance
        guidance_section = ""
        if self.guidance_text:
            guidance_section = f"""
AUTHOR GUIDANCE:
{self.guidance_text}

"""

        context = f"""CURRENT DOCUMENT STATE:
{self.document.get_summary()}

ESTABLISHED CONCEPTS:
{', '.join(self.concept_tracker.get_established_concepts()) or 'None yet'}

{guidance_section}AVAILABLE INSIGHTS TO INCORPORATE ({len(all_opportunities)} total):

{insights_text}
"""

        task = """Analyze the document and available insights. Decide what to do next.

CRITICAL: Check the EXISTING SECTIONS list above. Do NOT write prerequisite content that
duplicates or overlaps with what already exists. The document already contains this material.

OPTION A - If an insight is ready and fits naturally as the next step:
  Reply with DECISION: INCORPORATE
  INSIGHT_ID: [the insight ID]
  REASON: [why this is the right one to add now]

OPTION B - If NEW foundational material is needed (not already in EXISTING SECTIONS):
  Reply with DECISION: PREREQUISITE
  NEEDED: [what NEW content is needed - must be different from existing sections]
  CONTENT: [Draft the actual prerequisite content for publication - definitions, explanations,
           mathematical foundations. Write in authoritative expository style, NOT conversational.
           Do NOT include "Let me explain...", "Would you like...", or questions to the reader.]
  ESTABLISHES: [comma-separated list of concepts this content establishes]

OPTION C - If nothing is ready and all needed prerequisites already exist:
  Reply with DECISION: WAIT
  REASON: [why we should wait - what's missing or unclear]

Consider:
- What does the document already establish?
- Which insight has the fewest unmet dependencies?
- What foundational material would unlock multiple insights?
- Follow any author guidance provided."""

        result = await self.consensus.run(
            context,
            task,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
        )
        self.consensus_calls += 1

        if not result.success:
            log.warning("documenter.planning.failed", outcome=result.outcome)
            return None

        outcome = result.outcome.upper()

        # Parse the decision
        if "DECISION: INCORPORATE" in outcome or "DECISION:INCORPORATE" in outcome:
            # Extract insight ID
            id_match = re.search(r'INSIGHT_ID:\s*(\S+)', result.outcome, re.IGNORECASE)
            if id_match:
                insight_id = id_match.group(1).strip('[]')
                # Find the matching opportunity
                for opp in all_opportunities:
                    if opp.insight_id == insight_id or insight_id in str(opp.insight_id):
                        log.info("documenter.planning.selected_insight", insight_id=insight_id)
                        return {"type": "insight", "opportunity": opp}

            # Fallback: couldn't parse ID, try first opportunity
            log.warning("documenter.planning.id_parse_failed", trying_first=True)
            return {"type": "insight", "opportunity": all_opportunities[0]}

        elif "DECISION: PREREQUISITE" in outcome or "DECISION:PREREQUISITE" in outcome:
            # Extract the content to write - handle CONTENT: on its own line
            # Content ends at ESTABLISHES: or end of string
            content_match = re.search(r'CONTENT:\s*\n?(.*?)(?=\nESTABLISHES:|\Z)', result.outcome, re.IGNORECASE | re.DOTALL)
            if content_match:
                prerequisite_content = content_match.group(1).strip()
                if prerequisite_content:
                    # Also extract ESTABLISHES if present
                    establishes = []
                    est_match = re.search(r'ESTABLISHES:\s*\n?(.+?)(?=\n\n|\Z)', result.outcome, re.IGNORECASE | re.DOTALL)
                    if est_match:
                        establishes = [c.strip() for c in est_match.group(1).split(',') if c.strip()]

                    log.info(
                        "documenter.planning.prerequisite_needed",
                        content_length=len(prerequisite_content),
                        establishes=establishes,
                    )
                    return {"type": "prerequisite", "content": prerequisite_content, "establishes": establishes}

            # Fallback: try to extract content after NEEDED: if no CONTENT: found
            needed_match = re.search(r'NEEDED:\s*(.+?)(?=\n\n|\Z)', result.outcome, re.IGNORECASE | re.DOTALL)
            if needed_match:
                # The LLM told us what's needed - ask it to generate the content
                log.info("documenter.planning.prerequisite_description_only", needed=needed_match.group(1)[:100])
                # Generate the prerequisite content based on what's needed
                return await self._generate_prerequisite_content(needed_match.group(1).strip())

        elif "DECISION: WAIT" in outcome or "DECISION:WAIT" in outcome:
            # LLMs decided to wait - nothing is ready and prerequisites exist
            reason_match = re.search(r'REASON:\s*(.+?)(?=\n\n|\Z)', result.outcome, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "no reason given"
            log.info("documenter.planning.waiting", reason=reason)
            return None  # Signal to wait/do review instead

        log.warning("documenter.planning.unclear_decision", outcome=result.outcome[:200])
        return None

    async def _generate_prerequisite_content(self, needed_description: str) -> Optional[dict]:
        """
        Generate prerequisite content when the LLM only described what's needed
        but didn't provide the actual content.
        """
        log.info("documenter.prerequisite.generating", description=needed_description[:100])

        guidance_section = f"\nAUTHOR GUIDANCE:\n{self.guidance_text}\n" if self.guidance_text else ""
        annotations_context = self._get_annotations_context()

        context = f"""CURRENT DOCUMENT:
{self.document.get_summary()}

{guidance_section}
PREREQUISITE NEEDED:
{needed_description}
{annotations_context}"""

        task = """Write the foundational content described above.

IMPORTANT: This is for a FINAL PUBLISHED DOCUMENT, not a conversation.
- Do NOT include conversational phrases like "Let me explain...", "Next step:", "Would you like..."
- Do NOT ask questions or offer options
- Do NOT include meta-commentary about the writing process
- Write in a clear, authoritative expository style suitable for publication

This content should:
1. Start from first principles
2. Build up concepts step by step
3. Use clear, precise mathematical language
4. Follow the author guidance if provided
5. Establish the concepts needed for later insights

Format your response as:
CONTENT:
[The mathematical exposition to add to the document]

ESTABLISHES:
[Comma-separated list of key concepts this content establishes, e.g.: Fano plane, projective geometry, XOR arithmetic]"""

        result = await self.consensus.run(
            context,
            task,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
            select_best=True,  # Let LLMs vote on best content
        )
        self.consensus_calls += 1

        if result.success and result.outcome:
            log.info(
                "documenter.prerequisite.selected",
                selection_stats=result.selection_stats,
            )
            # Parse content and established concepts
            content = result.outcome
            establishes = []

            # Extract CONTENT section
            content_match = re.search(r'CONTENT:\s*\n?(.*?)(?=\nESTABLISHES:|$)', result.outcome, re.IGNORECASE | re.DOTALL)
            if content_match:
                content = content_match.group(1).strip()

            # Extract ESTABLISHES list
            est_match = re.search(r'ESTABLISHES:\s*\n?(.+)', result.outcome, re.IGNORECASE | re.DOTALL)
            if est_match:
                establishes = [c.strip() for c in est_match.group(1).split(',') if c.strip()]

            return {"type": "prerequisite", "content": content, "establishes": establishes}

        log.warning("documenter.prerequisite.generation_failed")
        return None

    async def _write_prerequisite(self, content: str, establishes: Optional[list[str]] = None):
        """
        Write prerequisite/foundational content to the document.

        This content was already generated through consensus in _generate_prerequisite_content
        or extracted from the planning consensus, so we trust it and write directly.
        No additional validation needed - that would be redundant.

        Args:
            content: The content to write
            establishes: List of concepts this content establishes (for tracking)
        """
        establishes = establishes or []

        # Check for duplicates before writing
        if self.dedup_checker:
            dedup_result = await self.dedup_checker.check_duplicate(
                content,
                item_id=f"prereq-pending",
                content_type=ContentType.PREREQUISITE,
            )
            if dedup_result.is_duplicate:
                log.info(
                    "documenter.prerequisite.duplicate_detected",
                    duplicate_of=dedup_result.duplicate_of,
                    method=dedup_result.check_method,
                    reason=dedup_result.reason,
                )
                return  # Skip this prerequisite, it's already covered

        # Fix math formatting if needed
        content = await self._fix_math_formatting(content)

        log.info(
            "documenter.prerequisite.writing",
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
        self.document.append_section(section, content)
        concepts_str = ", ".join(establishes) if establishes else "foundational content"
        self.document.save(f"Added prerequisite: {concepts_str}")

        # Refresh concept tracker by recreating it
        self.concept_tracker = ConceptTracker(self.document)

        # Add to dedup checker for future duplicate detection
        if self.dedup_checker:
            self.dedup_checker.add_content(ContentItem(
                id=section.id,
                text=content,
                content_type=ContentType.PREREQUISITE,
            ))

        log.info(
            "documenter.prerequisite.written",
            content_length=len(content),
            new_concepts=establishes,
        )

    async def run(self):
        """Main loop - grow the document."""
        await self.initialize()

        log.info("documenter.run.started")

        try:
            with correlation_context() as cid:
                while not self.exhausted:
                    # Check for snapshot
                    self.snapshot_manager.check_and_snapshot()

                    # Check consensus call budget
                    if self.consensus_calls >= self.max_consensus_calls:
                        log.info(
                            "documenter.run.budget_exhausted",
                            consensus_calls=self.consensus_calls,
                        )
                        self.exhausted = True
                        break

                    # Check for author comments first (highest priority)
                    comments = self.document.find_unresolved_comments()
                    if comments:
                        comment_text, line_num = comments[0]
                        await self._address_comment(comment_text, line_num)
                        continue

                    # Decide: new work or review?
                    if self.review_manager.should_do_review():
                        section = self.review_manager.select_section_for_review()
                        if section:
                            await self._review_section(section)
                            continue

                    # Work on new material - use intelligent planning
                    work_plan = await self._plan_next_work()
                    if work_plan:
                        if work_plan["type"] == "prerequisite":
                            # Generate foundational content first
                            establishes = work_plan.get("establishes", [])
                            await self._write_prerequisite(work_plan["content"], establishes)
                        elif work_plan["type"] == "insight":
                            # Work on the selected insight
                            await self._work_on_opportunity(work_plan["opportunity"])
                        continue

                    # Nothing to do - LLMs said WAIT or no clear decision
                    log.info("documenter.run.exhausted", reason="nothing_to_do")
                    self.exhausted = True

        finally:
            # Always cleanup
            await self.cleanup()

            # Final snapshot
            self.snapshot_manager.create_snapshot()

            # Log summary
            self._log_summary()

    async def _address_comment(self, comment_text: str, line_num: int):
        """Address an author comment."""
        log.info(
            "documenter.comment.addressing",
            comment=comment_text[:100],
            line=line_num,
        )

        # Find the section containing the comment
        section_text = self._get_section_around_line(line_num)

        # Build consensus task
        context, task, response_format = self.task_builder.build_address_comment(
            section_text,
            comment_text,
        )
        context = self.task_builder.truncate_context(context)

        # Run consensus
        result = await self.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
        )
        self.consensus_calls += 1

        if result.converged and "REVISED_SECTION:" in result.outcome:
            # Extract revised section
            match = re.search(r'REVISED_SECTION:\s*(.+)', result.outcome, re.DOTALL | re.IGNORECASE)
            if match and match.group(1).strip().upper() != "UNCHANGED":
                revised = match.group(1).strip()
                # Update document (simplified - just remove the comment marker)
                old_comment = f"<!-- COMMENT: {comment_text} -->"
                self.document.content = self.document.content.replace(old_comment, "", 1)
                self.document.save(f"Resolved comment: {comment_text[:50]}")
                log.info("documenter.comment.resolved", comment=comment_text[:50])
        else:
            # Mark as attempted
            old_comment = f"<!-- COMMENT: {comment_text} -->"
            new_comment = f"<!-- COMMENT: {comment_text} (attempted: true) -->"
            self.document.content = self.document.content.replace(old_comment, new_comment, 1)
            self.document.save(f"Attempted comment: {comment_text[:50]}")
            log.warning("documenter.comment.unresolved", comment=comment_text[:50])

    def _get_section_around_line(self, line_num: int, context_lines: int = 20) -> str:
        """Get document content around a line number."""
        lines = self.document.content.split('\n')
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines)
        return '\n'.join(lines[start:end])

    def _get_annotations_context(self) -> str:
        """
        Get user annotations formatted for LLM context.

        Returns formatted string with comments and protected regions,
        or empty string if no annotations.
        """
        if not self.annotation_manager:
            return ""

        # Reload annotations to get latest
        self.annotation_manager.load()

        # Use the built-in formatter
        context = self.annotation_manager.format_for_llm()

        if context:
            # Count annotations by type
            comments = sum(1 for a in self.annotation_manager.annotations.values() if a.type == "comment")
            protected = sum(1 for a in self.annotation_manager.annotations.values() if a.type == "protected")
            log.info(
                "documenter.annotations.included",
                comments=comments,
                protected=protected,
            )

        return context

    async def _fix_math_formatting(self, content: str) -> str:
        """
        Fix math delimiter issues in content if needed.

        Uses regex detection first, only calls LLM if issues found.
        """
        if not self.llm_client:
            return content

        return await fix_math_if_needed(content, self.llm_client)

    async def _review_section(self, section: Section):
        """Review an existing section."""
        log.info(
            "documenter.review.starting",
            section_id=section.id,
        )

        # Get concepts at creation and since
        concepts_at_creation = self.review_manager.get_concepts_at_creation(section)
        concepts_since = self.review_manager.get_concepts_since_creation(section)

        # Build consensus task
        context, task, response_format = self.task_builder.build_review_section(
            section.content,
            section.created.strftime("%Y-%m-%d"),
            concepts_at_creation,
            concepts_since,
        )
        context = self.task_builder.truncate_context(context)

        # Run consensus
        result = await self.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
        )
        self.consensus_calls += 1

        if result.converged:
            # Check decision
            if "DECISION: CONFIRMED" in result.outcome.upper():
                self.document.mark_section_reviewed(section.id)
                self.document.save(f"Reviewed section {section.id}: confirmed")
                log.info("documenter.review.confirmed", section_id=section.id)
            elif "DECISION: REVISE" in result.outcome.upper():
                # Extract revision notes and apply
                # For now, just mark as reviewed
                self.document.mark_section_reviewed(section.id)
                self.document.save(f"Reviewed section {section.id}: revision suggested")
                log.info("documenter.review.revision_suggested", section_id=section.id)
        else:
            log.warning("documenter.review.no_consensus", section_id=section.id)

    async def _work_on_opportunity(self, opportunity: Opportunity):
        """Work on a new opportunity."""
        log.info(
            "documenter.opportunity.working",
            type=opportunity.type,
            insight_id=opportunity.insight_id,
        )

        # Step 0: Check for duplicates with existing document content
        if self.dedup_checker:
            dedup_result = await self.dedup_checker.check_duplicate(
                opportunity.text,
                item_id=opportunity.insight_id or "unknown",
                content_type=ContentType.INSIGHT,
            )
            if dedup_result.is_duplicate:
                log.info(
                    "documenter.opportunity.duplicate_detected",
                    insight_id=opportunity.insight_id,
                    duplicate_of=dedup_result.duplicate_of,
                    method=dedup_result.check_method,
                    reason=dedup_result.reason,
                )
                # Mark as incorporated (skip it, don't dispute)
                self.opportunity_finder.mark_incorporated(opportunity)
                return

        # Step 1: Evaluate mathematics
        math_approved = await self._evaluate_math(opportunity)
        if not math_approved:
            self.opportunity_finder.mark_disputed(opportunity)
            return

        # Step 2: Draft content
        draft = await self._draft_content(opportunity)
        if not draft:
            self.opportunity_finder.mark_disputed(opportunity)
            return

        # Step 3: Evaluate exposition
        exposition_approved = await self._evaluate_exposition(draft)
        if not exposition_approved:
            self.opportunity_finder.mark_disputed(opportunity)
            return

        # Step 4: Add to document
        await self._add_to_document(opportunity, draft)

    async def _evaluate_math(self, opportunity: Opportunity) -> bool:
        """Evaluate if the mathematics should be included."""
        context, task, response_format = self.task_builder.build_math_evaluation(
            self.document.get_summary(),
            opportunity.text,
        )
        context = self.task_builder.truncate_context(context)

        result = await self.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
        )
        self.consensus_calls += 1

        # Check for errors first
        if not result.success:
            log.error(
                "documenter.math.error",
                insight_id=opportunity.insight_id,
                outcome=result.outcome,
                rounds=result.rounds,
            )
            return False

        if result.converged and "DECISION: INCLUDE" in result.outcome.upper():
            log.info("documenter.math.approved", insight_id=opportunity.insight_id)
            return True
        else:
            log.info(
                "documenter.math.rejected",
                insight_id=opportunity.insight_id,
                converged=result.converged,
                confidence=result.confidence,
            )
            return False

    async def _draft_content(self, opportunity: Opportunity) -> Optional[str]:
        """Draft content for the opportunity."""
        # Get preceding section
        preceding = ""
        if self.document.sections:
            preceding = self.document.sections[-1].content

        context, task, response_format = self.task_builder.build_draft_content(
            self.document.get_summary(),
            preceding,
            list(self.concept_tracker.get_established_concepts()),
            opportunity.text[:100],  # Topic
            opportunity.text,  # Full insight
        )

        # Add user annotations context
        annotations_context = self._get_annotations_context()
        if annotations_context:
            context += annotations_context

        context = self.task_builder.truncate_context(context)

        result = await self.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
            select_best=True,  # Let LLMs vote on best draft
        )
        self.consensus_calls += 1

        # Check for errors first
        if not result.success:
            log.error(
                "documenter.draft.error",
                insight_id=opportunity.insight_id,
                outcome=result.outcome,
            )
            return None

        log.info(
            "documenter.draft.selected",
            insight_id=opportunity.insight_id,
            selection_stats=result.selection_stats,
        )

        if "DRAFT:" in result.outcome.upper():
            # Extract draft
            match = re.search(r'DRAFT:\s*(.+?)(?=ESTABLISHES:|$)', result.outcome, re.DOTALL | re.IGNORECASE)
            if match:
                log.info("documenter.draft.created", insight_id=opportunity.insight_id)
                return result.outcome  # Return full outcome for parsing

        log.warning("documenter.draft.failed", insight_id=opportunity.insight_id)
        return None

    async def _evaluate_exposition(self, draft: str) -> bool:
        """Evaluate if the exposition is acceptable."""
        # Extract just the draft text
        match = re.search(r'DRAFT:\s*(.+?)(?=ESTABLISHES:|$)', draft, re.DOTALL | re.IGNORECASE)
        draft_text = match.group(1).strip() if match else draft

        # Get preceding section
        preceding = ""
        if self.document.sections:
            preceding = self.document.sections[-1].content

        context, task, response_format = self.task_builder.build_exposition_evaluation(
            self.document.get_summary(),
            preceding,
            list(self.concept_tracker.get_established_concepts()),
            draft_text,
        )
        context = self.task_builder.truncate_context(context)

        result = await self.consensus.run(
            context,
            task,
            response_format=response_format,
            max_rounds=2,
            use_deep_mode=self.use_deep_mode,
        )
        self.consensus_calls += 1

        # Check for errors first
        if not result.success:
            log.error("documenter.exposition.error", outcome=result.outcome)
            return False

        if result.converged and "DECISION: APPROVE" in result.outcome.upper():
            log.info("documenter.exposition.approved")
            return True
        else:
            log.info(
                "documenter.exposition.not_approved",
                converged=result.converged,
                confidence=result.confidence,
            )
            return False

    async def _add_to_document(self, opportunity: Opportunity, draft: str):
        """Add approved content to the document."""
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
            id=self.document.generate_next_section_id(),
            content=content,
            created=datetime.now(),
            status="provisional",
            establishes=establishes,
            requires=requires,
        )

        # Add to document
        self.document.append_section(section, content)
        concepts_str = ", ".join(establishes) if establishes else "new content"
        self.document.save(f"Incorporated insight {opportunity.insight_id}: {concepts_str}")

        # Register concepts
        self.concept_tracker.register_section(section)

        # Add to dedup checker for future duplicate detection
        if self.dedup_checker:
            self.dedup_checker.add_content(ContentItem(
                id=section.id,
                text=content,
                content_type=ContentType.SECTION,
            ))

        # Mark opportunity as incorporated
        self.opportunity_finder.mark_incorporated(opportunity)

        log.info(
            "documenter.content.added",
            section_id=section.id,
            establishes=establishes,
            insight_id=opportunity.insight_id,
        )

    def _log_summary(self):
        """Log session summary."""
        review_stats = self.review_manager.get_review_stats()

        log.info(
            "documenter.session.summary",
            sections=len(self.document.sections),
            concepts=len(self.concept_tracker.get_established_concepts()),
            consensus_calls=self.consensus_calls,
            pending_opportunities=self.opportunity_finder.get_pending_count(),
            **review_stats,
        )


async def main():
    """Entry point for running the documenter."""
    documenter = Documenter()
    await documenter.run()


if __name__ == "__main__":
    asyncio.run(main())
