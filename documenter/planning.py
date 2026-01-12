"""
Work planning for the documenter.

Handles intelligent selection of what to work on next:
- Batch insight selection
- Prerequisite identification and generation
- Work prioritization
"""

import re
from typing import Optional, TYPE_CHECKING

from shared.logging import get_logger

if TYPE_CHECKING:
    from .session import SessionManager

log = get_logger("documenter", "planning")


class WorkPlanner:
    """
    Plans what work to do next.

    Uses LLM consensus to intelligently select between:
    - Incorporating ready insights
    - Generating prerequisite content
    - Waiting for more material
    """

    def __init__(self, session: "SessionManager", guidance_text: str = ""):
        """
        Initialize work planner.

        Args:
            session: The session manager with all components
            guidance_text: Optional author guidance text
        """
        self.session = session
        self.guidance_text = guidance_text

    async def plan_next_work(self) -> Optional[dict]:
        """
        Plan what to work on next using batch insight selection.

        Shows all available insights to LLMs and asks them to either:
        1. Select an insight that's ready to incorporate
        2. Identify prerequisites that need to be written first

        Returns:
            dict with "type" ("insight" or "prerequisite") and content,
            or None if no work to do.
        """
        s = self.session  # Shorthand

        # Get all available opportunities
        all_opportunities = s.opportunity_finder._gather_all_opportunities()
        if not all_opportunities:
            return None

        # Log what we're showing to LLMs for debugging
        established = list(s.concept_tracker.get_established_concepts())
        recent_section = s.document.sections[-1] if s.document.sections else None
        log.info(
            "planning.started",
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
{s.document.get_summary()}

ESTABLISHED CONCEPTS:
{', '.join(s.concept_tracker.get_established_concepts()) or 'None yet'}

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

        result = await s.consensus.run(
            context,
            task,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
        )
        s.increment_consensus_calls()

        if not result.success:
            log.warning("planning.failed", outcome=result.outcome)
            return None

        outcome = result.outcome.upper()

        # Parse the decision
        if "DECISION: INCORPORATE" in outcome or "DECISION:INCORPORATE" in outcome:
            return self._parse_incorporate_decision(result.outcome, all_opportunities)

        elif "DECISION: PREREQUISITE" in outcome or "DECISION:PREREQUISITE" in outcome:
            return await self._parse_prerequisite_decision(result.outcome)

        elif "DECISION: WAIT" in outcome or "DECISION:WAIT" in outcome:
            reason_match = re.search(r'REASON:\s*(.+?)(?=\n\n|\Z)', result.outcome, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "no reason given"
            log.info("planning.waiting", reason=reason)
            return None

        log.warning("planning.unclear_decision", outcome=result.outcome[:200])
        return None

    def _parse_incorporate_decision(self, outcome: str, all_opportunities: list) -> Optional[dict]:
        """Parse an INCORPORATE decision from the LLM outcome."""
        id_match = re.search(r'INSIGHT_ID:\s*(\S+)', outcome, re.IGNORECASE)
        if id_match:
            insight_id = id_match.group(1).strip('[]')
            # Find the matching opportunity
            for opp in all_opportunities:
                if opp.insight_id == insight_id or insight_id in str(opp.insight_id):
                    log.info("planning.selected_insight", insight_id=insight_id)
                    return {"type": "insight", "opportunity": opp}

        # Fallback: couldn't parse ID, try first opportunity
        log.warning("planning.id_parse_failed", trying_first=True)
        return {"type": "insight", "opportunity": all_opportunities[0]}

    async def _parse_prerequisite_decision(self, outcome: str) -> Optional[dict]:
        """Parse a PREREQUISITE decision from the LLM outcome."""
        # Extract the content to write
        content_match = re.search(r'CONTENT:\s*\n?(.*?)(?=\nESTABLISHES:|\Z)', outcome, re.IGNORECASE | re.DOTALL)
        if content_match:
            prerequisite_content = content_match.group(1).strip()
            if prerequisite_content:
                # Also extract ESTABLISHES if present
                establishes = []
                est_match = re.search(r'ESTABLISHES:\s*\n?(.+?)(?=\n\n|\Z)', outcome, re.IGNORECASE | re.DOTALL)
                if est_match:
                    establishes = [c.strip() for c in est_match.group(1).split(',') if c.strip()]

                log.info(
                    "planning.prerequisite_needed",
                    content_length=len(prerequisite_content),
                    establishes=establishes,
                )
                return {"type": "prerequisite", "content": prerequisite_content, "establishes": establishes}

        # Fallback: try to extract content after NEEDED: if no CONTENT: found
        needed_match = re.search(r'NEEDED:\s*(.+?)(?=\n\n|\Z)', outcome, re.IGNORECASE | re.DOTALL)
        if needed_match:
            log.info("planning.prerequisite_description_only", needed=needed_match.group(1)[:100])
            return await self._generate_prerequisite_content(needed_match.group(1).strip())

        return None

    async def _generate_prerequisite_content(self, needed_description: str) -> Optional[dict]:
        """
        Generate prerequisite content when the LLM only described what's needed
        but didn't provide the actual content.
        """
        s = self.session
        log.info("planning.prerequisite.generating", description=needed_description[:100])

        guidance_section = f"\nAUTHOR GUIDANCE:\n{self.guidance_text}\n" if self.guidance_text else ""

        context = f"""CURRENT DOCUMENT:
{s.document.get_summary()}

{guidance_section}
PREREQUISITE NEEDED:
{needed_description}
"""

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

        result = await s.consensus.run(
            context,
            task,
            max_rounds=2,
            use_deep_mode=s.use_deep_mode,
            select_best=True,
        )
        s.increment_consensus_calls()

        if result.success and result.outcome:
            log.info(
                "planning.prerequisite.selected",
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

        log.warning("planning.prerequisite.generation_failed")
        return None
