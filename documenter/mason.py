"""
The Mason Agent - Content Drafter.

The Mason is responsible for:
- Writing atomic section content with rigorous LaTeX
- Ensuring smooth transitions (bridging text with prev/next sections)
- Emitting diagram requests for visual concepts
- Following the notation style guide

Backend: Claude 3.5 Sonnet (strong at LaTeX/coding)
"""

import re
from dataclasses import dataclass
from typing import Optional

from shared.logging import get_logger
from llm.src.client import LLMClient

from .repository import DocumentRepository, PlacementPlan, SectionContent

log = get_logger("documenter", "mason")


# Pattern to detect diagram requests emitted by Mason
DIAGRAM_PATTERN = re.compile(r'<<DIAGRAM_REQUEST:\s*(.+?)>>', re.IGNORECASE)


@dataclass
class DraftedContent:
    """Result of Mason's drafting work."""
    content: str
    title: str
    summary: str
    concepts_defined: list[str]
    concepts_required: list[str]
    diagram_requests: list[str]
    placement: PlacementPlan


class Mason:
    """
    The Mason Agent - writes atomic section content.

    Uses Claude for its strength in LaTeX formatting and
    precise technical writing. Focuses on mathematical rigor
    while maintaining readability.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        llm_client: LLMClient,
    ):
        """
        Initialize the Mason.

        Args:
            repository: Document repository for context
            llm_client: LLM client for backend communication
        """
        self.repository = repository
        self.llm_client = llm_client

    async def draft_content(
        self,
        insight_text: str,
        insight_id: str,
        placement: PlacementPlan,
        concepts_to_establish: list[str] = None,
    ) -> Optional[DraftedContent]:
        """
        Draft content for incorporation into the document.

        This is Phase 2 of the incorporate_insight workflow.
        Takes the Architect's placement plan and creates actual content.

        Args:
            insight_text: The insight to incorporate
            insight_id: Unique identifier
            placement: PlacementPlan from Architect
            concepts_to_establish: Optional list of concepts this should define

        Returns:
            DraftedContent ready for review, or None on failure
        """
        concepts_to_establish = concepts_to_establish or []

        # Build context
        context = self._build_drafting_context(placement)

        # Get notation guide for relevant concepts
        notation = self.repository.load_notation()
        notation_guide = self._format_notation_guide(notation)

        prompt = f"""{context}

## Notation Guide
{notation_guide}

## Insight to Incorporate
{insight_text}

## Placement Decision
Mode: {placement.mode}
{f"Target Section: {placement.target_section_id}" if placement.target_section_id else f"New Section: {placement.new_section_title}"}
{f"In Chapter: {placement.target_chapter_id}" if placement.target_chapter_id else ""}
Rationale: {placement.rationale}

## Concepts to Establish
{', '.join(concepts_to_establish) if concepts_to_establish else 'As appropriate from the insight'}
"""

        task = self._get_drafting_task(placement.mode)

        # Use Claude for drafting (strong at LaTeX)
        response = await self.llm_client.send(
            backend="claude",
            prompt=f"{prompt}\n\n{task}",
            timeout_seconds=90,
        )

        if not response.success:
            log.error(
                "mason.draft_failed",
                insight_id=insight_id,
                error=response.error,
            )
            return None

        return self._parse_draft(response.text, placement, concepts_to_establish)

    def _build_drafting_context(self, placement: PlacementPlan) -> str:
        """Build context from document structure for drafting."""
        parts = ["## Document Context"]

        if placement.mode == "INSERT":
            # For new sections, get chapter context and adjacent summaries
            chapter = self.repository.get_chapter(placement.target_chapter_id)
            if chapter:
                parts.append(f"\nChapter: {chapter['title']}")
                parts.append(f"Chapter Summary: {chapter['summary']}")

                if chapter["sections"]:
                    parts.append("\nExisting sections in this chapter:")
                    for sec in chapter["sections"]:
                        parts.append(f"  - {sec['id']}: {sec['title']}")
                        parts.append(f"    Summary: {sec['summary'][:100]}...")

        elif placement.mode in ("MERGE", "APPEND") and placement.target_section_id:
            # For merge/append, get the target section's content
            section = self.repository.get_section_content(placement.target_section_id)
            if section:
                parts.append(f"\nTarget Section: {section.title}")
                parts.append(f"Current Content:\n{section.content[:1500]}...")

                # Get adjacent summaries for smooth transitions
                prev_summary, next_summary = self.repository.get_adjacent_summaries(
                    placement.target_section_id
                )
                if prev_summary:
                    parts.append(f"\nPrevious Section Summary: {prev_summary}")
                if next_summary:
                    parts.append(f"\nNext Section Summary: {next_summary}")

        return "\n".join(parts)

    def _format_notation_guide(self, notation: dict) -> str:
        """Format notation guide for prompt."""
        if not notation:
            return "No specific notation conventions defined."

        lines = []
        for name, entry in notation.items():
            symbol = entry.get("symbol", "")
            latex = entry.get("latex", "")
            meaning = entry.get("meaning", "")
            lines.append(f"- {name}: {symbol} ({latex}) - {meaning}")

        return "\n".join(lines)

    def _get_drafting_task(self, mode: str) -> str:
        """Get the appropriate drafting task prompt for the mode."""
        base_instructions = """
IMPORTANT WRITING GUIDELINES:
1. Write in authoritative expository style suitable for a mathematical textbook
2. Do NOT include conversational phrases ("Let me explain...", "As we can see...")
3. Do NOT ask questions or address the reader directly
4. Use precise LaTeX notation following the notation guide
5. Build concepts step by step with clear logical flow
6. Include bridging text to connect with surrounding content

DIAGRAM REQUESTS:
If a concept would benefit from a visual diagram, emit a tag like:
<<DIAGRAM_REQUEST: Description of what the diagram should show>>

For example:
<<DIAGRAM_REQUEST: Fano plane with 7 points and 7 lines labeled>>
"""

        if mode == "INSERT":
            return f"""Write a NEW SECTION to be inserted into the document.

{base_instructions}

Format your response as:

TITLE: [Section title]

CONTENT:
[The section content in markdown with LaTeX math]

SUMMARY: [One sentence summary for table of contents]

ESTABLISHES: [Comma-separated concepts this section defines]

REQUIRES: [Comma-separated concepts this section assumes are known]
"""

        elif mode == "MERGE":
            return f"""MERGE the insight into the existing section content.

{base_instructions}

Seamlessly integrate the new insight while:
- Preserving valuable existing content
- Removing any duplication
- Ensuring logical flow throughout

Format your response as:

CONTENT:
[The complete merged section content in markdown with LaTeX math]

SUMMARY: [Updated one sentence summary]

ESTABLISHES: [Updated comma-separated concepts]

REQUIRES: [Updated comma-separated concepts]
"""

        else:  # APPEND
            return f"""APPEND to the existing section, continuing its exposition.

{base_instructions}

Add content that:
- Follows naturally from the existing ending
- Develops the insight as a continuation
- Maintains consistent voice and notation

Format your response as:

CONTENT:
[The content to APPEND - not the entire section, just the new part]

SUMMARY: [Updated one sentence summary for the expanded section]

ESTABLISHES: [Additional concepts defined by the new content]

REQUIRES: [Additional concepts assumed by the new content]
"""

    def _parse_draft(
        self,
        response_text: str,
        placement: PlacementPlan,
        concepts_to_establish: list[str],
    ) -> DraftedContent:
        """Parse Mason's response into DraftedContent."""
        # Parse title (for INSERT mode)
        title = placement.new_section_title or "Untitled Section"
        title_match = re.search(r'TITLE:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()

        # Parse content
        content = response_text
        content_match = re.search(
            r'CONTENT:\s*\n(.*?)(?=\nSUMMARY:|ESTABLISHES:|REQUIRES:|\Z)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if content_match:
            content = content_match.group(1).strip()

        # Parse summary
        summary = ""
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()

        # Parse establishes
        establishes = concepts_to_establish.copy()
        est_match = re.search(r'ESTABLISHES:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if est_match:
            new_concepts = [c.strip() for c in est_match.group(1).split(',') if c.strip()]
            for c in new_concepts:
                if c not in establishes:
                    establishes.append(c)

        # Parse requires
        requires = []
        req_match = re.search(r'REQUIRES:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if req_match:
            requires = [c.strip() for c in req_match.group(1).split(',') if c.strip()]

        # Extract diagram requests
        diagram_requests = DIAGRAM_PATTERN.findall(content)

        log.info(
            "mason.draft_complete",
            title=title,
            content_length=len(content),
            concepts_defined=len(establishes),
            diagram_requests=len(diagram_requests),
        )

        return DraftedContent(
            content=content,
            title=title,
            summary=summary or title,
            concepts_defined=establishes,
            concepts_required=requires,
            diagram_requests=diagram_requests,
            placement=placement,
        )

    async def revise_draft(
        self,
        draft: DraftedContent,
        feedback: str,
    ) -> Optional[DraftedContent]:
        """
        Revise a draft based on Consensus Board feedback.

        Called when the Board rejects a draft but wants revision
        rather than complete rejection.

        Args:
            draft: The original draft
            feedback: Feedback from the review board

        Returns:
            Revised DraftedContent or None on failure
        """
        prompt = f"""## Original Draft

Title: {draft.title}

Content:
{draft.content}

## Review Feedback

{feedback}

## Revision Task

Revise the draft to address the feedback above. Maintain the same structure
and mathematical rigor, but incorporate the suggested improvements.

Format your response as:

CONTENT:
[The revised content]

SUMMARY: [Updated summary if needed]

ESTABLISHES: [Updated concepts if needed]

REQUIRES: [Updated requirements if needed]
"""

        response = await self.llm_client.send(
            backend="claude",
            prompt=prompt,
            timeout_seconds=90,
        )

        if not response.success:
            log.error("mason.revision_failed", error=response.error)
            return None

        return self._parse_draft(response.text, draft.placement, draft.concepts_defined)

    def extract_diagram_requests(self, content: str) -> list[str]:
        """
        Extract all diagram request tags from content.

        Args:
            content: Text content to scan

        Returns:
            List of diagram descriptions
        """
        return DIAGRAM_PATTERN.findall(content)

    def strip_diagram_tags(self, content: str) -> str:
        """
        Remove diagram request tags from content.

        Called after diagrams have been generated and embedded.

        Args:
            content: Content with diagram tags

        Returns:
            Content with tags removed
        """
        return DIAGRAM_PATTERN.sub('', content)
