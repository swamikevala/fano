"""
The Architect Agent - Editorial Voice and Placement Decisions.

The Architect is a persistent agent that:
- Maintains the "Editorial Voice" across sessions (via thread persistence)
- Manages the document Outline (structure.json)
- Decides WHERE new content belongs (PlacementPlan)
- Detects conflicts between insights and existing content
- Submits seed questions to Explorer for conflict resolution

Backend: Gemini Deep Think (uses Deep Mode quota)
"""

import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from shared.logging import get_logger
from llm.src.client import LLMClient

from .repository import DocumentRepository, PlacementPlan, SectionContent

if TYPE_CHECKING:
    from .session import SessionManager

log = get_logger("documenter", "architect")


@dataclass
class InsightAnalysis:
    """Analysis result for a single insight."""
    insight_id: str
    can_incorporate: bool
    placement: Optional[PlacementPlan] = None
    missing_prerequisites: list[str] = None
    conflict_with_section: Optional[str] = None
    conflict_description: str = ""

    def __post_init__(self):
        if self.missing_prerequisites is None:
            self.missing_prerequisites = []


class Architect:
    """
    The Architect Agent - decides where content belongs.

    Uses structure.json summaries to maintain context without
    reading full section content. This allows it to fit the
    entire document structure in its context window (~2k tokens).
    """

    def __init__(
        self,
        repository: DocumentRepository,
        llm_client: LLMClient,
        thread_id: Optional[str] = None,
    ):
        """
        Initialize the Architect.

        Args:
            repository: Document repository for structure access
            llm_client: LLM client for backend communication
            thread_id: Optional thread ID for persistence
        """
        self.repository = repository
        self.llm_client = llm_client
        self.thread_id = thread_id

    async def analyze_insight(
        self,
        insight_text: str,
        insight_id: str,
        insight_concepts: list[str] = None,
    ) -> InsightAnalysis:
        """
        Analyze a blessed insight and decide on placement.

        This is Phase 1 of the incorporate_insight workflow.
        Uses Deep Think to reason about document structure.

        Args:
            insight_text: The insight content to incorporate
            insight_id: Unique identifier for tracking
            insight_concepts: Optional list of concepts the insight establishes

        Returns:
            InsightAnalysis with placement decision or conflict info
        """
        insight_concepts = insight_concepts or []

        # Check prerequisites first
        definitions = self.repository.load_definitions()
        structure_summary = self.repository.get_structure_summary()

        # Build context for Architect
        context = f"""# Document Structure Analysis

{structure_summary}

## Defined Concepts Registry
{', '.join(definitions.keys()) if definitions else 'No concepts defined yet'}

## New Insight to Place
ID: {insight_id}
Content Preview: {insight_text[:500]}...
Concepts it may establish: {', '.join(insight_concepts) if insight_concepts else 'Not specified'}
"""

        task = """Analyze this insight and decide where it belongs in the document.

STEP 1: Check Prerequisites
- What concepts does this insight REQUIRE to be understood?
- Are those concepts already defined in the registry?
- If prerequisites are missing, list them.

STEP 2: Check for Conflicts
- Does this insight contradict anything in the existing structure?
- Does it duplicate existing content?
- If there's a conflict, describe it clearly.

STEP 3: Decide Placement (if no blocking issues)
Choose ONE of these modes:
- INSERT: Create a new section (specify chapter and position)
- MERGE: Integrate into an existing section (specify section ID)
- APPEND: Add as a continuation of an existing section (specify section ID)

Reply in this EXACT format:

PREREQUISITES: [comma-separated list of required concepts, or "none"]
MISSING: [comma-separated list of undefined prerequisites, or "none"]

CONFLICT: [yes/no]
CONFLICT_SECTION: [section ID if conflict exists, or "none"]
CONFLICT_DESCRIPTION: [description of conflict, or "none"]

PLACEMENT_MODE: [INSERT/MERGE/APPEND]
TARGET_CHAPTER: [chapter ID for INSERT, or "n/a"]
TARGET_SECTION: [section ID for MERGE/APPEND, or "n/a"]
NEW_TITLE: [title for new section if INSERT, or "n/a"]
RATIONALE: [brief explanation of why this placement is appropriate]
"""

        # Use Gemini Deep Think for planning
        response = await self.llm_client.send(
            backend="gemini",
            prompt=f"{context}\n\n{task}",
            deep_think=True,
            thread_id=self.thread_id,
            timeout_seconds=120,
        )

        if not response.success:
            log.error(
                "architect.analysis_failed",
                insight_id=insight_id,
                error=response.error,
            )
            return InsightAnalysis(
                insight_id=insight_id,
                can_incorporate=False,
                conflict_description=f"Analysis failed: {response.error}",
            )

        return self._parse_analysis(response.text, insight_id)

    def _parse_analysis(self, response_text: str, insight_id: str) -> InsightAnalysis:
        """Parse LLM response into InsightAnalysis."""
        text = response_text.upper()

        # Parse missing prerequisites
        missing = []
        missing_match = re.search(r'MISSING:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if missing_match:
            missing_text = missing_match.group(1).strip()
            if missing_text.lower() != "none":
                missing = [m.strip() for m in missing_text.split(',') if m.strip()]

        # Parse conflict
        has_conflict = "CONFLICT: YES" in text or "CONFLICT:YES" in text
        conflict_section = None
        conflict_desc = ""

        if has_conflict:
            sec_match = re.search(r'CONFLICT_SECTION:\s*(\S+)', response_text, re.IGNORECASE)
            if sec_match and sec_match.group(1).lower() != "none":
                conflict_section = sec_match.group(1)

            desc_match = re.search(r'CONFLICT_DESCRIPTION:\s*(.+?)(?=\nPLACEMENT|\Z)', response_text, re.IGNORECASE | re.DOTALL)
            if desc_match:
                conflict_desc = desc_match.group(1).strip()

        # Can't incorporate if missing prereqs or conflict
        if missing or has_conflict:
            return InsightAnalysis(
                insight_id=insight_id,
                can_incorporate=False,
                missing_prerequisites=missing,
                conflict_with_section=conflict_section,
                conflict_description=conflict_desc,
            )

        # Parse placement
        placement = self._parse_placement(response_text)

        return InsightAnalysis(
            insight_id=insight_id,
            can_incorporate=True,
            placement=placement,
        )

    def _parse_placement(self, response_text: str) -> PlacementPlan:
        """Parse placement decision from response."""
        # Default values
        mode = "APPEND"
        target_section = None
        target_chapter = "ch_01"
        new_title = None
        rationale = ""

        # Parse mode
        mode_match = re.search(r'PLACEMENT_MODE:\s*(\w+)', response_text, re.IGNORECASE)
        if mode_match:
            mode = mode_match.group(1).upper()
            if mode not in ("INSERT", "MERGE", "APPEND"):
                mode = "APPEND"

        # Parse target chapter
        chapter_match = re.search(r'TARGET_CHAPTER:\s*(\S+)', response_text, re.IGNORECASE)
        if chapter_match and chapter_match.group(1).lower() not in ("n/a", "none"):
            target_chapter = chapter_match.group(1)

        # Parse target section
        section_match = re.search(r'TARGET_SECTION:\s*(\S+)', response_text, re.IGNORECASE)
        if section_match and section_match.group(1).lower() not in ("n/a", "none"):
            target_section = section_match.group(1)

        # Parse new title
        title_match = re.search(r'NEW_TITLE:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if title_match and title_match.group(1).strip().lower() not in ("n/a", "none"):
            new_title = title_match.group(1).strip()

        # Parse rationale
        rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=\n\n|\Z)', response_text, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        return PlacementPlan(
            mode=mode,
            target_section_id=target_section,
            target_chapter_id=target_chapter,
            new_section_title=new_title,
            rationale=rationale,
        )

    async def create_seed_question(
        self,
        conflict_description: str,
        section_id: str,
        insight_text: str,
    ) -> str:
        """
        Create a seed question for Explorer to investigate conflict.

        When the Architect detects a conflict between an insight and
        existing content, it formulates a question for deeper exploration.

        Args:
            conflict_description: What the conflict is
            section_id: Section that conflicts
            insight_text: The conflicting insight

        Returns:
            Seed question text for Explorer
        """
        # Get the conflicting section's summary
        section_meta = self.repository.get_section_meta(section_id)
        existing_summary = ""
        if section_meta:
            _, meta = section_meta
            existing_summary = meta.get("summary", "")

        context = f"""CONFLICT DETECTED IN DOCUMENT

Existing Section ({section_id}):
{existing_summary}

New Insight:
{insight_text[:300]}...

Conflict Description:
{conflict_description}
"""

        task = """Formulate a precise mathematical question that would help resolve this conflict.

The question should:
1. Be answerable through mathematical reasoning
2. Help determine which perspective is correct (or if both can coexist)
3. Be specific enough for focused exploration

Reply with just the question, nothing else."""

        response = await self.llm_client.send(
            backend="gemini",
            prompt=f"{context}\n\n{task}",
            deep_think=True,
            thread_id=self.thread_id,
            timeout_seconds=60,
        )

        if response.success:
            question = response.text.strip()
            log.info(
                "architect.seed_question_created",
                section_id=section_id,
                question_length=len(question),
            )
            return question

        # Fallback to generic question
        return f"How can we reconcile the apparent conflict between {section_id} and the new insight about {insight_text[:50]}...?"

    async def suggest_chapter_reorganization(self) -> list[dict]:
        """
        Analyze document structure and suggest reorganization.

        Called during global_refactor task when document has grown.

        Returns:
            List of reorganization suggestions
        """
        structure_summary = self.repository.get_structure_summary()

        context = f"""# Document Structure for Reorganization Analysis

{structure_summary}
"""

        task = """Analyze the document structure and identify opportunities for improvement:

1. MERGE: Sections that cover the same topic and should be combined
2. SPLIT: Sections that are too long and cover multiple distinct topics
3. REORDER: Sections that are in an illogical order for reader comprehension

For each suggestion, provide:
- TYPE: MERGE/SPLIT/REORDER
- SECTIONS: [list of section IDs involved]
- RATIONALE: [why this improves the document]

If no reorganization is needed, reply: NONE NEEDED

Format as a numbered list of suggestions."""

        response = await self.llm_client.send(
            backend="gemini",
            prompt=f"{context}\n\n{task}",
            deep_think=True,
            thread_id=self.thread_id,
            timeout_seconds=120,
        )

        if not response.success:
            log.error("architect.reorganization_failed", error=response.error)
            return []

        if "NONE NEEDED" in response.text.upper():
            log.info("architect.no_reorganization_needed")
            return []

        return self._parse_reorganization(response.text)

    def _parse_reorganization(self, response_text: str) -> list[dict]:
        """Parse reorganization suggestions from response."""
        suggestions = []

        # Split by numbered items
        items = re.split(r'\n\d+\.', response_text)

        for item in items[1:]:  # Skip first empty split
            suggestion = {"type": None, "sections": [], "rationale": ""}

            # Parse type
            type_match = re.search(r'TYPE:\s*(\w+)', item, re.IGNORECASE)
            if type_match:
                suggestion["type"] = type_match.group(1).upper()

            # Parse sections
            sections_match = re.search(r'SECTIONS:\s*\[?([^\]]+)\]?', item, re.IGNORECASE)
            if sections_match:
                sections_text = sections_match.group(1)
                suggestion["sections"] = [s.strip() for s in sections_text.split(',') if s.strip()]

            # Parse rationale
            rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=\n\d+\.|\Z)', item, re.IGNORECASE | re.DOTALL)
            if rationale_match:
                suggestion["rationale"] = rationale_match.group(1).strip()

            if suggestion["type"] and suggestion["sections"]:
                suggestions.append(suggestion)

        log.info("architect.reorganization_suggestions", count=len(suggestions))
        return suggestions

    def update_thread_id(self, thread_id: str):
        """Update thread ID for persistence."""
        self.thread_id = thread_id
        self.repository.architect_thread_id = thread_id
        log.info("architect.thread_updated", thread_id=thread_id)
