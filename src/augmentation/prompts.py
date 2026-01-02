"""
Prompts for augmentation analysis and generation.
"""

import re
from typing import Optional

from .models import (
    AugmentationAnalysis,
    DiagramType,
    TableType,
    CodePurpose,
)


def build_analysis_prompt(
    insight: str,
    tags: list[str],
    dependencies: list[str],
    review_summary: str,
) -> str:
    """Build prompt to analyze what augmentations would help an insight."""
    tags_str = ", ".join(tags) if tags else "none"
    deps_str = ", ".join(dependencies) if dependencies else "none"

    return f"""A mathematical insight has been blessed. Evaluate whether supplementary
materials would strengthen or clarify it.

BLESSED INSIGHT:
{insight}

TAGS: {tags_str}
DEPENDS ON: {deps_str}

REVIEW CONTEXT:
{review_summary}

EVALUATE EACH AUGMENTATION TYPE:

1. DIAGRAM
   Would a visual representation clarify this?
   Consider: graph structures, geometric relationships, mappings between sets,
   flow diagrams, hierarchies, symmetry illustrations

   DIAGRAM_HELPFUL: [yes/no]
   DIAGRAM_TYPE: [graph / geometry / mapping / hierarchy / flow / none]
   DIAGRAM_DESCRIPTION: [what it would show]

2. TABLE
   Would a systematic enumeration strengthen this?
   Consider: complete mappings, correspondences, case analysis,
   numerical relationships, comparison matrices

   TABLE_HELPFUL: [yes/no]
   TABLE_TYPE: [mapping / enumeration / comparison / correspondence / none]
   TABLE_DESCRIPTION: [rows, columns, what it demonstrates]

3. PROOF
   Can this be formally verified mathematically?
   Consider: the claim is precise enough to prove, there's a proof strategy,
   it's not just an observation but a theorem

   PROOF_POSSIBLE: [yes/partial/no]
   PROOF_STRATEGY: [approach if yes]
   PROOF_DEPENDENCIES: [what lemmas or known results it would use]

4. CODE
   Would executable code demonstrate or verify this?
   Consider: computation that confirms numerical claims, generation of
   structures, verification of properties

   CODE_HELPFUL: [yes/no]
   CODE_PURPOSE: [verify / generate / demonstrate / none]
   CODE_DESCRIPTION: [what it would do]

RECOMMENDATION:
[List which augmentations to generate, in priority order]"""


def parse_analysis_response(response: str) -> AugmentationAnalysis:
    """Parse the analysis response into an AugmentationAnalysis."""
    analysis = AugmentationAnalysis()

    # Parse diagram section
    diagram_helpful = _extract_field(response, "DIAGRAM_HELPFUL")
    analysis.diagram_helpful = diagram_helpful.lower() == "yes" if diagram_helpful else False

    diagram_type = _extract_field(response, "DIAGRAM_TYPE")
    if diagram_type:
        diagram_type = diagram_type.lower().strip()
        try:
            analysis.diagram_type = DiagramType(diagram_type)
        except ValueError:
            analysis.diagram_type = DiagramType.NONE

    analysis.diagram_description = _extract_field(response, "DIAGRAM_DESCRIPTION") or ""

    # Parse table section
    table_helpful = _extract_field(response, "TABLE_HELPFUL")
    analysis.table_helpful = table_helpful.lower() == "yes" if table_helpful else False

    table_type = _extract_field(response, "TABLE_TYPE")
    if table_type:
        table_type = table_type.lower().strip()
        try:
            analysis.table_type = TableType(table_type)
        except ValueError:
            analysis.table_type = TableType.NONE

    analysis.table_description = _extract_field(response, "TABLE_DESCRIPTION") or ""

    # Parse proof section
    proof_possible = _extract_field(response, "PROOF_POSSIBLE")
    analysis.proof_possible = proof_possible.lower() if proof_possible else "no"

    analysis.proof_strategy = _extract_field(response, "PROOF_STRATEGY") or ""

    proof_deps = _extract_field(response, "PROOF_DEPENDENCIES")
    if proof_deps:
        analysis.proof_dependencies = [d.strip() for d in proof_deps.split(",")]

    # Parse code section
    code_helpful = _extract_field(response, "CODE_HELPFUL")
    analysis.code_helpful = code_helpful.lower() == "yes" if code_helpful else False

    code_purpose = _extract_field(response, "CODE_PURPOSE")
    if code_purpose:
        code_purpose = code_purpose.lower().strip()
        try:
            analysis.code_purpose = CodePurpose(code_purpose)
        except ValueError:
            analysis.code_purpose = CodePurpose.NONE

    analysis.code_description = _extract_field(response, "CODE_DESCRIPTION") or ""

    # Parse recommendations
    rec_section = _extract_section(response, "RECOMMENDATION")
    if rec_section:
        # Split by newlines or commas
        recs = re.split(r"[\n,]", rec_section)
        analysis.recommendations = [r.strip().strip("-").strip() for r in recs if r.strip()]

    return analysis


def build_diagram_prompt(
    insight: str,
    diagram_type: str,
    diagram_description: str,
) -> str:
    """Build prompt to generate a diagram."""
    return f"""Generate a diagram to illustrate this mathematical insight.

INSIGHT: {insight}

DIAGRAM SPECIFICATION:
Type: {diagram_type}
Description: {diagram_description}

REQUIREMENTS:
- Clear, minimal, mathematically precise
- Labels must be readable
- Use color meaningfully (not decoratively)
- Include legend if multiple elements
- Export as SVG or PNG

Generate Python code using matplotlib/networkx/graphviz that produces this diagram.
The code must be self-contained and executable.

Respond with:
CODE:
```python
[complete Python code]
```

CAPTION: [1-2 sentence description for the diagram]"""


def parse_diagram_response(response: str) -> tuple[str, str]:
    """Parse diagram generation response into (code, caption)."""
    # Extract code block
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    # If no code block, try to find CODE: section
    if not code:
        code = _extract_section(response, "CODE") or ""

    # Extract caption
    caption = _extract_field(response, "CAPTION") or ""

    return code, caption


def build_table_prompt(
    insight: str,
    table_type: str,
    table_description: str,
) -> str:
    """Build prompt to generate a table."""
    return f"""Generate a table to systematize this mathematical insight.

INSIGHT: {insight}

TABLE SPECIFICATION:
Type: {table_type}
Description: {table_description}

REQUIREMENTS:
- Complete enumeration (all cases, not just examples)
- Clear column headers
- Sorted logically (by number, by category, by relationship)
- Include totals/summaries if applicable
- Markdown format

Respond with:
TABLE:
[complete markdown table]

NOTES: [any observations about patterns in the table]

VERIFICATION: [how to verify the table is complete/correct]"""


def parse_table_response(response: str) -> tuple[str, str, str]:
    """Parse table generation response into (table, notes, verification)."""
    # Extract table section
    table = _extract_section(response, "TABLE") or ""

    # Look for markdown table if not in TABLE section
    if not table or "|" not in table:
        # Try to find a markdown table anywhere
        table_match = re.search(r"(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)", response)
        if table_match:
            table = table_match.group(1).strip()

    notes = _extract_field(response, "NOTES") or ""
    verification = _extract_field(response, "VERIFICATION") or ""

    return table, notes, verification


def build_proof_prompt(
    insight: str,
    proof_strategy: str,
    proof_dependencies: list[str],
) -> str:
    """Build prompt to generate a formal proof."""
    deps_str = ", ".join(proof_dependencies) if proof_dependencies else "none specified"

    return f"""Generate a formal mathematical proof for this insight.

INSIGHT: {insight}

PROOF SPECIFICATION:
Strategy: {proof_strategy}
Dependencies: {deps_str}

REQUIREMENTS:
- State the theorem precisely
- List all assumptions/axioms used
- Each step must follow logically from previous
- Cite known results where used
- QED or explicit conclusion

FORMAT:
THEOREM: [precise statement]

ASSUMPTIONS:
- [list each assumption]

PROOF:
1. [first step with justification]
2. [second step with justification]
...
n. [conclusion] ∎

NOTES: [any caveats, alternative approaches, or open questions]"""


def parse_proof_response(response: str) -> tuple[str, str]:
    """Parse proof generation response into (full_proof, notes)."""
    # The full proof includes THEOREM, ASSUMPTIONS, and PROOF sections
    theorem = _extract_section(response, "THEOREM") or ""
    assumptions = _extract_section(response, "ASSUMPTIONS") or ""
    proof_steps = _extract_section(response, "PROOF") or ""
    notes = _extract_field(response, "NOTES") or ""

    # Combine into full proof
    full_proof = ""
    if theorem:
        full_proof += f"THEOREM: {theorem}\n\n"
    if assumptions:
        full_proof += f"ASSUMPTIONS:\n{assumptions}\n\n"
    if proof_steps:
        full_proof += f"PROOF:\n{proof_steps}"

    if not full_proof:
        # If parsing failed, return the whole response as the proof
        full_proof = response

    return full_proof.strip(), notes


def build_code_prompt(
    insight: str,
    code_purpose: str,
    code_description: str,
) -> str:
    """Build prompt to generate verification code."""
    return f"""Generate executable code to demonstrate or verify this insight.

INSIGHT: {insight}

CODE SPECIFICATION:
Purpose: {code_purpose}
Description: {code_description}

REQUIREMENTS:
- Self-contained (no external dependencies beyond numpy/standard lib)
- Clear comments explaining each step
- Print output that demonstrates the claim
- Include verification/assertion where applicable

Respond with:
CODE:
```python
[complete Python code]
```

EXPECTED_OUTPUT: [what running this should show]

WHAT_IT_PROVES: [how the output supports the insight]"""


def parse_code_response(response: str) -> tuple[str, str, str]:
    """Parse code generation response into (code, expected_output, what_it_proves)."""
    # Extract code block
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    # If no code block, try to find CODE: section
    if not code:
        code = _extract_section(response, "CODE") or ""

    expected = _extract_field(response, "EXPECTED_OUTPUT") or ""
    proves = _extract_field(response, "WHAT_IT_PROVES") or ""

    return code, expected, proves


def build_proof_verification_prompt(theorem: str, proof: str) -> str:
    """Build prompt to verify a mathematical proof."""
    return f"""Review this mathematical proof for correctness.

THEOREM: {theorem}

PROOF:
{proof}

CHECK:
1. Is the theorem statement precise and unambiguous?
2. Are all assumptions explicitly stated?
3. Does each step follow logically from previous steps?
4. Are cited results used correctly?
5. Does the conclusion actually follow?

Respond with:
VERDICT: [valid / flawed / needs revision]
ISSUES: [list any problems found, or "none" if valid]
SUGGESTIONS: [how to fix if flawed, or "none" if valid]"""


def parse_verification_response(response: str) -> tuple[str, str, str]:
    """Parse verification response into (verdict, issues, suggestions)."""
    verdict = _extract_field(response, "VERDICT") or "unknown"
    issues = _extract_field(response, "ISSUES") or ""
    suggestions = _extract_field(response, "SUGGESTIONS") or ""

    return verdict.lower().strip(), issues, suggestions


# Helper functions

def _extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a field value like 'FIELD_NAME: value' from text."""
    pattern = rf"{field_name}:\s*\[?([^\]\n]+)\]?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Also try without brackets
    pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _extract_section(text: str, section_name: str) -> Optional[str]:
    """Extract a multi-line section from text."""
    # Try to find section followed by content until next section or end
    pattern = rf"{section_name}:\s*\n?(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Also try inline version
    pattern = rf"{section_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None
