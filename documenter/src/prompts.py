"""Prompt templates for the Documenter engine.

All LLM interaction prompts are centralized here for easy review and tuning.
"""

DEDUP_CHECK = """You are a research document assistant. Given an insight and existing
document sections, determine if the insight is already represented.

INSIGHT:
{insight_text}

EXISTING SECTIONS:
{sections_text}

Respond in JSON:
{{"is_duplicate": true/false, "prerequisites": ["concept_name", ...], "concepts": ["concept_name", ...]}}

- is_duplicate: true if the insight's core claim is already covered
- prerequisites: concept names this insight depends on that are NOT in existing sections
- concepts: new concept names this insight establishes
"""

DRAFT_SECTION = """You are a research document writer. Draft a section incorporating
this insight into the document.

RESEARCH GOAL: {goal}

INSIGHT: {insight_text}

CONTEXT (relevant existing sections):
{context}

Write a clear, well-structured section with:
- A descriptive title
- Content that integrates this insight into the document narrative
- References to prerequisite concepts where relevant

Respond with ONLY the section content (no metadata).
"""

REVISE_SECTION = """You are a research document editor. Revise this section based
on the annotation.

RESEARCH GOAL: {goal}

CURRENT SECTION TITLE: {section_title}
CURRENT CONTENT:
{section_content}

ANNOTATION: {annotation_content}

CONTEXT (surrounding sections):
{context}

Produce a revised version of the section that addresses the annotation.
Respond with ONLY the revised content.
"""

REVIEW_SECTION = """You are a research document reviewer. Review this section for
quality and consistency.

RESEARCH GOAL: {goal}

SECTION TITLE: {section_title}
SECTION CONTENT:
{section_content}

CONTEXT (surrounding sections):
{context}

Does this section accurately represent the findings? Is it consistent with
the current document? Suggest improvements if needed.

If no changes needed, respond with "NO_CHANGES".
Otherwise, respond with the revised content.
"""

EVALUATE_DRAFT = """Evaluate the following draft section for a research document.

RESEARCH GOAL: {goal}

DRAFT:
{draft}

Score on these criteria (0.0 - 1.0):
{criteria}

Respond in JSON:
{{"verdict": "accept" or "reject", "scores": {{"criterion_name": score}}, "reasoning": "..."}}
"""
