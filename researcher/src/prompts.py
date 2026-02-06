"""Prompt templates for the Researcher Engine.

All prompts are parameterized templates. No hardcoded domain fallbacks.
"""

from __future__ import annotations

QUESTION_GENERATION = """Given this insight from a research project, generate questions \
that would help find external evidence to support, refute, or extend it.

PROJECT GOAL: {project_goal}
PROJECT CONTEXT: {project_context}
INSIGHT: {insight_text}
RESEARCH DOMAINS: {formatted_domains}

Generate up to {max_questions} specific, searchable questions.

Return a JSON array of strings, e.g. ["Question 1?", "Question 2?"].
Return ONLY the JSON array, no other text."""

QUESTION_GENERATION_FALLBACK = """Generate research questions about the following \
claim in the context of a research project.

PROJECT GOAL: {project_goal}
CLAIM: {insight_text}

Generate up to {max_questions} specific, searchable questions.

Return a JSON array of strings. Return ONLY the JSON array, no other text."""

DIRECTED_QUESTION_GENERATION = """Generate research questions for a directed \
research request.

PROJECT GOAL: {project_goal}
TOPIC: {topic}
CONTEXT: {context}
RESEARCH DOMAINS: {formatted_domains}

Generate up to {max_questions} specific, searchable questions.

Return a JSON array of strings. Return ONLY the JSON array, no other text."""

SEARCH_SIMULATION = """You are simulating a web search for a research project.
Given the research question, generate plausible search results that would be \
found on the internet.

QUESTION: {question}
RESEARCH DOMAINS: {formatted_domains}

Generate up to {max_results} results. For each result, provide:
- url: a plausible URL
- title: the page title
- snippet: a brief excerpt
- domain: the research domain it belongs to (or null)

Return a JSON array of objects. Return ONLY the JSON array, no other text."""

TRUST_EVALUATION = """Evaluate the trustworthiness of this source for a research project.

PROJECT GOAL: {project_goal}
SOURCE URL: {source_url}
SOURCE TITLE: {source_title}
CONTENT SUMMARY: {content_summary}

Rate on:
- Authority: Is the author/publisher credible in this domain?
- Accuracy: Does the content appear factually sound?
- Relevance: How relevant is this to the research goal?
- Recency: Is the information current?

Overall trust score: 0-100
Verdict: accept (trustworthy) / reject (not trustworthy) / uncertain

Return JSON: {{"score": <int>, "verdict": "<accept|reject|uncertain>", \
"reasoning": "<brief explanation>"}}
Return ONLY the JSON object, no other text."""

FINDING_EXTRACTION = """Given this source and the related insight, extract specific findings.

PROJECT GOAL: {project_goal}
INSIGHT: {insight_text}
SOURCE TITLE: {source_title}
SOURCE CONTENT: {content}

For each finding provide:
- summary: what was found
- confidence: 0.0 to 1.0
- type: "supports", "refutes", or "extends"

Extract up to {max_findings} findings.

Return a JSON array of objects. Return ONLY the JSON array, no other text."""
