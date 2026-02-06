"""Prompt template functions for the Explorer Engine.

All prompts are built from project config. Zero hardcoded domain content.
"""

from __future__ import annotations

from shared.models import Exchange, ExchangeRole, Project, Seed, Thread

ROLE_DESCRIPTIONS: dict[ExchangeRole, str] = {
    ExchangeRole.EXPLORER: (
        "Build on this conversation. Develop the idea further. "
        "Follow your curiosity. Propose novel connections."
    ),
    ExchangeRole.CRITIC: (
        "Challenge the reasoning. Find weaknesses. Push for rigor. "
        "Is this inevitable or forced?"
    ),
    ExchangeRole.SYNTHESIZER: (
        "Identify the core insight emerging. What's the atomic claim? "
        "What needs more development?"
    ),
}


def _format_criteria(project: Project) -> str:
    parts = []
    for c in project.evaluation_criteria:
        parts.append(f"- {c.name} (weight {c.weight}): {c.description}")
    return "\n".join(parts)


def _format_exchanges(exchanges: list[Exchange]) -> str:
    parts = []
    for ex in exchanges:
        parts.append(f"[{ex.role.value.upper()} — {ex.model}]:\n{ex.response}")
    return "\n\n".join(parts)


def build_exchange_prompt(
    project: Project,
    thread: Thread,
    exchanges: list[Exchange],
    role: ExchangeRole,
    seed: Seed,
) -> str:
    """Build prompt for one exploration exchange."""
    sections = [
        f"SYSTEM CONTEXT: {project.context}",
        f"RESEARCH GOAL: {project.goal}",
        f"EVALUATION CRITERIA:\n{_format_criteria(project)}",
        f"GUIDANCE: {project.exploration_guidance}",
        f"SEED BEING EXPLORED: {seed.text}",
    ]
    if exchanges:
        sections.append(f"CONVERSATION SO FAR:\n{_format_exchanges(exchanges)}")
    sections.append(f"YOUR ROLE: {ROLE_DESCRIPTIONS[role]}")
    sections.append("RESPOND:")
    return "\n\n".join(sections)


def build_extraction_prompt(
    project: Project,
    exchanges: list[Exchange],
) -> str:
    """Build prompt for extracting atomic insights from a thread."""
    return (
        "Given this research conversation, extract the atomic insights — claims "
        "that can be independently evaluated. For each insight, specify "
        "confidence (high/medium/low) and relevant tags.\n\n"
        f"PROJECT GOAL: {project.goal}\n\n"
        f"CONVERSATION:\n{_format_exchanges(exchanges)}\n\n"
        "Format each insight as:\n"
        "INSIGHT N: <concise, self-contained claim>\n"
        "CONFIDENCE: high|medium|low\n"
        "TAGS: tag1, tag2, ..."
    )
