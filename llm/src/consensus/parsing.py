"""Response parsing utilities for the consensus engine.

Multi-strategy parser: tries JSON block first, then regex, then gives up.
"""

from __future__ import annotations

import json
import re

from shared.models import ParsedResponse, Verdict

_ERROR_PATTERNS = re.compile(
    r"I apologize|I cannot|I'm sorry|I am sorry|I'm unable|I am unable",
    re.IGNORECASE,
)
_JSON_BLOCK = re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.DOTALL)
_VERDICT_PATTERN = re.compile(
    r"(?:verdict|rating|decision)\s*[:\-]\s*(\S+)", re.IGNORECASE,
)
_SCORE_TEMPLATE = r"{name}\s*[:\-]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?"

_ACCEPT_TERMS = {"accept", "approve", "bless", "\u26a1"}
_REJECT_TERMS = {"reject", "discard", "\u2717"}
_UNCERTAIN_TERMS = {"uncertain", "unsure", "?"}


def normalize_verdict(raw: str) -> Verdict | None:
    """Normalize a verdict string to a Verdict enum value.

    Returns None if the string is not recognized.
    """
    cleaned = raw.strip().lower()
    if cleaned in _ACCEPT_TERMS:
        return Verdict.ACCEPT
    if cleaned in _REJECT_TERMS:
        return Verdict.REJECT
    if cleaned in _UNCERTAIN_TERMS:
        return Verdict.UNCERTAIN
    return None


def parse_review_response(text: str, criteria_names: list[str]) -> ParsedResponse:
    """Multi-strategy parser for LLM review responses.

    Strategy 1: JSON block (```json ... ```)
    Strategy 2: Regex (verdict label + score patterns)
    Strategy 3: Return invalid

    Never returns is_valid=True for empty, error, or unparseable responses.
    """
    if not text or not text.strip():
        return ParsedResponse(
            is_valid=False, verdict=None, scores={},
            reasoning=None, error="Empty response",
        )
    if _ERROR_PATTERNS.search(text):
        return ParsedResponse(
            is_valid=False, verdict=None, scores={},
            reasoning=None, error="Error message detected",
        )

    for strategy in (_try_json, _try_regex):
        result = strategy(text, criteria_names)
        if result is not None:
            return result

    return ParsedResponse(
        is_valid=False, verdict=None, scores={},
        reasoning=None, error="Could not parse verdict",
    )


def _try_json(text: str, criteria_names: list[str]) -> ParsedResponse | None:
    """Parse a ```json``` block from the response."""
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None

    verdict = normalize_verdict(str(data.get("verdict", "")))
    if verdict is None:
        return None

    scores: dict[str, float] = {}
    raw_scores = data.get("scores", {})
    if isinstance(raw_scores, dict):
        for name in criteria_names:
            if name in raw_scores:
                scores[name] = float(raw_scores[name])

    return ParsedResponse(
        is_valid=True, verdict=verdict, scores=scores,
        reasoning=str(data.get("reasoning", "")), error=None,
    )


def _try_regex(text: str, criteria_names: list[str]) -> ParsedResponse | None:
    """Extract verdict and scores via regex patterns."""
    verdict_match = _VERDICT_PATTERN.search(text)
    if not verdict_match:
        return None
    verdict = normalize_verdict(verdict_match.group(1))
    if verdict is None:
        return None

    scores: dict[str, float] = {}
    for name in criteria_names:
        pat = re.compile(_SCORE_TEMPLATE.format(name=re.escape(name)), re.IGNORECASE)
        m = pat.search(text)
        if m:
            scores[name] = float(m.group(1))

    return ParsedResponse(
        is_valid=True, verdict=verdict, scores=scores,
        reasoning=text.strip(), error=None,
    )
