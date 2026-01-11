"""
Atomic chunking module.

Extracts multiple atomic insights (1-3 sentences each) from exploration threads,
with dependency tracking to blessed axioms.

Panel-based extraction uses all 3 LLMs to propose insights independently,
then consolidates to capture diverse perspectives and cross-domain bridges.
"""

from .models import (
    AtomicInsight,
    InsightStatus,
    InsightVersion,
    Refinement,
    VersionedInsight,
)
from .extractor import AtomicExtractor
from .panel_extractor import PanelExtractor, get_panel_extractor
from .dependencies import resolve_dependencies, find_keyword_matches
from .prompts import (
    build_refinement_prompt,
    build_post_refinement_review_prompt,
    parse_refinement_response,
    parse_post_refinement_review,
)
# Use shared deduplication module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.deduplication import (
    DeduplicationChecker,
    get_dedup_checker,
    is_similar_heuristic as is_likely_duplicate,  # Backward-compatible alias
    calculate_keyword_similarity,  # Backward-compatible function
)

__all__ = [
    # Models
    "AtomicInsight",
    "InsightStatus",
    "InsightVersion",
    "Refinement",
    "VersionedInsight",
    # Extractors
    "AtomicExtractor",
    "PanelExtractor",
    "get_panel_extractor",
    # Dependencies
    "resolve_dependencies",
    "find_keyword_matches",
    # Deduplication
    "DeduplicationChecker",
    "get_dedup_checker",
    "is_likely_duplicate",
    "calculate_keyword_similarity",
    # Refinement prompts
    "build_refinement_prompt",
    "build_post_refinement_review_prompt",
    "parse_refinement_response",
    "parse_post_refinement_review",
]
