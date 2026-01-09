"""
Insight Deduplication Module.

Detects semantic duplicates among insights using:
1. Fast keyword-based similarity (pre-filter)
2. LLM-based semantic comparison (confirmation)

This prevents near-duplicate insights like:
- "The 14 Maheshwara Sutras map to the 14 vertices of the Heawood graph..."
- "The 14 Maheshwar Sutras correspond to the 14 vertices of the Heawood graph..."
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Common mathematical/domain stop words to ignore
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "also", "now", "and", "but", "or", "if", "because", "until", "while",
    "this", "that", "these", "those", "which", "who", "whom", "whose",
    "what", "it", "its", "itself", "they", "them", "their", "we", "us",
    "our", "you", "your", "he", "him", "his", "she", "her", "i", "me", "my",
    # Domain-specific common words to reduce noise
    "structure", "structures", "relationship", "relationships",
    "connection", "connections", "represents", "represent",
    "corresponds", "correspond", "maps", "map", "mapping",
    "shows", "show", "demonstrates", "demonstrate",
    "exactly", "precisely", "naturally", "inherently",
}


@dataclass
class SimilarityResult:
    """Result of comparing two insights for similarity."""
    insight1_id: str
    insight2_id: str
    keyword_similarity: float  # 0.0 to 1.0
    is_duplicate: bool
    llm_confirmed: bool = False
    llm_explanation: str = ""


def normalize_word(word: str) -> str:
    """
    Normalize a word by removing common suffixes (simple stemming).

    This helps match "faces" with "face", "vertices" with "vertex", etc.
    """
    word = word.lower()

    # Handle special cases first
    if word == "vertices":
        return "vertex"
    if word == "indices":
        return "index"

    # Common suffix removals (simple stemming)
    suffixes = [
        ("ies", "y"),      # symmetries -> symmetry
        ("es", "e"),       # faces -> face (but keep the 'e')
        ("es", ""),        # edges -> edge... wait, this removes too much
        ("s", ""),         # points -> point
        ("ing", ""),       # mapping -> map
        ("ed", ""),        # connected -> connect
        ("tion", "t"),     # connection -> connect
        ("sion", "s"),     # expression -> express
    ]

    # Only apply suffixes that make sense
    if word.endswith("ices"):  # vertices, indices
        return word[:-4] + "ex"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        # edges -> edge, faces -> face
        return word[:-1]  # Just remove the 's', keep 'e'
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]

    return word


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from insight text.

    Returns normalized keywords (lowercase, no punctuation, stemmed).
    """
    # Normalize text
    text = text.lower()

    # Extract words (including numbers and hyphenated terms)
    words = re.findall(r'\b[\w-]+\b', text)

    # Filter stop words and very short words, then normalize
    keywords = set()
    for w in words:
        if w in STOP_WORDS or len(w) <= 2:
            continue
        if w.isdigit():
            continue  # Numbers handled separately

        # Normalize and add
        normalized = normalize_word(w)
        if len(normalized) > 2:
            keywords.add(normalized)

    # Also extract number-word combinations (e.g., "14", "7")
    numbers = re.findall(r'\b\d+\b', text)
    keywords.update(numbers)

    return keywords


def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity based on keyword overlap.

    Returns value between 0.0 (no overlap) and 1.0 (identical).
    """
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)

    if not keywords1 or not keywords2:
        return 0.0

    intersection = keywords1 & keywords2
    union = keywords1 | keywords2

    return len(intersection) / len(union)


def calculate_concept_overlap(text1: str, text2: str) -> float:
    """
    Calculate overlap of key mathematical/structural concepts.

    More sensitive to domain-specific terms than generic Jaccard.
    """
    # Key concept patterns to look for
    concept_patterns = [
        r'\b\d+\b',  # Numbers (7, 14, 168, etc.)
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns (Fano, Heawood, Klein, etc.)
        # Geometry/graph terms
        r'\b(?:point|line|vertex|vertices|edge|edges|face|faces|graph|plane|tiling|quartic|hyperbolic)\w*\b',
        # Group theory terms
        r'\b(?:group|orbit|stabilizer|subgroup|automorphism|isomorphism|symmetry|psl|order)\w*\b',
        # Structure/relationship terms
        r'\b(?:incidence|duality|correspondence|bijection|mapping|homeomorphism)\w*\b',
        # Sanskrit/music/yoga terms
        r'\b(?:sanskrit|phonetic|grammar|linguistic|sutra|sutras|swara|shruti|raga|chakra|nadi)\w*\b',
        # Mathematical object names
        r'\b(?:fano|heawood|klein|petersen|steiner|galois|mobius|riemann)\b',
    ]

    def extract_concepts(text: str) -> set[str]:
        concepts = set()
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                # Normalize the concept
                normalized = normalize_word(m.lower())
                concepts.add(normalized)
        return concepts

    concepts1 = extract_concepts(text1)
    concepts2 = extract_concepts(text2)

    if not concepts1 or not concepts2:
        return 0.0

    intersection = concepts1 & concepts2
    union = concepts1 | concepts2

    return len(intersection) / len(union)


def is_likely_duplicate(
    text1: str,
    text2: str,
    keyword_threshold: float = 0.6,
    concept_threshold: float = 0.7,
) -> tuple[bool, float]:
    """
    Quick check if two texts are likely duplicates.

    Args:
        text1: First insight text
        text2: Second insight text
        keyword_threshold: Minimum keyword similarity to flag
        concept_threshold: Minimum concept overlap to flag

    Returns:
        Tuple of (is_likely_duplicate, combined_score)
    """
    keyword_sim = calculate_keyword_similarity(text1, text2)
    concept_sim = calculate_concept_overlap(text1, text2)

    # Weight concept overlap more heavily (domain-specific)
    combined = (keyword_sim * 0.4) + (concept_sim * 0.6)

    is_likely = (
        keyword_sim >= keyword_threshold or
        concept_sim >= concept_threshold or
        combined >= 0.65
    )

    return is_likely, combined


async def confirm_duplicate_with_llm(
    insight1: str,
    insight2: str,
    claude_reviewer,
) -> tuple[bool, str]:
    """
    Use LLM to confirm if two insights are semantic duplicates.

    Args:
        insight1: First insight text
        insight2: Second insight text
        claude_reviewer: ClaudeReviewer instance for API calls

    Returns:
        Tuple of (is_duplicate, explanation)
    """
    prompt = f"""Compare these two mathematical insights and determine if they express the SAME core idea (even if worded differently).

INSIGHT A:
{insight1}

INSIGHT B:
{insight2}

Analyze:
1. Do they make the same fundamental claim?
2. Do they reference the same mathematical structures/relationships?
3. Would accepting both as separate insights be redundant?

Respond with:
DUPLICATE: [yes/no]
EXPLANATION: [1-2 sentences explaining your decision]
WHICH_IS_BETTER: [A/B/equal] (if duplicate, which wording is clearer/more precise)"""

    try:
        response = await claude_reviewer.send_message(prompt, extended_thinking=False)

        # Parse response
        is_duplicate = False
        explanation = ""
        better = "equal"

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("DUPLICATE:"):
                value = line.replace("DUPLICATE:", "").strip().lower()
                is_duplicate = value in ("yes", "true", "y")
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("WHICH_IS_BETTER:"):
                better = line.replace("WHICH_IS_BETTER:", "").strip().lower()

        if is_duplicate:
            explanation += f" (Better wording: {better})"

        return is_duplicate, explanation

    except Exception as e:
        logger.warning(f"LLM duplicate check failed: {e}")
        return False, f"LLM check failed: {e}"


async def batch_check_duplicates_with_llm(
    new_insight: str,
    existing_insights: list[dict],
    claude_reviewer,
    max_per_batch: int = 15,
) -> tuple[bool, Optional[str], str]:
    """
    Efficiently check a new insight against multiple existing insights in one LLM call.

    Args:
        new_insight: The new insight text to check
        existing_insights: List of dicts with 'id' and 'text' keys
        claude_reviewer: ClaudeReviewer instance
        max_per_batch: Maximum insights to compare in one call

    Returns:
        Tuple of (is_duplicate, duplicate_of_id or None, explanation)
    """
    if not existing_insights:
        return False, None, "No existing insights to compare"

    # Take most recent insights first (more likely to be similar)
    insights_to_check = existing_insights[:max_per_batch]

    # Build the comparison prompt
    existing_list = "\n".join([
        f"[{i+1}] {ins['text'][:200]}{'...' if len(ins['text']) > 200 else ''}"
        for i, ins in enumerate(insights_to_check)
    ])

    prompt = f"""You are checking if a NEW insight is a semantic duplicate of any EXISTING insight.

NEW INSIGHT:
{new_insight}

EXISTING INSIGHTS:
{existing_list}

A duplicate means the NEW insight expresses the SAME core mathematical claim as an existing one, even if:
- Different words are used
- One adds minor elaboration to the same idea
- The wording is rearranged

If the NEW insight makes a DISTINCT claim (even if related), it is NOT a duplicate.

Respond with EXACTLY:
IS_DUPLICATE: [yes/no]
DUPLICATE_OF: [number 1-{len(insights_to_check)} or 'none']
REASON: [one sentence explanation]"""

    try:
        response = await claude_reviewer.send_message(prompt, extended_thinking=False)

        is_duplicate = False
        duplicate_of_num = None
        reason = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("IS_DUPLICATE:"):
                value = line.replace("IS_DUPLICATE:", "").strip().lower()
                is_duplicate = value in ("yes", "true", "y")
            elif line.startswith("DUPLICATE_OF:"):
                value = line.replace("DUPLICATE_OF:", "").strip().lower()
                if value != "none":
                    try:
                        duplicate_of_num = int(re.search(r'\d+', value).group()) - 1
                    except (ValueError, AttributeError):
                        pass
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        if is_duplicate and duplicate_of_num is not None and 0 <= duplicate_of_num < len(insights_to_check):
            duplicate_id = insights_to_check[duplicate_of_num]["id"]
            return True, duplicate_id, reason
        elif is_duplicate:
            # Duplicate detected but couldn't identify which one
            return True, None, reason

        return False, None, reason

    except Exception as e:
        logger.warning(f"Batch LLM duplicate check failed: {e}")
        return False, None, f"LLM check failed: {e}"


class DeduplicationChecker:
    """
    Checks new insights against existing ones for duplicates.

    Uses a two-stage approach:
    1. Quick pre-filter with keyword/concept similarity (low threshold to catch more)
    2. LLM batch comparison for semantic duplicate detection
    """

    def __init__(
        self,
        claude_reviewer=None,
        keyword_threshold: float = 0.25,  # Lowered to let more through to LLM
        concept_threshold: float = 0.30,  # Lowered to let more through to LLM
        use_llm_confirmation: bool = True,
        use_batch_llm: bool = True,  # Use efficient batch checking
    ):
        """
        Initialize the deduplication checker.

        Args:
            claude_reviewer: ClaudeReviewer for LLM confirmation (optional)
            keyword_threshold: Threshold for keyword similarity flagging (lowered)
            concept_threshold: Threshold for concept overlap flagging (lowered)
            use_llm_confirmation: Whether to confirm with LLM
            use_batch_llm: Use batch LLM check against all known insights
        """
        self.claude = claude_reviewer
        self.keyword_threshold = keyword_threshold
        self.concept_threshold = concept_threshold
        self.use_llm = use_llm_confirmation and claude_reviewer is not None
        self.use_batch_llm = use_batch_llm

        # Cache of known insights (id -> text)
        self._known_insights: dict[str, str] = {}
        # Ordered list for batch checking
        self._known_insights_list: list[dict] = []

        logger.info(
            f"[dedup] Initialized (keyword_thresh={keyword_threshold}, "
            f"concept_thresh={concept_threshold}, llm={self.use_llm}, batch={use_batch_llm})"
        )

    def add_known_insight(self, insight_id: str, text: str):
        """Add an insight to the known set for future comparisons."""
        if insight_id not in self._known_insights:
            self._known_insights[insight_id] = text
            self._known_insights_list.append({"id": insight_id, "text": text})

    def load_known_insights(self, insights: list[dict]):
        """
        Load multiple known insights.

        Args:
            insights: List of dicts with 'id' and 'text' keys
        """
        for insight in insights:
            self.add_known_insight(insight["id"], insight["text"])
        logger.info(f"[dedup] Loaded {len(self._known_insights)} known insights")

    async def find_duplicates(
        self,
        new_text: str,
        new_id: str = "new",
    ) -> list[SimilarityResult]:
        """
        Find potential duplicates of a new insight among known insights.

        Args:
            new_text: The new insight text to check
            new_id: ID of the new insight (for logging)

        Returns:
            List of SimilarityResult for each duplicate found
        """
        duplicates = []

        for known_id, known_text in self._known_insights.items():
            # Quick pre-filter
            is_likely, score = is_likely_duplicate(
                new_text,
                known_text,
                self.keyword_threshold,
                self.concept_threshold,
            )

            if not is_likely:
                continue

            logger.info(f"[dedup] Potential duplicate: {new_id} <-> {known_id} (score={score:.2f})")

            # LLM confirmation if enabled
            llm_confirmed = False
            llm_explanation = ""

            if self.use_llm and self.claude:
                llm_confirmed, llm_explanation = await confirm_duplicate_with_llm(
                    new_text,
                    known_text,
                    self.claude,
                )
                logger.info(f"[dedup] LLM confirmation: {llm_confirmed} - {llm_explanation}")

            # Consider it a duplicate if LLM confirms, or if no LLM and score is high
            is_duplicate = llm_confirmed or (not self.use_llm and score >= 0.75)

            if is_duplicate:
                duplicates.append(SimilarityResult(
                    insight1_id=new_id,
                    insight2_id=known_id,
                    keyword_similarity=score,
                    is_duplicate=True,
                    llm_confirmed=llm_confirmed,
                    llm_explanation=llm_explanation,
                ))

        return duplicates

    async def is_duplicate(self, new_text: str, new_id: str = "new") -> tuple[bool, Optional[str]]:
        """
        Check if a new insight is a duplicate of any known insight.

        Uses batch LLM checking for efficiency when enabled.

        Args:
            new_text: The new insight text
            new_id: ID of the new insight

        Returns:
            Tuple of (is_duplicate, duplicate_of_id or None)
        """
        # If batch LLM mode is enabled and we have Claude, use efficient batch check
        if self.use_batch_llm and self.claude and self._known_insights_list:
            logger.info(f"[dedup] Batch checking {new_id} against {len(self._known_insights_list)} known insights")
            is_dup, dup_id, reason = await batch_check_duplicates_with_llm(
                new_text,
                self._known_insights_list,
                self.claude,
                max_per_batch=20,  # Check up to 20 at once
            )
            if is_dup:
                logger.info(f"[dedup] DUPLICATE FOUND: {new_id} duplicates {dup_id} - {reason}")
                return True, dup_id
            else:
                logger.info(f"[dedup] No duplicate found for {new_id}")
                return False, None

        # Fallback to original method (pre-filter + individual LLM checks)
        duplicates = await self.find_duplicates(new_text, new_id)

        if duplicates:
            # Return the first (highest confidence) duplicate
            dup = duplicates[0]
            return True, dup.insight2_id

        return False, None

    def clear(self):
        """Clear all known insights."""
        self._known_insights.clear()
        self._known_insights_list.clear()


def get_dedup_checker(
    claude_reviewer=None,
    config: dict = None,
) -> DeduplicationChecker:
    """
    Factory function to create a DeduplicationChecker.

    Args:
        claude_reviewer: ClaudeReviewer instance
        config: Configuration dict (optional)

    Returns:
        Configured DeduplicationChecker
    """
    config = config or {}
    dedup_config = config.get("deduplication", {})

    return DeduplicationChecker(
        claude_reviewer=claude_reviewer,
        keyword_threshold=dedup_config.get("keyword_threshold", 0.25),  # Lower default
        concept_threshold=dedup_config.get("concept_threshold", 0.30),  # Lower default
        use_llm_confirmation=dedup_config.get("use_llm_confirmation", True),
        use_batch_llm=dedup_config.get("use_batch_llm", True),  # Enable by default
    )
