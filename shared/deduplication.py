"""
Robust Deduplication Module for Fano.

Provides multi-layer duplicate detection that works across modules:
- Explorer: Check insights against each other
- Documenter: Check insights against document sections

Detection layers (in order of speed):
1. Content signatures (MD5 hash of normalized text) - instant
2. Keyword/concept overlap (Jaccard similarity) - fast
3. N-gram similarity (for partial matches) - fast
4. LLM semantic confirmation - thorough but slower

Usage:
    from shared.deduplication import DeduplicationChecker, ContentItem

    # Create checker with LLM callback
    async def llm_check(prompt: str) -> str:
        response = await llm_client.send("claude", prompt)
        return response.text

    checker = DeduplicationChecker(llm_callback=llm_check)

    # Add known content
    checker.add_content(ContentItem(id="section_1", text="...", content_type="section"))

    # Check for duplicates
    result = await checker.check_duplicate("new insight text...")
    if result.is_duplicate:
        print(f"Duplicates: {result.duplicate_of}")
"""

import hashlib
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Awaitable, Any

# Add shared module to path for logging
SHARED_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

log = get_logger("shared", "deduplication")


# =============================================================================
# Data Classes
# =============================================================================


class ContentType(Enum):
    """Types of content that can be deduplicated."""
    INSIGHT = "insight"
    SECTION = "section"
    PREREQUISITE = "prerequisite"
    COMMENT = "comment"
    UNKNOWN = "unknown"


@dataclass
class ContentItem:
    """
    A piece of content to check for duplicates.

    This is the unified representation used across all modules.
    """
    id: str
    text: str
    content_type: ContentType = ContentType.UNKNOWN
    created_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    # Computed on first access
    _signature: Optional[str] = field(default=None, repr=False)
    _keywords: Optional[set] = field(default=None, repr=False)
    _concepts: Optional[set] = field(default=None, repr=False)
    _ngrams: Optional[set] = field(default=None, repr=False)

    def __post_init__(self):
        # Normalize content_type if passed as string
        if isinstance(self.content_type, str):
            try:
                self.content_type = ContentType(self.content_type)
            except ValueError:
                self.content_type = ContentType.UNKNOWN

    @property
    def signature(self) -> str:
        """Get content signature (MD5 of normalized text)."""
        if self._signature is None:
            self._signature = compute_signature(self.text)
        return self._signature

    @property
    def keywords(self) -> set[str]:
        """Get extracted keywords."""
        if self._keywords is None:
            self._keywords = extract_keywords(self.text)
        return self._keywords

    @property
    def concepts(self) -> set[str]:
        """Get extracted domain concepts."""
        if self._concepts is None:
            self._concepts = extract_concepts(self.text)
        return self._concepts

    @property
    def ngrams(self) -> set[str]:
        """Get character n-grams for fuzzy matching."""
        if self._ngrams is None:
            self._ngrams = extract_ngrams(self.text, n=4)
        return self._ngrams


@dataclass
class SimilarityScore:
    """Detailed similarity scores between two content items."""
    signature_match: bool = False  # Exact normalized match
    keyword_similarity: float = 0.0  # Jaccard on keywords
    concept_similarity: float = 0.0  # Jaccard on domain concepts
    ngram_similarity: float = 0.0  # Jaccard on character n-grams
    combined_score: float = 0.0  # Weighted combination
    llm_confirmed: Optional[bool] = None  # LLM semantic check result
    llm_explanation: str = ""

    @property
    def passed_heuristics(self) -> bool:
        """Whether the heuristic checks suggest a duplicate."""
        return (
            self.signature_match or
            self.keyword_similarity >= 0.50 or
            self.concept_similarity >= 0.55 or
            self.combined_score >= 0.55
        )


@dataclass
class DuplicateResult:
    """Result of a duplicate check."""
    is_duplicate: bool
    checked_item_id: str
    duplicate_of: Optional[str] = None  # ID of the matching content
    similarity: Optional[SimilarityScore] = None
    check_method: str = "none"  # "signature", "heuristic", "llm", "batch_llm"
    reason: str = ""
    check_time_ms: float = 0.0


# =============================================================================
# Text Processing Functions
# =============================================================================


# Common stop words to ignore in keyword extraction
STOP_WORDS = frozenset({
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
    # Domain-specific common filler words
    "structure", "structures", "relationship", "relationships",
    "connection", "connections", "represents", "represent",
    "corresponds", "correspond", "maps", "map", "mapping",
    "shows", "show", "demonstrates", "demonstrate",
    "exactly", "precisely", "naturally", "inherently",
})


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove punctuation (except hyphens in words)
    - Collapse multiple spaces
    """
    text = text.lower()
    # Keep alphanumeric, spaces, and hyphens within words
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_signature(text: str) -> str:
    """
    Compute a content signature for fast duplicate detection.

    Uses MD5 of the first 500 chars of normalized text.
    """
    normalized = normalize_text(text)[:500]
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def stem_word(word: str) -> str:
    """
    Simple stemming for keyword normalization.

    Handles common English suffixes and mathematical plural forms.
    """
    word = word.lower()

    # Special cases for mathematical terms
    if word == "vertices":
        return "vertex"
    if word == "indices":
        return "index"
    if word == "matrices":
        return "matrix"

    # Common suffix handling
    if word.endswith("ices") and len(word) > 5:
        return word[:-4] + "ex"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-1]  # Keep the 'e' (edges -> edge)
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]

    return word


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from text.

    Returns normalized, stemmed keywords excluding stop words.
    """
    text = text.lower()
    words = re.findall(r'\b[\w-]+\b', text)

    keywords = set()
    for word in words:
        if word in STOP_WORDS or len(word) <= 2:
            continue
        if word.isdigit():
            # Keep numbers as-is (7, 14, 168 are significant)
            keywords.add(word)
            continue

        stemmed = stem_word(word)
        if len(stemmed) > 2:
            keywords.add(stemmed)

    return keywords


def extract_concepts(text: str) -> set[str]:
    """
    Extract domain-specific mathematical/structural concepts.

    More targeted than keywords - focuses on significant terms.
    """
    # Patterns for important domain concepts
    patterns = [
        r'\b\d+\b',  # Numbers (7, 14, 168)
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
        # Geometry terms
        r'\b(?:point|line|vertex|vertices|edge|face|graph|plane|'
        r'tiling|quartic|hyperbolic|projective|affine|euclidean)\w*\b',
        # Group theory
        r'\b(?:group|orbit|stabilizer|subgroup|automorphism|isomorphism|'
        r'symmetry|psl|order|cyclic|abelian|permutation)\w*\b',
        # Structure/relationship
        r'\b(?:incidence|duality|correspondence|bijection|homeomorphism|'
        r'isogeny|morphism|functor|embedding)\w*\b',
        # Sanskrit/music/yoga (for this specific project)
        r'\b(?:sanskrit|phonetic|grammar|sutra|sutras|swara|shruti|'
        r'raga|chakra|nadi|maheshwara)\w*\b',
        # Named mathematical objects
        r'\b(?:fano|heawood|klein|petersen|steiner|galois|mobius|'
        r'riemann|poincare|hilbert|octonion|quaternion)\b',
    ]

    concepts = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = stem_word(match.lower())
            if len(normalized) > 1:
                concepts.add(normalized)

    return concepts


def extract_ngrams(text: str, n: int = 4) -> set[str]:
    """
    Extract character n-grams for fuzzy matching.

    N-grams capture local text structure and help detect
    rearranged or slightly modified content.
    """
    normalized = normalize_text(text)
    # Remove spaces for character n-grams
    chars = normalized.replace(' ', '')

    if len(chars) < n:
        return {chars} if chars else set()

    return {chars[i:i+n] for i in range(len(chars) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def calculate_similarity(item1: ContentItem, item2: ContentItem) -> SimilarityScore:
    """
    Calculate comprehensive similarity between two content items.

    Uses multiple signals for robustness.
    """
    score = SimilarityScore()

    # 1. Signature match (exact normalized content)
    score.signature_match = item1.signature == item2.signature

    # 2. Keyword similarity
    score.keyword_similarity = jaccard_similarity(item1.keywords, item2.keywords)

    # 3. Concept similarity (more domain-specific)
    score.concept_similarity = jaccard_similarity(item1.concepts, item2.concepts)

    # 4. N-gram similarity (fuzzy match)
    score.ngram_similarity = jaccard_similarity(item1.ngrams, item2.ngrams)

    # 5. Combined weighted score
    # Weight concepts more heavily (domain-specific), n-grams less (noise)
    score.combined_score = (
        score.keyword_similarity * 0.25 +
        score.concept_similarity * 0.45 +
        score.ngram_similarity * 0.30
    )

    return score


# =============================================================================
# LLM Confirmation
# =============================================================================


def build_pairwise_llm_prompt(text1: str, text2: str) -> str:
    """Build prompt for pairwise LLM duplicate confirmation."""
    return f"""Compare these two pieces of mathematical content and determine if they express the SAME core idea.

CONTENT A:
{text1[:1000]}

CONTENT B:
{text2[:1000]}

A duplicate means they make the SAME fundamental claim, even if:
- Different words are used
- One has more detail than the other
- The structure/ordering differs

NOT a duplicate if they make distinct claims, even if related.

Respond EXACTLY in this format:
IS_DUPLICATE: [yes/no]
CONFIDENCE: [high/medium/low]
REASON: [one sentence explanation]
BETTER_VERSION: [A/B/equal] (if duplicate, which is clearer)"""


def build_batch_llm_prompt(
    new_text: str,
    existing_items: list[dict[str, str]],
) -> str:
    """
    Build prompt for batch LLM duplicate check.

    Args:
        new_text: The new content to check
        existing_items: List of dicts with 'id' and 'text' keys
    """
    items_text = "\n\n".join([
        f"[{i+1}] (ID: {item['id']})\n{item['text'][:300]}{'...' if len(item['text']) > 300 else ''}"
        for i, item in enumerate(existing_items)
    ])

    return f"""You are checking if NEW content is a semantic duplicate of any EXISTING content.

NEW CONTENT:
{new_text[:800]}

EXISTING CONTENT ({len(existing_items)} items):

{items_text}

A duplicate means the NEW content expresses the SAME core claim as an existing item.
Even if worded differently or with different detail level, if the core idea is the same, it's a duplicate.

If it makes a DISTINCT claim (even if related/adjacent), it is NOT a duplicate.

Respond EXACTLY in this format:
IS_DUPLICATE: [yes/no]
DUPLICATE_OF: [number 1-{len(existing_items)} or 'none']
CONFIDENCE: [high/medium/low]
REASON: [one sentence explanation]"""


def parse_llm_duplicate_response(response: str) -> tuple[bool, Optional[int], str, str]:
    """
    Parse LLM response for duplicate check.

    Returns: (is_duplicate, duplicate_index, confidence, reason)
    """
    is_duplicate = False
    duplicate_idx = None
    confidence = "low"
    reason = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("IS_DUPLICATE:"):
            value = line.replace("IS_DUPLICATE:", "").strip().lower()
            is_duplicate = value in ("yes", "true", "y")
        elif line.startswith("DUPLICATE_OF:"):
            value = line.replace("DUPLICATE_OF:", "").strip().lower()
            if value not in ("none", "n/a", "-"):
                try:
                    duplicate_idx = int(re.search(r'\d+', value).group()) - 1
                except (ValueError, AttributeError):
                    pass
        elif line.startswith("CONFIDENCE:"):
            confidence = line.replace("CONFIDENCE:", "").strip().lower()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return is_duplicate, duplicate_idx, confidence, reason


# =============================================================================
# Main Deduplication Checker
# =============================================================================


# Type alias for LLM callback
LLMCallback = Callable[[str], Awaitable[str]]


class DeduplicationChecker:
    """
    Multi-layer deduplication checker.

    Checks new content against known content using multiple strategies:
    1. Signature matching (instant, catches exact duplicates)
    2. Heuristic similarity (fast, catches near-duplicates)
    3. LLM semantic check (thorough, catches semantic duplicates)

    The checker is designed to be:
    - LLM-agnostic: Works with any LLM via a simple callback
    - Efficient: Fast heuristics filter before expensive LLM calls
    - Configurable: Thresholds and layers can be adjusted
    - Shared: Used by both explorer and documenter modules
    """

    def __init__(
        self,
        llm_callback: Optional[LLMCallback] = None,
        *,
        # Thresholds (lower = more aggressive matching)
        keyword_threshold: float = 0.40,
        concept_threshold: float = 0.45,
        combined_threshold: float = 0.50,
        # Features
        use_signature_check: bool = True,
        use_heuristic_check: bool = True,
        use_llm_check: bool = True,
        use_batch_llm: bool = True,
        batch_size: int = 20,
        # LLM confidence requirements
        require_high_confidence: bool = False,
    ):
        """
        Initialize the deduplication checker.

        Args:
            llm_callback: Async function that takes a prompt and returns LLM response text.
                         If None, LLM checking is disabled.
            keyword_threshold: Minimum keyword Jaccard to flag as potential duplicate
            concept_threshold: Minimum concept Jaccard to flag as potential duplicate
            combined_threshold: Minimum combined score to flag as potential duplicate
            use_signature_check: Enable exact signature matching
            use_heuristic_check: Enable keyword/concept/ngram heuristics
            use_llm_check: Enable LLM semantic confirmation
            use_batch_llm: Use batch LLM checking (more efficient)
            batch_size: Max items per batch LLM call
            require_high_confidence: Only accept LLM duplicates with high confidence
        """
        self.llm_callback = llm_callback
        self.keyword_threshold = keyword_threshold
        self.concept_threshold = concept_threshold
        self.combined_threshold = combined_threshold
        self.use_signature_check = use_signature_check
        self.use_heuristic_check = use_heuristic_check
        self.use_llm_check = use_llm_check and llm_callback is not None
        self.use_batch_llm = use_batch_llm
        self.batch_size = batch_size
        self.require_high_confidence = require_high_confidence

        # Known content registry
        self._known_items: dict[str, ContentItem] = {}
        self._signatures: dict[str, str] = {}  # signature -> item_id

        # Statistics
        self._stats = {
            "checks": 0,
            "duplicates_found": 0,
            "by_signature": 0,
            "by_heuristic": 0,
            "by_llm": 0,
        }

        log.info(
            "deduplication.checker.initialized",
            use_llm=self.use_llm_check,
            use_batch=self.use_batch_llm,
            thresholds={
                "keyword": keyword_threshold,
                "concept": concept_threshold,
                "combined": combined_threshold,
            },
        )

    def add_content(self, item: ContentItem) -> None:
        """
        Add content to the known registry.

        Args:
            item: ContentItem to add
        """
        if item.id in self._known_items:
            log.debug("deduplication.content.already_known", item_id=item.id)
            return

        self._known_items[item.id] = item
        self._signatures[item.signature] = item.id

        log.debug(
            "deduplication.content.added",
            item_id=item.id,
            content_type=item.content_type.value,
            keyword_count=len(item.keywords),
            concept_count=len(item.concepts),
        )

    def add_contents(self, items: list[ContentItem]) -> None:
        """Add multiple content items."""
        for item in items:
            self.add_content(item)

    def load_from_dicts(
        self,
        items: list[dict],
        content_type: ContentType = ContentType.UNKNOWN,
    ) -> None:
        """
        Load content from list of dicts.

        Expected dict format: {"id": str, "text": str, ...}
        """
        for item_dict in items:
            item = ContentItem(
                id=item_dict.get("id", str(len(self._known_items))),
                text=item_dict.get("text", item_dict.get("content", "")),
                content_type=content_type,
                metadata=item_dict,
            )
            self.add_content(item)

        log.info(
            "deduplication.content.loaded",
            count=len(items),
            content_type=content_type.value,
            total_known=len(self._known_items),
        )

    async def check_duplicate(
        self,
        text: str,
        item_id: str = "new",
        content_type: ContentType = ContentType.UNKNOWN,
        skip_llm: bool = False,
    ) -> DuplicateResult:
        """
        Check if text is a duplicate of any known content.

        Args:
            text: The text to check
            item_id: ID for logging/tracking
            content_type: Type of content being checked
            skip_llm: If True, skip LLM check even if enabled

        Returns:
            DuplicateResult with details of the check
        """
        import time
        start_time = time.time()

        self._stats["checks"] += 1

        # Create content item for the new text
        new_item = ContentItem(
            id=item_id,
            text=text,
            content_type=content_type,
        )

        # Layer 1: Signature check (instant)
        if self.use_signature_check:
            result = self._check_signature(new_item)
            if result.is_duplicate:
                result.check_time_ms = (time.time() - start_time) * 1000
                self._stats["duplicates_found"] += 1
                self._stats["by_signature"] += 1
                return result

        # Layer 2: Heuristic check (fast)
        if self.use_heuristic_check:
            result = self._check_heuristics(new_item)
            if result.is_duplicate:
                result.check_time_ms = (time.time() - start_time) * 1000
                self._stats["duplicates_found"] += 1
                self._stats["by_heuristic"] += 1
                return result
            # Even if not duplicate, track candidates for LLM check
            heuristic_candidates = result.similarity  # May be None or partial

        # Layer 3: LLM semantic check (thorough)
        if self.use_llm_check and not skip_llm and self._known_items:
            result = await self._check_with_llm(new_item)
            if result.is_duplicate:
                result.check_time_ms = (time.time() - start_time) * 1000
                self._stats["duplicates_found"] += 1
                self._stats["by_llm"] += 1
                return result

        # No duplicate found
        elapsed = (time.time() - start_time) * 1000
        log.debug(
            "deduplication.check.no_duplicate",
            item_id=item_id,
            known_count=len(self._known_items),
            time_ms=round(elapsed, 2),
        )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item_id,
            check_method="all",
            reason="No duplicate found",
            check_time_ms=elapsed,
        )

    def _check_signature(self, item: ContentItem) -> DuplicateResult:
        """Check for exact signature match."""
        if item.signature in self._signatures:
            dup_id = self._signatures[item.signature]
            log.info(
                "deduplication.signature.match",
                new_id=item.id,
                duplicate_of=dup_id,
            )
            return DuplicateResult(
                is_duplicate=True,
                checked_item_id=item.id,
                duplicate_of=dup_id,
                check_method="signature",
                reason="Exact content match (identical normalized text)",
            )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            check_method="signature",
        )

    def _check_heuristics(self, item: ContentItem) -> DuplicateResult:
        """Check using keyword/concept/ngram heuristics."""
        best_match: Optional[tuple[str, SimilarityScore]] = None
        best_score = 0.0

        for known_id, known_item in self._known_items.items():
            score = calculate_similarity(item, known_item)

            # Check if this is a potential duplicate
            is_potential = (
                score.keyword_similarity >= self.keyword_threshold or
                score.concept_similarity >= self.concept_threshold or
                score.combined_score >= self.combined_threshold
            )

            if is_potential and score.combined_score > best_score:
                best_score = score.combined_score
                best_match = (known_id, score)

        if best_match and best_score >= self.combined_threshold:
            dup_id, similarity = best_match
            log.info(
                "deduplication.heuristic.match",
                new_id=item.id,
                duplicate_of=dup_id,
                keyword_sim=round(similarity.keyword_similarity, 3),
                concept_sim=round(similarity.concept_similarity, 3),
                combined=round(similarity.combined_score, 3),
            )
            return DuplicateResult(
                is_duplicate=True,
                checked_item_id=item.id,
                duplicate_of=dup_id,
                similarity=similarity,
                check_method="heuristic",
                reason=f"High similarity (keyword={similarity.keyword_similarity:.2f}, "
                       f"concept={similarity.concept_similarity:.2f}, "
                       f"combined={similarity.combined_score:.2f})",
            )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            similarity=best_match[1] if best_match else None,
            check_method="heuristic",
        )

    async def _check_with_llm(self, item: ContentItem) -> DuplicateResult:
        """Check using LLM semantic comparison."""
        if not self.llm_callback:
            return DuplicateResult(
                is_duplicate=False,
                checked_item_id=item.id,
                check_method="llm_skipped",
                reason="No LLM callback configured",
            )

        if self.use_batch_llm:
            return await self._check_with_batch_llm(item)
        else:
            return await self._check_with_pairwise_llm(item)

    async def _check_with_batch_llm(self, item: ContentItem) -> DuplicateResult:
        """Efficient batch LLM check against all known content."""
        # Prepare items for batch check
        items_for_check = [
            {"id": k, "text": v.text}
            for k, v in list(self._known_items.items())[:self.batch_size]
        ]

        if not items_for_check:
            return DuplicateResult(
                is_duplicate=False,
                checked_item_id=item.id,
                check_method="batch_llm",
                reason="No known items to check against",
            )

        prompt = build_batch_llm_prompt(item.text, items_for_check)

        try:
            response = await self.llm_callback(prompt)
            is_dup, dup_idx, confidence, reason = parse_llm_duplicate_response(response)

            # Check confidence requirement
            if is_dup and self.require_high_confidence and confidence != "high":
                log.info(
                    "deduplication.llm.low_confidence",
                    new_id=item.id,
                    confidence=confidence,
                )
                is_dup = False
                reason = f"LLM match with {confidence} confidence (requires high)"

            if is_dup and dup_idx is not None and 0 <= dup_idx < len(items_for_check):
                dup_id = items_for_check[dup_idx]["id"]
                log.info(
                    "deduplication.llm.batch_match",
                    new_id=item.id,
                    duplicate_of=dup_id,
                    confidence=confidence,
                    reason=reason,
                )

                # Build similarity score with LLM confirmation
                known_item = self._known_items[dup_id]
                similarity = calculate_similarity(item, known_item)
                similarity.llm_confirmed = True
                similarity.llm_explanation = reason

                return DuplicateResult(
                    is_duplicate=True,
                    checked_item_id=item.id,
                    duplicate_of=dup_id,
                    similarity=similarity,
                    check_method="batch_llm",
                    reason=reason,
                )

            log.debug(
                "deduplication.llm.no_match",
                new_id=item.id,
                checked_count=len(items_for_check),
            )

        except Exception as e:
            log.warning(
                "deduplication.llm.error",
                new_id=item.id,
                error=str(e),
            )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            check_method="batch_llm",
            reason="No LLM-confirmed duplicate",
        )

    async def _check_with_pairwise_llm(self, item: ContentItem) -> DuplicateResult:
        """Pairwise LLM check (slower but more thorough)."""
        # First, find candidates with some heuristic similarity
        candidates = []
        for known_id, known_item in self._known_items.items():
            score = calculate_similarity(item, known_item)
            if score.combined_score >= 0.30:  # Low threshold for candidates
                candidates.append((known_id, known_item, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2].combined_score, reverse=True)

        # Check top candidates with LLM
        for known_id, known_item, heuristic_score in candidates[:5]:
            prompt = build_pairwise_llm_prompt(item.text, known_item.text)

            try:
                response = await self.llm_callback(prompt)
                is_dup, _, confidence, reason = parse_llm_duplicate_response(response)

                if is_dup:
                    if self.require_high_confidence and confidence != "high":
                        continue

                    heuristic_score.llm_confirmed = True
                    heuristic_score.llm_explanation = reason

                    log.info(
                        "deduplication.llm.pairwise_match",
                        new_id=item.id,
                        duplicate_of=known_id,
                        confidence=confidence,
                    )

                    return DuplicateResult(
                        is_duplicate=True,
                        checked_item_id=item.id,
                        duplicate_of=known_id,
                        similarity=heuristic_score,
                        check_method="pairwise_llm",
                        reason=reason,
                    )

            except Exception as e:
                log.warning(
                    "deduplication.llm.pairwise_error",
                    new_id=item.id,
                    known_id=known_id,
                    error=str(e),
                )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            check_method="pairwise_llm",
            reason="No LLM-confirmed duplicate",
        )

    def clear(self) -> None:
        """Clear all known content."""
        self._known_items.clear()
        self._signatures.clear()
        log.info("deduplication.checker.cleared")

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            **self._stats,
            "known_items": len(self._known_items),
        }

    @property
    def known_count(self) -> int:
        """Number of known content items."""
        return len(self._known_items)

    # =========================================================================
    # Backward-Compatible Methods (for explorer module migration)
    # =========================================================================

    def add_known_insight(self, insight_id: str, text: str) -> None:
        """
        Add an insight to the known set (backward-compatible method).

        This method provides compatibility with the explorer's existing interface.

        Args:
            insight_id: The insight's unique identifier
            text: The insight text
        """
        self.add_content(ContentItem(
            id=insight_id,
            text=text,
            content_type=ContentType.INSIGHT,
        ))

    def load_known_insights(self, insights: list[dict]) -> None:
        """
        Load multiple known insights (backward-compatible method).

        This method provides compatibility with the explorer's existing interface.

        Args:
            insights: List of dicts with 'id' and 'text' (or 'insight') keys
        """
        for insight in insights:
            insight_id = insight.get("id", str(len(self._known_items)))
            text = insight.get("text") or insight.get("insight", "")
            self.add_known_insight(insight_id, text)

        log.info(
            "deduplication.insights.loaded",
            count=len(insights),
            total_known=len(self._known_items),
        )

    async def is_duplicate(
        self,
        new_text: str,
        new_id: str = "new",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if text is a duplicate (backward-compatible method).

        This method provides compatibility with the explorer's existing interface.

        Args:
            new_text: The new insight text
            new_id: ID for the new insight

        Returns:
            Tuple of (is_duplicate, duplicate_of_id or None)
        """
        result = await self.check_duplicate(
            new_text,
            item_id=new_id,
            content_type=ContentType.INSIGHT,
        )
        return result.is_duplicate, result.duplicate_of


def get_dedup_checker(
    claude_reviewer=None,
    config: dict = None,
) -> DeduplicationChecker:
    """
    Factory function to create a DeduplicationChecker (backward-compatible).

    This provides compatibility with the explorer's existing interface while
    using the new shared implementation.

    Args:
        claude_reviewer: Object with send_message method (ClaudeReviewer interface)
        config: Configuration dict (optional)

    Returns:
        Configured DeduplicationChecker
    """
    config = config or {}
    dedup_config = config.get("deduplication", {})

    # Create LLM callback if claude_reviewer is provided
    llm_callback = None
    if claude_reviewer is not None:
        async def callback(prompt: str) -> str:
            return await claude_reviewer.send_message(prompt, extended_thinking=False)
        llm_callback = callback

    return DeduplicationChecker(
        llm_callback=llm_callback,
        keyword_threshold=dedup_config.get("keyword_threshold", 0.40),
        concept_threshold=dedup_config.get("concept_threshold", 0.45),
        combined_threshold=dedup_config.get("combined_threshold", 0.50),
        use_llm_check=dedup_config.get("use_llm_confirmation", True),
        use_batch_llm=dedup_config.get("use_batch_llm", True),
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_content_item(
    id: str,
    text: str,
    content_type: str = "unknown",
    **metadata,
) -> ContentItem:
    """
    Convenience function to create a ContentItem.

    Args:
        id: Unique identifier
        text: The text content
        content_type: Type of content ("insight", "section", etc.)
        **metadata: Additional metadata

    Returns:
        ContentItem instance
    """
    try:
        ct = ContentType(content_type)
    except ValueError:
        ct = ContentType.UNKNOWN

    return ContentItem(
        id=id,
        text=text,
        content_type=ct,
        metadata=metadata,
    )


async def quick_duplicate_check(
    new_text: str,
    existing_texts: list[dict[str, str]],
    llm_callback: Optional[LLMCallback] = None,
) -> tuple[bool, Optional[str], str]:
    """
    Quick one-shot duplicate check.

    Args:
        new_text: Text to check
        existing_texts: List of {"id": str, "text": str} dicts
        llm_callback: Optional LLM callback for semantic check

    Returns:
        (is_duplicate, duplicate_of_id, reason)
    """
    checker = DeduplicationChecker(llm_callback=llm_callback)
    checker.load_from_dicts(existing_texts)

    result = await checker.check_duplicate(new_text)
    return result.is_duplicate, result.duplicate_of, result.reason


def is_similar_heuristic(
    text1: str,
    text2: str,
    threshold: float = 0.50,
) -> tuple[bool, float]:
    """
    Quick heuristic similarity check between two texts.

    Args:
        text1: First text
        text2: Second text
        threshold: Combined score threshold

    Returns:
        (is_similar, combined_score)
    """
    item1 = ContentItem(id="1", text=text1)
    item2 = ContentItem(id="2", text=text2)

    score = calculate_similarity(item1, item2)
    return score.combined_score >= threshold, score.combined_score


def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity based on keyword overlap.

    Backward-compatible function for explorer module.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity between 0.0 and 1.0
    """
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    return jaccard_similarity(keywords1, keywords2)
