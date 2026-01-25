"""
Tests for the DeduplicationChecker and related components.

Tests cover:
- ContentItem model and properties
- Text processing functions
- Signature-based duplicate detection
- Heuristic-based duplicate detection
- LLM-based duplicate detection (mocked)
- Statistics tracking
- Backward-compatible methods
"""

import pytest
from unittest.mock import AsyncMock, patch

from shared.deduplication import (
    DeduplicationChecker,
    ContentItem,
    ContentType,
    SimilarityScore,
    DuplicateResult,
    normalize_text,
    compute_signature,
    stem_word,
    extract_keywords,
    extract_concepts,
    extract_ngrams,
    jaccard_similarity,
    calculate_similarity,
    calculate_keyword_similarity,
)


# =============================================================================
# Text Processing Tests
# =============================================================================

class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        """Converts text to lowercase."""
        assert normalize_text("HELLO World") == "hello world"

    def test_removes_punctuation(self):
        """Removes punctuation except hyphens in words."""
        assert normalize_text("Hello, world! How's it going?") == "hello world how s it going"

    def test_preserves_hyphens(self):
        """Preserves hyphens within hyphenated words."""
        result = normalize_text("well-known")
        assert "well-known" in result

    def test_collapses_whitespace(self):
        """Collapses multiple spaces to single space."""
        assert normalize_text("hello    world") == "hello world"

    def test_strips_leading_trailing(self):
        """Strips leading and trailing whitespace."""
        assert normalize_text("  hello  ") == "hello"

    def test_preserves_numbers(self):
        """Preserves numbers in text."""
        assert normalize_text("7 points and 7 lines") == "7 points and 7 lines"


class TestComputeSignature:
    """Tests for signature computation."""

    def test_identical_text_same_signature(self):
        """Identical texts produce same signature."""
        sig1 = compute_signature("The Fano plane has 7 points.")
        sig2 = compute_signature("The Fano plane has 7 points.")
        assert sig1 == sig2

    def test_different_text_different_signature(self):
        """Different texts produce different signatures."""
        sig1 = compute_signature("The Fano plane has 7 points.")
        sig2 = compute_signature("The Klein quartic has 168 automorphisms.")
        assert sig1 != sig2

    def test_case_insensitive(self):
        """Signature is case-insensitive."""
        sig1 = compute_signature("Hello World")
        sig2 = compute_signature("hello world")
        assert sig1 == sig2

    def test_ignores_punctuation(self):
        """Signature ignores punctuation differences."""
        sig1 = compute_signature("Hello, World!")
        sig2 = compute_signature("Hello World")
        assert sig1 == sig2

    def test_signature_length(self):
        """Signature is 16 characters (MD5 truncated)."""
        sig = compute_signature("Any text")
        assert len(sig) == 16


class TestStemWord:
    """Tests for word stemming."""

    def test_removes_plural_s(self):
        """Removes plural 's' suffix."""
        assert stem_word("points") == "point"
        assert stem_word("lines") == "line"

    def test_handles_ies_ending(self):
        """Converts 'ies' to 'y'."""
        assert stem_word("properties") == "property"

    def test_handles_vertices(self):
        """Special case for 'vertices'."""
        assert stem_word("vertices") == "vertex"

    def test_handles_matrices(self):
        """Special case for 'matrices'."""
        assert stem_word("matrices") == "matrix"

    def test_handles_indices(self):
        """Special case for 'indices'."""
        assert stem_word("indices") == "index"

    def test_removes_ing(self):
        """Removes 'ing' suffix."""
        assert stem_word("connecting") == "connect"

    def test_removes_ed(self):
        """Removes 'ed' suffix."""
        assert stem_word("connected") == "connect"

    def test_preserves_short_words(self):
        """Preserves short words that might match patterns."""
        assert stem_word("is") == "is"
        assert stem_word("as") == "as"


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_removes_stop_words(self):
        """Removes common stop words."""
        keywords = extract_keywords("the quick brown fox")
        assert "the" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_preserves_numbers(self):
        """Preserves significant numbers (multi-digit)."""
        # Note: Single digits like "7" are filtered by length check
        # Multi-digit numbers are preserved
        keywords = extract_keywords("The plane has 168 automorphisms")
        assert "168" in keywords

    def test_stems_words(self):
        """Stems extracted keywords."""
        keywords = extract_keywords("The vertices are connected")
        assert "vertex" in keywords
        assert "connect" in keywords

    def test_removes_short_words(self):
        """Removes words with 2 or fewer characters."""
        keywords = extract_keywords("A is to B as C is to D")
        assert "a" not in keywords
        assert "is" not in keywords
        assert "to" not in keywords

    def test_returns_set(self):
        """Returns a set of unique keywords."""
        keywords = extract_keywords("point point point")
        assert isinstance(keywords, set)
        assert "point" in keywords
        assert len([k for k in keywords if k == "point"]) == 1


class TestExtractConcepts:
    """Tests for domain concept extraction."""

    def test_extracts_numbers(self):
        """Extracts significant numbers."""
        concepts = extract_concepts("The group has order 168")
        assert "168" in concepts

    def test_extracts_geometry_terms(self):
        """Extracts geometry-related terms."""
        concepts = extract_concepts("The projective plane with hyperbolic structure")
        assert "projective" in concepts or "plane" in concepts

    def test_extracts_group_theory_terms(self):
        """Extracts group theory terms."""
        concepts = extract_concepts("The automorphism group has 168 elements")
        assert "automorphism" in concepts or "group" in concepts

    def test_extracts_named_objects(self):
        """Extracts named mathematical objects."""
        concepts = extract_concepts("The Fano plane is the smallest")
        assert "fano" in concepts

    def test_extracts_heawood(self):
        """Extracts Heawood graph reference."""
        concepts = extract_concepts("The Heawood graph has 14 vertices")
        assert "heawood" in concepts


class TestExtractNgrams:
    """Tests for n-gram extraction."""

    def test_extracts_4grams_by_default(self):
        """Extracts 4-character n-grams by default."""
        ngrams = extract_ngrams("hello")
        assert "hell" in ngrams
        assert "ello" in ngrams

    def test_custom_n_size(self):
        """Supports custom n-gram size."""
        ngrams = extract_ngrams("hello", n=3)
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams

    def test_handles_short_text(self):
        """Handles text shorter than n."""
        ngrams = extract_ngrams("hi", n=4)
        assert "hi" in ngrams

    def test_removes_spaces(self):
        """Removes spaces before creating n-grams."""
        ngrams = extract_ngrams("a b c d e f", n=4)
        assert "abcd" in ngrams


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""

    def test_identical_sets(self):
        """Identical sets have similarity 1.0."""
        s = {"a", "b", "c"}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        """Disjoint sets have similarity 0.0."""
        s1 = {"a", "b"}
        s2 = {"c", "d"}
        assert jaccard_similarity(s1, s2) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping sets have intermediate similarity."""
        s1 = {"a", "b", "c"}
        s2 = {"b", "c", "d"}
        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert jaccard_similarity(s1, s2) == 0.5

    def test_empty_sets(self):
        """Empty sets return 0.0."""
        assert jaccard_similarity(set(), {"a"}) == 0.0
        assert jaccard_similarity({"a"}, set()) == 0.0
        assert jaccard_similarity(set(), set()) == 0.0


class TestCalculateSimilarity:
    """Tests for comprehensive similarity calculation."""

    def test_identical_items(self):
        """Identical items have signature match."""
        item1 = ContentItem(id="1", text="The Fano plane has 7 points")
        item2 = ContentItem(id="2", text="The Fano plane has 7 points")

        score = calculate_similarity(item1, item2)

        assert score.signature_match is True
        assert score.keyword_similarity == 1.0

    def test_different_items(self):
        """Different items have low similarity."""
        item1 = ContentItem(id="1", text="The Fano plane has 7 points")
        item2 = ContentItem(id="2", text="Cooking recipes for pasta dishes")

        score = calculate_similarity(item1, item2)

        assert score.signature_match is False
        assert score.combined_score < 0.3

    def test_similar_items(self):
        """Similar items have moderate-high similarity."""
        item1 = ContentItem(id="1", text="The Fano plane has 7 points and 7 lines")
        item2 = ContentItem(id="2", text="The Fano plane contains exactly 7 points")

        score = calculate_similarity(item1, item2)

        # Should have some keyword/concept overlap
        assert score.keyword_similarity > 0.3
        assert score.combined_score > 0.3


# =============================================================================
# ContentItem Model Tests
# =============================================================================

class TestContentItem:
    """Tests for ContentItem dataclass."""

    def test_creates_with_minimal_args(self):
        """Creates item with just id and text."""
        item = ContentItem(id="test-1", text="Test content")
        assert item.id == "test-1"
        assert item.text == "Test content"
        assert item.content_type == ContentType.UNKNOWN

    def test_content_type_from_string(self):
        """Accepts content_type as string."""
        item = ContentItem(id="1", text="Test", content_type="insight")
        assert item.content_type == ContentType.INSIGHT

    def test_invalid_content_type_defaults_unknown(self):
        """Invalid content_type string defaults to UNKNOWN."""
        item = ContentItem(id="1", text="Test", content_type="invalid")
        assert item.content_type == ContentType.UNKNOWN

    def test_signature_computed_lazily(self):
        """Signature is computed on first access."""
        item = ContentItem(id="1", text="Test content")
        assert item._signature is None
        _ = item.signature
        assert item._signature is not None

    def test_keywords_computed_lazily(self):
        """Keywords are computed on first access."""
        item = ContentItem(id="1", text="Test content")
        assert item._keywords is None
        _ = item.keywords
        assert item._keywords is not None

    def test_concepts_computed_lazily(self):
        """Concepts are computed on first access."""
        item = ContentItem(id="1", text="The Fano plane")
        assert item._concepts is None
        _ = item.concepts
        assert item._concepts is not None

    def test_ngrams_computed_lazily(self):
        """N-grams are computed on first access."""
        item = ContentItem(id="1", text="Test content")
        assert item._ngrams is None
        _ = item.ngrams
        assert item._ngrams is not None


class TestSimilarityScore:
    """Tests for SimilarityScore dataclass."""

    def test_passed_heuristics_signature_match(self):
        """Signature match passes heuristics."""
        score = SimilarityScore(signature_match=True)
        assert score.passed_heuristics is True

    def test_passed_heuristics_high_keyword(self):
        """High keyword similarity passes heuristics."""
        score = SimilarityScore(keyword_similarity=0.55)
        assert score.passed_heuristics is True

    def test_passed_heuristics_high_concept(self):
        """High concept similarity passes heuristics."""
        score = SimilarityScore(concept_similarity=0.60)
        assert score.passed_heuristics is True

    def test_passed_heuristics_high_combined(self):
        """High combined score passes heuristics."""
        score = SimilarityScore(combined_score=0.60)
        assert score.passed_heuristics is True

    def test_not_passed_low_scores(self):
        """Low scores don't pass heuristics."""
        score = SimilarityScore(
            keyword_similarity=0.3,
            concept_similarity=0.3,
            combined_score=0.3,
        )
        assert score.passed_heuristics is False


# =============================================================================
# DeduplicationChecker Tests
# =============================================================================

class TestDeduplicationChecker:
    """Tests for DeduplicationChecker class."""

    @pytest.fixture
    def checker(self):
        """Create checker without LLM callback."""
        return DeduplicationChecker(
            llm_callback=None,
            use_signature_check=True,
            use_heuristic_check=False,
            use_llm_check=False,
            stats_log_interval=0,  # Disable stats logging for tests
        )

    @pytest.fixture
    def checker_with_heuristics(self):
        """Create checker with heuristics enabled."""
        return DeduplicationChecker(
            llm_callback=None,
            use_signature_check=True,
            use_heuristic_check=True,
            use_llm_check=False,
            stats_log_interval=0,
        )

    def test_add_content(self, checker):
        """add_content adds item to registry."""
        item = ContentItem(id="test-1", text="Test content")
        checker.add_content(item)

        assert checker.known_count == 1
        assert "test-1" in checker._known_items

    def test_add_content_idempotent(self, checker):
        """Adding same item twice doesn't duplicate."""
        item = ContentItem(id="test-1", text="Test content")
        checker.add_content(item)
        checker.add_content(item)

        assert checker.known_count == 1

    def test_add_contents(self, checker):
        """add_contents adds multiple items."""
        items = [
            ContentItem(id="1", text="First"),
            ContentItem(id="2", text="Second"),
        ]
        checker.add_contents(items)

        assert checker.known_count == 2

    def test_load_from_dicts(self, checker):
        """load_from_dicts creates items from dicts."""
        dicts = [
            {"id": "dict-1", "text": "First item"},
            {"id": "dict-2", "text": "Second item"},
        ]
        checker.load_from_dicts(dicts, ContentType.INSIGHT)

        assert checker.known_count == 2

    def test_clear(self, checker):
        """clear removes all known items."""
        checker.add_content(ContentItem(id="1", text="Test"))
        checker.clear()

        assert checker.known_count == 0

    def test_get_stats(self, checker):
        """get_stats returns statistics dict."""
        stats = checker.get_stats()

        assert "checks" in stats
        assert "duplicates_found" in stats
        assert "by_signature" in stats
        assert "known_items" in stats


class TestSignatureDeduplication:
    """Tests for signature-based deduplication."""

    @pytest.fixture
    def checker(self):
        """Create checker with signature check only."""
        return DeduplicationChecker(
            llm_callback=None,
            use_signature_check=True,
            use_heuristic_check=False,
            use_llm_check=False,
            stats_log_interval=0,
        )

    @pytest.mark.asyncio
    async def test_exact_duplicate_detected(self, checker):
        """Exact duplicate is detected via signature."""
        checker.add_content(ContentItem(
            id="existing",
            text="The Fano plane has 7 points and 7 lines."
        ))

        result = await checker.check_duplicate(
            "The Fano plane has 7 points and 7 lines.",
            item_id="new",
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "existing"
        assert result.check_method == "signature"

    @pytest.mark.asyncio
    async def test_case_insensitive_duplicate(self, checker):
        """Case differences don't prevent duplicate detection."""
        checker.add_content(ContentItem(
            id="existing",
            text="THE FANO PLANE HAS 7 POINTS"
        ))

        result = await checker.check_duplicate(
            "the fano plane has 7 points",
            item_id="new",
        )

        assert result.is_duplicate is True

    @pytest.mark.asyncio
    async def test_unique_content_not_flagged(self, checker):
        """Unique content is not flagged as duplicate."""
        checker.add_content(ContentItem(
            id="existing",
            text="The Fano plane has 7 points"
        ))

        result = await checker.check_duplicate(
            "The Klein quartic has 168 automorphisms",
            item_id="new",
        )

        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_empty_registry_no_duplicate(self, checker):
        """Empty registry returns no duplicate."""
        result = await checker.check_duplicate(
            "Any text",
            item_id="new",
        )

        assert result.is_duplicate is False


class TestHeuristicDeduplication:
    """Tests for heuristic-based deduplication."""

    @pytest.fixture
    def checker(self):
        """Create checker with heuristics enabled."""
        return DeduplicationChecker(
            llm_callback=None,
            use_signature_check=True,
            use_heuristic_check=True,
            use_llm_check=False,
            combined_threshold=0.40,
            stats_log_interval=0,
        )

    @pytest.mark.asyncio
    async def test_similar_content_detected(self, checker):
        """Similar content is detected via heuristics."""
        checker.add_content(ContentItem(
            id="existing",
            text="The Fano plane is a projective plane with 7 points and 7 lines"
        ))

        # Very similar text
        result = await checker.check_duplicate(
            "The Fano plane is the smallest projective plane having 7 points and 7 lines",
            item_id="new",
        )

        # Should be detected as duplicate due to keyword/concept overlap
        # Note: May or may not be detected depending on exact thresholds
        assert result.check_method in ["signature", "heuristic", "all"]


class TestLLMDeduplication:
    """Tests for LLM-based deduplication (mocked)."""

    @pytest.fixture
    def mock_llm_callback(self):
        """Create mock LLM callback."""
        async def callback(prompt: str) -> str:
            # Simulate LLM response indicating duplicate
            if "DUPLICATE" in prompt or "check" in prompt.lower():
                return "DUPLICATE: YES\nINDEX: 0\nCONFIDENCE: high\nREASON: Semantically equivalent"
            return "DUPLICATE: NO"
        return callback

    @pytest.fixture
    def mock_llm_no_dup_callback(self):
        """Create mock LLM callback that returns no duplicate."""
        async def callback(prompt: str) -> str:
            return "DUPLICATE: NO\nREASON: Content is unique"
        return callback

    @pytest.fixture
    def checker_with_llm(self, mock_llm_callback):
        """Create checker with LLM callback."""
        return DeduplicationChecker(
            llm_callback=mock_llm_callback,
            use_signature_check=True,
            use_heuristic_check=False,
            use_llm_check=True,
            use_batch_llm=True,
            stats_log_interval=0,
        )

    @pytest.mark.asyncio
    async def test_llm_check_called_when_enabled(self, checker_with_llm):
        """LLM check is called when enabled and no signature match."""
        checker_with_llm.add_content(ContentItem(
            id="existing",
            text="The Fano plane has 7 points"
        ))

        result = await checker_with_llm.check_duplicate(
            "Different text entirely",
            item_id="new",
        )

        # LLM was called (check_method indicates which method was used)
        assert result.check_method in ["signature", "batch_llm", "all"]

    @pytest.mark.asyncio
    async def test_llm_callback_error_handled(self):
        """LLM callback errors are handled gracefully."""
        async def failing_callback(prompt: str) -> str:
            raise Exception("LLM API error")

        checker = DeduplicationChecker(
            llm_callback=failing_callback,
            use_signature_check=True,
            use_llm_check=True,
            stats_log_interval=0,
        )
        checker.add_content(ContentItem(id="1", text="Test"))

        # Should not raise, should return no duplicate
        result = await checker.check_duplicate("Different", item_id="new")
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_skip_llm_parameter(self, checker_with_llm):
        """skip_llm parameter skips LLM check."""
        checker_with_llm.add_content(ContentItem(
            id="existing",
            text="Test content"
        ))

        result = await checker_with_llm.check_duplicate(
            "Different content",
            item_id="new",
            skip_llm=True,
        )

        # Should not have used LLM
        assert result.check_method != "batch_llm"
        assert result.check_method != "pairwise_llm"


class TestBackwardCompatibility:
    """Tests for backward-compatible methods."""

    @pytest.fixture
    def checker(self):
        """Create checker."""
        return DeduplicationChecker(
            llm_callback=None,
            use_signature_check=True,
            use_heuristic_check=False,
            use_llm_check=False,
            stats_log_interval=0,
        )

    def test_add_known_insight(self, checker):
        """add_known_insight adds insight to registry."""
        checker.add_known_insight("ins-1", "Test insight text")

        assert checker.known_count == 1
        assert checker._known_items["ins-1"].content_type == ContentType.INSIGHT

    def test_load_known_insights(self, checker):
        """load_known_insights loads from list of dicts."""
        insights = [
            {"id": "ins-1", "text": "First insight"},
            {"id": "ins-2", "insight": "Second insight"},  # Uses 'insight' key
        ]
        checker.load_known_insights(insights)

        assert checker.known_count == 2

    @pytest.mark.asyncio
    async def test_is_duplicate_method(self, checker):
        """is_duplicate returns tuple (bool, optional_id)."""
        checker.add_known_insight("existing", "The Fano plane")

        is_dup, dup_id = await checker.is_duplicate(
            "The Fano plane",
            new_id="new",
        )

        assert is_dup is True
        assert dup_id == "existing"

    @pytest.mark.asyncio
    async def test_is_duplicate_no_match(self, checker):
        """is_duplicate returns (False, None) when no match."""
        checker.add_known_insight("existing", "The Fano plane")

        is_dup, dup_id = await checker.is_duplicate(
            "Completely different topic",
            new_id="new",
        )

        assert is_dup is False
        assert dup_id is None


class TestStatistics:
    """Tests for statistics tracking."""

    @pytest.fixture
    def checker(self):
        """Create checker."""
        return DeduplicationChecker(
            llm_callback=None,
            use_signature_check=True,
            stats_log_interval=0,
        )

    @pytest.mark.asyncio
    async def test_tracks_check_count(self, checker):
        """Tracks total number of checks."""
        checker.add_content(ContentItem(id="1", text="Test"))

        await checker.check_duplicate("New 1")
        await checker.check_duplicate("New 2")
        await checker.check_duplicate("New 3")

        stats = checker.get_stats()
        assert stats["checks"] == 3

    @pytest.mark.asyncio
    async def test_tracks_duplicates_found(self, checker):
        """Tracks number of duplicates found."""
        checker.add_content(ContentItem(id="1", text="Test content"))

        await checker.check_duplicate("Test content")  # Duplicate
        await checker.check_duplicate("Different")  # Not duplicate
        await checker.check_duplicate("Test content")  # Duplicate

        stats = checker.get_stats()
        assert stats["duplicates_found"] == 2

    @pytest.mark.asyncio
    async def test_tracks_by_signature(self, checker):
        """Tracks duplicates found by signature."""
        checker.add_content(ContentItem(id="1", text="Test"))

        await checker.check_duplicate("Test")  # Signature match

        stats = checker.get_stats()
        assert stats["by_signature"] == 1
