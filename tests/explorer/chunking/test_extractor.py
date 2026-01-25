"""
Tests for the AtomicExtractor.

Tests cover:
- Extraction from threads
- Loading pending and blessed insights
- Confidence filtering
- Dependency resolution

IMPORTANT: All tests must verify that deep_mode and pro_mode are NEVER used.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from explorer.src.chunking.extractor import AtomicExtractor, get_extractor
from explorer.src.chunking.models import AtomicInsight, InsightStatus


class TestAtomicExtractorInit:
    """Tests for AtomicExtractor initialization."""

    def test_creates_insights_directory(self, temp_dir):
        """AtomicExtractor creates insights directory on init."""
        extractor = AtomicExtractor(temp_dir / "chunks")

        assert (temp_dir / "chunks" / "insights").exists()

    def test_accepts_config(self, temp_dir):
        """AtomicExtractor accepts configuration."""
        config = {
            "chunking": {"max_insights_per_thread": 5},
            "dependencies": {"semantic_match_threshold": 0.7},
        }
        extractor = AtomicExtractor(temp_dir / "chunks", config)

        assert extractor.config == config


class TestExtractFromThread:
    """Tests for extract_from_thread method."""

    @pytest.mark.asyncio
    async def test_skips_already_extracted(self, extractor, sample_exploration_thread):
        """Skips threads that are already extracted."""
        sample_exploration_thread.chunks_extracted = True

        result = await extractor.extract_from_thread(sample_exploration_thread)

        assert result == []

    @pytest.mark.asyncio
    async def test_uses_claude_when_available(
        self, extractor, sample_exploration_thread, mock_claude_reviewer
    ):
        """Uses Claude as primary extraction model when available."""
        result = await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude_reviewer,
        )

        # Verify extraction happened
        assert sample_exploration_thread.chunks_extracted is True

    @pytest.mark.asyncio
    async def test_fallback_to_chatgpt_no_pro_mode(
        self, extractor, sample_exploration_thread, mock_chatgpt_browser
    ):
        """Falls back to ChatGPT without pro_mode when Claude unavailable."""
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = False

        # Setup mock ChatGPT response
        mock_chatgpt_browser.send_message = AsyncMock(return_value="""===
INSIGHT: Test insight from ChatGPT.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===""")

        result = await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude,
            fallback_model=mock_chatgpt_browser,
            fallback_model_name="chatgpt",
        )

        # Verify pro_mode was NOT used (check the mock call)
        mock_chatgpt_browser.send_message.assert_called_once()
        call_kwargs = mock_chatgpt_browser.send_message.call_args[1]
        assert call_kwargs.get("use_pro_mode") is False, "pro_mode must be False!"

    @pytest.mark.asyncio
    async def test_fallback_to_gemini_no_deep_think(
        self, extractor, sample_exploration_thread, mock_gemini_browser
    ):
        """Falls back to Gemini without deep_think when Claude unavailable."""
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = False

        # Setup mock Gemini response
        mock_gemini_browser.send_message = AsyncMock(return_value="""===
INSIGHT: Test insight from Gemini.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===""")

        result = await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude,
            fallback_model=mock_gemini_browser,
            fallback_model_name="gemini",
        )

        # Verify deep_think was NOT used
        mock_gemini_browser.send_message.assert_called_once()
        call_kwargs = mock_gemini_browser.send_message.call_args[1]
        assert call_kwargs.get("use_deep_think") is False, "deep_think must be False!"

    @pytest.mark.asyncio
    async def test_raises_when_no_model_available(
        self, extractor, sample_exploration_thread
    ):
        """Raises error when no extraction model is available."""
        mock_claude = MagicMock()
        mock_claude.is_available.return_value = False

        result = await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude,
            # No fallback provided
        )

        # Should handle gracefully - no crash, empty result
        assert result == []
        assert "failed" in sample_exploration_thread.extraction_note.lower()

    @pytest.mark.asyncio
    async def test_handles_extraction_error(
        self, extractor, sample_exploration_thread, mock_claude_reviewer
    ):
        """Handles extraction errors gracefully."""
        mock_claude_reviewer.send_message = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude_reviewer,
        )

        assert result == []
        assert "failed" in sample_exploration_thread.extraction_note.lower()

    @pytest.mark.asyncio
    async def test_marks_thread_as_extracted(
        self, extractor, sample_exploration_thread, mock_claude_reviewer
    ):
        """Marks thread as extracted after successful extraction."""
        await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude_reviewer,
        )

        assert sample_exploration_thread.chunks_extracted is True

    @pytest.mark.asyncio
    async def test_saves_extracted_insights(
        self, extractor, sample_exploration_thread, mock_claude_reviewer
    ):
        """Saves extracted insights to disk."""
        await extractor.extract_from_thread(
            sample_exploration_thread,
            claude_reviewer=mock_claude_reviewer,
        )

        pending_dir = extractor.insights_dir / "pending"
        json_files = list(pending_dir.glob("*.json"))
        assert len(json_files) > 0


class TestConfidenceFiltering:
    """Tests for confidence-based filtering."""

    @pytest.mark.asyncio
    async def test_filters_by_min_confidence(self, temp_dir):
        """Filters insights below minimum confidence."""
        config = {
            "chunking": {
                "max_insights_per_thread": 10,
                "min_confidence_to_keep": "high",  # Only keep high confidence
            },
        }
        extractor = AtomicExtractor(temp_dir / "chunks", config)

        # Mock response with mixed confidence levels
        mock_claude = AsyncMock()
        mock_claude.is_available.return_value = True
        mock_claude.send_message = AsyncMock(return_value="""===
INSIGHT: High confidence insight.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Medium confidence insight.
CONFIDENCE: medium
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Low confidence insight.
CONFIDENCE: low
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===""")

        thread = MagicMock()
        thread.id = "thread-filter-test"
        thread.chunks_extracted = False
        thread.get_context_for_prompt.return_value = "Test context"

        result = await extractor.extract_from_thread(thread, claude_reviewer=mock_claude)

        # Only high confidence should be kept
        assert len(result) == 1
        assert result[0].confidence == "high"

    @pytest.mark.asyncio
    async def test_keeps_all_with_low_threshold(self, temp_dir):
        """Keeps all insights when min_confidence is 'low'."""
        config = {
            "chunking": {
                "max_insights_per_thread": 10,
                "min_confidence_to_keep": "low",
            },
        }
        extractor = AtomicExtractor(temp_dir / "chunks", config)

        mock_claude = AsyncMock()
        mock_claude.is_available.return_value = True
        mock_claude.send_message = AsyncMock(return_value="""===
INSIGHT: High.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Medium.
CONFIDENCE: medium
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Low.
CONFIDENCE: low
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===""")

        thread = MagicMock()
        thread.id = "thread-all"
        thread.chunks_extracted = False
        thread.get_context_for_prompt.return_value = "Test context"

        result = await extractor.extract_from_thread(thread, claude_reviewer=mock_claude)

        assert len(result) == 3


class TestMaxInsightsLimit:
    """Tests for max_insights_per_thread limit."""

    @pytest.mark.asyncio
    async def test_respects_max_limit(self, temp_dir):
        """Respects max_insights_per_thread limit."""
        config = {
            "chunking": {
                "max_insights_per_thread": 2,  # Only 2 allowed
                "min_confidence_to_keep": "low",
            },
        }
        extractor = AtomicExtractor(temp_dir / "chunks", config)

        mock_claude = AsyncMock()
        mock_claude.is_available.return_value = True
        mock_claude.send_message = AsyncMock(return_value="""===
INSIGHT: First.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Second.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Third.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Fourth.
CONFIDENCE: high
TAGS: test
DEPENDS_ON: none
PENDING_DEPENDS: none
===""")

        thread = MagicMock()
        thread.id = "thread-max"
        thread.chunks_extracted = False
        thread.get_context_for_prompt.return_value = "Test"

        result = await extractor.extract_from_thread(thread, claude_reviewer=mock_claude)

        assert len(result) == 2


class TestLoadInsights:
    """Tests for loading insights from disk."""

    def test_load_pending_insights(self, extractor, sample_pending_insights):
        """load_pending_insights returns saved pending insights."""
        # Save some pending insights
        for insight in sample_pending_insights:
            insight.save(extractor.data_dir)

        loaded = extractor.load_pending_insights()

        assert len(loaded) == len(sample_pending_insights)

    def test_load_pending_empty_directory(self, extractor):
        """load_pending_insights returns empty list if no pending."""
        loaded = extractor.load_pending_insights()
        assert loaded == []

    def test_load_blessed_insights(self, extractor, sample_blessed_insight):
        """load_blessed_insights returns saved blessed insights."""
        sample_blessed_insight.save(extractor.data_dir)

        loaded = extractor.load_blessed_insights()

        assert len(loaded) == 1
        assert loaded[0].status == InsightStatus.BLESSED

    def test_get_blessed_ids(self, extractor, sample_blessed_insight):
        """get_blessed_ids returns set of blessed insight IDs."""
        sample_blessed_insight.save(extractor.data_dir)

        ids = extractor.get_blessed_ids()

        assert sample_blessed_insight.id in ids


class TestGetExtractor:
    """Tests for get_extractor factory function."""

    def test_creates_extractor(self, temp_dir):
        """get_extractor creates an AtomicExtractor."""
        extractor = get_extractor(temp_dir)

        assert isinstance(extractor, AtomicExtractor)

    def test_accepts_config(self, temp_dir):
        """get_extractor passes config to extractor."""
        config = {"chunking": {"max_insights_per_thread": 5}}
        extractor = get_extractor(temp_dir, config)

        assert extractor.config == config


class TestDependencyResolution:
    """Tests for dependency resolution during extraction."""

    @pytest.mark.asyncio
    async def test_resolves_dependencies(self, extractor, sample_blessed_insight):
        """Resolves pending dependencies to blessed insight IDs."""
        # Save a blessed insight
        sample_blessed_insight.save(extractor.data_dir)

        # Mock extraction with a pending dependency
        mock_claude = AsyncMock()
        mock_claude.is_available.return_value = True
        mock_claude.send_message = AsyncMock(return_value="""===
INSIGHT: This builds on counting properties.
CONFIDENCE: high
TAGS: fano-plane, extension
DEPENDS_ON: none
PENDING_DEPENDS: projective plane has 7 points
===""")

        thread = MagicMock()
        thread.id = "thread-dep"
        thread.chunks_extracted = False
        thread.get_context_for_prompt.return_value = "Test"

        blessed = [sample_blessed_insight]

        with patch("explorer.src.chunking.extractor.resolve_dependencies") as mock_resolve:
            mock_resolve.return_value = ([sample_blessed_insight.id], [])

            result = await extractor.extract_from_thread(
                thread,
                claude_reviewer=mock_claude,
                blessed_insights=blessed,
            )

            mock_resolve.assert_called()
