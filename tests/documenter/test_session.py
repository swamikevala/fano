"""
Tests for documenter SessionManager.

Tests cover:
- Configuration loading
- Component initialization
- Budget checking
- Resource cleanup
- Deduplication setup
"""

import asyncio
from datetime import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_creates_session(self, temp_dir):
        """SessionManager initializes with default values."""
        config_content = """
documenter:
  document:
    path: document/main.md
    archive_dir: document/archive
    snapshot_time: "00:00"
  inputs:
    blessed_insights_dir: blessed_insights
    guidance_file: document/guidance.md
  context:
    max_tokens: 8000
  termination:
    max_consecutive_disputes: 3
    max_consensus_calls_per_session: 100
  review:
    max_age_days: 7
  work_allocation:
    review_existing: 30
llm:
  consensus:
    use_deep_mode: false
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)

        assert session.document is None
        assert session.consensus_calls == 0
        assert session.exhausted is False

    def test_init_handles_missing_config(self, temp_dir):
        """SessionManager handles missing config file."""
        nonexistent = temp_dir / "nonexistent.yaml"

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(nonexistent)

        # Should have empty config
        assert session.config == {}


class TestSessionManagerConfigExtraction:
    """Tests for configuration extraction."""

    def test_extracts_document_path(self, temp_dir):
        """Extracts document path from config."""
        config_content = """
documenter:
  document:
    path: custom/doc.md
    archive_dir: custom/archive
    snapshot_time: "12:30"
  inputs:
    blessed_insights_dir: insights
    guidance_file: guidance.md
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)

        assert session.doc_path == Path("custom/doc.md")
        assert session.archive_dir == Path("custom/archive")
        assert session.snapshot_time == time(12, 30)

    def test_extracts_limits(self, temp_dir):
        """Extracts limits and thresholds from config."""
        config_content = """
documenter:
  context:
    max_tokens: 5000
  termination:
    max_consecutive_disputes: 5
    max_consensus_calls_per_session: 50
  review:
    max_age_days: 14
  work_allocation:
    review_existing: 40
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)

        assert session.max_context_tokens == 5000
        assert session.max_disputes == 5
        assert session.max_consensus_calls == 50
        assert session.max_age_days == 14
        assert session.review_allocation == 40


class TestSessionManagerLoadGuidance:
    """Tests for guidance file loading."""

    def test_loads_guidance_file(self, temp_dir):
        """_load_guidance loads guidance text."""
        guidance_content = "# Guidance\n\nFollow these rules..."
        guidance_path = temp_dir / "guidance.md"
        guidance_path.write_text(guidance_content)

        config_content = f"""
documenter:
  inputs:
    guidance_file: {guidance_path}
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.guidance_path = guidance_path

            result = session._load_guidance()

        assert result == guidance_content

    def test_returns_empty_if_missing(self, temp_dir):
        """_load_guidance returns empty string if file missing."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.guidance_path = temp_dir / "nonexistent.md"

            result = session._load_guidance()

        assert result == ""


class TestSessionManagerInitialize:
    """Tests for initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_loads_document(self, temp_dir):
        """initialize() loads the document."""
        # Create minimal document
        doc_dir = temp_dir / "document"
        doc_dir.mkdir()
        doc_path = doc_dir / "main.md"
        doc_path.write_text("# Test Document\n\nContent here.")

        config_content = f"""
documenter:
  document:
    path: {doc_path}
    archive_dir: {temp_dir / 'archive'}
    snapshot_time: "00:00"
  inputs:
    blessed_insights_dir: {temp_dir / 'blessed'}
    guidance_file: {temp_dir / 'guidance.md'}
  context:
    max_tokens: 8000
  termination:
    max_consecutive_disputes: 3
    max_consensus_calls_per_session: 100
  review:
    max_age_days: 7
  work_allocation:
    review_existing: 30
llm:
  pool:
    host: "127.0.0.1"
    port: 9000
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        # Mock LLMClient
        mock_llm_client = MagicMock()
        mock_llm_client.get_available_backends = AsyncMock(return_value=["gemini", "chatgpt"])
        mock_llm_client.close = AsyncMock()

        mock_dedup = MagicMock()
        mock_dedup.known_count = 0
        mock_dedup.add_content = MagicMock()

        with patch("documenter.session.FANO_ROOT", temp_dir):
            with patch("documenter.session.LLMClient", return_value=mock_llm_client):
                with patch("documenter.session.ConsensusReviewer"):
                    with patch("documenter.session.DeduplicationChecker", return_value=mock_dedup):
                        with patch("documenter.session.load_dedup_config", return_value={}):
                            from documenter.session import SessionManager
                            session = SessionManager(config_path)

                            await session.initialize()

        assert session.document is not None

    @pytest.mark.asyncio
    async def test_initialize_raises_if_insufficient_backends(self, temp_dir):
        """initialize() raises if fewer than 2 backends available."""
        doc_dir = temp_dir / "document"
        doc_dir.mkdir()
        doc_path = doc_dir / "main.md"
        doc_path.write_text("# Test")

        config_content = f"""
documenter:
  document:
    path: {doc_path}
  inputs:
    blessed_insights_dir: {temp_dir / 'blessed'}
    guidance_file: {temp_dir / 'guidance.md'}
llm:
  pool:
    host: "127.0.0.1"
    port: 9000
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        # Mock LLMClient with only 1 backend
        mock_llm_client = MagicMock()
        mock_llm_client.get_available_backends = AsyncMock(return_value=["claude"])
        mock_llm_client.close = AsyncMock()

        with patch("documenter.session.FANO_ROOT", temp_dir):
            with patch("documenter.session.LLMClient", return_value=mock_llm_client):
                from documenter.session import SessionManager
                session = SessionManager(config_path)

                with pytest.raises(RuntimeError, match="at least 2 LLM backends"):
                    await session.initialize()


class TestSessionManagerBudget:
    """Tests for budget checking."""

    def test_check_budget_returns_false_when_available(self, temp_dir):
        """check_budget returns False when budget available."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("""
documenter:
  termination:
    max_consensus_calls_per_session: 100
""")

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.consensus_calls = 50

            result = session.check_budget()

        assert result is False
        assert session.exhausted is False

    def test_check_budget_returns_true_when_exhausted(self, temp_dir):
        """check_budget returns True when budget exhausted."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("""
documenter:
  termination:
    max_consensus_calls_per_session: 100
""")

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.consensus_calls = 100

            result = session.check_budget()

        assert result is True
        assert session.exhausted is True

    def test_increment_consensus_calls(self, temp_dir):
        """increment_consensus_calls increases counter."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)

            assert session.consensus_calls == 0
            session.increment_consensus_calls()
            assert session.consensus_calls == 1
            session.increment_consensus_calls()
            assert session.consensus_calls == 2


class TestSessionManagerCleanup:
    """Tests for cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_closes_llm_client(self, temp_dir):
        """cleanup() closes the LLM client."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        mock_llm_client = MagicMock()
        mock_llm_client.close = AsyncMock()

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.llm_client = mock_llm_client

            await session.cleanup()

            mock_llm_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_handles_no_client(self, temp_dir):
        """cleanup() handles case when no client initialized."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)

            # Should not raise
            await session.cleanup()


class TestSessionManagerCreateSeedDocument:
    """Tests for seed document creation."""

    def test_creates_seed_if_missing(self, temp_dir):
        """_create_seed_document creates seed content."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        doc_path = temp_dir / "document" / "main.md"

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            from documenter.document import Document

            session = SessionManager(config_path)
            session.document = Document(doc_path)

            session._create_seed_document()

        assert doc_path.exists()
        content = doc_path.read_text()
        assert "Principles of Creation" in content
        assert "geometry" in content


class TestSessionManagerLogSummary:
    """Tests for log_summary method."""

    def test_log_summary_calls_logger(self, temp_dir):
        """log_summary logs session statistics."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)

            # Mock document and components
            mock_document = MagicMock()
            mock_document.sections = [MagicMock(), MagicMock()]

            mock_concept_tracker = MagicMock()
            mock_concept_tracker.get_established_concepts = MagicMock(return_value=["c1", "c2", "c3"])

            mock_opportunity_finder = MagicMock()
            mock_opportunity_finder.get_pending_count = MagicMock(return_value=5)

            mock_review_manager = MagicMock()
            mock_review_manager.get_review_stats = MagicMock(return_value={
                "reviewed": 10,
                "pending_review": 2,
            })

            session.document = mock_document
            session.concept_tracker = mock_concept_tracker
            session.opportunity_finder = mock_opportunity_finder
            session.review_manager = mock_review_manager
            session.consensus_calls = 42

            # Should not raise
            session.log_summary()

            mock_review_manager.get_review_stats.assert_called_once()


class TestSessionManagerDedupCallback:
    """Tests for deduplication LLM callback."""

    @pytest.mark.asyncio
    async def test_dedup_callback_sends_to_llm(self, temp_dir):
        """_dedup_llm_callback sends prompt to LLM."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = "No duplicates found"

        mock_llm_client = MagicMock()
        mock_llm_client.send = AsyncMock(return_value=mock_response)

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.llm_client = mock_llm_client
            session._dedup_config = {"model": "claude-sonnet", "llm_timeout": 30}

            result = await session._dedup_llm_callback("Check for duplicates")

        assert result == "No duplicates found"
        mock_llm_client.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_dedup_callback_raises_on_failure(self, temp_dir):
        """_dedup_llm_callback raises on LLM failure."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("documenter: {}")

        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error = "API error"

        mock_llm_client = MagicMock()
        mock_llm_client.send = AsyncMock(return_value=mock_response)

        with patch("documenter.session.FANO_ROOT", temp_dir):
            from documenter.session import SessionManager
            session = SessionManager(config_path)
            session.llm_client = mock_llm_client
            session._dedup_config = {"model": "claude-sonnet", "llm_timeout": 30}

            with pytest.raises(RuntimeError, match="LLM call failed"):
                await session._dedup_llm_callback("Check for duplicates")
