"""
Tests for Researcher Orchestrator.

Tests cover:
- Initialization
- Config loading
- Component initialization
- Research loop
- Question processing
- Source processing
- Status reporting
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""

    def test_init_detects_base_path(self, temp_dir):
        """Orchestrator detects base path if not provided."""
        # Create mock config directory
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("observer:\n  polling_interval_seconds: 30")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

        assert orch.base_path == temp_dir
        assert orch._running is False

    def test_init_sets_initial_state(self, temp_dir):
        """Orchestrator sets initial state correctly."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

        assert orch._idle_count == 0
        assert orch._last_search_time is None
        assert orch._questions_processed == 0


class TestOrchestratorLoadConfig:
    """Tests for config loading."""

    def test_load_config_from_file(self, temp_dir):
        """Loads config from settings.yaml."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("""
observer:
  polling_interval_seconds: 60
limits:
  max_questions_per_cycle: 10
""")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

        assert orch.config["observer"]["polling_interval_seconds"] == 60
        assert orch.config["limits"]["max_questions_per_cycle"] == 10

    def test_load_config_handles_missing_file(self, temp_dir):
        """Returns empty dict if config file missing."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        # Don't create settings.yaml

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

        assert orch.config == {}


class TestOrchestratorSetLLMClient:
    """Tests for set_llm_client method."""

    def test_set_llm_client_updates_components(self, temp_dir):
        """set_llm_client updates trust evaluator and extractor."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_trust_evaluator = MagicMock()
        mock_extractor = MagicMock()

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator", return_value=mock_trust_evaluator):
                                with patch("researcher.src.orchestrator.ContentExtractor", return_value=mock_extractor):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            mock_client = MagicMock()
                                            orch.set_llm_client(mock_client)

        assert mock_trust_evaluator.llm_client is mock_client
        assert mock_extractor.llm_client is mock_client


class TestOrchestratorStop:
    """Tests for stop method."""

    def test_stop_sets_running_false(self, temp_dir):
        """stop() sets _running to False."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)
                                            orch._running = True

                                            orch.stop()

        assert orch._running is False


class TestOrchestratorGetStatus:
    """Tests for get_status method."""

    def test_get_status_returns_dict(self, temp_dir):
        """get_status returns status dictionary."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_db = MagicMock()
        mock_db.get_statistics = MagicMock(return_value={"findings": 10, "sources": 5})

        mock_cache = MagicMock()
        mock_cache.get_stats = MagicMock(return_value={"items": 100})

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache", return_value=mock_cache):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase", return_value=mock_db):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)
                                            orch._questions_processed = 42
                                            orch._idle_count = 3

                                            status = orch.get_status()

        assert status["running"] is False
        assert status["questions_processed"] == 42
        assert status["idle_count"] == 3
        assert status["database"]["findings"] == 10


class TestOrchestratorShouldResearch:
    """Tests for _should_research method."""

    def test_should_research_true_if_never_searched(self, temp_dir):
        """Returns True if no prior searches."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            result = orch._should_research()

        assert result is True

    def test_should_research_true_after_10_minutes(self, temp_dir):
        """Returns True if 10+ minutes since last search."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)
                                            orch._last_search_time = datetime.now() - timedelta(minutes=15)

                                            result = orch._should_research()

        assert result is True

    def test_should_research_false_if_recent(self, temp_dir):
        """Returns False if searched recently."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)
                                            orch._last_search_time = datetime.now() - timedelta(minutes=5)

                                            result = orch._should_research()

        assert result is False


class TestOrchestratorResearchCycle:
    """Tests for _research_cycle method."""

    @pytest.mark.asyncio
    async def test_research_cycle_generates_questions(self, temp_dir):
        """_research_cycle generates questions from context."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("limits:\n  max_questions_per_cycle: 20")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_question_gen = MagicMock()
        mock_question_gen.generate = MagicMock(return_value=[])
        mock_question_gen.prioritize_by_context = MagicMock(return_value=[])

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator", return_value=mock_question_gen):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            mock_context = MagicMock()
                                            await orch._research_cycle(mock_context)

        mock_question_gen.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_cycle_processes_questions(self, temp_dir):
        """_research_cycle processes prioritized questions."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("limits:\n  max_questions_per_cycle: 20")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_questions = [
            {"query": "What is the Fano plane?", "source": "domain", "source_value": "math"},
            {"query": "Properties of projective geometry", "source": "domain", "source_value": "math"},
        ]

        mock_question_gen = MagicMock()
        mock_question_gen.generate = MagicMock(return_value=mock_questions)
        mock_question_gen.prioritize_by_context = MagicMock(return_value=mock_questions)

        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value=[])

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator", return_value=mock_question_gen):
                with patch("researcher.src.orchestrator.WebSearcher", return_value=mock_searcher):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            mock_context = MagicMock()
                                            await orch._research_cycle(mock_context)

        # Should have processed questions (called search for each)
        assert mock_searcher.search.call_count == 2


class TestOrchestratorProcessSource:
    """Tests for _process_source method."""

    @pytest.mark.asyncio
    async def test_process_source_skips_existing(self, temp_dir):
        """Skips sources that already exist in database."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("test: true")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_db = MagicMock()
        mock_db.get_source_by_url = MagicMock(return_value=MagicMock())  # Existing source

        mock_fetcher = MagicMock()
        mock_fetcher.fetch = AsyncMock()

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher", return_value=mock_fetcher):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase", return_value=mock_db):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            search_result = {"url": "https://example.com/test"}
                                            mock_context = MagicMock()

                                            await orch._process_source(search_result, mock_context)

        # Should not have fetched (source already exists)
        mock_fetcher.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_source_skips_low_trust(self, temp_dir):
        """Skips sources with low trust score."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("trust:\n  min_trust_score: 50")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_db = MagicMock()
        mock_db.get_source_by_url = MagicMock(return_value=None)  # New source
        mock_db.save_source = MagicMock()

        mock_fetcher = MagicMock()
        mock_fetcher.fetch = AsyncMock(return_value={
            "domain": "example.com",
            "content_hash": "abc123",
        })

        mock_trust = MagicMock()
        mock_trust.evaluate = AsyncMock(return_value={
            "trust_score": 30,  # Below threshold
            "trust_tier": "low",
            "reasoning": "Unverified source",
        })

        with patch("researcher.src.orchestrator.ContextAggregator"):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher", return_value=mock_fetcher):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator", return_value=mock_trust):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase", return_value=mock_db):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            search_result = {"url": "https://untrusted.com/page"}
                                            mock_context = MagicMock()

                                            await orch._process_source(search_result, mock_context)

        # Should not have saved the low-trust source
        mock_db.save_source.assert_not_called()


class TestOrchestratorRun:
    """Tests for run method."""

    @pytest.mark.asyncio
    async def test_run_sets_running_true(self, temp_dir):
        """run() sets _running to True."""
        config_path = temp_dir / "researcher" / "config"
        config_path.mkdir(parents=True)
        (config_path / "settings.yaml").write_text("""
observer:
  polling_interval_seconds: 0.1
  idle_polling_interval_seconds: 0.1
  idle_threshold_checks: 1
""")

        data_path = temp_dir / "researcher" / "data"
        data_path.mkdir(parents=True)

        mock_context_agg = MagicMock()
        mock_context_agg.update_context = MagicMock(return_value=MagicMock())
        mock_context_agg.is_idle = MagicMock(return_value=True)

        with patch("researcher.src.orchestrator.ContextAggregator", return_value=mock_context_agg):
            with patch("researcher.src.orchestrator.QuestionGenerator"):
                with patch("researcher.src.orchestrator.WebSearcher"):
                    with patch("researcher.src.orchestrator.ContentFetcher"):
                        with patch("researcher.src.orchestrator.ContentCache"):
                            with patch("researcher.src.orchestrator.TrustEvaluator"):
                                with patch("researcher.src.orchestrator.ContentExtractor"):
                                    with patch("researcher.src.orchestrator.CrossReferenceDetector"):
                                        with patch("researcher.src.orchestrator.ResearcherDatabase"):
                                            from researcher.src.orchestrator import Orchestrator
                                            orch = Orchestrator(base_path=temp_dir)

                                            # Run briefly then stop
                                            async def stop_after_delay():
                                                await asyncio.sleep(0.05)
                                                orch.stop()

                                            with patch.object(orch, "_research_cycle", new_callable=AsyncMock):
                                                await asyncio.gather(
                                                    orch.run(),
                                                    stop_after_delay(),
                                                )

        # Should have called update_context at least once
        mock_context_agg.update_context.assert_called()
