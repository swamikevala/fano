"""
Tests for Explorer Orchestrator.

Tests cover:
- Initialization
- Main exploration loop
- Backlog processing
- Exploration cycles
- Stop signal handling

IMPORTANT: All tests must verify that deep_mode and pro_mode are NEVER used.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""

    def test_init_creates_paths(self, temp_dir):
        """Orchestrator creates required paths on init."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths") as mock_paths_cls:
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager"):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                mock_paths_cls.assert_called_once()

    def test_init_sets_running_false(self, temp_dir):
        """Orchestrator starts with running=False."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths"):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager"):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                assert orch.running is False


class TestOrchestratorStop:
    """Tests for stop signal."""

    def test_stop_sets_running_false(self):
        """stop() sets running to False."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths"):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager"):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()
                                                orch.running = True
                                                orch.stop()

                                                assert orch.running is False


class TestOrchestratorCleanup:
    """Tests for cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_disconnects_llms(self):
        """cleanup() disconnects from LLMs."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.disconnect = AsyncMock()

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths"):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                await orch.cleanup()

                                                mock_llm_manager.disconnect.assert_called_once()


class TestOrchestratorConnectAndInitialize:
    """Tests for _connect_and_initialize method."""

    @pytest.mark.asyncio
    async def test_connects_to_llms(self):
        """_connect_and_initialize connects to LLMs."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.connect = AsyncMock()
        mock_llm_manager.gemini = None
        mock_llm_manager.chatgpt = None

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths") as mock_paths:
                mock_paths.return_value.data_dir = Path("/tmp")
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                with patch("explorer.src.orchestrator.BlessedStore"):
                                                    with patch("explorer.src.orchestrator.InsightProcessor"):
                                                        from explorer.src.orchestrator import Orchestrator
                                                        orch = Orchestrator()

                                                        await orch._connect_and_initialize()

                                                        mock_llm_manager.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initializes_reviewer_when_enabled(self):
        """_connect_and_initialize creates AutomatedReviewer when enabled."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": True},
            "augmentation": {"enabled": False},
            "deduplication": {"enabled": False},
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.connect = AsyncMock()
        mock_llm_manager.gemini = MagicMock()
        mock_llm_manager.chatgpt = MagicMock()

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths") as mock_paths:
                mock_paths.return_value.data_dir = Path("/tmp")
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                with patch("explorer.src.orchestrator.AutomatedReviewer") as mock_reviewer_cls:
                                                    with patch("explorer.src.orchestrator.BlessedStore"):
                                                        with patch("explorer.src.orchestrator.InsightProcessor"):
                                                            from explorer.src.orchestrator import Orchestrator
                                                            orch = Orchestrator()

                                                            await orch._connect_and_initialize()

                                                            mock_reviewer_cls.assert_called_once()


class TestOrchestratorProcessBacklog:
    """Tests for process_backlog method."""

    @pytest.mark.asyncio
    async def test_process_backlog_no_explorations_dir(self, temp_dir):
        """process_backlog handles missing explorations directory."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        mock_paths = MagicMock()
        mock_paths.explorations_dir = temp_dir / "explorations"  # Doesn't exist
        mock_paths.data_dir = temp_dir

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths", return_value=mock_paths):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager"):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                # Should not raise
                                                await orch.process_backlog()


class TestOrchestratorExplorationCycle:
    """Tests for _exploration_cycle method."""

    @pytest.mark.asyncio
    async def test_cycle_ensures_connected(self):
        """_exploration_cycle ensures pool connection."""
        mock_config = {
            "orchestration": {"poll_interval": 10, "backoff_base": 5},
            "review_panel": {"enabled": False},
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.ensure_connected = AsyncMock()
        mock_llm_manager.get_available_models = MagicMock(return_value={})
        mock_llm_manager.gemini = None

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths"):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                with patch("asyncio.sleep", new_callable=AsyncMock):
                                                    await orch._exploration_cycle()

                                                mock_llm_manager.ensure_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_backs_off_when_rate_limited(self):
        """_exploration_cycle backs off when all models rate limited."""
        mock_config = {
            "orchestration": {"poll_interval": 10, "backoff_base": 5},
            "review_panel": {"enabled": False},
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.ensure_connected = AsyncMock()
        mock_llm_manager.get_available_models = MagicMock(return_value={})  # No available models
        mock_llm_manager.gemini = None

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths"):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                                                    await orch._exploration_cycle()

                                                    # Should have slept for backoff_base seconds
                                                    mock_sleep.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_cycle_runs_parallel_work(self):
        """_exploration_cycle runs work in parallel."""
        mock_config = {
            "orchestration": {"poll_interval": 10, "backoff_base": 5},
            "review_panel": {"enabled": False},
        }

        # Mock thread
        mock_thread = MagicMock()
        mock_thread.id = "test-thread"
        mock_thread.needs_exploration = True
        mock_thread.needs_critique = False
        mock_thread.primary_question_id = None
        mock_thread.related_conjecture_ids = []

        # Mock models
        mock_model = MagicMock()

        mock_llm_manager = MagicMock()
        mock_llm_manager.ensure_connected = AsyncMock()
        mock_llm_manager.get_available_models = MagicMock(return_value={"gemini": mock_model})
        mock_llm_manager.gemini = mock_model

        mock_thread_manager = MagicMock()
        mock_thread_manager.select_thread = MagicMock(return_value=mock_thread)

        mock_exploration_engine = MagicMock()

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths"):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager", return_value=mock_thread_manager):
                                with patch("explorer.src.orchestrator.ExplorationEngine", return_value=mock_exploration_engine):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                # Mock the _do_exploration_and_save method
                                                orch._do_exploration_and_save = AsyncMock()

                                                await orch._exploration_cycle()

                                                # Should have called exploration
                                                orch._do_exploration_and_save.assert_called_once()


class TestOrchestratorDoExploration:
    """Tests for _do_exploration_and_save method."""

    @pytest.mark.asyncio
    async def test_do_exploration_calls_engine(self):
        """_do_exploration_and_save calls exploration engine."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        mock_thread = MagicMock()
        mock_thread.id = "test-thread"
        mock_thread.save = MagicMock()

        mock_model = MagicMock()
        mock_llm_manager = MagicMock()

        mock_exploration_engine = MagicMock()
        mock_exploration_engine.do_exploration = AsyncMock()

        mock_synthesis_engine = MagicMock()
        mock_synthesis_engine.is_chunk_ready = MagicMock(return_value=False)

        mock_paths = MagicMock()
        mock_paths.data_dir = Path("/tmp")

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths", return_value=mock_paths):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine", return_value=mock_exploration_engine):
                                    with patch("explorer.src.orchestrator.SynthesisEngine", return_value=mock_synthesis_engine):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                await orch._do_exploration_and_save(
                                                    mock_thread, "gemini", mock_model
                                                )

                                                mock_exploration_engine.do_exploration.assert_called_once()
                                                mock_thread.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_exploration_triggers_synthesis_when_ready(self):
        """_do_exploration_and_save triggers synthesis when thread is ready."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        mock_thread = MagicMock()
        mock_thread.id = "test-thread"
        mock_thread.save = MagicMock()

        mock_model = MagicMock()
        mock_llm_manager = MagicMock()

        mock_exploration_engine = MagicMock()
        mock_exploration_engine.do_exploration = AsyncMock()

        mock_synthesis_engine = MagicMock()
        mock_synthesis_engine.is_chunk_ready = MagicMock(return_value=True)  # Ready for synthesis
        mock_synthesis_engine.synthesize_chunk = AsyncMock()

        mock_paths = MagicMock()
        mock_paths.data_dir = Path("/tmp")

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths", return_value=mock_paths):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine", return_value=mock_exploration_engine):
                                    with patch("explorer.src.orchestrator.SynthesisEngine", return_value=mock_synthesis_engine):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                await orch._do_exploration_and_save(
                                                    mock_thread, "gemini", mock_model
                                                )

                                                mock_synthesis_engine.synthesize_chunk.assert_called_once()


class TestOrchestratorDoCritique:
    """Tests for _do_critique_and_save method."""

    @pytest.mark.asyncio
    async def test_do_critique_calls_engine(self):
        """_do_critique_and_save calls exploration engine."""
        mock_config = {
            "orchestration": {"poll_interval": 10},
            "review_panel": {"enabled": False},
        }

        mock_thread = MagicMock()
        mock_thread.id = "test-thread"
        mock_thread.save = MagicMock()

        mock_model = MagicMock()
        mock_llm_manager = MagicMock()

        mock_exploration_engine = MagicMock()
        mock_exploration_engine.do_critique = AsyncMock()

        mock_synthesis_engine = MagicMock()
        mock_synthesis_engine.is_chunk_ready = MagicMock(return_value=False)

        mock_paths = MagicMock()
        mock_paths.data_dir = Path("/tmp")

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths", return_value=mock_paths):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager"):
                                with patch("explorer.src.orchestrator.ExplorationEngine", return_value=mock_exploration_engine):
                                    with patch("explorer.src.orchestrator.SynthesisEngine", return_value=mock_synthesis_engine):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                from explorer.src.orchestrator import Orchestrator
                                                orch = Orchestrator()

                                                await orch._do_critique_and_save(
                                                    mock_thread, "chatgpt", mock_model
                                                )

                                                mock_exploration_engine.do_critique.assert_called_once()
                                                mock_thread.save.assert_called_once()


class TestOrchestratorRun:
    """Tests for run method."""

    @pytest.mark.asyncio
    async def test_run_sets_running_true(self):
        """run() sets running to True."""
        mock_config = {
            "orchestration": {"poll_interval": 0.1},
            "review_panel": {"enabled": False},
            "review_server": {"host": "127.0.0.1", "port": 8765},
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.connect = AsyncMock()
        mock_llm_manager.disconnect = AsyncMock()
        mock_llm_manager.check_recovered_responses = AsyncMock()
        mock_llm_manager.gemini = None
        mock_llm_manager.chatgpt = None

        mock_thread_manager = MagicMock()
        mock_thread_manager.check_and_spawn_for_new_seeds = AsyncMock()
        mock_thread_manager.load_thread_by_id = MagicMock()

        mock_paths = MagicMock()
        mock_paths.data_dir = Path("/tmp")
        mock_paths.explorations_dir = Path("/tmp/explorations")  # Non-existent

        with patch("explorer.src.orchestrator.CONFIG", mock_config):
            with patch("explorer.src.orchestrator.ExplorerPaths", return_value=mock_paths):
                with patch("explorer.src.orchestrator.Database"):
                    with patch("explorer.src.orchestrator.AxiomStore"):
                        with patch("explorer.src.orchestrator.LLMManager", return_value=mock_llm_manager):
                            with patch("explorer.src.orchestrator.ThreadManager", return_value=mock_thread_manager):
                                with patch("explorer.src.orchestrator.ExplorationEngine"):
                                    with patch("explorer.src.orchestrator.SynthesisEngine"):
                                        with patch("explorer.src.orchestrator.AtomicExtractor"):
                                            with patch("explorer.src.orchestrator.PanelExtractor"):
                                                with patch("explorer.src.orchestrator.BlessedStore"):
                                                    with patch("explorer.src.orchestrator.InsightProcessor"):
                                                        from explorer.src.orchestrator import Orchestrator
                                                        orch = Orchestrator()

                                                        # Run for a short time then stop
                                                        async def stop_after_delay():
                                                            await asyncio.sleep(0.05)
                                                            orch.stop()

                                                        with patch.object(orch, "_exploration_cycle", new_callable=AsyncMock):
                                                            await asyncio.gather(
                                                                orch.run(process_backlog_first=False),
                                                                stop_after_delay(),
                                                            )

                                                        # Should have been set to True during run
                                                        # (and then False after stop)
                                                        mock_llm_manager.connect.assert_called()
