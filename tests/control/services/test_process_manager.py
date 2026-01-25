"""
Tests for ProcessManager - thread-safe process lifecycle management.

Tests cover:
- Process start/stop operations
- PID tracking
- Running state detection
- Dependency checking
- Health waiting
- Thread safety
- Log file management
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestProcessManagerBasic:
    """Basic ProcessManager functionality tests."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager with temp log directory."""
        with patch("control.services.process_manager.LOGS_DIR", temp_dir / "logs"):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                return ProcessManager()

    def test_initial_state(self, pm):
        """ProcessManager starts with no running processes."""
        assert pm.is_running("pool") is False
        assert pm.is_running("explorer") is False
        assert pm.is_running("documenter") is False
        assert pm.get_pid("pool") is None

    def test_is_running_false_for_unknown(self, pm):
        """is_running returns False for unknown components."""
        assert pm.is_running("unknown_component") is False

    def test_get_returns_none_initially(self, pm):
        """get() returns None for unstarted components."""
        assert pm.get("pool") is None
        assert pm.get("explorer") is None


class TestProcessStart:
    """Tests for starting processes."""

    @pytest.fixture
    def pm(self, temp_dir, mock_subprocess):
        """Create ProcessManager with mocked subprocess."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_start_pool(self, pm, mock_subprocess):
        """start_pool creates subprocess and registers it."""
        mock_popen, mock_proc = mock_subprocess

        proc = pm.start_pool()

        assert proc is mock_proc
        mock_popen.assert_called_once()
        # After starting, should be registered
        assert pm.get("pool") is mock_proc

    def test_start_explorer(self, pm, mock_subprocess):
        """start_explorer creates subprocess."""
        mock_popen, mock_proc = mock_subprocess

        proc = pm.start_explorer()

        assert proc is mock_proc
        mock_popen.assert_called_once()
        assert pm.get("explorer") is mock_proc

    def test_start_explorer_with_mode(self, pm, mock_subprocess):
        """start_explorer accepts mode parameter."""
        mock_popen, mock_proc = mock_subprocess

        pm.start_explorer(mode="resume")

        # Check the call included the mode argument
        call_args = mock_popen.call_args
        assert "resume" in call_args[0][0]

    def test_start_documenter(self, pm, mock_subprocess):
        """start_documenter creates subprocess."""
        mock_popen, mock_proc = mock_subprocess

        proc = pm.start_documenter()

        assert proc is mock_proc
        assert pm.get("documenter") is mock_proc

    def test_start_researcher(self, pm, mock_subprocess):
        """start_researcher creates subprocess."""
        mock_popen, mock_proc = mock_subprocess

        proc = pm.start_researcher()

        assert proc is mock_proc
        assert pm.get("researcher") is mock_proc

    def test_start_orchestrator(self, pm, mock_subprocess):
        """start_orchestrator creates subprocess."""
        mock_popen, mock_proc = mock_subprocess

        proc = pm.start_orchestrator()

        assert proc is mock_proc
        assert pm.get("orchestrator") is mock_proc


class TestProcessRunningState:
    """Tests for process running state detection."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_is_running_with_active_process(self, pm):
        """is_running returns True for process with poll() = None."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=None)  # Still running

        pm.set("pool", mock_proc)

        assert pm.is_running("pool") is True

    def test_is_running_with_exited_process(self, pm):
        """is_running returns False for exited process."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=0)  # Exited with code 0

        pm.set("pool", mock_proc)

        assert pm.is_running("pool") is False

    def test_get_pid_running_process(self, pm):
        """get_pid returns PID for running process."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.pid = 54321

        pm.set("pool", mock_proc)

        assert pm.get_pid("pool") == 54321

    def test_get_pid_exited_process(self, pm):
        """get_pid returns None for exited process."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=0)
        mock_proc.pid = 54321

        pm.set("pool", mock_proc)

        assert pm.get_pid("pool") is None


class TestProcessStop:
    """Tests for stopping processes."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_stop_running_process(self, pm):
        """stop() terminates running process and returns True."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        pm.set("pool", mock_proc)

        result = pm.stop("pool")

        assert result is True
        mock_proc.terminate.assert_called_once()
        assert pm.get("pool") is None

    def test_stop_not_running_process(self, pm):
        """stop() returns False if process not running."""
        result = pm.stop("pool")
        assert result is False

    def test_stop_already_exited(self, pm):
        """stop() returns False if process already exited."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=0)

        pm.set("pool", mock_proc)

        result = pm.stop("pool")
        assert result is False

    def test_stop_kills_on_timeout(self, pm):
        """stop() kills process if terminate times out."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock(side_effect=TimeoutError())
        mock_proc.kill = MagicMock()

        pm.set("pool", mock_proc)

        result = pm.stop("pool")

        assert result is True
        mock_proc.kill.assert_called_once()


class TestCleanupAll:
    """Tests for cleanup_all() method."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_cleanup_all_stops_all(self, pm):
        """cleanup_all() terminates all running processes."""
        mock_procs = {}
        for name in ["pool", "explorer", "documenter"]:
            mock_proc = MagicMock()
            mock_proc.poll = MagicMock(return_value=None)
            mock_proc.terminate = MagicMock()
            mock_proc.wait = MagicMock()
            pm.set(name, mock_proc)
            mock_procs[name] = mock_proc

        pm.cleanup_all()

        for name, proc in mock_procs.items():
            proc.terminate.assert_called()
            assert pm.get(name) is None

    def test_cleanup_all_handles_errors(self, pm):
        """cleanup_all() continues despite errors."""
        mock_proc1 = MagicMock()
        mock_proc1.poll = MagicMock(return_value=None)
        mock_proc1.terminate = MagicMock(side_effect=Exception("Error"))
        mock_proc1.kill = MagicMock()

        mock_proc2 = MagicMock()
        mock_proc2.poll = MagicMock(return_value=None)
        mock_proc2.terminate = MagicMock()
        mock_proc2.wait = MagicMock()

        pm.set("pool", mock_proc1)
        pm.set("explorer", mock_proc2)

        # Should not raise
        pm.cleanup_all()

        # Both should be cleared
        assert pm.get("pool") is None
        assert pm.get("explorer") is None


class TestDependencyChecking:
    """Tests for dependency checking methods."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_check_dependencies_pool_has_none(self, pm):
        """Pool has no dependencies."""
        ready, missing = pm.check_dependencies_ready("pool")

        assert ready is True
        assert missing == []

    def test_check_dependencies_explorer_needs_pool(self, pm):
        """Explorer requires pool to be running."""
        ready, missing = pm.check_dependencies_ready("explorer")

        assert ready is False
        assert "pool" in missing

    def test_check_dependencies_satisfied(self, pm):
        """Dependencies satisfied when pool is running."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=None)
        pm.set("pool", mock_proc)

        ready, missing = pm.check_dependencies_ready("explorer")

        assert ready is True
        assert missing == []

    def test_service_dependencies_defined(self, pm):
        """All expected service dependencies are defined."""
        deps = pm.SERVICE_DEPENDENCIES

        assert "pool" in deps
        assert "explorer" in deps
        assert "documenter" in deps
        assert deps["pool"] == []  # Pool has no deps


class TestStartWithDeps:
    """Tests for start_with_deps() method."""

    @pytest.fixture
    def pm(self, temp_dir, mock_subprocess):
        """Create ProcessManager with mocked subprocess."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_start_with_deps_auto_starts_pool(self, pm, mock_subprocess, sample_config):
        """start_with_deps auto-starts pool for explorer."""
        mock_popen, mock_proc = mock_subprocess

        with patch.object(pm, "wait_for_pool_health", return_value=True):
            result = pm.start_with_deps("explorer", sample_config)

        # Pool should have been started
        assert result is True

    def test_start_with_deps_fails_if_pool_unhealthy(self, pm, mock_subprocess, sample_config):
        """start_with_deps fails if pool doesn't become healthy."""
        mock_popen, mock_proc = mock_subprocess

        with patch.object(pm, "wait_for_pool_health", return_value=False):
            result = pm.start_with_deps("explorer", sample_config)

        assert result is False

    def test_start_with_deps_pool_has_no_deps(self, pm, mock_subprocess, sample_config):
        """start_with_deps for pool doesn't require other services."""
        mock_popen, mock_proc = mock_subprocess

        result = pm.start_with_deps("pool", sample_config)

        assert result is True


class TestHealthWaiting:
    """Tests for health waiting methods."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_wait_for_pool_health_success(self, pm):
        """wait_for_pool_health returns True when healthy."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = pm.wait_for_pool_health(timeout_seconds=5)

        assert result is True

    def test_wait_for_pool_health_timeout(self, pm):
        """wait_for_pool_health returns False on timeout."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = pm.wait_for_pool_health(timeout_seconds=1, poll_interval=0.1)

        assert result is False

    def test_wait_for_orchestrator_health_success(self, pm):
        """wait_for_orchestrator_health returns True when healthy."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = pm.wait_for_orchestrator_health(timeout_seconds=5)

        assert result is True


class TestThreadSafety:
    """Tests for thread-safe access."""

    @pytest.fixture
    def pm(self, temp_dir):
        """Create ProcessManager."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager()

    def test_concurrent_access_no_errors(self, pm):
        """Concurrent access doesn't cause errors."""
        errors = []

        def reader():
            try:
                for _ in range(100):
                    pm.is_running("pool")
                    pm.get_pid("explorer")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                mock_proc = MagicMock()
                mock_proc.poll = MagicMock(return_value=None)
                for _ in range(100):
                    pm.set("pool", mock_proc)
                    pm.set("pool", None)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_stop_is_atomic(self, pm):
        """stop() operation is atomic under concurrent access."""
        mock_proc = MagicMock()
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        pm.set("pool", mock_proc)

        results = []

        def try_stop():
            results.append(pm.stop("pool"))

        threads = [threading.Thread(target=try_stop) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should return True (the one that actually stopped it)
        assert results.count(True) == 1
        assert results.count(False) == 4


class TestLogFileManagement:
    """Tests for log file handling."""

    @pytest.fixture
    def pm(self, temp_dir, mock_subprocess):
        """Create ProcessManager with mocked subprocess."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.services.process_manager.LOGS_DIR", logs_dir):
            with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                from control.services.process_manager import ProcessManager
                yield ProcessManager(), logs_dir

    def test_log_file_created_on_start(self, pm, mock_subprocess):
        """Log file is created when process starts."""
        process_manager, logs_dir = pm
        mock_popen, mock_proc = mock_subprocess

        process_manager.start_pool()

        log_file = logs_dir / "pool.log"
        assert log_file.exists()

    def test_log_file_truncated_on_restart(self, pm, mock_subprocess):
        """Log file is truncated when process restarts."""
        process_manager, logs_dir = pm
        mock_popen, mock_proc = mock_subprocess

        # Create existing log file with content
        log_file = logs_dir / "pool.log"
        log_file.write_text("Old log content that should be removed")

        process_manager.start_pool()

        # Log file should be truncated (empty or minimal content)
        content = log_file.read_text()
        assert "Old log content" not in content
