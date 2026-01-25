"""
Tests for the status blueprint API endpoints.

Tests cover:
- GET /api/status - Get status of all components
- GET /api/logs/<component> - Get recent logs
- GET /api/config - Get configuration
- POST /api/config - Update configuration
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestStatusEndpoint:
    """Tests for GET /api/status."""

    def test_status_returns_all_components(self, flask_client, mock_process_manager):
        """Status endpoint returns info for all components."""
        response = flask_client.get("/api/status")
        assert response.status_code == 200

        data = response.get_json()
        assert "pool" in data
        assert "explorer" in data
        assert "documenter" in data
        assert "researcher" in data
        assert "orchestrator" in data
        assert "backends" in data
        assert "stats" in data

    def test_status_shows_pool_not_running(self, flask_client, mock_process_manager):
        """Status shows pool as not running when process not started."""
        mock_process_manager.is_running.return_value = False

        # Also mock health check to ensure no external pool is detected
        with patch("control.blueprints.status.check_pool_health", return_value=False):
            response = flask_client.get("/api/status")

        data = response.get_json()
        assert data["pool"]["running"] is False

    def test_status_shows_pool_running(self, flask_client, mock_process_manager):
        """Status shows pool as running when process is active."""
        mock_process_manager.is_running.side_effect = lambda x: x == "pool"
        mock_process_manager.get_pid.return_value = 12345

        with patch("control.blueprints.status.check_pool_health", return_value=True):
            response = flask_client.get("/api/status")

        data = response.get_json()
        assert data["pool"]["running"] is True
        assert data["pool"]["pid"] == 12345

    def test_status_detects_external_pool(self, flask_client, mock_process_manager):
        """Status detects externally running pool (not started by us)."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.get_pid.return_value = None

        with patch("control.blueprints.status.check_pool_health", return_value=True):
            response = flask_client.get("/api/status")

        data = response.get_json()
        assert data["pool"]["running"] is True
        assert data["pool"]["external"] is True

    def test_status_shows_explorer_state(self, flask_client, mock_process_manager):
        """Status shows explorer running state."""
        mock_process_manager.is_running.side_effect = lambda x: x == "explorer"
        mock_process_manager.get_pid.side_effect = lambda x: 9999 if x == "explorer" else None

        response = flask_client.get("/api/status")
        data = response.get_json()

        assert data["explorer"]["running"] is True
        assert data["explorer"]["pid"] == 9999

    def test_status_shows_documenter_state(self, flask_client, mock_process_manager):
        """Status shows documenter running state."""
        mock_process_manager.is_running.side_effect = lambda x: x == "documenter"
        mock_process_manager.get_pid.side_effect = lambda x: 8888 if x == "documenter" else None

        response = flask_client.get("/api/status")
        data = response.get_json()

        assert data["documenter"]["running"] is True
        assert data["documenter"]["pid"] == 8888

    def test_status_shows_researcher_state(self, flask_client, mock_process_manager):
        """Status shows researcher running state."""
        mock_process_manager.is_running.side_effect = lambda x: x == "researcher"
        mock_process_manager.get_pid.side_effect = lambda x: 7777 if x == "researcher" else None

        response = flask_client.get("/api/status")
        data = response.get_json()

        assert data["researcher"]["running"] is True
        assert data["researcher"]["pid"] == 7777

    def test_status_includes_backend_info(self, flask_client, mock_process_manager):
        """Status includes backend configuration."""
        response = flask_client.get("/api/status")
        data = response.get_json()

        assert "backends" in data
        # Backends come from config, so they may be empty or populated
        assert isinstance(data["backends"], dict)

    def test_status_includes_stats(self, flask_client, mock_process_manager):
        """Status includes statistics."""
        response = flask_client.get("/api/status")
        data = response.get_json()

        assert "stats" in data
        assert "documenter" in data["stats"]
        assert "explorer" in data["stats"]
        assert "researcher" in data["stats"]


class TestLogsEndpoint:
    """Tests for GET /api/logs/<component>."""

    def test_logs_valid_component(self, flask_client, temp_dir):
        """Logs endpoint returns logs for valid component."""
        # Create a mock log file
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "pool.jsonl"
        log_entry = {"event_type": "pool.test", "level": "INFO", "message": "Test log"}
        log_file.write_text(json.dumps(log_entry) + "\n")

        with patch("control.blueprints.status.LOGS_DIR", logs_dir):
            response = flask_client.get("/api/logs/pool")

        assert response.status_code == 200
        data = response.get_json()
        assert "logs" in data
        assert len(data["logs"]) == 1
        assert data["logs"][0]["event_type"] == "pool.test"

    def test_logs_invalid_component_returns_400(self, flask_client):
        """Logs endpoint returns 400 for invalid component."""
        response = flask_client.get("/api/logs/invalid_component")

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "Unknown component" in data["error"]

    def test_logs_missing_file_returns_empty(self, flask_client, temp_dir):
        """Logs endpoint returns empty list if log file doesn't exist."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.blueprints.status.LOGS_DIR", logs_dir):
            response = flask_client.get("/api/logs/explorer")

        assert response.status_code == 200
        data = response.get_json()
        assert data["logs"] == []

    def test_logs_respects_limit_parameter(self, flask_client, temp_dir):
        """Logs endpoint respects limit query parameter."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "documenter.jsonl"

        # Write 10 log entries
        with open(log_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"event_type": f"event.{i}", "index": i}) + "\n")

        with patch("control.blueprints.status.LOGS_DIR", logs_dir):
            response = flask_client.get("/api/logs/documenter?limit=3")

        assert response.status_code == 200
        data = response.get_json()
        assert len(data["logs"]) == 3
        # Should return last 3 entries (7, 8, 9)
        assert data["logs"][0]["index"] == 7

    def test_logs_handles_malformed_json(self, flask_client, temp_dir):
        """Logs endpoint skips malformed JSON lines."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "llm.jsonl"

        with open(log_file, "w") as f:
            f.write(json.dumps({"event_type": "valid.event"}) + "\n")
            f.write("this is not valid json\n")
            f.write(json.dumps({"event_type": "another.valid"}) + "\n")

        with patch("control.blueprints.status.LOGS_DIR", logs_dir):
            response = flask_client.get("/api/logs/llm")

        assert response.status_code == 200
        data = response.get_json()
        assert len(data["logs"]) == 2

    def test_logs_accepts_all_valid_components(self, flask_client, temp_dir):
        """Logs endpoint accepts all valid component names."""
        valid_components = ["pool", "explorer", "documenter", "orchestrator", "llm", "researcher"]
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with patch("control.blueprints.status.LOGS_DIR", logs_dir):
            for component in valid_components:
                response = flask_client.get(f"/api/logs/{component}")
                assert response.status_code == 200, f"Failed for component: {component}"


class TestConfigEndpoint:
    """Tests for GET/POST /api/config."""

    def test_get_config_returns_config(self, flask_client):
        """GET /api/config returns current configuration."""
        mock_config = {
            "llm": {"backends": {"gemini": {"enabled": True}}},
            "explorer": {"enabled": True},
        }

        with patch("control.blueprints.status.load_config", return_value=mock_config):
            response = flask_client.get("/api/config")

        assert response.status_code == 200
        data = response.get_json()
        assert data == mock_config

    def test_post_config_updates_config(self, flask_client):
        """POST /api/config updates configuration."""
        new_config = {
            "llm": {"backends": {"claude": {"enabled": True}}},
        }

        with patch("control.blueprints.status.save_config") as mock_save:
            response = flask_client.post(
                "/api/config",
                data=json.dumps(new_config),
                content_type="application/json",
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "updated"
        mock_save.assert_called_once_with(new_config)

    def test_post_config_handles_save_error(self, flask_client):
        """POST /api/config returns 500 on save error."""
        with patch("control.blueprints.status.save_config", side_effect=IOError("Write failed")):
            response = flask_client.post(
                "/api/config",
                data=json.dumps({"test": "config"}),
                content_type="application/json",
            )

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data


class TestHealthChecks:
    """Tests for health check utility functions."""

    def test_check_pool_health_success(self):
        """check_pool_health returns True when pool responds."""
        from control.blueprints.status import check_pool_health

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = check_pool_health("127.0.0.1", 9000)

        assert result is True

    def test_check_pool_health_failure(self):
        """check_pool_health returns False when pool doesn't respond."""
        from control.blueprints.status import check_pool_health

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = check_pool_health("127.0.0.1", 9000)

        assert result is False

    def test_check_orchestrator_health_success(self):
        """check_orchestrator_health returns True when orchestrator responds."""
        from control.blueprints.status import check_orchestrator_health

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = check_orchestrator_health()

        assert result is True

    def test_check_orchestrator_health_failure(self):
        """check_orchestrator_health returns False when orchestrator doesn't respond."""
        from control.blueprints.status import check_orchestrator_health

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = check_orchestrator_health()

        assert result is False


class TestGetStats:
    """Tests for get_stats utility function."""

    def test_get_stats_returns_structure(self, temp_dir):
        """get_stats returns expected structure."""
        from control.blueprints.status import get_stats

        with patch("control.blueprints.status.LOGS_DIR", temp_dir / "logs"), \
             patch("control.blueprints.status.FANO_ROOT", temp_dir):
            stats = get_stats()

        assert "documenter" in stats
        assert "explorer" in stats
        assert "researcher" in stats
        assert "sections" in stats["documenter"]
        assert "blessed" in stats["explorer"]

    def test_get_stats_counts_blessed_insights(self, temp_dir):
        """get_stats counts blessed insight files."""
        from control.blueprints.status import get_stats

        # Create blessed directory with some files
        blessed_dir = temp_dir / "explorer" / "data" / "chunks" / "insights" / "blessed"
        blessed_dir.mkdir(parents=True, exist_ok=True)
        (blessed_dir / "insight1.json").write_text("{}")
        (blessed_dir / "insight2.json").write_text("{}")
        (blessed_dir / "insight3.json").write_text("{}")

        with patch("control.blueprints.status.LOGS_DIR", temp_dir / "logs"), \
             patch("control.blueprints.status.FANO_ROOT", temp_dir):
            stats = get_stats()

        assert stats["explorer"]["blessed"] == 3

    def test_get_stats_counts_active_threads(self, temp_dir):
        """get_stats counts active exploration threads."""
        from control.blueprints.status import get_stats

        # Create explorations directory with threads
        explorations_dir = temp_dir / "explorer" / "data" / "explorations"
        explorations_dir.mkdir(parents=True, exist_ok=True)
        (explorations_dir / "thread1.json").write_text(json.dumps({"status": "active"}))
        (explorations_dir / "thread2.json").write_text(json.dumps({"status": "active"}))
        (explorations_dir / "thread3.json").write_text(json.dumps({"status": "completed"}))

        with patch("control.blueprints.status.LOGS_DIR", temp_dir / "logs"), \
             patch("control.blueprints.status.FANO_ROOT", temp_dir):
            stats = get_stats()

        assert stats["explorer"]["threads"] == 2

    def test_get_stats_reads_documenter_log(self, temp_dir):
        """get_stats reads documenter stats from log."""
        from control.blueprints.status import get_stats

        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        doc_log = logs_dir / "documenter.jsonl"

        with open(doc_log, "w") as f:
            f.write(json.dumps({"event_type": "other.event"}) + "\n")
            f.write(json.dumps({
                "event_type": "documenter.session.summary",
                "sections": 15,
                "consensus_calls": 42
            }) + "\n")

        with patch("control.blueprints.status.LOGS_DIR", logs_dir), \
             patch("control.blueprints.status.FANO_ROOT", temp_dir):
            stats = get_stats()

        assert stats["documenter"]["sections"] == 15
        assert stats["documenter"]["consensus_calls"] == 42
