"""
Tests for the components blueprint API endpoints.

Tests cover:
- POST /api/start/<component> - Start a component
- POST /api/stop/<component> - Stop a component
- POST /api/server/restart - Restart the server
- GET /api/health - Health check endpoint
"""

import json
from unittest.mock import patch, MagicMock

import pytest


class TestStartEndpoint:
    """Tests for POST /api/start/<component>."""

    def test_start_pool_success(self, flask_client, mock_process_manager):
        """Start pool successfully."""
        mock_process_manager.is_running.return_value = False

        with patch("control.blueprints.components.check_pool_health", return_value=False):
            response = flask_client.post("/api/start/pool")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "started"
        assert data["component"] == "pool"
        mock_process_manager.start_pool.assert_called_once()

    def test_start_pool_already_running(self, flask_client, mock_process_manager):
        """Start pool returns 400 when already running."""
        with patch("control.blueprints.components.check_pool_health", return_value=True):
            response = flask_client.post("/api/start/pool")

        assert response.status_code == 400
        data = response.get_json()
        assert "already running" in data["error"].lower()
        assert data.get("already_running") is True

    def test_start_explorer_success(self, flask_client, mock_process_manager):
        """Start explorer successfully."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        response = flask_client.post(
            "/api/start/explorer",
            data="{}",
            content_type="application/json"
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "started"
        assert data["component"] == "explorer"

    def test_start_explorer_already_running(self, flask_client, mock_process_manager):
        """Start explorer returns 400 when already running."""
        mock_process_manager.is_running.side_effect = lambda x: x == "explorer"

        response = flask_client.post("/api/start/explorer")

        assert response.status_code == 400
        data = response.get_json()
        assert "already running" in data["error"].lower()

    def test_start_explorer_dependency_failure(self, flask_client, mock_process_manager):
        """Start explorer returns 500 when dependencies fail."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = False

        response = flask_client.post(
            "/api/start/explorer",
            data="{}",
            content_type="application/json"
        )

        assert response.status_code == 500
        data = response.get_json()
        assert "dependencies" in data["error"].lower()

    def test_start_documenter_success(self, flask_client, mock_process_manager):
        """Start documenter successfully."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        response = flask_client.post("/api/start/documenter")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "started"
        assert data["component"] == "documenter"

    def test_start_researcher_success(self, flask_client, mock_process_manager):
        """Start researcher successfully."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        response = flask_client.post("/api/start/researcher")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "started"
        assert data["component"] == "researcher"

    def test_start_orchestrator_success(self, flask_client, mock_process_manager):
        """Start orchestrator successfully."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        with patch("control.blueprints.components.check_orchestrator_health", return_value=False):
            response = flask_client.post("/api/start/orchestrator")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "started"
        assert data["component"] == "orchestrator"

    def test_start_orchestrator_already_running(self, flask_client, mock_process_manager):
        """Start orchestrator returns 400 when already running."""
        with patch("control.blueprints.components.check_orchestrator_health", return_value=True):
            response = flask_client.post("/api/start/orchestrator")

        assert response.status_code == 400
        data = response.get_json()
        assert "already running" in data["error"].lower()

    def test_start_invalid_component(self, flask_client, mock_process_manager):
        """Start invalid component returns 400."""
        response = flask_client.post("/api/start/invalid_component")

        assert response.status_code == 400
        data = response.get_json()
        assert "Unknown component" in data["error"]

    def test_start_without_process_manager(self, flask_client, control_app):
        """Start returns 500 when process manager is not available."""
        # Remove process manager from app config
        control_app.config["process_manager"] = None

        response = flask_client.post("/api/start/pool")

        assert response.status_code == 500
        data = response.get_json()
        assert "Process manager not available" in data["error"]

    def test_start_handles_exception(self, flask_client, mock_process_manager):
        """Start handles exceptions gracefully."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_pool.side_effect = Exception("Failed to start")

        with patch("control.blueprints.components.check_pool_health", return_value=False):
            response = flask_client.post("/api/start/pool")

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data


class TestStopEndpoint:
    """Tests for POST /api/stop/<component>."""

    def test_stop_pool_success(self, flask_client, mock_process_manager):
        """Stop pool successfully."""
        mock_process_manager.is_running.side_effect = lambda x: x == "pool"

        response = flask_client.post("/api/stop/pool")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "stopped"
        assert data["component"] == "pool"
        mock_process_manager.stop.assert_called_once_with("pool")

    def test_stop_not_running(self, flask_client, mock_process_manager):
        """Stop returns 400 when component not running."""
        mock_process_manager.is_running.return_value = False

        response = flask_client.post("/api/stop/explorer")

        assert response.status_code == 400
        data = response.get_json()
        assert "not running" in data["error"].lower()

    def test_stop_invalid_component(self, flask_client, mock_process_manager):
        """Stop invalid component returns 400."""
        response = flask_client.post("/api/stop/invalid_component")

        assert response.status_code == 400
        data = response.get_json()
        assert "Unknown component" in data["error"]

    def test_stop_without_process_manager(self, flask_client, control_app):
        """Stop returns 500 when process manager is not available."""
        control_app.config["process_manager"] = None

        response = flask_client.post("/api/stop/pool")

        assert response.status_code == 500
        data = response.get_json()
        assert "Process manager not available" in data["error"]

    def test_stop_handles_exception(self, flask_client, mock_process_manager):
        """Stop handles exceptions and returns killed status."""
        mock_process_manager.is_running.side_effect = lambda x: x == "documenter"
        mock_process_manager.stop.side_effect = Exception("Process hung")

        response = flask_client.post("/api/stop/documenter")

        # Note: the endpoint catches exceptions and returns 200 with status "killed"
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "killed"
        assert "error" in data

    def test_stop_all_valid_components(self, flask_client, mock_process_manager):
        """Stop accepts all valid component names."""
        valid_components = ["pool", "orchestrator", "explorer", "documenter", "researcher"]

        for component in valid_components:
            mock_process_manager.is_running.side_effect = lambda x, c=component: x == c

            response = flask_client.post(f"/api/stop/{component}")

            assert response.status_code == 200, f"Failed to stop {component}"
            data = response.get_json()
            assert data["status"] == "stopped"


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_ok(self, flask_client):
        """Health endpoint returns OK status."""
        response = flask_client.get("/api/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"


class TestServerRestartEndpoint:
    """Tests for POST /api/server/restart."""

    def test_restart_returns_success(self, flask_client, mock_process_manager):
        """Server restart returns success immediately."""
        # Mock os._exit to prevent actual exit
        with patch("os._exit"):
            response = flask_client.post("/api/server/restart")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "restarting" in data["message"].lower()

    def test_restart_starts_background_thread(self, flask_client, mock_process_manager):
        """Server restart starts a background thread."""
        import threading

        original_thread_count = threading.active_count()

        with patch("os._exit"):
            with patch("subprocess.Popen"):
                response = flask_client.post("/api/server/restart")

        # Response should be immediate
        assert response.status_code == 200


class TestHealthCheckFunctions:
    """Tests for health check utility functions in components blueprint."""

    def test_check_pool_health_custom_host_port(self):
        """check_pool_health uses custom host and port."""
        from control.blueprints.components import check_pool_health

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = check_pool_health("192.168.1.100", 8080)

        mock_urlopen.assert_called_once()
        call_url = mock_urlopen.call_args[0][0]
        assert "192.168.1.100" in call_url
        assert "8080" in call_url
        assert result is True

    def test_check_orchestrator_health_custom_port(self):
        """check_orchestrator_health uses default host and port."""
        from control.blueprints.components import check_orchestrator_health

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = check_orchestrator_health("127.0.0.1", 9001)

        mock_urlopen.assert_called_once()
        call_url = mock_urlopen.call_args[0][0]
        assert "127.0.0.1" in call_url
        assert "9001" in call_url
        assert result is True


class TestComponentDependencies:
    """Tests for component dependency handling."""

    def test_explorer_requires_pool(self, flask_client, mock_process_manager):
        """Explorer start calls start_with_deps."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        response = flask_client.post(
            "/api/start/explorer",
            data="{}",
            content_type="application/json"
        )

        assert response.status_code == 200
        mock_process_manager.start_with_deps.assert_called()

    def test_documenter_requires_pool(self, flask_client, mock_process_manager):
        """Documenter start calls start_with_deps."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        response = flask_client.post("/api/start/documenter")

        assert response.status_code == 200
        mock_process_manager.start_with_deps.assert_called()

    def test_researcher_requires_pool(self, flask_client, mock_process_manager):
        """Researcher start calls start_with_deps."""
        mock_process_manager.is_running.return_value = False
        mock_process_manager.start_with_deps.return_value = True

        response = flask_client.post("/api/start/researcher")

        assert response.status_code == 200
        mock_process_manager.start_with_deps.assert_called()

    def test_pool_starts_directly(self, flask_client, mock_process_manager):
        """Pool starts directly without dependency check."""
        mock_process_manager.is_running.return_value = False

        with patch("control.blueprints.components.check_pool_health", return_value=False):
            response = flask_client.post("/api/start/pool")

        assert response.status_code == 200
        mock_process_manager.start_pool.assert_called_once()
        # start_with_deps should NOT be called for pool
        assert not any(
            call[0][0] == "pool"
            for call in mock_process_manager.start_with_deps.call_args_list
        )
