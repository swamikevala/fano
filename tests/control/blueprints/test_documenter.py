"""
Tests for the documenter blueprint API endpoints.

Tests cover:
- GET /api/document - Get document content
- GET /api/documenter/activity - Get documenter activity
- GET /api/documenter/pipeline - Get pipeline status
- GET /api/document/versions - Get version history
- POST /api/document/fix-formatting - Start formatting fix
- GET /api/document/fix-formatting/status - Check formatting status
- Thread safety for formatting fix operations
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestDocumentEndpoint:
    """Tests for GET /api/document."""

    def test_document_returns_content(self, flask_client, temp_dir):
        """Document endpoint returns document content."""
        doc_path = temp_dir / "document" / "main.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text("# Test Document\n\nSection 1.\n\nSection 2.")

        with patch("control.blueprints.documenter.DOC_PATH", doc_path):
            response = flask_client.get("/api/document")

        assert response.status_code == 200
        data = response.get_json()
        assert "content" in data
        assert "# Test Document" in data["content"]
        assert data["sections"] >= 0

    def test_document_missing_file(self, flask_client, temp_dir):
        """Document endpoint handles missing file."""
        doc_path = temp_dir / "nonexistent" / "document.md"

        with patch("control.blueprints.documenter.DOC_PATH", doc_path):
            response = flask_client.get("/api/document")

        assert response.status_code == 200
        data = response.get_json()
        assert "error" in data


class TestDocumenterActivity:
    """Tests for GET /api/documenter/activity."""

    def test_activity_returns_data(self, flask_client, mock_process_manager):
        """Activity endpoint returns documenter status."""
        mock_process_manager.is_running.side_effect = lambda x: x == "documenter"

        response = flask_client.get("/api/documenter/activity")

        assert response.status_code == 200
        data = response.get_json()
        # API returns is_running, not running
        assert "is_running" in data

    def test_activity_not_running(self, flask_client, mock_process_manager):
        """Activity endpoint shows not running."""
        mock_process_manager.is_running.return_value = False

        response = flask_client.get("/api/documenter/activity")

        assert response.status_code == 200
        data = response.get_json()
        assert data["is_running"] is False


class TestDocumenterPipeline:
    """Tests for GET /api/documenter/pipeline."""

    def test_pipeline_orchestrator_unavailable(self, flask_client):
        """Pipeline returns error when orchestrator unavailable."""
        import requests as requests_module
        with patch("control.blueprints.documenter.get_orchestrator_url",
                   return_value="http://127.0.0.1:9001"):
            with patch("control.blueprints.documenter.requests.get",
                       side_effect=requests_module.RequestException("Connection refused")):
                response = flask_client.get("/api/documenter/pipeline")

        assert response.status_code == 503
        data = response.get_json()
        assert "error" in data

    def test_pipeline_returns_data(self, flask_client, mock_orchestrator_response):
        """Pipeline returns task data from orchestrator."""
        mock_resp = mock_orchestrator_response({
            "tasks": [
                {"id": "task-1", "module": "documenter", "phase": "draft"},
            ]
        })

        with patch("control.blueprints.documenter.get_orchestrator_url",
                   return_value="http://127.0.0.1:9001"):
            with patch("requests.get", return_value=mock_resp):
                response = flask_client.get("/api/documenter/pipeline")

        assert response.status_code == 200


class TestFormattingFix:
    """Tests for formatting fix endpoints."""

    def test_fix_formatting_starts_background_task(self, flask_client, control_app):
        """POST /api/document/fix-formatting starts background task."""
        # Reset formatting state
        control_app.config["formatting_fix_state"] = {
            "in_progress": False,
            "result": None,
            "progress": None,
            "sections_fixed": 0,
        }

        with patch("control.blueprints.documenter._run_formatting_fix"):
            response = flask_client.post("/api/document/fix-formatting")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["status"] == "started"

    def test_fix_formatting_already_running(self, flask_client, control_app):
        """POST /api/document/fix-formatting returns already_running when in progress."""
        control_app.config["formatting_fix_state"] = {
            "in_progress": True,
            "result": None,
            "progress": None,
            "sections_fixed": 0,
        }

        response = flask_client.post("/api/document/fix-formatting")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "already_running"

    def test_fix_formatting_status_idle(self, flask_client, control_app):
        """GET /api/document/fix-formatting/status returns idle when not running."""
        control_app.config["formatting_fix_state"] = {
            "in_progress": False,
            "result": None,
            "progress": None,
            "sections_fixed": 0,
        }

        response = flask_client.get("/api/document/fix-formatting/status")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "idle"

    def test_fix_formatting_status_in_progress(self, flask_client, control_app):
        """GET /api/document/fix-formatting/status returns progress when running."""
        control_app.config["formatting_fix_state"] = {
            "in_progress": True,
            "result": None,
            "progress": {"current": 5, "total": 10},
            "sections_fixed": 3,
        }

        response = flask_client.get("/api/document/fix-formatting/status")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "in_progress"
        assert data["progress"]["current"] == 5
        assert data["sections_fixed"] == 3

    def test_fix_formatting_status_complete(self, flask_client, control_app):
        """GET /api/document/fix-formatting/status returns result when complete."""
        control_app.config["formatting_fix_state"] = {
            "in_progress": False,
            "result": {"success": True, "issues_fixed": 5, "message": "Fixed 5 sections"},
            "progress": None,
            "sections_fixed": 5,
        }

        response = flask_client.get("/api/document/fix-formatting/status")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "complete"
        assert data["success"] is True
        assert data["issues_fixed"] == 5

    def test_fix_formatting_status_clears_result_after_read(self, flask_client, control_app):
        """Status endpoint clears result after reading (one-time retrieval)."""
        control_app.config["formatting_fix_state"] = {
            "in_progress": False,
            "result": {"success": True, "issues_fixed": 3},
            "progress": None,
            "sections_fixed": 3,
        }

        # First read should return result
        response1 = flask_client.get("/api/document/fix-formatting/status")
        assert response1.get_json()["status"] == "complete"

        # Second read should return idle (result cleared)
        response2 = flask_client.get("/api/document/fix-formatting/status")
        assert response2.get_json()["status"] == "idle"


class TestFormattingFixThreadSafety:
    """Tests for thread safety of formatting fix operations."""

    def test_concurrent_start_requests_only_one_starts(self, flask_client, control_app):
        """Only one formatting fix starts when concurrent requests arrive."""
        control_app.config["formatting_fix_state"] = {
            "in_progress": False,
            "result": None,
            "progress": None,
            "sections_fixed": 0,
        }

        results = []
        errors = []

        def make_request():
            try:
                with control_app.test_client() as client:
                    response = client.post("/api/document/fix-formatting")
                    results.append(response.get_json())
            except Exception as e:
                errors.append(e)

        with patch("control.blueprints.documenter._run_formatting_fix"):
            # Start multiple threads simultaneously
            threads = [threading.Thread(target=make_request) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        # Count how many got "started" vs "already_running"
        started = sum(1 for r in results if r.get("status") == "started")
        already_running = sum(1 for r in results if r.get("status") == "already_running")

        # Exactly one should have started, rest should be already_running
        assert started == 1
        assert already_running == 4


class TestVersionHistory:
    """Tests for version history endpoints."""

    def test_get_versions_returns_list(self, flask_client, temp_dir):
        """GET /api/document/versions returns version list."""
        versions_dir = temp_dir / "document" / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        (versions_dir / "version_001.md").write_text("# V1")
        (versions_dir / "version_001.json").write_text(json.dumps({
            "version_id": "version_001",
            "timestamp": "2024-01-15T10:00:00",
            "description": "Initial version",
            "content_length": 100,
        }))

        with patch("control.blueprints.documenter.DOC_PATH", temp_dir / "document" / "main.md"):
            with patch("control.blueprints.documenter.FANO_ROOT", temp_dir):
                response = flask_client.get("/api/document/versions")

        assert response.status_code == 200
        data = response.get_json()
        assert "versions" in data

    def test_get_version_by_id_not_found(self, flask_client, temp_dir):
        """GET /api/document/versions/<id> returns 404 for non-existent version."""
        # Version not found test - verifies 404 handling
        doc_path = temp_dir / "document" / "main.md"

        with patch("control.blueprints.documenter.DOC_PATH", doc_path):
            response = flask_client.get("/api/document/versions/nonexistent_version")

        # Non-existent versions should return 404
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data

    def test_get_version_not_found(self, flask_client, temp_dir):
        """GET /api/document/versions/<id> returns 404 for missing version."""
        with patch("control.blueprints.documenter.DOC_PATH", temp_dir / "document" / "main.md"):
            with patch("control.blueprints.documenter.FANO_ROOT", temp_dir):
                response = flask_client.get("/api/document/versions/nonexistent")

        data = response.get_json()
        assert "error" in data


class TestOrchestratorUrl:
    """Tests for orchestrator URL configuration."""

    def test_get_orchestrator_url_from_config(self):
        """get_orchestrator_url reads from config."""
        from control.blueprints.documenter import get_orchestrator_url

        mock_config = {
            "orchestrator": {
                "host": "192.168.1.100",
                "port": 9999,
            }
        }

        with patch("control.blueprints.documenter.load_config", return_value=mock_config):
            url = get_orchestrator_url()

        assert url == "http://192.168.1.100:9999"

    def test_get_orchestrator_url_uses_defaults(self):
        """get_orchestrator_url uses defaults when not in config."""
        from control.blueprints.documenter import get_orchestrator_url

        with patch("control.blueprints.documenter.load_config", return_value={}):
            url = get_orchestrator_url()

        assert url == "http://127.0.0.1:9001"

    def test_get_orchestrator_url_partial_config(self):
        """get_orchestrator_url handles partial config."""
        from control.blueprints.documenter import get_orchestrator_url

        mock_config = {"orchestrator": {"host": "10.0.0.1"}}

        with patch("control.blueprints.documenter.load_config", return_value=mock_config):
            url = get_orchestrator_url()

        # Port should use default
        assert url == "http://10.0.0.1:9001"
