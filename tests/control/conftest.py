"""
Control module test fixtures.

Provides fixtures for:
- Flask test client with mocked dependencies
- Mock ProcessManager
- Mock pool/orchestrator responses
"""

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@pytest.fixture
def mock_process_manager():
    """
    Create a mock ProcessManager.

    All methods return sensible defaults for testing.
    """
    pm = MagicMock()
    pm.is_running = MagicMock(return_value=False)
    pm.get_pid = MagicMock(return_value=None)
    pm.get = MagicMock(return_value=None)

    # Process start methods return mock Popen
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll = MagicMock(return_value=None)  # Still running
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock()
    mock_proc.kill = MagicMock()

    pm.start_pool = MagicMock(return_value=mock_proc)
    pm.start_explorer = MagicMock(return_value=mock_proc)
    pm.start_documenter = MagicMock(return_value=mock_proc)
    pm.start_researcher = MagicMock(return_value=mock_proc)
    pm.start_orchestrator = MagicMock(return_value=mock_proc)

    pm.stop = MagicMock(return_value=True)
    pm.cleanup_all = MagicMock()
    pm.wait_for_pool_health = MagicMock(return_value=True)
    pm.wait_for_orchestrator_health = MagicMock(return_value=True)
    pm.check_dependencies_ready = MagicMock(return_value=(True, []))
    pm.start_with_deps = MagicMock(return_value=True)

    return pm


@pytest.fixture
def mock_subprocess():
    """
    Mock subprocess.Popen for process tests.

    Prevents actual process spawning during tests.
    """
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll = MagicMock(return_value=None)
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock()
    mock_proc.kill = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        yield mock_popen, mock_proc


@pytest.fixture
def control_app(mock_process_manager, sample_config, temp_dir):
    """
    Create Flask test app with mocked dependencies.

    Use this fixture to test the control panel API.
    """
    # Set up test paths
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    doc_path = temp_dir / "document.md"
    doc_path.write_text("# Test Document\n\nTest content.")
    explorer_data_dir = temp_dir / "explorer" / "data"
    explorer_data_dir.mkdir(parents=True, exist_ok=True)

    with patch("control.server.load_config", return_value=sample_config):
        with patch("control.server.ProcessManager", return_value=mock_process_manager):
            with patch("control.services.process_manager.LOGS_DIR", logs_dir):
                with patch("control.services.process_manager.FANO_ROOT", temp_dir):
                    from control.server import create_app
                    app = create_app(process_manager=mock_process_manager)
                    app.config["TESTING"] = True
                    yield app


@pytest.fixture
def flask_client(control_app):
    """Create Flask test client."""
    return control_app.test_client()


@pytest.fixture
def mock_pool_response():
    """
    Factory for creating mock pool HTTP responses.

    Usage:
        response = mock_pool_response({"gemini": {"available": True}})
    """
    def _create(json_data: dict[str, Any], status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.status = status_code
        mock_resp.read = MagicMock(return_value=json.dumps(json_data).encode())
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)
        return mock_resp
    return _create


@pytest.fixture
def mock_orchestrator_response():
    """
    Factory for creating mock orchestrator HTTP responses.

    Usage:
        response = mock_orchestrator_response({"status": "healthy"})
    """
    def _create(json_data: dict[str, Any], status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            from requests import HTTPError
            mock_resp.raise_for_status.side_effect = HTTPError()
        return mock_resp
    return _create


@pytest.fixture
def sample_insight_files(temp_dir) -> Path:
    """
    Create sample insight files for testing explorer endpoints.

    Returns the explorer data directory path.
    """
    data_dir = temp_dir / "explorer" / "data"
    insights_dir = data_dir / "chunks" / "insights"

    # Create pending insights
    pending_dir = insights_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / "insight-001.json").write_text(json.dumps({
        "id": "insight-001",
        "insight": "The Fano plane has 7 points",
        "confidence": "high",
        "tags": ["fano", "projective"],
        "extracted_at": "2024-01-15T10:00:00",
    }))

    # Create blessed insights
    blessed_dir = insights_dir / "blessed"
    blessed_dir.mkdir(parents=True, exist_ok=True)
    (blessed_dir / "insight-002.json").write_text(json.dumps({
        "id": "insight-002",
        "insight": "Klein quartic has 168 automorphisms",
        "confidence": "high",
        "tags": ["klein", "automorphism"],
        "extracted_at": "2024-01-14T10:00:00",
        "blessed_at": "2024-01-14T12:00:00",
    }))

    return data_dir


@pytest.fixture
def sample_thread_files(temp_dir) -> Path:
    """
    Create sample thread files for testing explorer endpoints.

    Returns the explorer data directory path.
    """
    data_dir = temp_dir / "explorer" / "data"
    threads_dir = data_dir / "explorations"
    threads_dir.mkdir(parents=True, exist_ok=True)

    (threads_dir / "thread-001.json").write_text(json.dumps({
        "id": "thread-001",
        "topic": "Exploring the Fano plane",
        "status": "active",
        "priority": 5,
        "exchanges": [
            {
                "id": "ex-1",
                "role": "explorer",
                "model": "gemini",
                "prompt": "What is the Fano plane?",
                "response": "The Fano plane is...",
            }
        ],
        "created_at": "2024-01-15T09:00:00",
    }))

    return data_dir


@pytest.fixture
def sample_seed_files(temp_dir) -> Path:
    """
    Create sample seed/axiom files for testing.

    Returns the explorer data directory path.
    """
    data_dir = temp_dir / "explorer" / "data"
    seeds_dir = data_dir / "axioms"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    (seeds_dir / "axioms.json").write_text(json.dumps({
        "seeds": [
            {
                "id": "seed-001",
                "text": "The Fano plane is the smallest projective plane",
                "type": "axiom",
                "priority": 8,
                "tags": ["fano", "projective"],
                "confidence": "high",
                "source": "user",
            }
        ]
    }))

    return data_dir
