"""
Root-level shared fixtures for all Fano tests.

This file provides common fixtures used across multiple test modules.
Module-specific fixtures should be defined in their respective conftest.py files.
"""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, AsyncMock

import pytest


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test data.

    Automatically cleaned up after test completion.
    """
    temp_path = Path(tempfile.mkdtemp(prefix="fano_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """
    Standard test configuration.

    Provides a minimal configuration suitable for most tests.
    Deep mode and pro mode are available but should NOT be used in tests.
    """
    return {
        "llm": {
            "pool": {"host": "127.0.0.1", "port": 9000},
            "backends": {
                "gemini": {
                    "enabled": True,
                    "type": "browser",
                    "deep_mode": {"daily_limit": 20},
                },
                "chatgpt": {
                    "enabled": True,
                    "type": "browser",
                    "pro_mode": {"daily_limit": 100},
                },
                "claude": {
                    "enabled": True,
                    "type": "api",
                    "model": "claude-sonnet-4-20250514",
                },
            },
        },
        "services": {
            "pool": {
                "auto_start": False,
                "health_timeout_seconds": 5,
            },
        },
        "explorer": {
            "data_dir": "explorer/data",
        },
        "documenter": {
            "document": {"path": "document.md"},
            "max_consensus_calls": 100,
        },
    }


@pytest.fixture
def mock_llm_response():
    """
    Factory for creating mock LLM responses.

    IMPORTANT: Tests must NEVER use deep_mode or pro_mode.
    All mocked responses simulate basic model usage.

    Usage:
        response = mock_llm_response("Test response text")
    """
    def _create(
        text: str,
        success: bool = True,
        backend: str = "claude",
        deep_mode_used: bool = False,
    ) -> dict[str, Any]:
        # Ensure tests don't accidentally use deep/pro modes
        assert not deep_mode_used, "Tests must NOT use deep_mode"

        return {
            "success": success,
            "response": text if success else None,
            "error": None if success else "mock_error",
            "metadata": {
                "backend": backend,
                "deep_mode_used": False,  # Always False in tests
                "response_time_seconds": 1.5,
                "session_id": "test-session-123",
            },
            "recovered": False,
        }
    return _create


@pytest.fixture
def mock_browser():
    """
    Create a generic mock browser interface.

    Provides all common browser methods as AsyncMock/MagicMock.
    Deep think and pro mode methods are mocked but should not be called.
    """
    browser = MagicMock()
    browser.page = MagicMock()
    browser.page.url = "https://test.example.com/chat/123"
    browser.page.goto = AsyncMock()
    browser.page.wait_for_load_state = AsyncMock()
    browser.page.wait_for_selector = AsyncMock()
    browser.page.evaluate = AsyncMock(return_value="Evaluated result")
    browser.page.screenshot = AsyncMock()
    browser.page.fill = AsyncMock()
    browser.page.click = AsyncMock()
    browser.page.keyboard = MagicMock()
    browser.page.keyboard.press = AsyncMock()

    browser.connect = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.start_new_chat = AsyncMock()
    browser.send_message = AsyncMock(return_value="Test response from browser")
    browser.try_get_response = AsyncMock(return_value="Recovered response")
    browser.is_generating = AsyncMock(return_value=False)

    # Deep/pro mode methods exist but should not be used in tests
    browser.enable_deep_think = AsyncMock()
    browser.enable_pro_mode = AsyncMock()

    browser._check_rate_limit = MagicMock(return_value=False)
    browser.chat_logger = MagicMock()
    browser.chat_logger.get_session_id = MagicMock(return_value="session-123")

    return browser


@pytest.fixture
def mock_anthropic_client():
    """
    Create a mock Anthropic client.

    Returns a MagicMock configured to simulate Anthropic API responses.
    """
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="Claude API response")]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(input_tokens=100, output_tokens=50)
    client.messages.create = MagicMock(return_value=response)
    return client


@pytest.fixture
def sample_insight_data() -> dict[str, Any]:
    """Sample atomic insight data for testing."""
    return {
        "id": "insight-test-001",
        "insight": "The Fano plane is the smallest projective plane with 7 points and 7 lines.",
        "confidence": "high",
        "tags": ["projective_geometry", "fano_plane", "incidence"],
        "source_thread_id": "thread-test-001",
        "extraction_model": "claude",
        "extracted_at": datetime.now().isoformat(),
        "status": "pending",
    }


@pytest.fixture
def sample_thread_data() -> dict[str, Any]:
    """Sample exploration thread data for testing."""
    return {
        "id": "thread-test-001",
        "topic": "Exploring properties of the Fano plane",
        "status": "active",
        "seed_axioms": ["seed-001"],
        "exchanges": [
            {
                "id": "ex-001",
                "role": "explorer",
                "model": "gemini",
                "prompt": "What are the key properties of the Fano plane?",
                "response": "The Fano plane has several important properties...",
                "timestamp": datetime.now().isoformat(),
                "deep_mode_used": False,
            },
        ],
        "created_at": datetime.now().isoformat(),
        "priority": 5,
    }


@pytest.fixture
def sample_seed_data() -> dict[str, Any]:
    """Sample seed/axiom data for testing."""
    return {
        "id": "seed-test-001",
        "text": "Every projective plane satisfies the incidence axioms.",
        "type": "axiom",
        "priority": 8,
        "tags": ["projective_geometry", "axiom"],
        "confidence": "high",
        "source": "user",
        "notes": "Foundational axiom for projective geometry exploration",
        "created_at": datetime.now().isoformat(),
    }


# Utility functions for tests

def create_test_file(path: Path, content: str | dict) -> Path:
    """
    Helper to create a test file with content.

    Args:
        path: Path to create the file at
        content: String content or dict to serialize as JSON

    Returns:
        The path to the created file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        path.write_text(json.dumps(content, indent=2))
    else:
        path.write_text(content)
    return path


def assert_no_deep_mode(mock_browser: MagicMock) -> None:
    """
    Assert that deep_think was never called on a mock browser.

    Use this in tests to verify we're not accidentally using Pro features.
    """
    mock_browser.enable_deep_think.assert_not_called()


def assert_no_pro_mode(mock_browser: MagicMock) -> None:
    """
    Assert that pro_mode was never called on a mock browser.

    Use this in tests to verify we're not accidentally using Pro features.
    """
    mock_browser.enable_pro_mode.assert_not_called()
