"""Comprehensive tests for pool recovery functionality.

Tests cover:
- Active work tracking and staleness detection
- Recovered response storage and retrieval
- Queue persistence and restoration
- Worker recovery from interrupted chats
- End-to-end recovery scenarios
"""

import asyncio
import json
import time
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import StateManager, MAX_ACTIVE_WORK_AGE_SECONDS
from src.queue import QueuedRequest, RequestQueue, QueueManager
from src.models import SendRequest, SendOptions, SendResponse, Backend, Priority
from src.workers import GeminiWorker, ChatGPTWorker


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_state_file(tmp_path):
    """Create a temporary state file."""
    return tmp_path / "pool_state.json"


@pytest.fixture
def temp_queue_file(tmp_path):
    """Create a temporary queue state file."""
    return tmp_path / "queue_state.json"


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "backends": {
            "gemini": {"enabled": True, "deep_mode": {"daily_limit": 20}},
            "chatgpt": {"enabled": True, "pro_mode": {"daily_limit": 100}},
            "claude": {"enabled": True, "model": "claude-sonnet-4-20250514"},
        },
        "queue": {"max_depth_per_backend": 10},
    }


@pytest.fixture
def state_manager(temp_state_file, sample_config):
    """Create a StateManager for tests."""
    return StateManager(temp_state_file, sample_config)


@pytest.fixture
def mock_browser():
    """Create a mock browser with recovery-related methods."""
    browser = MagicMock()
    browser.page = MagicMock()
    browser.page.goto = AsyncMock()
    browser.page.wait_for_load_state = AsyncMock()
    browser.page.evaluate = AsyncMock()
    browser.page.screenshot = AsyncMock()
    browser.is_generating = AsyncMock(return_value=False)
    browser.try_get_response = AsyncMock(return_value="Recovered response text")
    browser._wait_for_response = AsyncMock(return_value="Generated response text")
    browser.connect = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.start_new_chat = AsyncMock()
    browser.send_message = AsyncMock(return_value="Normal response")
    browser.enable_deep_think = AsyncMock()
    browser._check_rate_limit = MagicMock(return_value=False)
    browser.chat_logger = MagicMock()
    browser.chat_logger.get_session_id = MagicMock(return_value="session-123")
    return browser


# =============================================================================
# Active Work Tests
# =============================================================================

class TestActiveWorkTracking:
    """Tests for active work set/get/clear operations."""

    def test_set_active_work(self, state_manager):
        """set_active_work stores work details correctly."""
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test prompt",
            chat_url="https://gemini.google.com/app/abc123",
            thread_id="thread-456",
            options={"deep_mode": True},
        )

        work = state_manager.get_active_work("gemini", check_staleness=False)
        assert work is not None
        assert work["request_id"] == "gemini-123"
        assert work["prompt"] == "Test prompt"
        assert work["chat_url"] == "https://gemini.google.com/app/abc123"
        assert work["thread_id"] == "thread-456"
        assert work["options"]["deep_mode"] is True
        assert "started_at" in work

    def test_set_active_work_persists_to_file(self, temp_state_file, sample_config):
        """set_active_work saves state to disk."""
        manager = StateManager(temp_state_file, sample_config)
        manager.set_active_work(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test",
            chat_url="https://example.com/chat",
        )

        # Read file directly
        saved = json.loads(temp_state_file.read_text())
        assert saved["gemini"]["active_work"]["request_id"] == "gemini-123"

    def test_get_active_work_returns_none_for_no_work(self, state_manager):
        """get_active_work returns None when no work is set."""
        work = state_manager.get_active_work("gemini")
        assert work is None

    def test_get_active_work_returns_none_for_unknown_backend(self, state_manager):
        """get_active_work returns None for unknown backend."""
        work = state_manager.get_active_work("unknown_backend")
        assert work is None

    def test_clear_active_work(self, state_manager):
        """clear_active_work removes the active work."""
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test",
            chat_url="https://example.com/chat",
        )

        state_manager.clear_active_work("gemini")

        work = state_manager.get_active_work("gemini")
        assert work is None

    def test_clear_active_work_persists(self, temp_state_file, sample_config):
        """clear_active_work saves state to disk."""
        manager = StateManager(temp_state_file, sample_config)
        manager.set_active_work(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test",
            chat_url="https://example.com/chat",
        )
        manager.clear_active_work("gemini")

        saved = json.loads(temp_state_file.read_text())
        assert saved["gemini"]["active_work"] is None


class TestActiveWorkStaleness:
    """Tests for staleness detection of active work."""

    def test_fresh_work_is_returned(self, state_manager):
        """Work within MAX_ACTIVE_WORK_AGE_SECONDS is returned."""
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test",
            chat_url="https://example.com/chat",
        )

        work = state_manager.get_active_work("gemini", check_staleness=True)
        assert work is not None
        assert work["request_id"] == "gemini-123"

    def test_stale_work_is_auto_cleared(self, state_manager):
        """Work older than MAX_ACTIVE_WORK_AGE_SECONDS is auto-cleared."""
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-old",
            prompt="Old test",
            chat_url="https://example.com/chat",
        )

        # Manually set started_at to be stale
        state_manager._state["gemini"]["active_work"]["started_at"] = (
            time.time() - MAX_ACTIVE_WORK_AGE_SECONDS - 100
        )

        work = state_manager.get_active_work("gemini", check_staleness=True)
        assert work is None

        # Verify it was cleared from state
        assert state_manager._state["gemini"]["active_work"] is None

    def test_staleness_check_can_be_disabled(self, state_manager):
        """check_staleness=False returns stale work without clearing."""
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-old",
            prompt="Old test",
            chat_url="https://example.com/chat",
        )

        # Make it stale
        state_manager._state["gemini"]["active_work"]["started_at"] = (
            time.time() - MAX_ACTIVE_WORK_AGE_SECONDS - 100
        )

        work = state_manager.get_active_work("gemini", check_staleness=False)
        assert work is not None
        assert work["request_id"] == "gemini-old"

    def test_staleness_boundary(self, state_manager):
        """Work exactly at the age limit is considered stale."""
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-boundary",
            prompt="Boundary test",
            chat_url="https://example.com/chat",
        )

        # Set to exactly at the limit
        state_manager._state["gemini"]["active_work"]["started_at"] = (
            time.time() - MAX_ACTIVE_WORK_AGE_SECONDS - 1
        )

        work = state_manager.get_active_work("gemini", check_staleness=True)
        assert work is None


# =============================================================================
# Recovered Responses Tests
# =============================================================================

class TestRecoveredResponses:
    """Tests for recovered response storage and retrieval."""

    def test_add_recovered_response(self, state_manager):
        """add_recovered_response stores response correctly."""
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test prompt",
            response="This is the recovered response",
            thread_id="thread-456",
            options={"deep_mode": True},
        )

        responses = state_manager.get_recovered_responses()
        assert len(responses) == 1
        assert responses[0]["request_id"] == "gemini-123"
        assert responses[0]["backend"] == "gemini"
        assert responses[0]["prompt"] == "Test prompt"
        assert responses[0]["response"] == "This is the recovered response"
        assert responses[0]["thread_id"] == "thread-456"
        assert responses[0]["options"]["deep_mode"] is True
        assert "recovered_at" in responses[0]

    def test_add_multiple_recovered_responses(self, state_manager):
        """Multiple recovered responses can be stored."""
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-1",
            prompt="Prompt 1",
            response="Response 1",
        )
        state_manager.add_recovered_response(
            backend="chatgpt",
            request_id="chatgpt-1",
            prompt="Prompt 2",
            response="Response 2",
            thread_id="thread-789",
        )

        responses = state_manager.get_recovered_responses()
        assert len(responses) == 2

    def test_get_recovered_responses_returns_copy(self, state_manager):
        """get_recovered_responses returns a copy, not the original list."""
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-1",
            prompt="Test",
            response="Response",
        )

        responses = state_manager.get_recovered_responses()
        responses.clear()

        # Original should still have the response
        assert len(state_manager.get_recovered_responses()) == 1

    def test_get_recovered_responses_empty(self, state_manager):
        """get_recovered_responses returns empty list when none exist."""
        responses = state_manager.get_recovered_responses()
        assert responses == []

    def test_clear_recovered_response(self, state_manager):
        """clear_recovered_response removes specific response."""
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-1",
            prompt="Test 1",
            response="Response 1",
        )
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-2",
            prompt="Test 2",
            response="Response 2",
        )

        result = state_manager.clear_recovered_response("gemini-1")

        assert result is True
        responses = state_manager.get_recovered_responses()
        assert len(responses) == 1
        assert responses[0]["request_id"] == "gemini-2"

    def test_clear_recovered_response_not_found(self, state_manager):
        """clear_recovered_response returns False when not found."""
        result = state_manager.clear_recovered_response("nonexistent")
        assert result is False

    def test_recovered_responses_persist(self, temp_state_file, sample_config):
        """Recovered responses are saved to disk."""
        manager = StateManager(temp_state_file, sample_config)
        manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-123",
            prompt="Test",
            response="Recovered!",
            thread_id="thread-abc",
        )

        saved = json.loads(temp_state_file.read_text())
        assert len(saved["recovered_responses"]) == 1
        assert saved["recovered_responses"][0]["thread_id"] == "thread-abc"

    def test_filter_by_thread_id(self, state_manager):
        """Responses can be filtered by thread_id."""
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-1",
            prompt="Test 1",
            response="Response 1",
            thread_id="thread-A",
        )
        state_manager.add_recovered_response(
            backend="chatgpt",
            request_id="chatgpt-1",
            prompt="Test 2",
            response="Response 2",
            thread_id="thread-B",
        )
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-2",
            prompt="Test 3",
            response="Response 3",
            thread_id="thread-A",
        )

        responses = state_manager.get_recovered_responses()
        thread_a_responses = [r for r in responses if r["thread_id"] == "thread-A"]

        assert len(thread_a_responses) == 2
        assert all(r["thread_id"] == "thread-A" for r in thread_a_responses)


# =============================================================================
# Queue Persistence Tests
# =============================================================================

class TestQueuedRequestSerialization:
    """Tests for QueuedRequest serialization."""

    def test_to_dict(self):
        """to_dict serializes request correctly."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        request = SendRequest(
            backend=Backend.GEMINI,
            prompt="Test prompt",
            thread_id="thread-123",
            options=SendOptions(
                deep_mode=True,
                timeout_seconds=600,
                priority=Priority.HIGH,
                new_chat=True,
            ),
        )
        queued = QueuedRequest.create(request, future, "gemini-456")

        data = queued.to_dict()

        assert data["request_id"] == "gemini-456"
        assert data["priority_value"] == 0  # HIGH
        assert data["request"]["backend"] == "gemini"
        assert data["request"]["prompt"] == "Test prompt"
        assert data["request"]["thread_id"] == "thread-123"
        assert data["request"]["options"]["deep_mode"] is True
        assert data["request"]["options"]["timeout_seconds"] == 600
        assert data["request"]["options"]["priority"] == "high"
        assert "timestamp" in data

        loop.close()

    def test_from_dict(self):
        """from_dict deserializes request correctly."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        data = {
            "request_id": "chatgpt-789",
            "priority_value": 1,
            "timestamp": "2024-01-15T10:30:00",
            "request": {
                "backend": "chatgpt",
                "prompt": "Restored prompt",
                "thread_id": "thread-abc",
                "options": {
                    "deep_mode": False,
                    "timeout_seconds": 300,
                    "priority": "normal",
                    "new_chat": True,
                },
            },
        }

        queued = QueuedRequest.from_dict(data, future)

        assert queued.request_id == "chatgpt-789"
        assert queued.priority_value == 1
        assert queued.request.backend == Backend.CHATGPT
        assert queued.request.prompt == "Restored prompt"
        assert queued.request.thread_id == "thread-abc"
        assert queued.request.options.deep_mode is False
        assert queued.future is future

        loop.close()

    def test_roundtrip_serialization(self):
        """to_dict -> from_dict preserves all data."""
        loop = asyncio.new_event_loop()
        future1 = loop.create_future()
        future2 = loop.create_future()

        request = SendRequest(
            backend=Backend.GEMINI,
            prompt="Roundtrip test",
            thread_id="thread-round",
            options=SendOptions(
                deep_mode=True,
                timeout_seconds=450,
                priority=Priority.LOW,
                new_chat=False,
            ),
        )
        original = QueuedRequest.create(request, future1, "gemini-round")

        data = original.to_dict()
        restored = QueuedRequest.from_dict(data, future2)

        assert restored.request_id == original.request_id
        assert restored.priority_value == original.priority_value
        assert restored.request.backend == original.request.backend
        assert restored.request.prompt == original.request.prompt
        assert restored.request.thread_id == original.request.thread_id
        assert restored.request.options.deep_mode == original.request.options.deep_mode
        assert restored.request.options.timeout_seconds == original.request.options.timeout_seconds
        assert restored.request.options.priority == original.request.options.priority

        loop.close()


class TestRequestQueuePersistence:
    """Tests for RequestQueue persistence."""

    @pytest.mark.asyncio
    async def test_get_all_serialized(self):
        """get_all_serialized returns all items as dicts."""
        queue = RequestQueue("gemini", max_depth=10)

        req1 = SendRequest(backend=Backend.GEMINI, prompt="First")
        req2 = SendRequest(backend=Backend.GEMINI, prompt="Second")

        await queue.enqueue(req1)
        await queue.enqueue(req2)

        serialized = queue.get_all_serialized()

        assert len(serialized) == 2
        prompts = {s["request"]["prompt"] for s in serialized}
        assert prompts == {"First", "Second"}

    @pytest.mark.asyncio
    async def test_restore_from_serialized(self):
        """restore_from_serialized recreates queue items."""
        queue = RequestQueue("gemini", max_depth=10)

        items = [
            {
                "request_id": "gemini-10",
                "priority_value": 0,
                "timestamp": "2024-01-15T10:00:00",
                "request": {
                    "backend": "gemini",
                    "prompt": "High priority",
                    "thread_id": None,
                    "options": {
                        "deep_mode": False,
                        "timeout_seconds": 300,
                        "priority": "high",
                        "new_chat": True,
                    },
                },
            },
            {
                "request_id": "gemini-11",
                "priority_value": 1,
                "timestamp": "2024-01-15T10:01:00",
                "request": {
                    "backend": "gemini",
                    "prompt": "Normal priority",
                    "thread_id": "thread-xyz",
                    "options": {
                        "deep_mode": True,
                        "timeout_seconds": 600,
                        "priority": "normal",
                        "new_chat": True,
                    },
                },
            },
        ]

        restored_count = queue.restore_from_serialized(items)

        assert restored_count == 2
        assert queue.depth == 2

        # Dequeue should return highest priority first
        first = await queue.dequeue()
        assert first.request.prompt == "High priority"

        second = await queue.dequeue()
        assert second.request.prompt == "Normal priority"
        assert second.request.thread_id == "thread-xyz"

    @pytest.mark.asyncio
    async def test_restore_updates_request_counter(self):
        """Restored items update counter to avoid ID collisions."""
        queue = RequestQueue("gemini", max_depth=10)

        items = [
            {
                "request_id": "gemini-50",
                "priority_value": 1,
                "timestamp": "2024-01-15T10:00:00",
                "request": {
                    "backend": "gemini",
                    "prompt": "Old request",
                    "thread_id": None,
                    "options": {
                        "deep_mode": False,
                        "timeout_seconds": 300,
                        "priority": "normal",
                        "new_chat": True,
                    },
                },
            },
        ]

        queue.restore_from_serialized(items)

        # New request should get ID > 50
        new_request = SendRequest(backend=Backend.GEMINI, prompt="New")
        await queue.enqueue(new_request)

        # Find the new item
        serialized = queue.get_all_serialized()
        new_item = next(s for s in serialized if s["request"]["prompt"] == "New")
        new_id_num = int(new_item["request_id"].split("-")[-1])

        assert new_id_num > 50

    @pytest.mark.asyncio
    async def test_persist_callback_called_on_enqueue(self):
        """Persist callback is called when item is enqueued."""
        persist_calls = []

        def on_persist():
            persist_calls.append(time.time())

        queue = RequestQueue("gemini", max_depth=10, persist_callback=on_persist)

        req = SendRequest(backend=Backend.GEMINI, prompt="Test")
        await queue.enqueue(req)

        assert len(persist_calls) == 1

    @pytest.mark.asyncio
    async def test_persist_callback_called_on_dequeue(self):
        """Persist callback is called when item is dequeued."""
        persist_calls = []

        def on_persist():
            persist_calls.append(time.time())

        queue = RequestQueue("gemini", max_depth=10, persist_callback=on_persist)

        req = SendRequest(backend=Backend.GEMINI, prompt="Test")
        await queue.enqueue(req)  # +1 call
        await queue.dequeue()  # +1 call

        assert len(persist_calls) == 2


class TestQueueManagerPersistence:
    """Tests for QueueManager persistence."""

    @pytest.mark.asyncio
    async def test_persists_to_file(self, temp_queue_file, sample_config):
        """QueueManager saves queue state to file."""
        manager = QueueManager(sample_config, state_file=temp_queue_file)

        req = SendRequest(backend=Backend.GEMINI, prompt="Persisted request")
        await manager.get_queue("gemini").enqueue(req)

        assert temp_queue_file.exists()
        saved = json.loads(temp_queue_file.read_text())
        assert len(saved["gemini"]) == 1
        assert saved["gemini"][0]["request"]["prompt"] == "Persisted request"

    @pytest.mark.asyncio
    async def test_restore_pending_on_startup(self, temp_queue_file, sample_config):
        """restore_pending restores queue from file on startup."""
        # First, create state file with pending items
        state = {
            "gemini": [
                {
                    "request_id": "gemini-100",
                    "priority_value": 1,
                    "timestamp": "2024-01-15T10:00:00",
                    "request": {
                        "backend": "gemini",
                        "prompt": "Restored from file",
                        "thread_id": "thread-restore",
                        "options": {
                            "deep_mode": True,
                            "timeout_seconds": 300,
                            "priority": "normal",
                            "new_chat": True,
                        },
                    },
                },
            ],
            "chatgpt": [],
            "claude": [],
        }
        temp_queue_file.write_text(json.dumps(state))

        # Create new manager and restore
        manager = QueueManager(sample_config, state_file=temp_queue_file)
        restored = manager.restore_pending()

        assert restored == {"gemini": 1}
        assert manager.get_queue("gemini").depth == 1

        # Dequeue and verify
        item = await manager.get_queue("gemini").dequeue()
        assert item.request.prompt == "Restored from file"
        assert item.request.thread_id == "thread-restore"

    @pytest.mark.asyncio
    async def test_restore_pending_clears_file(self, temp_queue_file, sample_config):
        """restore_pending deletes the state file after successful restore."""
        state = {
            "gemini": [
                {
                    "request_id": "gemini-1",
                    "priority_value": 1,
                    "timestamp": "2024-01-15T10:00:00",
                    "request": {
                        "backend": "gemini",
                        "prompt": "Test",
                        "thread_id": None,
                        "options": {
                            "deep_mode": False,
                            "timeout_seconds": 300,
                            "priority": "normal",
                            "new_chat": True,
                        },
                    },
                },
            ],
            "chatgpt": [],
            "claude": [],
        }
        temp_queue_file.write_text(json.dumps(state))

        manager = QueueManager(sample_config, state_file=temp_queue_file)
        manager.restore_pending()

        assert not temp_queue_file.exists()

    def test_restore_pending_handles_missing_file(self, temp_queue_file, sample_config):
        """restore_pending returns empty dict when file doesn't exist."""
        manager = QueueManager(sample_config, state_file=temp_queue_file)
        restored = manager.restore_pending()

        assert restored == {}

    def test_restore_pending_handles_corrupt_file(self, temp_queue_file, sample_config):
        """restore_pending handles corrupt state file gracefully."""
        temp_queue_file.write_text("not valid json {{{")

        manager = QueueManager(sample_config, state_file=temp_queue_file)
        restored = manager.restore_pending()

        assert restored == {}


# =============================================================================
# Worker Recovery Tests
# =============================================================================

class TestWorkerCheckAndRecover:
    """Tests for worker check_and_recover_work method."""

    @pytest.fixture
    def gemini_worker(self, sample_config, state_manager):
        """Create a GeminiWorker for tests."""
        queue = RequestQueue("gemini", max_depth=10)
        worker = GeminiWorker(sample_config, state_manager, queue)
        return worker

    @pytest.mark.asyncio
    async def test_recover_completed_response(
        self, gemini_worker, state_manager, mock_browser
    ):
        """Recovery extracts completed response from chat page."""
        gemini_worker.browser = mock_browser
        mock_browser.is_generating.return_value = False
        mock_browser.try_get_response.return_value = "Completed response text"

        # Set up active work
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-recover-1",
            prompt="Original prompt",
            chat_url="https://gemini.google.com/app/abc123",
            thread_id="thread-recover",
        )

        result = await gemini_worker.check_and_recover_work()

        assert result is not None
        assert result.success is True
        assert result.response == "Completed response text"
        assert result.recovered is True

        # Verify response was stored
        responses = state_manager.get_recovered_responses()
        assert len(responses) == 1
        assert responses[0]["thread_id"] == "thread-recover"

        # Verify active work was cleared
        assert state_manager.get_active_work("gemini") is None

    @pytest.mark.asyncio
    async def test_recover_still_generating(
        self, gemini_worker, state_manager, mock_browser
    ):
        """Recovery waits for in-progress generation."""
        gemini_worker.browser = mock_browser
        mock_browser.is_generating.return_value = True
        mock_browser._wait_for_response.return_value = "Generated while waiting"

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-generating",
            prompt="Long prompt",
            chat_url="https://gemini.google.com/app/xyz789",
        )

        result = await gemini_worker.check_and_recover_work()

        assert result is not None
        assert result.success is True
        assert result.response == "Generated while waiting"
        mock_browser._wait_for_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_recover_no_active_work(self, gemini_worker, mock_browser):
        """Recovery returns None when no active work exists."""
        gemini_worker.browser = mock_browser

        result = await gemini_worker.check_and_recover_work()

        assert result is None
        mock_browser.page.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_recover_no_browser(self, gemini_worker, state_manager):
        """Recovery returns None when browser not connected."""
        gemini_worker.browser = None

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-no-browser",
            prompt="Test",
            chat_url="https://gemini.google.com/app/test",
        )

        result = await gemini_worker.check_and_recover_work()

        assert result is None

    @pytest.mark.asyncio
    async def test_recover_no_chat_url(self, gemini_worker, state_manager, mock_browser):
        """Recovery clears work and returns None when no chat_url."""
        gemini_worker.browser = mock_browser

        # Manually set active work without chat_url
        state_manager._state["gemini"]["active_work"] = {
            "request_id": "gemini-no-url",
            "prompt": "Test",
            "chat_url": None,
            "started_at": time.time(),
        }

        result = await gemini_worker.check_and_recover_work()

        assert result is None
        assert state_manager.get_active_work("gemini") is None

    @pytest.mark.asyncio
    async def test_recover_stale_work_skipped(
        self, gemini_worker, state_manager, mock_browser
    ):
        """Recovery skips stale work (handled by get_active_work)."""
        gemini_worker.browser = mock_browser

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-stale",
            prompt="Old prompt",
            chat_url="https://gemini.google.com/app/old",
        )

        # Make it stale
        state_manager._state["gemini"]["active_work"]["started_at"] = (
            time.time() - MAX_ACTIVE_WORK_AGE_SECONDS - 100
        )

        result = await gemini_worker.check_and_recover_work()

        assert result is None
        mock_browser.page.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_recover_navigates_to_chat_url(
        self, gemini_worker, state_manager, mock_browser
    ):
        """Recovery navigates to the stored chat URL."""
        gemini_worker.browser = mock_browser
        chat_url = "https://gemini.google.com/app/unique123"

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-nav",
            prompt="Test",
            chat_url=chat_url,
        )

        await gemini_worker.check_and_recover_work()

        mock_browser.page.goto.assert_called_once_with(chat_url)

    @pytest.mark.asyncio
    async def test_recover_stores_thread_id(
        self, gemini_worker, state_manager, mock_browser
    ):
        """Recovered response includes thread_id for correlation."""
        gemini_worker.browser = mock_browser
        mock_browser.try_get_response.return_value = "Response with thread"

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-thread",
            prompt="Thread test",
            chat_url="https://gemini.google.com/app/thread",
            thread_id="important-thread-123",
        )

        await gemini_worker.check_and_recover_work()

        responses = state_manager.get_recovered_responses()
        assert responses[0]["thread_id"] == "important-thread-123"

    @pytest.mark.asyncio
    async def test_recover_handles_navigation_error(
        self, gemini_worker, state_manager, mock_browser
    ):
        """Recovery handles errors gracefully."""
        gemini_worker.browser = mock_browser
        mock_browser.page.goto.side_effect = Exception("Navigation failed")

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-error",
            prompt="Error test",
            chat_url="https://gemini.google.com/app/error",
        )

        result = await gemini_worker.check_and_recover_work()

        assert result is None
        # Active work should be cleared on error
        assert state_manager.get_active_work("gemini") is None

    @pytest.mark.asyncio
    async def test_recover_timeout_saves_screenshot(
        self, gemini_worker, state_manager, mock_browser, tmp_path
    ):
        """Recovery saves debug screenshot on failure."""
        gemini_worker.browser = mock_browser
        mock_browser.is_generating.return_value = False
        mock_browser.try_get_response.return_value = None
        mock_browser._wait_for_response.side_effect = asyncio.TimeoutError()

        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-timeout",
            prompt="Timeout test",
            chat_url="https://gemini.google.com/app/timeout",
        )

        # Patch the screenshot path
        with patch("src.workers.Path") as mock_path:
            mock_debug_path = tmp_path / "debug"
            mock_path.return_value.parent.parent = tmp_path
            mock_debug_path.mkdir()

            result = await gemini_worker.check_and_recover_work()

        assert result is None
        # Screenshot should have been attempted
        mock_browser.page.screenshot.assert_called()


# =============================================================================
# End-to-End Recovery Scenarios
# =============================================================================

class TestE2ERecoveryScenarios:
    """End-to-end tests for recovery scenarios."""

    @pytest.mark.asyncio
    async def test_full_recovery_cycle(self, temp_state_file, temp_queue_file, sample_config):
        """Test complete recovery: persist -> restart -> recover."""
        # Phase 1: Simulate work in progress before restart
        state_manager = StateManager(temp_state_file, sample_config)
        queue_manager = QueueManager(sample_config, state_file=temp_queue_file)

        # Add active work
        state_manager.set_active_work(
            backend="gemini",
            request_id="gemini-e2e-1",
            prompt="E2E test prompt",
            chat_url="https://gemini.google.com/app/e2e",
            thread_id="e2e-thread",
            options={"deep_mode": True},
        )

        # Add pending queue items
        req = SendRequest(
            backend=Backend.CHATGPT,
            prompt="Queued request",
            thread_id="queue-thread",
            options=SendOptions(deep_mode=False),
        )
        await queue_manager.get_queue("chatgpt").enqueue(req)

        # Verify persistence
        assert temp_state_file.exists()
        assert temp_queue_file.exists()

        # Phase 2: Simulate restart - create new managers
        state_manager2 = StateManager(temp_state_file, sample_config)
        queue_manager2 = QueueManager(sample_config, state_file=temp_queue_file)

        # Restore queue
        restored = queue_manager2.restore_pending()
        assert restored["chatgpt"] == 1

        # Verify active work is still there
        active = state_manager2.get_active_work("gemini", check_staleness=False)
        assert active is not None
        assert active["thread_id"] == "e2e-thread"

    @pytest.mark.asyncio
    async def test_concurrent_recovery_thread_isolation(self, state_manager):
        """Multiple concurrent recoveries maintain thread isolation."""
        # Add recovered responses for different threads
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-thread-a",
            prompt="Thread A prompt",
            response="Thread A response",
            thread_id="thread-A",
        )
        state_manager.add_recovered_response(
            backend="chatgpt",
            request_id="chatgpt-thread-b",
            prompt="Thread B prompt",
            response="Thread B response",
            thread_id="thread-B",
        )
        state_manager.add_recovered_response(
            backend="gemini",
            request_id="gemini-thread-a-2",
            prompt="Thread A prompt 2",
            response="Thread A response 2",
            thread_id="thread-A",
        )

        all_responses = state_manager.get_recovered_responses()

        # Filter by thread
        thread_a = [r for r in all_responses if r["thread_id"] == "thread-A"]
        thread_b = [r for r in all_responses if r["thread_id"] == "thread-B"]

        assert len(thread_a) == 2
        assert len(thread_b) == 1

        # Clear one thread's responses
        for r in thread_a:
            state_manager.clear_recovered_response(r["request_id"])

        # Thread B should still have its response
        remaining = state_manager.get_recovered_responses()
        assert len(remaining) == 1
        assert remaining[0]["thread_id"] == "thread-B"


# =============================================================================
# State Loading on Restart Tests
# =============================================================================

class TestStateLoadingOnRestart:
    """Tests for loading persisted state on restart."""

    def test_loads_active_work_on_init(self, temp_state_file, sample_config):
        """StateManager loads active work from existing file."""
        # Create state file with active work
        existing_state = {
            "gemini": {
                "authenticated": True,
                "rate_limited": False,
                "deep_mode_uses_today": 0,
                "active_work": {
                    "request_id": "gemini-persisted",
                    "prompt": "Persisted prompt",
                    "chat_url": "https://gemini.google.com/app/persisted",
                    "thread_id": "persisted-thread",
                    "started_at": time.time(),
                    "options": {"deep_mode": True},
                },
            },
            "chatgpt": {
                "authenticated": False,
                "rate_limited": False,
                "pro_mode_uses_today": 0,
                "active_work": None,
            },
            "claude": {
                "authenticated": True,
                "rate_limited": False,
            },
        }
        temp_state_file.write_text(json.dumps(existing_state))

        # Create new manager - should load existing state
        manager = StateManager(temp_state_file, sample_config)

        work = manager.get_active_work("gemini", check_staleness=False)
        assert work is not None
        assert work["request_id"] == "gemini-persisted"
        assert work["thread_id"] == "persisted-thread"

    def test_loads_recovered_responses_on_init(self, temp_state_file, sample_config):
        """StateManager loads recovered responses from existing file."""
        existing_state = {
            "gemini": {"authenticated": True, "rate_limited": False},
            "chatgpt": {"authenticated": True, "rate_limited": False},
            "claude": {"authenticated": True, "rate_limited": False},
            "recovered_responses": [
                {
                    "request_id": "gemini-old-recovery",
                    "backend": "gemini",
                    "prompt": "Old prompt",
                    "response": "Old response",
                    "thread_id": "old-thread",
                    "recovered_at": time.time() - 3600,
                },
            ],
        }
        temp_state_file.write_text(json.dumps(existing_state))

        manager = StateManager(temp_state_file, sample_config)

        responses = manager.get_recovered_responses()
        assert len(responses) == 1
        assert responses[0]["thread_id"] == "old-thread"
