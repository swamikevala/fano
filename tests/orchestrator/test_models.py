"""
Tests for orchestrator/models.py
"""

import time
import pytest

from orchestrator.models import (
    Task,
    TaskState,
    ConversationState,
    get_base_priority,
    BASE_PRIORITIES,
)


class TestTaskState:
    """Tests for TaskState enum."""

    def test_task_states_exist(self):
        """All expected task states should exist."""
        assert TaskState.PENDING == "pending"
        assert TaskState.RUNNING == "running"
        assert TaskState.FAILED == "failed"
        assert TaskState.COMPLETED == "completed"

    def test_task_state_is_string_enum(self):
        """TaskState should be a string enum."""
        assert isinstance(TaskState.PENDING.value, str)
        assert TaskState.PENDING.value == "pending"


class TestConversationState:
    """Tests for ConversationState dataclass."""

    def test_conversation_state_creation(self, sample_conversation):
        """ConversationState should be created with correct fields."""
        assert sample_conversation.llm == "gemini"
        assert sample_conversation.external_thread_id == "https://gemini.google.com/chat/abc123"
        assert sample_conversation.thread_title == "Mathematical Exploration"
        assert sample_conversation.turn_count == 3

    def test_conversation_state_defaults(self):
        """ConversationState should have sensible defaults."""
        conv = ConversationState(llm="chatgpt")
        assert conv.external_thread_id is None
        assert conv.thread_title is None
        assert conv.turn_count == 0

    def test_conversation_state_to_dict(self, sample_conversation):
        """ConversationState should serialize to dict."""
        data = sample_conversation.to_dict()
        assert data["llm"] == "gemini"
        assert data["external_thread_id"] == "https://gemini.google.com/chat/abc123"
        assert data["thread_title"] == "Mathematical Exploration"
        assert data["turn_count"] == 3

    def test_conversation_state_from_dict(self):
        """ConversationState should deserialize from dict."""
        data = {
            "llm": "claude",
            "external_thread_id": "https://claude.ai/chat/xyz",
            "thread_title": "Test Chat",
            "turn_count": 5,
        }
        conv = ConversationState.from_dict(data)
        assert conv.llm == "claude"
        assert conv.external_thread_id == "https://claude.ai/chat/xyz"
        assert conv.thread_title == "Test Chat"
        assert conv.turn_count == 5

    def test_conversation_state_roundtrip(self, sample_conversation):
        """ConversationState should survive serialization roundtrip."""
        data = sample_conversation.to_dict()
        restored = ConversationState.from_dict(data)
        assert restored.llm == sample_conversation.llm
        assert restored.external_thread_id == sample_conversation.external_thread_id
        assert restored.thread_title == sample_conversation.thread_title
        assert restored.turn_count == sample_conversation.turn_count


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self, sample_task):
        """Task should be created with correct fields."""
        assert sample_task.id == "task-001"
        assert sample_task.key == "test:sample"
        assert sample_task.module == "explorer"
        assert sample_task.task_type == "exploration"
        assert sample_task.priority == 50
        assert sample_task.state == TaskState.PENDING
        assert sample_task.payload == {"thread_id": "thread-123"}

    def test_task_defaults(self):
        """Task should have sensible defaults."""
        task = Task(
            id="t1",
            key="k1",
            module="explorer",
            task_type="exploration",
            priority=50,
        )
        assert task.state == TaskState.PENDING
        assert task.payload == {}
        assert task.conversation is None
        assert task.pool_request_id is None
        assert task.attempts == 0
        assert task.max_attempts == 3
        assert task.requires_deep_mode is False
        assert task.preferred_backend is None
        assert task.result is None
        assert task.error is None

    def test_task_with_deep_mode(self, sample_task_documenter):
        """Task should support deep mode flag."""
        assert sample_task_documenter.requires_deep_mode is True

    def test_task_to_dict(self, sample_task):
        """Task should serialize to dict."""
        data = sample_task.to_dict()
        assert data["id"] == "task-001"
        assert data["key"] == "test:sample"
        assert data["module"] == "explorer"
        assert data["task_type"] == "exploration"
        assert data["priority"] == 50
        assert data["state"] == "pending"
        assert data["payload"] == {"thread_id": "thread-123"}

    def test_task_from_dict(self):
        """Task should deserialize from dict."""
        data = {
            "id": "task-abc",
            "key": "test:key",
            "module": "documenter",
            "task_type": "incorporate_insight",
            "priority": 60,
            "state": "running",
            "payload": {"insight_id": "i1"},
            "attempts": 2,
            "requires_deep_mode": True,
        }
        task = Task.from_dict(data)
        assert task.id == "task-abc"
        assert task.module == "documenter"
        assert task.state == TaskState.RUNNING
        assert task.attempts == 2
        assert task.requires_deep_mode is True

    def test_task_roundtrip(self, sample_task):
        """Task should survive serialization roundtrip."""
        data = sample_task.to_dict()
        restored = Task.from_dict(data)
        assert restored.id == sample_task.id
        assert restored.key == sample_task.key
        assert restored.module == sample_task.module
        assert restored.task_type == sample_task.task_type
        assert restored.priority == sample_task.priority
        assert restored.state == sample_task.state

    def test_task_with_conversation(self, sample_task, sample_conversation):
        """Task should serialize conversation state."""
        sample_task.conversation = sample_conversation
        data = sample_task.to_dict()
        assert data["conversation"] is not None
        assert data["conversation"]["llm"] == "gemini"

        restored = Task.from_dict(data)
        assert restored.conversation is not None
        assert restored.conversation.llm == "gemini"
        assert restored.conversation.turn_count == 3

    def test_mark_running(self, sample_task):
        """mark_running should update state and increment attempts."""
        assert sample_task.state == TaskState.PENDING
        assert sample_task.attempts == 0

        sample_task.mark_running("pool-req-123")

        assert sample_task.state == TaskState.RUNNING
        assert sample_task.attempts == 1
        assert sample_task.pool_request_id == "pool-req-123"
        assert sample_task.started_at is not None

    def test_mark_completed(self, sample_task):
        """mark_completed should update state and store result."""
        sample_task.mark_running()
        sample_task.mark_completed("Success result")

        assert sample_task.state == TaskState.COMPLETED
        assert sample_task.result == "Success result"
        assert sample_task.completed_at is not None

    def test_mark_failed(self, sample_task):
        """mark_failed should update state and store error."""
        sample_task.mark_running()
        sample_task.mark_failed("Connection timeout")

        assert sample_task.state == TaskState.FAILED
        assert sample_task.error == "Connection timeout"
        assert sample_task.completed_at is not None

    def test_can_retry_within_limit(self, sample_task):
        """can_retry should return True when under max_attempts."""
        assert sample_task.can_retry() is True
        sample_task.mark_running()
        assert sample_task.can_retry() is True
        sample_task.mark_running()
        assert sample_task.can_retry() is True

    def test_can_retry_at_limit(self, sample_task):
        """can_retry should return False when at max_attempts."""
        sample_task.attempts = 3
        assert sample_task.can_retry() is False

    def test_update_conversation(self, sample_task):
        """update_conversation should create or update conversation state."""
        assert sample_task.conversation is None

        sample_task.update_conversation(
            thread_url="https://chat.example.com/123",
            thread_title="New Thread",
            backend="gemini",
        )

        assert sample_task.conversation is not None
        assert sample_task.conversation.llm == "gemini"
        assert sample_task.conversation.external_thread_id == "https://chat.example.com/123"
        assert sample_task.conversation.thread_title == "New Thread"
        assert sample_task.conversation.turn_count == 1

        # Update again
        sample_task.update_conversation(
            thread_url="https://chat.example.com/123",
            thread_title="New Thread",
            backend="gemini",
        )
        assert sample_task.conversation.turn_count == 2


class TestBasePriority:
    """Tests for base priority lookup."""

    def test_known_task_types(self):
        """Known task types should have defined priorities."""
        assert get_base_priority("exploration") == 50
        assert get_base_priority("synthesis") == 60
        assert get_base_priority("address_comment") == 70
        assert get_base_priority("incorporate_insight") == 55

    def test_unknown_task_type(self):
        """Unknown task types should return default priority."""
        assert get_base_priority("unknown_type") == 50

    def test_all_base_priorities_defined(self):
        """All documented task types should be in BASE_PRIORITIES."""
        expected_types = [
            "exploration", "synthesis", "review", "critique",
            "address_comment", "incorporate_insight", "review_section", "draft_section",
        ]
        for task_type in expected_types:
            assert task_type in BASE_PRIORITIES, f"Missing: {task_type}"
