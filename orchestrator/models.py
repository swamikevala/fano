"""
Data models for the Orchestrator.

Based on v4.0 design specification.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import json


class TaskState(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class ConversationState:
    """
    State for resumable multi-turn conversations.

    Stores thread URL and title for server-side navigation.
    We explicitly reject context replay - we rely on provider's
    server-side state via URL navigation.
    """
    # LLM backend this conversation is with
    llm: str

    # Thread URL for direct navigation (e.g., https://chatgpt.com/c/123-abc)
    external_thread_id: Optional[str] = None

    # Thread title for sidebar fallback navigation
    thread_title: Optional[str] = None

    # Turn count in this conversation
    turn_count: int = 0

    def to_dict(self) -> dict:
        return {
            "llm": self.llm,
            "external_thread_id": self.external_thread_id,
            "thread_title": self.thread_title,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        return cls(
            llm=data["llm"],
            external_thread_id=data.get("external_thread_id"),
            thread_title=data.get("thread_title"),
            turn_count=data.get("turn_count", 0),
        )


@dataclass
class Task:
    """
    A preemptible unit of work.

    Tasks are the smallest schedulable units. They may require
    multiple LLM interactions to complete.
    """
    # Unique identifier
    id: str

    # Deduplication key - modules repeatedly propose same work
    key: str

    # Module that owns this task: "explorer" or "documenter"
    module: str

    # Task type (e.g., "exploration", "synthesis", "incorporate_insight")
    task_type: str

    # Computed priority score (higher = more urgent)
    priority: int

    # Current state
    state: TaskState = TaskState.PENDING

    # Task-specific payload (thread_id, insight_id, etc.)
    payload: dict = field(default_factory=dict)

    # Conversation state for multi-turn tasks
    conversation: Optional[ConversationState] = None

    # Recovery handles
    pool_request_id: Optional[str] = None

    # Retry tracking
    attempts: int = 0
    max_attempts: int = 3

    # Timestamps
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # LLM requirements
    requires_deep_mode: bool = False
    preferred_backend: Optional[str] = None  # Preferred LLM backend

    # Result storage
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "key": self.key,
            "module": self.module,
            "task_type": self.task_type,
            "priority": self.priority,
            "state": self.state.value,
            "payload": self.payload,
            "conversation": self.conversation.to_dict() if self.conversation else None,
            "pool_request_id": self.pool_request_id,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "requires_deep_mode": self.requires_deep_mode,
            "preferred_backend": self.preferred_backend,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Deserialize task from dictionary."""
        conversation = None
        if data.get("conversation"):
            conversation = ConversationState.from_dict(data["conversation"])

        return cls(
            id=data["id"],
            key=data["key"],
            module=data["module"],
            task_type=data["task_type"],
            priority=data["priority"],
            state=TaskState(data["state"]),
            payload=data.get("payload", {}),
            conversation=conversation,
            pool_request_id=data.get("pool_request_id"),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            created_at=data.get("created_at", datetime.now().timestamp()),
            updated_at=data.get("updated_at", datetime.now().timestamp()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            requires_deep_mode=data.get("requires_deep_mode", False),
            preferred_backend=data.get("preferred_backend"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def mark_running(self, pool_request_id: Optional[str] = None):
        """Mark task as running."""
        self.state = TaskState.RUNNING
        self.started_at = datetime.now().timestamp()
        self.updated_at = self.started_at
        self.attempts += 1
        if pool_request_id:
            self.pool_request_id = pool_request_id

    def mark_completed(self, result: Optional[str] = None):
        """Mark task as completed."""
        self.state = TaskState.COMPLETED
        self.completed_at = datetime.now().timestamp()
        self.updated_at = self.completed_at
        self.result = result

    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.state = TaskState.FAILED
        self.completed_at = datetime.now().timestamp()
        self.updated_at = self.completed_at
        self.error = error

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.attempts < self.max_attempts

    def update_conversation(self, thread_url: Optional[str], thread_title: Optional[str], backend: str):
        """Update conversation state after LLM interaction."""
        if not self.conversation:
            self.conversation = ConversationState(llm=backend)

        self.conversation.external_thread_id = thread_url
        self.conversation.thread_title = thread_title
        self.conversation.turn_count += 1
        self.updated_at = datetime.now().timestamp()


# Base priorities by task type (from design spec)
BASE_PRIORITIES = {
    # Explorer tasks
    "exploration": 50,
    "synthesis": 60,      # Higher: produces blessed insights
    "review": 55,
    "critique": 45,

    # Documenter tasks
    "address_comment": 70,  # Highest: human feedback
    "incorporate_insight": 55,
    "review_section": 40,
    "draft_section": 50,

    # Researcher tasks (Phase 2)
    "evaluate_source": 45,
    "extract_content": 40,
    "generate_questions": 35,
    "synthesize_findings": 50,
}


def get_base_priority(task_type: str) -> int:
    """Get base priority for a task type."""
    return BASE_PRIORITIES.get(task_type, 50)
