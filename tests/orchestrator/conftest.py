"""
Shared fixtures for orchestrator tests.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import pytest

from orchestrator.models import Task, TaskState, ConversationState
from orchestrator.state import StateManager
from orchestrator.scheduler import Scheduler, RecentFailureCache
from orchestrator.allocator import QuotaAllocator, QuotaBudget


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        id="task-001",
        key="test:sample",
        module="explorer",
        task_type="exploration",
        priority=50,
        payload={"thread_id": "thread-123"},
    )


@pytest.fixture
def sample_task_documenter() -> Task:
    """Create a sample documenter task for testing."""
    return Task(
        id="task-002",
        key="doc:insight:123",
        module="documenter",
        task_type="incorporate_insight",
        priority=55,
        payload={"insight_id": "insight-123"},
        requires_deep_mode=True,
    )


@pytest.fixture
def sample_conversation() -> ConversationState:
    """Create a sample conversation state."""
    return ConversationState(
        llm="gemini",
        external_thread_id="https://gemini.google.com/chat/abc123",
        thread_title="Mathematical Exploration",
        turn_count=3,
    )


@pytest.fixture
def state_manager(temp_dir: Path) -> StateManager:
    """Create a StateManager with temporary storage."""
    return StateManager(
        checkpoint_dir=str(temp_dir),
        checkpoint_interval=60,
    )


@pytest.fixture
def failure_cache() -> RecentFailureCache:
    """Create a failure cache for testing."""
    return RecentFailureCache(ttl_seconds=3600, max_size=100)


@pytest.fixture
def scheduler(state_manager: StateManager) -> Scheduler:
    """Create a Scheduler with the given state manager."""
    return Scheduler(state_manager=state_manager)


@pytest.fixture
def allocator(state_manager: StateManager) -> QuotaAllocator:
    """Create a QuotaAllocator with the given state manager."""
    return QuotaAllocator(state_manager=state_manager)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Pytest configuration for async tests
def pytest_configure(config):
    """Configure pytest for async tests."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an asyncio test"
    )
