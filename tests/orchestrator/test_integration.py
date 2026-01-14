"""
Integration tests for orchestrator/main.py
"""

import asyncio
import tempfile
import shutil
from pathlib import Path

import pytest

from orchestrator.main import Orchestrator
from orchestrator.adapters import ModuleInterface, PromptContext, TaskResult
from orchestrator.models import Task, TaskState


class MockModule(ModuleInterface):
    """Mock module for integration testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._initialized = False
        self._shutdown = False
        self._pending_work = []
        self._prompts_built = []
        self._results_handled = []
        self._tasks_started = []

    @property
    def module_name(self) -> str:
        return self._name

    @property
    def supported_task_types(self) -> list[str]:
        return ["test_task"]

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def shutdown(self):
        self._shutdown = True

    async def get_pending_work(self) -> list[dict]:
        work = self._pending_work
        self._pending_work = []  # Clear after returning
        return work

    async def build_prompt(self, task: Task) -> PromptContext:
        self._prompts_built.append(task.id)
        return PromptContext(
            prompt=f"Test prompt for {task.id}",
            requires_deep_mode=False,
        )

    async def handle_result(self, task: Task, result: TaskResult) -> bool:
        self._results_handled.append((task.id, result.success))
        return True

    async def on_task_failed(self, task: Task, error: str):
        pass

    async def on_task_started(self, task: Task):
        self._tasks_started.append(task.id)

    async def get_system_state(self) -> dict:
        return {
            "blessed_insights_pending": 5,
            "comments_pending": 2,
        }

    def add_work(self, work: dict):
        self._pending_work.append(work)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""

    def test_orchestrator_creation(self, temp_data_dir):
        """Orchestrator should initialize with default config."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            pool_base_url="http://localhost:8765",
        )

        assert orch.data_dir == temp_data_dir
        assert orch.pool_base_url == "http://localhost:8765"
        assert orch.backends == ["gemini", "chatgpt", "claude"]
        assert orch._running is False

    def test_orchestrator_custom_backends(self, temp_data_dir):
        """Orchestrator should accept custom backends."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=["gemini"],
        )

        assert orch.backends == ["gemini"]

    def test_orchestrator_custom_config(self, temp_data_dir):
        """Orchestrator should accept custom config."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            config={
                "checkpoint_interval": 120,
                "work_poll_interval": 10.0,
            },
        )

        assert orch.config["checkpoint_interval"] == 120
        assert orch._work_poll_interval == 10.0


class TestOrchestratorModuleRegistration:
    """Tests for module registration."""

    def test_register_module(self, temp_data_dir):
        """register_module should add module to registry."""
        orch = Orchestrator(data_dir=str(temp_data_dir))
        mock = MockModule("test")

        orch.register_module(mock)

        assert orch.registry.get_module("test") is mock

    def test_register_multiple_modules(self, temp_data_dir):
        """Multiple modules can be registered."""
        orch = Orchestrator(data_dir=str(temp_data_dir))
        mock1 = MockModule("module1")
        mock2 = MockModule("module2")

        orch.register_module(mock1)
        orch.register_module(mock2)

        assert len(orch.registry.get_all_modules()) == 2


class TestOrchestratorLifecycle:
    """Tests for orchestrator start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_initializes_modules(self, temp_data_dir):
        """start should initialize all registered modules."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],  # No workers to simplify test
        )
        mock = MockModule("test")
        orch.register_module(mock)

        await orch.start()
        try:
            assert mock._initialized is True
            assert orch._running is True
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_modules(self, temp_data_dir):
        """stop should shutdown all modules."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )
        mock = MockModule("test")
        orch.register_module(mock)

        await orch.start()
        await orch.stop()

        assert mock._shutdown is True
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_double_start_ignored(self, temp_data_dir):
        """Calling start twice should be safe."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        await orch.start()  # Should not raise
        try:
            assert orch._running is True
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_double_stop_ignored(self, temp_data_dir):
        """Calling stop twice should be safe."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        await orch.stop()
        await orch.stop()  # Should not raise

        assert orch._running is False


class TestOrchestratorTaskSubmission:
    """Tests for task submission."""

    @pytest.mark.asyncio
    async def test_submit_task(self, temp_data_dir):
        """submit_task should create and schedule task."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            task = orch.submit_task(
                module="test",
                task_type="test_task",
                key="test:key:1",
                payload={"data": "value"},
            )

            assert task is not None
            assert task.module == "test"
            assert task.task_type == "test_task"
            assert task.key == "test:key:1"
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_submit_duplicate_rejected(self, temp_data_dir):
        """Duplicate keys should be rejected."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            task1 = orch.submit_task(
                module="test",
                task_type="test_task",
                key="same:key",
            )
            task2 = orch.submit_task(
                module="test",
                task_type="test_task",
                key="same:key",
            )

            assert task1 is not None
            assert task2 is None
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_submit_continuation(self, temp_data_dir):
        """submit_continuation should preserve conversation state."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            task1 = orch.submit_task(
                module="test",
                task_type="test_task",
                key="original",
            )
            task1.update_conversation(
                thread_url="http://example.com/chat",
                thread_title="Test Chat",
                backend="gemini",
            )

            task2 = orch.submit_continuation(task1, {"new": "payload"})

            assert task2 is not None
            assert task2.conversation is not None
            assert task2.conversation.llm == "gemini"
        finally:
            await orch.stop()


class TestOrchestratorStatus:
    """Tests for orchestrator status."""

    @pytest.mark.asyncio
    async def test_get_status(self, temp_data_dir):
        """get_status should return comprehensive status."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )
        mock = MockModule("test")
        orch.register_module(mock)

        await orch.start()
        try:
            status = orch.get_status()

            assert status["running"] is True
            assert "queue" in status
            assert "quotas" in status
            assert "modules" in status
            assert "test" in status["modules"]
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_get_pending_count(self, temp_data_dir):
        """get_pending_count should return correct count."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            assert orch.get_pending_count() == 0

            orch.submit_task(
                module="test",
                task_type="test_task",
                key="k1",
            )

            assert orch.get_pending_count() == 1
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_get_task(self, temp_data_dir):
        """get_task should return task by ID."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            task = orch.submit_task(
                module="test",
                task_type="test_task",
                key="k1",
            )

            retrieved = orch.get_task(task.id)
            assert retrieved is not None
            assert retrieved.id == task.id
        finally:
            await orch.stop()


class TestOrchestratorConfiguration:
    """Tests for orchestrator configuration."""

    @pytest.mark.asyncio
    async def test_update_module_weights(self, temp_data_dir):
        """update_module_weights should update scheduler."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            orch.update_module_weights({"explorer": 0.5, "documenter": 1.5})

            assert orch.scheduler.module_weights["explorer"] == 0.5
            assert orch.scheduler.module_weights["documenter"] == 1.5
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_clear_failure_cache(self, temp_data_dir):
        """clear_failure_cache should clear scheduler's cache."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )

        await orch.start()
        try:
            orch.scheduler.failure_cache.add("key1")
            orch.scheduler.failure_cache.add("key2")

            orch.clear_failure_cache("key1")
            assert orch.scheduler.failure_cache.contains("key1") is False
            assert orch.scheduler.failure_cache.contains("key2") is True

            orch.clear_failure_cache()  # Clear all
            assert orch.scheduler.failure_cache.contains("key2") is False
        finally:
            await orch.stop()


class TestOrchestratorWorkPolling:
    """Tests for work polling from modules."""

    @pytest.mark.asyncio
    async def test_poll_modules_for_work(self, temp_data_dir):
        """Polling should submit work from modules."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
            config={"work_poll_interval": 0.1},
        )
        mock = MockModule("test")
        mock.add_work({
            "task_type": "test_task",
            "key": "polled:work:1",
            "payload": {"source": "module"},
        })
        orch.register_module(mock)

        await orch.start()
        try:
            # Manually poll
            await orch._poll_modules_for_work()

            assert orch.get_pending_count() == 1
            task = orch.state.get_pending_tasks()[0]
            assert task.key == "polled:work:1"
        finally:
            await orch.stop()


class TestOrchestratorPromptAndResult:
    """Tests for prompt building and result handling."""

    @pytest.mark.asyncio
    async def test_build_prompt_for_task(self, temp_data_dir):
        """Prompt building should delegate to module."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )
        mock = MockModule("test")
        orch.register_module(mock)

        await orch.start()
        try:
            task = orch.submit_task(
                module="test",
                task_type="test_task",
                key="k1",
            )

            prompt_data = await orch._build_prompt_for_task(task)

            assert "k1" in mock._tasks_started or task.id in mock._prompts_built
            assert "prompt" in prompt_data
        finally:
            await orch.stop()

    @pytest.mark.asyncio
    async def test_handle_task_result(self, temp_data_dir):
        """Result handling should delegate to module."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )
        mock = MockModule("test")
        orch.register_module(mock)

        await orch.start()
        try:
            task = orch.submit_task(
                module="test",
                task_type="test_task",
                key="k1",
            )

            result = {
                "success": True,
                "response": "Test response",
            }
            await orch._handle_task_result(task, result)

            assert len(mock._results_handled) == 1
            assert mock._results_handled[0][0] == task.id
            assert mock._results_handled[0][1] is True
        finally:
            await orch.stop()


class TestOrchestratorSystemState:
    """Tests for system state updates."""

    @pytest.mark.asyncio
    async def test_update_system_state(self, temp_data_dir):
        """System state should be collected from modules."""
        orch = Orchestrator(
            data_dir=str(temp_data_dir),
            backends=[],
        )
        mock = MockModule("test")
        orch.register_module(mock)

        await orch.start()
        try:
            await orch._update_system_state()

            assert orch.scheduler.blessed_insights_pending == 5
            assert orch.scheduler.comments_pending == 2
        finally:
            await orch.stop()
