"""
Tests for orchestrator/adapters.py
"""

import pytest
from typing import Optional

from orchestrator.adapters import (
    ModuleInterface,
    ModuleRegistry,
    PromptContext,
    TaskResult,
    TaskType,
    run_in_executor,
)
from orchestrator.models import Task, TaskState


class MockModuleAdapter(ModuleInterface):
    """Mock module adapter for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._initialized = False
        self._pending_work = []
        self._handled_results = []
        self._failed_tasks = []

    @property
    def module_name(self) -> str:
        return self._name

    @property
    def supported_task_types(self) -> list[str]:
        return ["mock_task", "another_task"]

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def shutdown(self):
        self._initialized = False

    async def get_pending_work(self) -> list[dict]:
        return self._pending_work

    async def build_prompt(self, task: Task) -> PromptContext:
        return PromptContext(
            prompt=f"Mock prompt for {task.id}",
            images=[],
            requires_deep_mode=task.requires_deep_mode,
        )

    async def handle_result(self, task: Task, result: TaskResult) -> bool:
        self._handled_results.append((task, result))
        return True

    async def on_task_failed(self, task: Task, error: str):
        self._failed_tasks.append((task, error))

    def set_pending_work(self, work: list[dict]):
        self._pending_work = work


class TestTaskType:
    """Tests for TaskType enum."""

    def test_explorer_task_types(self):
        """Explorer task types should be defined."""
        assert TaskType.EXPLORATION.value == "exploration"
        assert TaskType.CRITIQUE.value == "critique"
        assert TaskType.SYNTHESIS.value == "synthesis"
        assert TaskType.REVIEW.value == "review"

    def test_documenter_task_types(self):
        """Documenter task types should be defined."""
        assert TaskType.ADDRESS_COMMENT.value == "address_comment"
        assert TaskType.INCORPORATE_INSIGHT.value == "incorporate_insight"
        assert TaskType.REVIEW_SECTION.value == "review_section"
        assert TaskType.DRAFT_SECTION.value == "draft_section"


class TestPromptContext:
    """Tests for PromptContext dataclass."""

    def test_prompt_context_defaults(self):
        """PromptContext should have sensible defaults."""
        ctx = PromptContext(prompt="Test prompt")
        assert ctx.prompt == "Test prompt"
        assert ctx.images == []
        assert ctx.system_context is None
        assert ctx.thread_context is None
        assert ctx.requires_deep_mode is False
        assert ctx.preferred_backend is None
        assert ctx.metadata == {}

    def test_prompt_context_with_images(self):
        """PromptContext should store images."""
        images = [{"data": "base64..."}]
        ctx = PromptContext(prompt="Test", images=images)
        assert ctx.images == images


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_success(self):
        """TaskResult should represent success."""
        result = TaskResult(
            success=True,
            response="LLM response text",
            thread_url="https://chat.example.com/123",
            thread_title="Test Chat",
            deep_mode_used=True,
        )
        assert result.success is True
        assert result.response == "LLM response text"
        assert result.error is None
        assert result.needs_continuation is False

    def test_task_result_failure(self):
        """TaskResult should represent failure."""
        result = TaskResult(
            success=False,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"

    def test_task_result_continuation(self):
        """TaskResult should support continuation."""
        result = TaskResult(
            success=True,
            response="Partial response",
            needs_continuation=True,
            continuation_payload={"next_prompt": "Continue..."},
        )
        assert result.needs_continuation is True
        assert result.continuation_payload == {"next_prompt": "Continue..."}


class TestModuleRegistry:
    """Tests for ModuleRegistry."""

    def test_register_module(self):
        """register should add module to registry."""
        registry = ModuleRegistry()
        mock = MockModuleAdapter("test_module")

        registry.register(mock)

        assert "test_module" in registry._modules
        assert registry.get_module("test_module") is mock

    def test_get_module_for_task(self):
        """get_module_for_task should return correct module."""
        registry = ModuleRegistry()
        mock = MockModuleAdapter("test_module")
        registry.register(mock)

        module = registry.get_module_for_task("mock_task")
        assert module is mock

        module = registry.get_module_for_task("another_task")
        assert module is mock

    def test_get_module_for_unknown_task(self):
        """get_module_for_task should return None for unknown tasks."""
        registry = ModuleRegistry()

        module = registry.get_module_for_task("unknown_task")
        assert module is None

    def test_get_all_modules(self):
        """get_all_modules should return all registered modules."""
        registry = ModuleRegistry()
        mock1 = MockModuleAdapter("module1")
        mock2 = MockModuleAdapter("module2")

        registry.register(mock1)
        registry.register(mock2)

        modules = registry.get_all_modules()
        assert len(modules) == 2
        assert mock1 in modules
        assert mock2 in modules

    @pytest.mark.asyncio
    async def test_initialize_all(self):
        """initialize_all should initialize all modules."""
        registry = ModuleRegistry()
        mock1 = MockModuleAdapter("module1")
        mock2 = MockModuleAdapter("module2")
        registry.register(mock1)
        registry.register(mock2)

        result = await registry.initialize_all()

        assert result is True
        assert mock1._initialized is True
        assert mock2._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_all_failure(self):
        """initialize_all should return False if any module fails."""
        registry = ModuleRegistry()
        mock = MockModuleAdapter("failing")

        # Make initialize fail
        async def failing_init():
            return False
        mock.initialize = failing_init

        registry.register(mock)

        result = await registry.initialize_all()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """shutdown_all should shutdown all modules."""
        registry = ModuleRegistry()
        mock1 = MockModuleAdapter("module1")
        mock2 = MockModuleAdapter("module2")
        registry.register(mock1)
        registry.register(mock2)

        await registry.initialize_all()
        await registry.shutdown_all()

        assert mock1._initialized is False
        assert mock2._initialized is False


class TestMockModuleAdapter:
    """Tests for MockModuleAdapter functionality."""

    @pytest.mark.asyncio
    async def test_build_prompt(self):
        """build_prompt should return PromptContext."""
        mock = MockModuleAdapter()
        task = Task(
            id="t1", key="k1", module="mock",
            task_type="mock_task", priority=50,
            requires_deep_mode=True,
        )

        ctx = await mock.build_prompt(task)

        assert "t1" in ctx.prompt
        assert ctx.requires_deep_mode is True

    @pytest.mark.asyncio
    async def test_handle_result(self):
        """handle_result should track handled results."""
        mock = MockModuleAdapter()
        task = Task(id="t1", key="k1", module="mock",
                   task_type="mock_task", priority=50)
        result = TaskResult(success=True, response="Test response")

        handled = await mock.handle_result(task, result)

        assert handled is True
        assert len(mock._handled_results) == 1
        assert mock._handled_results[0][0].id == "t1"

    @pytest.mark.asyncio
    async def test_on_task_failed(self):
        """on_task_failed should track failed tasks."""
        mock = MockModuleAdapter()
        task = Task(id="t1", key="k1", module="mock",
                   task_type="mock_task", priority=50)

        await mock.on_task_failed(task, "Connection error")

        assert len(mock._failed_tasks) == 1
        assert mock._failed_tasks[0][1] == "Connection error"

    @pytest.mark.asyncio
    async def test_get_pending_work(self):
        """get_pending_work should return configured work."""
        mock = MockModuleAdapter()
        mock.set_pending_work([
            {"task_type": "mock_task", "key": "k1", "payload": {}},
            {"task_type": "mock_task", "key": "k2", "payload": {}},
        ])

        work = await mock.get_pending_work()

        assert len(work) == 2
        assert work[0]["key"] == "k1"


class TestRunInExecutor:
    """Tests for run_in_executor helper."""

    @pytest.mark.asyncio
    async def test_run_sync_function(self):
        """run_in_executor should run sync function."""
        def sync_function(x, y):
            return x + y

        result = await run_in_executor(sync_function, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_run_with_kwargs(self):
        """run_in_executor should support kwargs."""
        def sync_function(x, y=10):
            return x * y

        result = await run_in_executor(sync_function, 5, y=3)
        assert result == 15
