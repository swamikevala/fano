"""
Tests for orchestrator/allocator.py
"""

import pytest

from orchestrator.allocator import (
    QuotaAllocator,
    QuotaBudget,
    AllocationResult,
    DEFAULT_BUDGETS,
)


class TestQuotaBudget:
    """Tests for QuotaBudget dataclass."""

    def test_budget_defaults(self):
        """QuotaBudget should have zero defaults."""
        budget = QuotaBudget()
        assert budget.explorer == 0
        assert budget.documenter == 0
        assert budget.buffer == 0

    def test_budget_total(self):
        """QuotaBudget total should sum all allocations."""
        budget = QuotaBudget(explorer=10, documenter=5, buffer=5)
        assert budget.total == 20

    def test_default_budgets_defined(self):
        """Default budgets should be defined for all backends."""
        assert "gemini_deep" in DEFAULT_BUDGETS
        assert "chatgpt_pro" in DEFAULT_BUDGETS
        assert "claude_extended" in DEFAULT_BUDGETS

    def test_default_gemini_budget(self):
        """Gemini deep budget should match spec."""
        budget = DEFAULT_BUDGETS["gemini_deep"]
        assert budget.explorer == 10
        assert budget.documenter == 5
        assert budget.buffer == 5


class TestAllocationResult:
    """Tests for AllocationResult dataclass."""

    def test_allocation_granted(self):
        """AllocationResult should represent granted allocation."""
        result = AllocationResult(
            granted=True,
            quota_type="gemini_deep",
            module="explorer",
            current_usage=5,
            budget_limit=10,
        )
        assert result.granted is True
        assert result.used_buffer is False

    def test_allocation_denied(self):
        """AllocationResult should represent denied allocation."""
        result = AllocationResult(
            granted=False,
            quota_type="gemini_deep",
            module="explorer",
            message="Quota exhausted",
        )
        assert result.granted is False


class TestQuotaAllocator:
    """Tests for QuotaAllocator."""

    @pytest.mark.asyncio
    async def test_can_allocate_within_budget(self, allocator, state_manager):
        """can_allocate should grant when within budget."""
        await state_manager.startup()
        try:
            result = allocator.can_allocate("gemini_deep", "explorer")
            assert result.granted is True
            assert result.used_buffer is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_can_allocate_exhausted_uses_buffer(self, allocator, state_manager):
        """can_allocate should use buffer when module budget exhausted."""
        await state_manager.startup()
        try:
            # Exhaust explorer budget
            for i in range(10):
                state_manager.increment_quota("gemini_deep", "explorer")

            result = allocator.can_allocate("gemini_deep", "explorer")
            assert result.granted is True
            assert result.used_buffer is True
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_can_allocate_all_exhausted(self, allocator, state_manager):
        """can_allocate should deny when all quotas exhausted."""
        await state_manager.startup()
        try:
            # Exhaust explorer budget
            for i in range(10):
                state_manager.increment_quota("gemini_deep", "explorer")

            # Exhaust buffer
            for i in range(5):
                state_manager.increment_quota("gemini_deep", "buffer")

            result = allocator.can_allocate("gemini_deep", "explorer")
            assert result.granted is False
            assert "exhausted" in result.message.lower()
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_can_allocate_non_rationed(self, allocator, state_manager):
        """can_allocate should always grant for non-rationed resources."""
        await state_manager.startup()
        try:
            result = allocator.can_allocate("unknown_backend", "explorer")
            assert result.granted is True
            assert result.message == "Non-rationed resource"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_allocate_consumes_quota(self, allocator, state_manager):
        """allocate should consume quota."""
        await state_manager.startup()
        try:
            initial = state_manager.get_quota_usage("gemini_deep", "explorer")

            result = allocator.allocate("gemini_deep", "explorer")

            assert result.granted is True
            assert state_manager.get_quota_usage("gemini_deep", "explorer") == initial + 1
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_allocate_consumes_buffer(self, allocator, state_manager):
        """allocate should consume buffer when module exhausted."""
        await state_manager.startup()
        try:
            # Exhaust explorer budget
            for i in range(10):
                state_manager.increment_quota("gemini_deep", "explorer")

            initial_buffer = state_manager.get_quota_usage("gemini_deep", "buffer")

            result = allocator.allocate("gemini_deep", "explorer")

            assert result.granted is True
            assert result.used_buffer is True
            assert state_manager.get_quota_usage("gemini_deep", "buffer") == initial_buffer + 1
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_allocate_denied(self, allocator, state_manager):
        """allocate should not consume when denied."""
        await state_manager.startup()
        try:
            # Exhaust all quotas
            for i in range(10):
                state_manager.increment_quota("gemini_deep", "explorer")
            for i in range(5):
                state_manager.increment_quota("gemini_deep", "buffer")

            result = allocator.allocate("gemini_deep", "explorer")

            assert result.granted is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_remaining(self, allocator, state_manager):
        """get_remaining should return correct remaining quota."""
        await state_manager.startup()
        try:
            # Fresh state: explorer=10, buffer=5 = 15 remaining
            remaining = allocator.get_remaining("gemini_deep", "explorer")
            assert remaining == 15

            # Use 3 from explorer
            for i in range(3):
                state_manager.increment_quota("gemini_deep", "explorer")

            remaining = allocator.get_remaining("gemini_deep", "explorer")
            assert remaining == 12  # 7 explorer + 5 buffer
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_remaining_non_rationed(self, allocator, state_manager):
        """get_remaining should return infinity for non-rationed."""
        await state_manager.startup()
        try:
            remaining = allocator.get_remaining("unknown", "explorer")
            assert remaining == float('inf')
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_status(self, allocator, state_manager):
        """get_status should return quota status for all types."""
        await state_manager.startup()
        try:
            state_manager.increment_quota("gemini_deep", "explorer")

            status = allocator.get_status()

            assert "gemini_deep" in status
            assert status["gemini_deep"]["explorer"]["used"] == 1
            assert status["gemini_deep"]["explorer"]["budget"] == 10
            assert status["gemini_deep"]["explorer"]["remaining"] == 9
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_status_specific_type(self, allocator, state_manager):
        """get_status with quota_type should return only that type."""
        await state_manager.startup()
        try:
            status = allocator.get_status("gemini_deep")

            assert "gemini_deep" in status
            assert "chatgpt_pro" not in status
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_set_budget(self, allocator, state_manager):
        """set_budget should update budget limits."""
        await state_manager.startup()
        try:
            allocator.set_budget("gemini_deep", explorer=20, documenter=10)

            assert allocator.budgets["gemini_deep"].explorer == 20
            assert allocator.budgets["gemini_deep"].documenter == 10
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_set_budget_new_type(self, allocator, state_manager):
        """set_budget should create new budget type if missing."""
        await state_manager.startup()
        try:
            allocator.set_budget("new_backend", explorer=5, buffer=2)

            assert "new_backend" in allocator.budgets
            assert allocator.budgets["new_backend"].explorer == 5
            assert allocator.budgets["new_backend"].buffer == 2
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_should_use_deep_mode_exploration(self, allocator, state_manager):
        """should_use_deep_mode should recommend deep for exploration."""
        await state_manager.startup()
        try:
            should_use, backend = allocator.should_use_deep_mode(
                module="explorer",
                task_type="exploration",
            )
            assert should_use is True
            assert backend == "gemini_deep"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_should_use_deep_mode_review(self, allocator, state_manager):
        """should_use_deep_mode should not recommend deep for review."""
        await state_manager.startup()
        try:
            should_use, reason = allocator.should_use_deep_mode(
                module="explorer",
                task_type="review",
            )
            assert should_use is False
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_should_use_deep_mode_with_available_quota(self, allocator, state_manager):
        """should_use_deep_mode should return preferred backend when quota available."""
        await state_manager.startup()
        try:
            # Use some quota but not all
            for i in range(5):
                allocator.allocate("gemini_deep", "explorer")

            should_use, backend = allocator.should_use_deep_mode(
                module="explorer",
                task_type="exploration",
                preferred_backend="gemini_deep",
            )

            # Should still use preferred backend
            assert should_use is True
            assert backend == "gemini_deep"
        finally:
            await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_should_use_deep_mode_all_exhausted(self, allocator, state_manager):
        """should_use_deep_mode should return False when all exhausted."""
        await state_manager.startup()
        try:
            # Exhaust all backends
            for backend in ["gemini_deep", "chatgpt_pro", "claude_extended"]:
                budget = allocator.budgets[backend]
                for _ in range(budget.explorer + budget.buffer):
                    state_manager.increment_quota(backend, "explorer")
                    state_manager.increment_quota(backend, "buffer")

            should_use, reason = allocator.should_use_deep_mode(
                module="explorer",
                task_type="exploration",
            )

            assert should_use is False
            assert "exhausted" in reason.lower()
        finally:
            await state_manager.shutdown()
