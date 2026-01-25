"""
Explorer test fixtures.

Provides mocks and sample data for testing the explorer module.
Key constraints:
- Browser mocks must NEVER use deep_mode or pro_mode
- All LLM responses should be mocked, never make real calls
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_atomic_insight():
    """Create a sample AtomicInsight for testing."""
    from explorer.src.chunking.models import AtomicInsight

    return AtomicInsight.create(
        insight="The Fano plane exhibits duality between points and lines.",
        confidence="high",
        tags=["fano-plane", "duality", "projective-geometry"],
        source_thread_id="thread-abc123",
        extraction_model="claude",
        source_exchange_indices=[0, 1, 2],
    )


@pytest.fixture
def sample_versioned_insight(sample_atomic_insight):
    """Create a sample VersionedInsight for testing refinement."""
    from explorer.src.chunking.models import VersionedInsight

    return VersionedInsight.from_insight(sample_atomic_insight)


@pytest.fixture
def sample_blessed_insight():
    """Create a blessed insight for testing dependencies."""
    from explorer.src.chunking.models import AtomicInsight, InsightStatus

    insight = AtomicInsight.create(
        insight="Every projective plane has exactly 7 points and 7 lines.",
        confidence="high",
        tags=["projective-plane", "fano-plane", "counting"],
        source_thread_id="thread-blessed1",
        extraction_model="claude",
    )
    insight.apply_rating("⚡", notes="Verified mathematical fact")
    return insight


@pytest.fixture
def sample_pending_insights():
    """Create multiple pending insights for batch testing."""
    from explorer.src.chunking.models import AtomicInsight

    insights = [
        AtomicInsight.create(
            insight="Points and lines can be interchanged in the Fano plane.",
            confidence="high",
            tags=["fano-plane", "duality"],
            source_thread_id="thread-001",
            extraction_model="claude",
        ),
        AtomicInsight.create(
            insight="The automorphism group of the Fano plane has order 168.",
            confidence="medium",
            tags=["fano-plane", "automorphism", "group-theory"],
            source_thread_id="thread-002",
            extraction_model="gemini",
        ),
        AtomicInsight.create(
            insight="Three points determine a unique line in projective geometry.",
            confidence="low",
            tags=["projective-geometry", "collinearity"],
            source_thread_id="thread-003",
            extraction_model="chatgpt",
        ),
    ]
    return insights


@pytest.fixture
def sample_exploration_thread():
    """Create a mock exploration thread."""
    thread = MagicMock()
    thread.id = "test-thread-123"
    thread.topic = "Properties of the Fano plane"
    thread.chunks_extracted = False
    thread.extraction_note = None
    thread.get_context_for_prompt.return_value = """
    Exchange 1:
    User: What makes the Fano plane special?
    LLM: The Fano plane is the smallest projective plane with 7 points and 7 lines...

    Exchange 2:
    User: How does duality work in the Fano plane?
    LLM: Duality in the Fano plane means you can swap points and lines...
    """
    return thread


# ---------------------------------------------------------------------------
# Mock Browser Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini_browser():
    """
    Mock Gemini browser that NEVER uses deep_think mode.

    IMPORTANT: All tests must assert that use_deep_think=False
    """
    browser = AsyncMock()
    browser.name = "gemini"
    browser.is_available = MagicMock(return_value=True)

    async def mock_send(prompt, use_deep_think=False):
        # CRITICAL: Verify deep_think is never enabled
        assert use_deep_think is False, "Tests must never use deep_think mode!"
        return "Mock Gemini response for: " + prompt[:50]

    browser.send_message = mock_send
    return browser


@pytest.fixture
def mock_chatgpt_browser():
    """
    Mock ChatGPT browser that NEVER uses pro_mode.

    IMPORTANT: All tests must assert that use_pro_mode=False
    """
    browser = AsyncMock()
    browser.name = "chatgpt"
    browser.is_available = MagicMock(return_value=True)

    async def mock_send(prompt, use_pro_mode=False, use_thinking_mode=False):
        # CRITICAL: Verify pro_mode is never enabled
        assert use_pro_mode is False, "Tests must never use pro_mode!"
        return "Mock ChatGPT response for: " + prompt[:50]

    browser.send_message = mock_send
    return browser


@pytest.fixture
def mock_claude_reviewer():
    """
    Mock Claude API reviewer for extraction and review operations.
    """
    reviewer = AsyncMock()
    reviewer.name = "claude"
    reviewer.is_available = MagicMock(return_value=True)

    async def mock_send(prompt, extended_thinking=False):
        # Return a mock extraction response in expected format
        # Format must match parse_extraction_response expectations
        if "extract" in prompt.lower() or "insight" in prompt.lower():
            return """===
INSIGHT: The Fano plane demonstrates perfect duality between points and lines.
CONFIDENCE: high
TAGS: fano-plane, duality, projective-geometry
DEPENDS_ON: none
PENDING_DEPENDS: none
===
INSIGHT: Each line in the Fano plane contains exactly 3 points.
CONFIDENCE: high
TAGS: fano-plane, incidence, counting
DEPENDS_ON: none
PENDING_DEPENDS: none
==="""
        return "Mock Claude response"

    reviewer.send_message = mock_send
    return reviewer


# ---------------------------------------------------------------------------
# Mock Pool Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pool_client():
    """Mock pool client for sending requests."""
    client = AsyncMock()

    async def mock_send(backend, prompt, **kwargs):
        # Verify no pro/deep modes
        if "use_deep_think" in kwargs:
            assert kwargs["use_deep_think"] is False
        if "use_pro_mode" in kwargs:
            assert kwargs["use_pro_mode"] is False

        return {
            "response": f"Pool response from {backend}",
            "backend": backend,
            "success": True,
        }

    client.send = mock_send
    return client


# ---------------------------------------------------------------------------
# Data Directory Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def explorer_data_dir(temp_dir):
    """Create explorer data directory structure."""
    data_dir = temp_dir / "explorer" / "data"
    chunks_dir = data_dir / "chunks"

    # Create insight directories
    (chunks_dir / "insights" / "pending").mkdir(parents=True, exist_ok=True)
    (chunks_dir / "insights" / "blessed").mkdir(parents=True, exist_ok=True)
    (chunks_dir / "insights" / "interesting").mkdir(parents=True, exist_ok=True)
    (chunks_dir / "insights" / "rejected").mkdir(parents=True, exist_ok=True)

    # Create explorations directory
    (data_dir / "explorations").mkdir(parents=True, exist_ok=True)

    return data_dir


@pytest.fixture
def extractor(explorer_data_dir):
    """Create an AtomicExtractor with test data directory."""
    from explorer.src.chunking.extractor import AtomicExtractor

    config = {
        "chunking": {
            "max_insights_per_thread": 10,
            "min_confidence_to_keep": "low",
        },
        "dependencies": {
            "semantic_match_threshold": 0.5,
        },
    }
    return AtomicExtractor(explorer_data_dir / "chunks", config)


# ---------------------------------------------------------------------------
# Review Panel Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_review_panel():
    """Mock review panel for testing review operations."""
    panel = MagicMock()
    panel.run_review = AsyncMock(return_value={
        "final_rating": "⚡",
        "is_disputed": False,
        "notes": "All reviewers agreed",
        "individual_ratings": {
            "gemini": "⚡",
            "chatgpt": "⚡",
            "claude": "⚡",
        },
    })
    return panel


@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for different review scenarios."""
    return {
        "unanimous_blessing": {
            "gemini": {"rating": "⚡", "critique": "Solid mathematical foundation"},
            "chatgpt": {"rating": "⚡", "critique": "Well-articulated insight"},
            "claude": {"rating": "⚡", "critique": "Precise and accurate"},
        },
        "unanimous_rejection": {
            "gemini": {"rating": "✗", "critique": "Not specific to Fano plane"},
            "chatgpt": {"rating": "✗", "critique": "Too general"},
            "claude": {"rating": "✗", "critique": "Lacks mathematical depth"},
        },
        "disputed_interesting": {
            "gemini": {"rating": "⚡", "critique": "Good insight"},
            "chatgpt": {"rating": "?", "critique": "Needs verification"},
            "claude": {"rating": "⚡", "critique": "Accurate but could be clearer"},
        },
        "mixed_with_rejection": {
            "gemini": {"rating": "⚡", "critique": "Good"},
            "chatgpt": {"rating": "✗", "critique": "Not accurate"},
            "claude": {"rating": "?", "critique": "Uncertain"},
        },
    }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_no_deep_mode():
    """Helper to verify deep_think mode is never used."""
    def checker(calls):
        for call in calls:
            if "use_deep_think" in call.kwargs:
                assert call.kwargs["use_deep_think"] is False, \
                    "Tests must never enable deep_think mode!"
    return checker


@pytest.fixture
def assert_no_pro_mode():
    """Helper to verify pro_mode is never used."""
    def checker(calls):
        for call in calls:
            if "use_pro_mode" in call.kwargs:
                assert call.kwargs["use_pro_mode"] is False, \
                    "Tests must never enable pro_mode!"
    return checker
