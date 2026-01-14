"""
Tests for documenter WorkPlanner.

Tests cover:
- Planning next work
- Parsing incorporate decisions
- Parsing prerequisite decisions
- Generating prerequisite content
"""

import re
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestWorkPlannerInit:
    """Tests for WorkPlanner initialization."""

    def test_init_stores_session(self):
        """WorkPlanner stores session reference."""
        from documenter.planning import WorkPlanner

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session, "guidance text")

        assert planner.session is mock_session
        assert planner.guidance_text == "guidance text"

    def test_init_default_guidance(self):
        """WorkPlanner defaults to empty guidance."""
        from documenter.planning import WorkPlanner

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session)

        assert planner.guidance_text == ""


class TestWorkPlannerPlanNextWork:
    """Tests for plan_next_work method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_opportunities(self):
        """Returns None when no opportunities available."""
        from documenter.planning import WorkPlanner

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[])

        planner = WorkPlanner(mock_session)
        result = await planner.plan_next_work()

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_consensus_with_context(self):
        """Calls consensus reviewer with proper context."""
        from documenter.planning import WorkPlanner

        # Create mock opportunity
        mock_opp = MagicMock()
        mock_opp.insight_id = "insight-123"
        mock_opp.text = "Test insight text"
        mock_opp.requires = []

        # Create mock consensus result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """DECISION: INCORPORATE
INSIGHT_ID: insight-123
REASON: This fits perfectly"""

        # Mock session components
        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=["concept1"])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Document summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)
        result = await planner.plan_next_work()

        mock_session.consensus.run.assert_called_once()
        assert result is not None
        assert result["type"] == "insight"

    @pytest.mark.asyncio
    async def test_handles_incorporate_decision(self):
        """Parses INCORPORATE decision correctly."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test-insight"
        mock_opp.text = "Test insight"
        mock_opp.requires = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """DECISION: INCORPORATE
INSIGHT_ID: test-insight
REASON: Good fit"""

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=[])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)
        result = await planner.plan_next_work()

        assert result["type"] == "insight"
        assert result["opportunity"] is mock_opp

    @pytest.mark.asyncio
    async def test_handles_prerequisite_decision(self):
        """Parses PREREQUISITE decision correctly."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test"
        mock_opp.text = "Test"
        mock_opp.requires = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """DECISION: PREREQUISITE
NEEDED: Define projective geometry
CONTENT:
## Projective Geometry

Projective geometry extends Euclidean geometry by adding points at infinity.

ESTABLISHES: projective geometry, point at infinity"""

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=[])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)
        result = await planner.plan_next_work()

        assert result["type"] == "prerequisite"
        assert "Projective Geometry" in result["content"]
        assert "projective geometry" in result["establishes"]

    @pytest.mark.asyncio
    async def test_handles_wait_decision(self):
        """Parses WAIT decision correctly."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test"
        mock_opp.text = "Test"
        mock_opp.requires = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """DECISION: WAIT
REASON: Need more insights about basic concepts first"""

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=[])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)
        result = await planner.plan_next_work()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        """Returns None when consensus fails."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test"
        mock_opp.text = "Test"
        mock_opp.requires = []

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.outcome = "Failed"

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=[])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)
        result = await planner.plan_next_work()

        assert result is None


class TestParseIncorporateDecision:
    """Tests for _parse_incorporate_decision method."""

    def test_parses_insight_id(self):
        """Correctly parses INSIGHT_ID from outcome."""
        from documenter.planning import WorkPlanner

        mock_opp1 = MagicMock()
        mock_opp1.insight_id = "insight-1"
        mock_opp2 = MagicMock()
        mock_opp2.insight_id = "insight-2"

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session)

        outcome = """DECISION: INCORPORATE
INSIGHT_ID: insight-2
REASON: Best fit"""

        result = planner._parse_incorporate_decision(outcome, [mock_opp1, mock_opp2])

        assert result is not None
        assert result["type"] == "insight"
        assert result["opportunity"] is mock_opp2

    def test_handles_bracketed_id(self):
        """Handles INSIGHT_ID with brackets."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test-id"

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session)

        outcome = """DECISION: INCORPORATE
INSIGHT_ID: [test-id]
REASON: Good"""

        result = planner._parse_incorporate_decision(outcome, [mock_opp])

        assert result["opportunity"] is mock_opp

    def test_returns_none_on_parse_failure(self):
        """Returns None if ID not found (triggers retry)."""
        from documenter.planning import WorkPlanner

        mock_opp1 = MagicMock()
        mock_opp1.insight_id = "first"
        mock_opp2 = MagicMock()
        mock_opp2.insight_id = "second"

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session)

        outcome = """DECISION: INCORPORATE
INSIGHT_ID: nonexistent
REASON: Test"""

        result = planner._parse_incorporate_decision(outcome, [mock_opp1, mock_opp2])

        # Should return None to trigger retry, not arbitrary first opportunity
        assert result is None


class TestParsePrerequisiteDecision:
    """Tests for _parse_prerequisite_decision method."""

    @pytest.mark.asyncio
    async def test_parses_content_and_establishes(self):
        """Correctly parses CONTENT and ESTABLISHES."""
        from documenter.planning import WorkPlanner

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session)

        outcome = """DECISION: PREREQUISITE
NEEDED: Define the Fano plane
CONTENT:
## The Fano Plane

The Fano plane is the smallest projective plane.

ESTABLISHES: Fano plane, projective plane, finite geometry"""

        result = await planner._parse_prerequisite_decision(outcome)

        assert result is not None
        assert result["type"] == "prerequisite"
        assert "Fano Plane" in result["content"]
        assert "Fano plane" in result["establishes"]

    @pytest.mark.asyncio
    async def test_falls_back_to_generate_if_no_content(self):
        """Generates content if only NEEDED provided."""
        from documenter.planning import WorkPlanner

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """CONTENT:
Generated content here.

ESTABLISHES: concept1, concept2"""
        mock_result.selection_stats = {}

        mock_session = MagicMock()
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)

        outcome = """DECISION: PREREQUISITE
NEEDED: Basic projective geometry concepts"""

        result = await planner._parse_prerequisite_decision(outcome)

        assert result is not None
        assert result["type"] == "prerequisite"
        mock_session.consensus.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_if_no_content_found(self):
        """Returns None if no content can be parsed."""
        from documenter.planning import WorkPlanner

        mock_session = MagicMock()
        planner = WorkPlanner(mock_session)

        outcome = """DECISION: PREREQUISITE
Some random text without proper formatting"""

        result = await planner._parse_prerequisite_decision(outcome)

        assert result is None


class TestGeneratePrerequisiteContent:
    """Tests for _generate_prerequisite_content method."""

    @pytest.mark.asyncio
    async def test_generates_content_with_consensus(self):
        """Generates content using consensus reviewer."""
        from documenter.planning import WorkPlanner

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """CONTENT:
## Introduction to Projective Geometry

Projective geometry is a type of geometry that studies geometric properties.

ESTABLISHES: projective geometry, geometric properties"""
        mock_result.selection_stats = {"selected_by": "quality"}

        mock_session = MagicMock()
        mock_session.document.get_summary = MagicMock(return_value="Document summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session, "Author guidance here")

        result = await planner._generate_prerequisite_content("Define projective geometry")

        assert result is not None
        assert result["type"] == "prerequisite"
        assert "Projective Geometry" in result["content"]
        assert "projective geometry" in result["establishes"]

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        """Returns None when consensus fails."""
        from documenter.planning import WorkPlanner

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.outcome = None

        mock_session = MagicMock()
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)

        result = await planner._generate_prerequisite_content("Something")

        assert result is None

    @pytest.mark.asyncio
    async def test_increments_consensus_calls(self):
        """Increments consensus call counter."""
        from documenter.planning import WorkPlanner

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = "CONTENT: test\n\nESTABLISHES: test"
        mock_result.selection_stats = {}

        mock_session = MagicMock()
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        planner = WorkPlanner(mock_session)

        await planner._generate_prerequisite_content("Test")

        mock_session.increment_consensus_calls.assert_called_once()


class TestWorkPlannerWithGuidance:
    """Tests for guidance text handling."""

    @pytest.mark.asyncio
    async def test_includes_guidance_in_context(self):
        """Includes guidance text in planning context."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test"
        mock_opp.text = "Test insight"
        mock_opp.requires = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """DECISION: WAIT
REASON: Following guidance"""

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=[])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False

        guidance = "Focus on mathematical rigor and elegant exposition."
        planner = WorkPlanner(mock_session, guidance)

        await planner.plan_next_work()

        # Check that guidance was included in the context
        call_args = mock_session.consensus.run.call_args
        context = call_args[0][0]
        assert "AUTHOR GUIDANCE" in context
        assert guidance in context


class TestDeepModeUsage:
    """Tests to verify deep_mode is used correctly (from config)."""

    @pytest.mark.asyncio
    async def test_uses_deep_mode_from_session(self):
        """Uses deep_mode setting from session."""
        from documenter.planning import WorkPlanner

        mock_opp = MagicMock()
        mock_opp.insight_id = "test"
        mock_opp.text = "Test"
        mock_opp.requires = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.outcome = """DECISION: WAIT
REASON: Test"""

        mock_session = MagicMock()
        mock_session.opportunity_finder._gather_all_opportunities = MagicMock(return_value=[mock_opp])
        mock_session.concept_tracker.get_established_concepts = MagicMock(return_value=[])
        mock_session.document.sections = []
        mock_session.document.get_summary = MagicMock(return_value="Summary")
        mock_session.consensus.run = AsyncMock(return_value=mock_result)
        mock_session.increment_consensus_calls = MagicMock()
        mock_session.use_deep_mode = False  # Important: should be False

        planner = WorkPlanner(mock_session)
        await planner.plan_next_work()

        # Verify deep_mode was passed as False
        call_kwargs = mock_session.consensus.run.call_args[1]
        assert call_kwargs["use_deep_mode"] is False
