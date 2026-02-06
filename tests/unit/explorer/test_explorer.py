"""Tests for the Explorer Engine module (Design Spec Section 8.9)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import Config
from shared.errors import ConsensusError
from shared.events import EventBus
from shared.models import (
    Confidence,
    ConsensusResult,
    EvaluationCriterion,
    Event,
    Exchange,
    ExchangeRole,
    Insight,
    InsightStatus,
    LLMResponse,
    ParsedResponse,
    Project,
    ProjectStatus,
    ResearchDomain,
    RoundResult,
    Seed,
    SeedStatus,
    SeedType,
    Thread,
    ThreadStatus,
    TokenUsage,
    ValidatedResponse,
    Verdict,
)
from shared.store import StateStore
from ulid import ULID


def _gen_id() -> str:
    return str(ULID())

# ── Helpers ──────────────────────────────────────────────────

NOW = datetime.now(timezone.utc)

SAMPLE_PROJECT = Project(
    id="proj-001",
    name="Test Project",
    goal="Explore test hypotheses.",
    context="Test context.",
    evaluation_criteria=[
        EvaluationCriterion(name="rigor", description="Logical soundness", weight=1.0),
        EvaluationCriterion(name="depth", description="Structural depth", weight=0.8),
    ],
    exploration_guidance="Follow curiosity.",
    document_guidance="Write clearly.",
    seed_modification_enabled=True,
    seed_modification_require_approval=False,
    research_domains=[ResearchDomain(name="math", keywords=["math"], source_types=["paper"])],
    status=ProjectStatus.ACTIVE,
    created_at=NOW,
    updated_at=NOW,
)

import os
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

TEST_CONFIG_DATA = {
    "llm": {
        "api_key_env": "OPENROUTER_API_KEY",
        "models": {"gemini": "google/gemini-flash", "claude": "anthropic/claude-sonnet"},
        "default_timeout_seconds": 30,
    },
    "consensus": {
        "backends": ["gemini", "claude"],
        "convergence_threshold": 0.7,
        "minimum_agreement": 0.66,
    },
    "explorer": {
        "max_active_threads": 3,
        "min_exchanges_for_chunk": 4,
        "max_exchanges_per_thread": 12,
        "thread_retirement": {
            "max_idle_hours": 48,
            "max_reexplore_count": 3,
        },
        "model_weights": {
            "exploration": {"gemini": 1.0},
            "critique": {"claude": 1.0},
            "synthesis": {"gemini": 0.5, "claude": 0.5},
        },
    },
}

TEST_PROJECT_DATA = {
    "id": SAMPLE_PROJECT.id,
    "name": SAMPLE_PROJECT.name,
    "goal": SAMPLE_PROJECT.goal,
    "context": SAMPLE_PROJECT.context,
    "evaluation_criteria": [
        {"name": "rigor", "description": "Logical soundness", "weight": 1.0},
        {"name": "depth", "description": "Structural depth", "weight": 0.8},
    ],
    "exploration_guidance": SAMPLE_PROJECT.exploration_guidance,
    "document_guidance": SAMPLE_PROJECT.document_guidance,
    "seed_modification_enabled": True,
    "seed_modification_require_approval": False,
    "research_domains": [{"name": "math", "keywords": ["math"], "source_types": ["paper"]}],
    "status": "active",
    "created_at": NOW.isoformat(),
    "updated_at": NOW.isoformat(),
}


def make_config() -> Config:
    return Config.from_dict(TEST_CONFIG_DATA, project_data=TEST_PROJECT_DATA)


def make_seed(priority: int = 7, exploration_count: int = 0, status: SeedStatus = SeedStatus.ACTIVE) -> Seed:
    return Seed(
        id=_gen_id(), project_id=SAMPLE_PROJECT.id,
        text=f"Seed hypothesis (priority={priority})",
        type=SeedType.CONJECTURE, priority=priority, tags=["test"],
        confidence=Confidence.MEDIUM, source="test", notes=None,
        status=status, parent_seed_id=None, modification_reason=None,
        exploration_count=exploration_count,
        created_at=NOW, updated_at=NOW,
    )


def make_thread(seed_id: str, exchange_count: int = 0, priority: int = 5) -> Thread:
    return Thread(
        id=_gen_id(), project_id=SAMPLE_PROJECT.id, seed_id=seed_id,
        status=ThreadStatus.ACTIVE, priority=priority,
        exchange_count=exchange_count, last_completed_sequence=exchange_count,
        created_at=NOW, updated_at=NOW, retired_at=None, retire_reason=None,
    )


def make_exchange(thread_id: str, seq: int, role: ExchangeRole) -> Exchange:
    return Exchange(
        id=_gen_id(), thread_id=thread_id, sequence=seq,
        role=role, model="gemini",
        prompt=f"prompt-{seq}", response=f"response-{seq}",
        created_at=NOW,
    )


def make_insight(thread_id: str, text: str = "A test insight", status: InsightStatus = InsightStatus.EXTRACTED) -> Insight:
    return Insight(
        id=_gen_id(), project_id=SAMPLE_PROJECT.id,
        text=text, confidence=Confidence.HIGH, tags=["test"],
        source_thread_id=thread_id, extraction_model="gemini",
        status=status, evaluation_scores={}, dispute_count=0,
        transient_failure_count=0, review_record=None,
        blessed_at=None, incorporated_at=None, incorporated_in_section=None,
        created_at=NOW, updated_at=NOW,
    )


def make_llm_response(text: str = "response") -> LLMResponse:
    return LLMResponse(
        success=True, text=text, backend="gemini", model="google/gemini-flash",
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, estimated_cost_usd=0.001),
        error=None,
    )


def make_consensus_result(verdict: Verdict, confidence: float = 1.0) -> ConsensusResult:
    vr = ValidatedResponse(backend="gemini", verdict=verdict, scores={"rigor": 8.0}, reasoning="Good", raw_text="ok")
    rr = RoundResult(round_num=1, responses=[vr], is_converged=True, verdict=verdict)
    return ConsensusResult(
        verdict=verdict, confidence=confidence, scores={"rigor": 8.0},
        rounds_completed=1, round_history=[rr], valid_vote_count=1,
    )


@pytest.fixture
async def store() -> StateStore:
    s = StateStore(":memory:")
    await s.connect()
    await s.create_project(SAMPLE_PROJECT)
    yield s
    await s.close()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus(store=None)


@pytest.fixture
def mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.send = AsyncMock(return_value=make_llm_response())
    client.get_available_backends = MagicMock(return_value=["gemini", "claude"])
    return client


@pytest.fixture
def mock_consensus() -> AsyncMock:
    engine = AsyncMock()
    engine.run = AsyncMock(return_value=make_consensus_result(Verdict.ACCEPT))
    return engine


@pytest.fixture
def config() -> Config:
    return make_config()


# ═══════════════════════════════════════════════════════════════
# SeedManager Tests
# ═══════════════════════════════════════════════════════════════

class TestSeedManager:

    async def test_select_seed_highest_priority(self, store: StateStore, event_bus: EventBus, config: Config) -> None:
        from explorer.src.seed_manager import SeedManager

        low = make_seed(priority=3)
        high = make_seed(priority=9)
        await store.create_seed(low)
        await store.create_seed(high)

        sm = SeedManager(store, event_bus, config)
        selected = await sm.select_next_seed(SAMPLE_PROJECT.id)
        assert selected is not None
        assert selected.id == high.id

    async def test_select_seed_skips_exhausted(self, store: StateStore, event_bus: EventBus, config: Config) -> None:
        from explorer.src.seed_manager import SeedManager

        exhausted = make_seed(priority=10, exploration_count=3)
        available = make_seed(priority=5, exploration_count=0)
        await store.create_seed(exhausted)
        await store.create_seed(available)

        sm = SeedManager(store, event_bus, config)
        selected = await sm.select_next_seed(SAMPLE_PROJECT.id)
        assert selected is not None
        assert selected.id == available.id

    async def test_seed_modification_auto_approve(self, store: StateStore, event_bus: EventBus, config: Config) -> None:
        from explorer.src.seed_manager import SeedManager

        seed = make_seed(priority=7)
        await store.create_seed(seed)
        # Need a real thread in DB for foreign key constraint
        thread = make_thread(seed.id)
        await store.create_thread(thread)
        sm = SeedManager(store, event_bus, config)

        mod = await sm.propose_modification(
            seed=seed,
            proposed_text="Refined hypothesis",
            reasoning="Better formulation",
            proposing_thread_id=thread.id,
            agreement_ratio=0.9,
        )
        # Auto-approve: project does not require approval and ratio >= 0.66
        assert mod.status == "approved"
        assert mod.child_seed_id is not None
        # Original seed should be EVOLVED
        updated_seed = await store.get_seed(seed.id)
        assert updated_seed.status == SeedStatus.EVOLVED


# ═══════════════════════════════════════════════════════════════
# ThreadManager Tests
# ═══════════════════════════════════════════════════════════════

class TestThreadManager:

    async def test_thread_creation(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.thread_manager import ThreadManager

        seed = make_seed(priority=7)
        await store.create_seed(seed)

        tm = ThreadManager(store, mock_llm, event_bus, config)
        thread = await tm.create_thread(seed, SAMPLE_PROJECT)
        assert thread.seed_id == seed.id
        assert thread.status == ThreadStatus.ACTIVE
        assert thread.project_id == SAMPLE_PROJECT.id

    async def test_exchange_role_rotation(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.thread_manager import ThreadManager

        seed = make_seed()
        await store.create_seed(seed)
        tm = ThreadManager(store, mock_llm, event_bus, config)
        thread = await tm.create_thread(seed, SAMPLE_PROJECT)

        roles = []
        for _ in range(6):
            exchange = await tm.run_exchange(thread, SAMPLE_PROJECT)
            roles.append(exchange.role)
            # Refresh thread state from store
            thread = await store.get_thread(thread.id)

        assert roles == [
            ExchangeRole.EXPLORER,
            ExchangeRole.CRITIC,
            ExchangeRole.SYNTHESIZER,
            ExchangeRole.EXPLORER,
            ExchangeRole.CRITIC,
            ExchangeRole.SYNTHESIZER,
        ]

    async def test_chunk_ready_threshold(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.thread_manager import ThreadManager

        seed = make_seed()
        await store.create_seed(seed)
        tm = ThreadManager(store, mock_llm, event_bus, config)
        thread = await tm.create_thread(seed, SAMPLE_PROJECT)

        # Before min_exchanges_for_chunk (4), not ready
        for i in range(3):
            await tm.run_exchange(thread, SAMPLE_PROJECT)
            thread = await store.get_thread(thread.id)
        assert await tm.is_chunk_ready(thread) is False

        # After 4 exchanges, ready
        await tm.run_exchange(thread, SAMPLE_PROJECT)
        thread = await store.get_thread(thread.id)
        assert await tm.is_chunk_ready(thread) is True

    async def test_thread_retirement_max_exchanges(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.thread_manager import ThreadManager

        seed = make_seed()
        await store.create_seed(seed)
        tm = ThreadManager(store, mock_llm, event_bus, config)
        thread = make_thread(seed.id, exchange_count=12)
        await store.create_thread(thread)

        should, reason = await tm.should_retire(thread)
        assert should is True
        assert reason == "max_exchanges"


# ═══════════════════════════════════════════════════════════════
# InsightExtractor Tests
# ═══════════════════════════════════════════════════════════════

class TestInsightExtractor:

    async def test_insight_extraction_basic(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.insight_extractor import InsightExtractor

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id, exchange_count=4)
        await store.create_thread(thread)
        for i in range(1, 5):
            role = [ExchangeRole.EXPLORER, ExchangeRole.CRITIC, ExchangeRole.SYNTHESIZER][i % 3]
            await store.create_exchange(make_exchange(thread.id, i, role))

        mock_llm.send.return_value = make_llm_response(
            "INSIGHT 1: The structure is triadic.\n"
            "CONFIDENCE: high\n"
            "TAGS: symmetry, structure\n\n"
            "INSIGHT 2: Commutativity fails here.\n"
            "CONFIDENCE: medium\n"
            "TAGS: algebra"
        )

        extractor = InsightExtractor(store, mock_llm, event_bus, config)
        insights = await extractor.extract(thread, SAMPLE_PROJECT)
        assert len(insights) == 2
        assert insights[0].text == "The structure is triadic."
        assert insights[0].confidence == Confidence.HIGH
        assert insights[1].text == "Commutativity fails here."

    async def test_insight_dedup_within_thread(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.insight_extractor import InsightExtractor

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id, exchange_count=4)
        await store.create_thread(thread)
        for i in range(1, 5):
            await store.create_exchange(make_exchange(thread.id, i, ExchangeRole.EXPLORER))

        # Create a pre-existing insight with same text
        existing = make_insight(thread.id, text="The structure is triadic.")
        await store.create_insight(existing)

        mock_llm.send.return_value = make_llm_response(
            "INSIGHT 1: The structure is triadic.\n"
            "CONFIDENCE: high\n"
            "TAGS: symmetry\n\n"
            "INSIGHT 2: New unique insight.\n"
            "CONFIDENCE: medium\n"
            "TAGS: novel"
        )

        extractor = InsightExtractor(store, mock_llm, event_bus, config)
        insights = await extractor.extract(thread, SAMPLE_PROJECT)
        # Only the new one should come through
        assert len(insights) == 1
        assert insights[0].text == "New unique insight."


# ═══════════════════════════════════════════════════════════════
# ReviewPanel Tests
# ═══════════════════════════════════════════════════════════════

class TestReviewPanel:

    async def test_review_attested(
        self, store: StateStore, event_bus: EventBus, mock_consensus: AsyncMock,
    ) -> None:
        from explorer.src.review import ReviewPanel

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id)
        await store.create_thread(thread)
        insight = make_insight(thread.id)
        await store.create_insight(insight)

        mock_consensus.run.return_value = make_consensus_result(Verdict.ACCEPT, confidence=0.9)

        panel = ReviewPanel(mock_consensus, store, event_bus)
        status = await panel.review(insight, SAMPLE_PROJECT)
        assert status == InsightStatus.ATTESTED

    async def test_review_discarded(
        self, store: StateStore, event_bus: EventBus, mock_consensus: AsyncMock,
    ) -> None:
        from explorer.src.review import ReviewPanel

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id)
        await store.create_thread(thread)
        insight = make_insight(thread.id)
        await store.create_insight(insight)

        mock_consensus.run.return_value = make_consensus_result(Verdict.REJECT, confidence=1.0)

        panel = ReviewPanel(mock_consensus, store, event_bus)
        status = await panel.review(insight, SAMPLE_PROJECT)
        assert status == InsightStatus.DISCARDED

    async def test_review_disputed(
        self, store: StateStore, event_bus: EventBus, mock_consensus: AsyncMock,
    ) -> None:
        from explorer.src.review import ReviewPanel

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id)
        await store.create_thread(thread)
        insight = make_insight(thread.id)
        await store.create_insight(insight)

        mock_consensus.run.return_value = make_consensus_result(Verdict.UNCERTAIN, confidence=0.5)

        panel = ReviewPanel(mock_consensus, store, event_bus)
        status = await panel.review(insight, SAMPLE_PROJECT)
        assert status == InsightStatus.DISPUTED

    async def test_review_transient_failure(
        self, store: StateStore, event_bus: EventBus, mock_consensus: AsyncMock,
    ) -> None:
        from explorer.src.review import ReviewPanel

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id)
        await store.create_thread(thread)
        insight = make_insight(thread.id)
        await store.create_insight(insight)

        mock_consensus.run.side_effect = ConsensusError("Failed", rounds_completed=1)

        panel = ReviewPanel(mock_consensus, store, event_bus)
        status = await panel.review(insight, SAMPLE_PROJECT)
        assert status == InsightStatus.TRANSIENT_FAILURE


# ═══════════════════════════════════════════════════════════════
# Event Subscription Tests
# ═══════════════════════════════════════════════════════════════

class TestEventSubscriptions:

    async def test_user_endorse_boosts_thread(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock,
        mock_consensus: AsyncMock, config: Config,
    ) -> None:
        from explorer.src.engine import ExplorerEngine

        seed = make_seed()
        await store.create_seed(seed)
        thread = make_thread(seed.id, priority=5)
        await store.create_thread(thread)
        insight = make_insight(thread.id)
        await store.create_insight(insight)

        engine = ExplorerEngine(store, event_bus, mock_llm, mock_consensus, config)
        await engine.initialize()

        event = Event(
            topic="user.insight.endorsed",
            timestamp=NOW,
            source="test",
            payload={"insight_id": insight.id},
            correlation_id="corr-1",
        )
        await event_bus.publish(event)

        updated = await store.get_thread(thread.id)
        assert updated.priority > 5


# ═══════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════

class TestExplorationLoop:

    async def test_exploration_loop_integration(
        self, store: StateStore, event_bus: EventBus, mock_llm: AsyncMock,
        mock_consensus: AsyncMock, config: Config,
    ) -> None:
        """Full cycle: seed -> thread -> exchanges -> extract -> review."""
        from explorer.src.engine import ExplorerEngine

        seed = make_seed(priority=8)
        await store.create_seed(seed)

        # Mock LLM to return exchange responses and then extraction response
        call_count = 0
        min_exchanges = config.get("explorer.min_exchanges_for_chunk", 4)

        async def mock_send(backend: str, prompt: str, **kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            # First min_exchanges calls are exchange responses
            if call_count <= min_exchanges:
                return make_llm_response(f"Exchange response {call_count}")
            # Next call is extraction
            return make_llm_response(
                "INSIGHT 1: A novel property discovered.\n"
                "CONFIDENCE: high\n"
                "TAGS: novel"
            )

        mock_llm.send.side_effect = mock_send
        mock_consensus.run.return_value = make_consensus_result(Verdict.ACCEPT, confidence=0.9)

        engine = ExplorerEngine(store, event_bus, mock_llm, mock_consensus, config)
        await engine.initialize()

        # Run one iteration of the exploration loop
        await engine.run_one_cycle()

        # Verify: thread was created
        threads = await store.list_threads(SAMPLE_PROJECT.id)
        assert len(threads) >= 1

        # Verify: exchanges were created
        exchanges = await store.get_exchanges(threads[0].id)
        assert len(exchanges) >= min_exchanges

        # Verify: insight was extracted and reviewed
        insights = await store.list_insights(SAMPLE_PROJECT.id)
        assert len(insights) >= 1
        assert insights[0].status == InsightStatus.ATTESTED
