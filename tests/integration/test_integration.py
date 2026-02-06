"""Integration tests for cross-module communication.

These tests assemble REAL module instances (StateStore, EventBus, engines)
with mocked LLM HTTP calls, and verify the full event-driven pipeline works.
Each test creates its own in-memory store and event bus.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import Config
from shared.events import EventBus
from shared.models import (
    BackendState,
    BackendStatus,
    Confidence,
    ConsensusResult,
    Event,
    EvaluationCriterion,
    Finding,
    Insight,
    InsightStatus,
    LLMResponse,
    Project,
    ProjectStatus,
    ResearchDomain,
    RoundResult,
    SearchResult,
    Section,
    SectionStatus,
    Seed,
    SeedStatus,
    SeedType,
    Thread,
    ThreadStatus,
    TokenUsage,
    ValidatedResponse,
    Verdict,
    generate_id,
)
from shared.store import StateStore

from explorer.src.engine import ExplorerEngine
from documenter.src.engine import DocumenterEngine
from researcher.src.engine import ResearcherEngine
from orchestrator.src.main import Orchestrator
from orchestrator.src.quota import QuotaAllocator


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

os.environ.setdefault("FANO_TEST_KEY", "test-key-value")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> Config:
    return Config.from_dict({
        "llm": {
            "api_key_env": "FANO_TEST_KEY",
            "models": {
                "claude": {"model": "claude-3"},
                "gemini": {"model": "gemini-2"},
            },
        },
        "consensus": {
            "backends": ["claude", "gemini"],
            "max_rounds": 2,
            "min_valid_responses": 1,
        },
        "explorer": {
            "max_active_threads": 2,
            "min_exchanges_for_chunk": 2,
            "max_exchanges_per_thread": 6,
            "thread_retirement": {
                "max_reexplore_count": 3,
                "max_idle_hours": 48,
            },
        },
        "documenter": {
            "document_dir": "data/document",
            "context": {"max_tokens": 4000},
        },
        "researcher": {
            "max_questions_per_insight": 2,
            "max_searches_per_question": 1,
            "max_findings_per_source": 3,
            "trust": {"min_trust_score": 50},
            "idle_polling_interval_seconds": 1,
        },
        "quotas": {
            "daily_budget_usd": 10.0,
            "per_module_weights": {
                "explorer": 50,
                "documenter": 35,
                "researcher": 15,
            },
            "alert_at_percent": 80,
        },
    })


NOW = datetime.now(timezone.utc)


def _make_project() -> Project:
    return Project(
        id="proj-integ-001",
        name="Integration Test Project",
        goal="Explore mathematical structures in test context.",
        context="This is an integration-test project.",
        evaluation_criteria=[
            EvaluationCriterion(
                name="rigor", description="Logical soundness", weight=1.0,
            ),
            EvaluationCriterion(
                name="depth", description="Structural depth", weight=0.8,
            ),
        ],
        exploration_guidance="Follow curiosity.",
        document_guidance="Write clearly.",
        seed_modification_enabled=True,
        seed_modification_require_approval=False,
        research_domains=[
            ResearchDomain(
                name="test_domain",
                keywords=["test", "math"],
                source_types=["academic_paper"],
            ),
        ],
        status=ProjectStatus.ACTIVE,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_seed(project: Project) -> Seed:
    return Seed(
        id=generate_id(),
        project_id=project.id,
        text="Every finite group has a natural triadic structure.",
        type=SeedType.CONJECTURE,
        priority=7,
        tags=["group-theory", "triadic"],
        confidence=Confidence.MEDIUM,
        source="integration test",
        notes=None,
        status=SeedStatus.ACTIVE,
        parent_seed_id=None,
        modification_reason=None,
        exploration_count=0,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_attested_insight(project: Project, thread_id: str | None = None) -> Insight:
    return Insight(
        id=generate_id(),
        project_id=project.id,
        text="The triadic decomposition reveals hidden algebraic symmetry.",
        confidence=Confidence.HIGH,
        tags=["symmetry", "algebra"],
        source_thread_id=thread_id,
        extraction_model="gemini",
        status=InsightStatus.ATTESTED,
        evaluation_scores={"rigor": 0.9, "depth": 0.85},
        dispute_count=0,
        transient_failure_count=0,
        review_record={"rounds": 1, "confidence": 0.9, "verdict": "accept"},
        blessed_at=NOW,
        incorporated_at=None,
        incorporated_in_section=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_llm_client() -> MagicMock:
    """Create a mock LLM client that returns plausible responses."""
    client = MagicMock()

    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.002,
    )

    # Default send returns an exploration response.
    # IMPORTANT: Conditions are ordered from most specific to least specific
    # to avoid early matches on generic terms like "extract".
    async def _send(backend: str, prompt: str, **kwargs) -> LLMResponse:
        lower = prompt.lower()

        # Finding extraction (researcher) -- must check BEFORE generic "extract"
        # The FINDING_EXTRACTION prompt contains "extract specific findings"
        if "extract specific findings" in lower or "finding" in lower:
            return LLMResponse(
                success=True,
                text=json.dumps([
                    {
                        "summary": "Triadic decomposition is provably valid for all finite groups of order < 100.",
                        "type": "supports",
                        "confidence": 0.85,
                    },
                ]),
                backend=backend,
                model=f"{backend}-model",
                token_usage=usage,
                error=None,
            )
        # Insight extraction (explorer) -- "extract the atomic insights"
        if "atomic" in lower or ("extract" in lower and "insight" in lower):
            return LLMResponse(
                success=True,
                text=(
                    "INSIGHT 1: The triadic decomposition reveals hidden algebraic symmetry.\n"
                    "CONFIDENCE: high\n"
                    "TAGS: symmetry, algebra\n\n"
                    "INSIGHT 2: Finite group actions preserve triadic structure under homomorphism.\n"
                    "CONFIDENCE: medium\n"
                    "TAGS: group-theory, homomorphism"
                ),
                backend=backend,
                model=f"{backend}-model",
                token_usage=usage,
                error=None,
            )
        # Search simulation (researcher) -- "simulating a web search"
        if "simulating a web search" in lower or "search result" in lower:
            return LLMResponse(
                success=True,
                text=json.dumps([
                    {
                        "url": "https://arxiv.org/abs/2301.12345",
                        "title": "Triadic Structures in Finite Groups",
                        "snippet": "We prove that every finite group admits a triadic decomposition...",
                        "domain": "arxiv.org",
                    },
                ]),
                backend=backend,
                model=f"{backend}-model",
                token_usage=usage,
                error=None,
            )
        # Question generation (researcher) -- "generate questions" or "generate research questions"
        if "generate" in lower and "question" in lower:
            return LLMResponse(
                success=True,
                text=json.dumps([
                    "What evidence supports triadic decomposition in finite groups?",
                    "Are there known counterexamples to triadic symmetry claims?",
                ]),
                backend=backend,
                model=f"{backend}-model",
                token_usage=usage,
                error=None,
            )
        # Dedup check for documenter
        if "duplicate" in lower or "dedup" in lower:
            return LLMResponse(
                success=True,
                text=json.dumps({
                    "is_duplicate": False,
                    "prerequisites": [],
                    "concepts": ["triadic_symmetry"],
                }),
                backend=backend,
                model=f"{backend}-model",
                token_usage=usage,
                error=None,
            )
        # Draft section for documenter
        if "draft" in lower or "section" in lower:
            return LLMResponse(
                success=True,
                text="# Triadic Symmetry\n\nThe triadic decomposition reveals hidden algebraic symmetry...",
                backend=backend,
                model=f"{backend}-model",
                token_usage=usage,
                error=None,
            )
        # Default generic response (exploration exchanges, etc.)
        return LLMResponse(
            success=True,
            text="The mathematical structure exhibits interesting properties that merit further investigation.",
            backend=backend,
            model=f"{backend}-model",
            token_usage=usage,
            error=None,
        )

    client.send = AsyncMock(side_effect=_send)

    async def _send_structured(backend: str, prompt: str, schema: dict) -> LLMResponse:
        return LLMResponse(
            success=True,
            text=json.dumps({
                "is_duplicate": False,
                "prerequisites": [],
                "concepts": ["triadic_symmetry"],
            }),
            backend=backend,
            model=f"{backend}-model",
            token_usage=usage,
            error=None,
        )

    client.send_structured = AsyncMock(side_effect=_send_structured)
    client.get_available_backends = MagicMock(return_value=["claude", "gemini"])
    client.get_backend_status = MagicMock(return_value=BackendStatus(
        name="claude",
        state=BackendState.OPEN,
        requests_per_minute=10,
        avg_latency_ms=500.0,
        failure_count=0,
        last_failure_at=None,
    ))
    return client


def _make_consensus_accept() -> MagicMock:
    """Create a mock ConsensusEngine that always accepts."""
    consensus = MagicMock()

    result = ConsensusResult(
        verdict=Verdict.ACCEPT,
        confidence=0.9,
        scores={"rigor": 0.9, "depth": 0.85},
        rounds_completed=1,
        round_history=[
            RoundResult(
                round_num=1,
                responses=[
                    ValidatedResponse(
                        backend="claude",
                        verdict=Verdict.ACCEPT,
                        scores={"rigor": 0.9, "depth": 0.85},
                        reasoning="Strong logical foundation.",
                        raw_text='{"verdict":"accept","scores":{"rigor":9,"depth":8.5},"reasoning":"Strong."}',
                    ),
                ],
                is_converged=True,
                verdict=Verdict.ACCEPT,
            ),
        ],
        valid_vote_count=2,
    )
    consensus.run = AsyncMock(return_value=result)
    return consensus


def _make_consensus_with_trust(trust_score: float = 0.75) -> MagicMock:
    """Consensus mock returning specified confidence (used for trust eval)."""
    consensus = MagicMock()
    result = ConsensusResult(
        verdict=Verdict.ACCEPT,
        confidence=trust_score,
        scores={"trust": trust_score},
        rounds_completed=1,
        round_history=[],
        valid_vote_count=2,
    )
    consensus.run = AsyncMock(return_value=result)
    return consensus


async def _make_store_and_bus() -> tuple[StateStore, EventBus]:
    """Create fresh in-memory store and event bus."""
    store = StateStore(":memory:")
    await store.connect()
    bus = EventBus(store)
    return store, bus


def _config_with_project(config: Config, project: Project) -> Config:
    """Return a config whose .project property returns the given project."""
    config._project = project
    return config


# ---------------------------------------------------------------------------
# Test 1: Seed -> Explore -> Insight -> Review -> ATTESTED
# ---------------------------------------------------------------------------

class TestSeedToInsight:
    """Store + EventBus + Explorer: full exploration pipeline."""

    @pytest.mark.asyncio
    async def test_seed_to_insight(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            seed = _make_seed(project)
            config = _config_with_project(_make_config(), project)

            await store.create_project(project)
            await store.create_seed(seed)

            llm_client = _make_llm_client()
            consensus = _make_consensus_accept()

            explorer = ExplorerEngine(store, bus, llm_client, consensus, config)
            await explorer.initialize()

            # Run one exploration cycle
            await explorer.run_one_cycle()

            # Verify thread was created
            threads = await store.list_threads(project.id)
            assert len(threads) >= 1, "Explorer should have created at least one thread"

            # Verify exchanges were created (min_exchanges_for_chunk = 2)
            thread = threads[0]
            exchanges = await store.get_exchanges(thread.id)
            assert len(exchanges) >= 2, (
                f"Expected >= 2 exchanges, got {len(exchanges)}"
            )

            # Verify insights were extracted
            insights = await store.list_insights(project.id)
            assert len(insights) >= 1, "Explorer should have extracted at least one insight"

            # Verify at least one insight was reviewed (attested or other post-review status)
            reviewed_statuses = {
                InsightStatus.ATTESTED,
                InsightStatus.INTERESTING,
                InsightStatus.DISCARDED,
                InsightStatus.DISPUTED,
            }
            reviewed = [i for i in insights if i.status in reviewed_statuses]
            assert len(reviewed) >= 1, (
                f"Expected reviewed insights, statuses: {[i.status for i in insights]}"
            )

            # With our consensus mock (always ACCEPT, confidence=0.9), expect ATTESTED
            attested = [i for i in insights if i.status == InsightStatus.ATTESTED]
            assert len(attested) >= 1, (
                f"Expected ATTESTED insights, statuses: {[i.status for i in insights]}"
            )

            # Verify the explorer.insight.attested event was published
            events = await store.list_events(topic="explorer.insight.attested")
            assert len(events) >= 1, "Expected explorer.insight.attested event"
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test 2: Attested Insight -> Documenter picks up -> creates section
# ---------------------------------------------------------------------------

class TestInsightToDocument:
    """Store + EventBus + Documenter: insight incorporation."""

    @pytest.mark.asyncio
    async def test_insight_to_document(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            config = _config_with_project(_make_config(), project)
            insight = _make_attested_insight(project)

            await store.create_project(project)
            await store.create_insight(insight)

            llm_client = _make_llm_client()
            consensus = _make_consensus_accept()

            documenter = DocumenterEngine(store, bus, llm_client, consensus, config)
            await documenter.initialize()

            # Publish the attested event -- the documenter should pick it up
            attested_event = Event(
                topic="explorer.insight.attested",
                timestamp=NOW,
                source="explorer.review_panel",
                payload={"insight_id": insight.id, "verdict": "accept"},
                correlation_id=generate_id(),
            )
            await bus.publish(attested_event)

            # The documenter queues the insight internally; verify it was received
            assert documenter._insight_queue.qsize() == 1, (
                "Documenter should have queued the insight"
            )

            # Verify the event was persisted
            stored_events = await store.list_events(
                topic="explorer.insight.attested",
            )
            assert len(stored_events) >= 1, (
                "Attested event should be persisted in store"
            )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test 3: Attested Insight -> Researcher generates questions -> findings
# ---------------------------------------------------------------------------

class TestInsightToResearch:
    """Store + EventBus + Researcher: insight research pipeline."""

    @pytest.mark.asyncio
    async def test_insight_to_research(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            config = _config_with_project(_make_config(), project)
            insight = _make_attested_insight(project)

            await store.create_project(project)
            await store.create_insight(insight)

            llm_client = _make_llm_client()
            # Trust evaluator needs high confidence so sources pass trust check
            consensus = _make_consensus_with_trust(0.75)

            researcher = ResearcherEngine(store, bus, llm_client, consensus, config)
            await researcher.initialize()

            # Publish attested event -- researcher should pick up the insight
            attested_event = Event(
                topic="explorer.insight.attested",
                timestamp=NOW,
                source="explorer.review_panel",
                payload={"insight_id": insight.id, "verdict": "accept"},
                correlation_id=generate_id(),
            )
            await bus.publish(attested_event)

            # Verify the insight was queued
            assert len(researcher._insight_queue) == 1, (
                "Researcher should have queued the insight"
            )

            # Process the queued work item
            work = researcher._get_next_work_item()
            assert work is not None, "Should have a work item"
            assert work["type"] == "insight", f"Expected insight work, got {work['type']}"

            await researcher._process_work_item(work)

            # Verify LLM was called for question generation
            assert llm_client.send.call_count >= 1, (
                "LLM should have been called for question generation"
            )

            # Verify events were published (researcher.finding.stored or
            # researcher.evidence.supports/contradicts)
            all_events = await store.list_events()
            researcher_events = [
                e for e in all_events
                if e.topic.startswith("researcher.")
            ]
            assert len(researcher_events) >= 1, (
                f"Expected researcher events, got topics: "
                f"{[e.topic for e in all_events]}"
            )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test 4: Directed research request -> Researcher processes
# ---------------------------------------------------------------------------

class TestDirectedResearch:
    """Store + EventBus + Researcher: directed research request."""

    @pytest.mark.asyncio
    async def test_directed_research(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            config = _config_with_project(_make_config(), project)

            await store.create_project(project)

            llm_client = _make_llm_client()
            consensus = _make_consensus_with_trust(0.75)

            researcher = ResearcherEngine(store, bus, llm_client, consensus, config)
            await researcher.initialize()

            # Publish user.research.requested event
            request_event = Event(
                topic="user.research.requested",
                timestamp=NOW,
                source="control.api",
                payload={
                    "topic": "triadic decomposition in algebra",
                    "context": "Looking for foundational references",
                    "project_id": project.id,
                },
                correlation_id=generate_id(),
            )
            await bus.publish(request_event)

            # Verify the directed request was queued
            assert len(researcher._directed_queue) == 1, (
                "Researcher should have queued the directed request"
            )

            # Process it
            work = researcher._get_next_work_item()
            assert work is not None, "Should have a work item"
            assert work["type"] == "directed", f"Expected directed work, got {work['type']}"

            await researcher._process_work_item(work)

            # Verify LLM was called
            assert llm_client.send.call_count >= 1, (
                "LLM should have been called for directed question generation"
            )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test 5: User endorses insight -> Explorer receives event
# ---------------------------------------------------------------------------

class TestUserEndorsesInsight:
    """Store + EventBus + Explorer: user endorsement event flow."""

    @pytest.mark.asyncio
    async def test_user_endorses_insight(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            config = _config_with_project(_make_config(), project)

            await store.create_project(project)

            # Create a thread and an insight linked to it
            seed = _make_seed(project)
            await store.create_seed(seed)
            thread = Thread(
                id=generate_id(),
                project_id=project.id,
                seed_id=seed.id,
                status=ThreadStatus.ACTIVE,
                priority=5,
                exchange_count=0,
                last_completed_sequence=0,
                created_at=NOW,
                updated_at=NOW,
                retired_at=None,
                retire_reason=None,
            )
            await store.create_thread(thread)

            insight = Insight(
                id=generate_id(),
                project_id=project.id,
                text="Triadic symmetry is fundamental.",
                confidence=Confidence.HIGH,
                tags=["symmetry"],
                source_thread_id=thread.id,
                extraction_model="gemini",
                status=InsightStatus.ATTESTED,
                evaluation_scores={},
                dispute_count=0,
                transient_failure_count=0,
                review_record=None,
                blessed_at=NOW,
                incorporated_at=None,
                incorporated_in_section=None,
                created_at=NOW,
                updated_at=NOW,
            )
            await store.create_insight(insight)

            llm_client = _make_llm_client()
            consensus = _make_consensus_accept()

            explorer = ExplorerEngine(store, bus, llm_client, consensus, config)
            await explorer.initialize()

            original_thread = await store.get_thread(thread.id)
            original_priority = original_thread.priority

            # Publish user.insight.endorsed event
            endorse_event = Event(
                topic="user.insight.endorsed",
                timestamp=NOW,
                source="control.api",
                payload={"insight_id": insight.id},
                correlation_id=generate_id(),
            )
            await bus.publish(endorse_event)

            # Verify the thread priority was adjusted upward (+2)
            updated_thread = await store.get_thread(thread.id)
            assert updated_thread.priority == min(10, original_priority + 2), (
                f"Thread priority should be {min(10, original_priority + 2)}, "
                f"got {updated_thread.priority}"
            )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test 6: Budget exhaustion tracking
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    """Store + EventBus + QuotaAllocator: budget tracking and warning."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            config = _make_config()

            quota = QuotaAllocator(config=config, event_bus=bus)
            await quota.initialize()

            # Track warning events
            warnings_received: list[Event] = []

            async def _capture_warning(event: Event) -> None:
                if event.topic == "system.budget.warning":
                    warnings_received.append(event)

            bus.subscribe("system.budget.warning", _capture_warning)

            # The explorer budget is 50% of $10 = $5.00
            # Alert threshold is 80% = $4.00
            # Send LLM events below threshold first
            for i in range(3):
                event = Event(
                    topic="llm.request.completed",
                    timestamp=NOW,
                    source="explorer",
                    payload={
                        "module": "explorer",
                        "cost_usd": 1.0,
                        "backend": "claude",
                    },
                    correlation_id=generate_id(),
                )
                await bus.publish(event)

            # At $3.00 / $5.00 = 60%, no warning yet
            assert len(warnings_received) == 0, (
                f"Should not warn at 60%, got {len(warnings_received)} warnings"
            )
            assert quota.is_within_budget("explorer"), "Explorer should still be within budget"

            # Push past 80% threshold
            event = Event(
                topic="llm.request.completed",
                timestamp=NOW,
                source="explorer",
                payload={
                    "module": "explorer",
                    "cost_usd": 1.5,
                    "backend": "claude",
                },
                correlation_id=generate_id(),
            )
            await bus.publish(event)

            # At $4.50 / $5.00 = 90%, warning should fire
            assert len(warnings_received) == 1, (
                f"Expected 1 warning at 90%, got {len(warnings_received)}"
            )
            assert warnings_received[0].payload["module"] == "explorer"
            assert warnings_received[0].payload["percent_used"] == 90.0

            # Further spend should not trigger another warning (already warned)
            event2 = Event(
                topic="llm.request.completed",
                timestamp=NOW,
                source="explorer",
                payload={
                    "module": "explorer",
                    "cost_usd": 0.3,
                    "backend": "claude",
                },
                correlation_id=generate_id(),
            )
            await bus.publish(event2)

            assert len(warnings_received) == 1, (
                "Should not re-warn for same module"
            )

            # Verify remaining budget
            remaining = quota.get_remaining("explorer")
            assert remaining < 0.3, f"Remaining should be < 0.3, got {remaining}"
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test 7: Full pipeline: Seed -> Explore -> Attest -> Research -> Document
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """All modules assembled: end-to-end with mocked LLM."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            config = _config_with_project(_make_config(), project)
            seed = _make_seed(project)

            await store.create_project(project)
            await store.create_seed(seed)

            llm_client = _make_llm_client()
            consensus = _make_consensus_accept()

            # --- Create all engines ---
            explorer = ExplorerEngine(store, bus, llm_client, consensus, config)
            documenter = DocumenterEngine(store, bus, llm_client, consensus, config)
            researcher = ResearcherEngine(store, bus, llm_client, consensus, config)

            # --- Initialize all (subscribes to events) ---
            await explorer.initialize()
            await documenter.initialize()
            await researcher.initialize()

            # Phase 1: Explorer runs one cycle -- seed -> thread -> insights -> review
            await explorer.run_one_cycle()

            # Verify insights were created and attested
            insights = await store.list_insights(project.id)
            assert len(insights) >= 1, "Explorer should have produced insights"

            attested = [
                i for i in insights if i.status == InsightStatus.ATTESTED
            ]
            assert len(attested) >= 1, (
                f"Expected ATTESTED insights, statuses: {[i.status for i in insights]}"
            )

            # Phase 2: Verify documenter received the insight.attested event
            # The EventBus is synchronous delivery, so the documenter handler
            # already ran during the explorer's publish call.
            assert documenter._insight_queue.qsize() >= 1, (
                "Documenter should have received attested insight via event"
            )

            # Phase 3: Verify researcher received the insight.attested event
            assert len(researcher._insight_queue) >= 1, (
                "Researcher should have received attested insight via event"
            )

            # Phase 4: Process researcher work
            work = researcher._get_next_work_item()
            assert work is not None
            await researcher._process_work_item(work)

            # Verify researcher events were published
            all_events = await store.list_events()
            researcher_events = [
                e for e in all_events if e.topic.startswith("researcher.")
            ]
            assert len(researcher_events) >= 1, (
                "Researcher should have published finding/evidence events"
            )

            # Verify LLM was called multiple times across modules
            assert llm_client.send.call_count >= 4, (
                f"Expected >= 4 LLM calls across modules, got {llm_client.send.call_count}"
            )

            # Verify event pipeline integrity: events span multiple modules
            event_topics = [e.topic for e in all_events]
            explorer_topics = [t for t in event_topics if t.startswith("explorer.")]
            researcher_topics = [t for t in event_topics if t.startswith("researcher.")]
            assert len(explorer_topics) >= 1, "Expected explorer events"
            assert len(researcher_topics) >= 1, "Expected researcher events"
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test: Orchestrator module lifecycle
# ---------------------------------------------------------------------------

class TestOrchestratorLifecycle:
    """Orchestrator registers, initializes, and queries module health."""

    @pytest.mark.asyncio
    async def test_orchestrator_registers_modules(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            project = _make_project()
            config = _config_with_project(_make_config(), project)

            await store.create_project(project)

            llm_client = _make_llm_client()
            consensus = _make_consensus_accept()

            explorer = ExplorerEngine(store, bus, llm_client, consensus, config)
            documenter = DocumenterEngine(store, bus, llm_client, consensus, config)
            researcher = ResearcherEngine(store, bus, llm_client, consensus, config)

            orchestrator = Orchestrator(store, bus, config)
            orchestrator.register_module(explorer)
            orchestrator.register_module(documenter)
            orchestrator.register_module(researcher)

            # Verify all modules registered
            assert len(orchestrator._modules) == 3
            assert orchestrator._module_names == {"explorer", "documenter", "researcher"}

            # Verify duplicate registration is rejected
            with pytest.raises(ValueError, match="already registered"):
                orchestrator.register_module(explorer)

            # Initialize modules through orchestrator-like pattern
            for module in orchestrator._modules:
                result = await module.initialize()
                assert result is True, f"{module.module_name} initialization failed"

            # Verify health checks
            for module in orchestrator._modules:
                health = await module.health_check()
                assert health.module == module.module_name
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test: Event routing with wildcards
# ---------------------------------------------------------------------------

class TestEventRouting:
    """Verify EventBus wildcard routing delivers to correct subscribers."""

    @pytest.mark.asyncio
    async def test_wildcard_routing(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            received: list[str] = []

            async def on_explorer(event: Event) -> None:
                received.append(f"explorer:{event.topic}")

            async def on_all_insights(event: Event) -> None:
                received.append(f"all_insights:{event.topic}")

            async def on_everything(event: Event) -> None:
                received.append(f"everything:{event.topic}")

            bus.subscribe("explorer.**", on_explorer)
            bus.subscribe("explorer.insight.*", on_all_insights)
            bus.subscribe("**", on_everything)

            event = Event(
                topic="explorer.insight.attested",
                timestamp=NOW,
                source="test",
                payload={"insight_id": "test-123"},
                correlation_id="corr-1",
            )
            await bus.publish(event)

            # explorer.** should match explorer.insight.attested
            assert "explorer:explorer.insight.attested" in received
            # explorer.insight.* should match explorer.insight.attested
            assert "all_insights:explorer.insight.attested" in received
            # ** should match everything
            assert "everything:explorer.insight.attested" in received
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Test: Event persistence and replay
# ---------------------------------------------------------------------------

class TestEventPersistence:
    """Verify events are persisted and can be replayed."""

    @pytest.mark.asyncio
    async def test_event_persistence_and_replay(self) -> None:
        store, bus = await _make_store_and_bus()
        try:
            # Publish several events
            topics = [
                "explorer.insight.attested",
                "researcher.finding.stored",
                "system.budget.warning",
            ]
            for topic in topics:
                event = Event(
                    topic=topic,
                    timestamp=NOW,
                    source="test",
                    payload={"test": True},
                    correlation_id=generate_id(),
                )
                await bus.publish(event)

            # Verify persistence
            stored = await store.list_events()
            assert len(stored) == 3, f"Expected 3 events, got {len(stored)}"

            # Set up a new handler and replay
            replayed: list[str] = []

            async def on_replay(event: Event) -> None:
                replayed.append(event.topic)

            bus.subscribe("**", on_replay)
            await bus.replay(since=datetime.min.replace(tzinfo=timezone.utc))

            # All 3 events should be replayed
            assert len(replayed) == 3, f"Expected 3 replayed events, got {len(replayed)}"
            for topic in topics:
                assert topic in replayed, f"Missing replayed topic: {topic}"
        finally:
            await store.close()
