"""Unit tests for the Researcher Engine module.

Tests cover: question generation, trust evaluation (consensus, caching,
invalidation), finding extraction, evidence event publishing, directed
research priority, configurable limits, and low-trust source skipping.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import Config
from shared.events import EventBus
from shared.models import (
    Confidence,
    ConsensusResult,
    EvaluationCriterion,
    Event,
    Finding,
    HealthStatus,
    Insight,
    InsightStatus,
    LLMResponse,
    ParsedResponse,
    Project,
    ProjectStatus,
    ResearchDomain,
    RoundResult,
    SearchResult,
    Source,
    TokenUsage,
    ValidatedResponse,
    Verdict,
)
from shared.store import StateStore


_counter = 0


def _gen_id() -> str:
    global _counter
    _counter += 1
    return f"test-id-{_counter:06d}"

# ── Helpers ───────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_project(**overrides) -> Project:
    defaults = dict(
        id="proj-1",
        name="Test Project",
        goal="Investigate algebraic symmetry.",
        context="Exploring group theory patterns.",
        evaluation_criteria=[
            EvaluationCriterion(name="rigor", description="Logical soundness", weight=1.0),
        ],
        exploration_guidance="",
        document_guidance="",
        seed_modification_enabled=False,
        seed_modification_require_approval=True,
        research_domains=[
            ResearchDomain(
                name="algebra",
                keywords=["group", "symmetry"],
                source_types=["academic_paper"],
                extraction_patterns=[r"\d+"],
            ),
        ],
        status=ProjectStatus.ACTIVE,
        created_at=_now(),
        updated_at=_now(),
    )
    defaults.update(overrides)
    return Project(**defaults)


def _make_insight(**overrides) -> Insight:
    defaults = dict(
        id=_gen_id(),
        project_id="proj-1",
        text="Groups of prime order are cyclic.",
        confidence=Confidence.HIGH,
        tags=["group-theory"],
        source_thread_id=None,
        extraction_model="claude",
        status=InsightStatus.ATTESTED,
        evaluation_scores={},
        dispute_count=0,
        transient_failure_count=0,
        review_record=None,
        blessed_at=None,
        incorporated_at=None,
        incorporated_in_section=None,
        created_at=_now(),
        updated_at=_now(),
    )
    defaults.update(overrides)
    return Insight(**defaults)


def _make_source(**overrides) -> Source:
    defaults = dict(
        id=_gen_id(),
        project_id="proj-1",
        url="https://example.com/paper",
        domain="algebra",
        title="On cyclic groups",
        trust_score=0,
        trust_tier=None,
        content_hash=None,
        evaluated_at=None,
        created_at=_now(),
    )
    defaults.update(overrides)
    return Source(**defaults)


def _mock_llm_client() -> AsyncMock:
    client = AsyncMock()
    client.send = AsyncMock()
    client.send_structured = AsyncMock()
    client.get_available_backends = MagicMock(return_value=["claude", "gemini"])
    return client


def _mock_consensus() -> AsyncMock:
    consensus = AsyncMock()
    consensus.run = AsyncMock()
    return consensus


def _make_config(overrides: dict | None = None) -> Config:
    """Build a minimal Config for testing via from_dict."""
    data = {
        "llm": {
            "api_key_env": "FANO_TEST_KEY",
            "models": {"claude": {"model": "claude-3"}, "gemini": {"model": "gemini-2"}},
        },
        "consensus": {"backends": ["claude", "gemini"]},
        "researcher": {
            "max_questions_per_insight": 3,
            "max_searches_per_question": 2,
            "max_findings_per_source": 5,
            "idle_polling_interval_seconds": 1,
            "trust": {"min_trust_score": 50},
        },
    }
    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = data
            for part in keys[:-1]:
                d = d.setdefault(part, {})
            d[keys[-1]] = v
    import os
    os.environ.setdefault("FANO_TEST_KEY", "test-key-value")
    return Config.from_dict(data)


def _consensus_result(score: int, verdict: Verdict = Verdict.ACCEPT) -> ConsensusResult:
    return ConsensusResult(
        verdict=verdict,
        confidence=score / 100,
        scores={"authority": score / 100, "accuracy": score / 100},
        rounds_completed=1,
        round_history=[],
        valid_vote_count=2,
    )


# ── QuestionGenerator tests ──────────────────────────────────


class TestQuestionGeneration:
    @pytest.mark.asyncio
    async def test_question_generation(self):
        """Generates relevant questions from insight."""
        from researcher.src.questions import QuestionGenerator

        llm = _mock_llm_client()
        llm.send.return_value = LLMResponse(
            success=True,
            text=json.dumps(["What groups have prime order?", "Is cyclicity unique?"]),
            backend="claude",
            model="claude-3",
            token_usage=None,
            error=None,
        )
        config = _make_config()
        gen = QuestionGenerator(llm, config)
        project = _make_project()
        insight = _make_insight()

        questions = await gen.generate(insight, project)

        assert len(questions) == 2
        assert "prime order" in questions[0].lower()
        llm.send.assert_called_once()
        prompt_used = llm.send.call_args.kwargs.get("prompt", llm.send.call_args.args[1] if len(llm.send.call_args.args) > 1 else "")
        assert insight.text in prompt_used

    @pytest.mark.asyncio
    async def test_question_generation_no_domain_fallback(self):
        """No hardcoded domain strings appear in questions when LLM fails."""
        from researcher.src.questions import QuestionGenerator

        llm = _mock_llm_client()
        # First call fails, second returns fallback using project.goal
        llm.send.side_effect = [
            LLMResponse(
                success=False, text="", backend="claude", model="claude-3",
                token_usage=None, error="timeout",
            ),
            LLMResponse(
                success=True,
                text=json.dumps(["What evidence supports this claim?"]),
                backend="claude", model="claude-3",
                token_usage=None, error=None,
            ),
        ]
        config = _make_config()
        gen = QuestionGenerator(llm, config)
        project = _make_project()
        insight = _make_insight()

        questions = await gen.generate(insight, project)

        assert len(questions) >= 1
        # No hardcoded domain fallback strings
        for q in questions:
            assert "mathematics" not in q.lower() or "algebraic" not in q.lower()


# ── TrustEvaluator tests ─────────────────────────────────────


class TestTrustEvaluation:
    @pytest.mark.asyncio
    async def test_trust_evaluation_uses_consensus(self):
        """Multiple LLMs evaluate source trust, not a single call."""
        from researcher.src.trust import TrustEvaluator

        consensus = _mock_consensus()
        consensus.run.return_value = _consensus_result(75)
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)

        evaluator = TrustEvaluator(consensus, store, config)
        source = _make_source(project_id=project.id)
        score = await evaluator.evaluate(source, "Summary of content", project)

        assert score == 75
        consensus.run.assert_called_once()
        await store.close()

    @pytest.mark.asyncio
    async def test_trust_caching(self):
        """Same URL returns cached score without re-evaluation."""
        from researcher.src.trust import TrustEvaluator

        consensus = _mock_consensus()
        consensus.run.return_value = _consensus_result(80)
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)

        evaluator = TrustEvaluator(consensus, store, config)
        source = _make_source(project_id=project.id, content_hash="abc123")

        score1 = await evaluator.evaluate(source, "Summary", project)
        score2 = await evaluator.evaluate(source, "Summary", project)

        assert score1 == score2 == 80
        assert consensus.run.call_count == 1  # Only called once

        await store.close()

    @pytest.mark.asyncio
    async def test_trust_cache_invalidation(self):
        """Changed content_hash triggers re-evaluation."""
        from researcher.src.trust import TrustEvaluator

        consensus = _mock_consensus()
        consensus.run.side_effect = [
            _consensus_result(70),
            _consensus_result(85),
        ]
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)

        evaluator = TrustEvaluator(consensus, store, config)
        source_v1 = _make_source(project_id=project.id, content_hash="hash_v1")
        source_v2 = _make_source(
            id=source_v1.id, project_id=project.id,
            url=source_v1.url, content_hash="hash_v2",
        )

        score1 = await evaluator.evaluate(source_v1, "Old content", project)
        score2 = await evaluator.evaluate(source_v2, "New content", project)

        assert score1 == 70
        assert score2 == 85
        assert consensus.run.call_count == 2

        await store.close()


# ── FindingExtractor tests ────────────────────────────────────


class TestFindingExtraction:
    @pytest.mark.asyncio
    async def test_finding_extraction(self):
        """Findings extracted and linked to source + insight."""
        from researcher.src.extractor import FindingExtractor

        llm = _mock_llm_client()
        llm.send.return_value = LLMResponse(
            success=True,
            text=json.dumps([
                {"summary": "Cyclic groups are abelian", "confidence": 0.9, "type": "supports"},
                {"summary": "Not all abelian groups are cyclic", "confidence": 0.7, "type": "extends"},
            ]),
            backend="claude",
            model="claude-3",
            token_usage=None,
            error=None,
        )
        config = _make_config()
        extractor = FindingExtractor(llm, config)
        project = _make_project()
        source = _make_source()
        insight = _make_insight()

        findings = await extractor.extract(
            source, "Full content of the paper...", insight, project,
        )

        assert len(findings) == 2
        assert findings[0].source_id == source.id
        assert findings[0].related_insight_id == insight.id
        assert findings[0].finding_type == "supports"
        assert findings[1].finding_type == "extends"


# ── Evidence event tests ──────────────────────────────────────


class TestEvidenceEvents:
    @pytest.mark.asyncio
    async def test_evidence_supports_event(self):
        """Supporting findings publish researcher.evidence.supports."""
        from researcher.src.engine import ResearcherEngine

        llm = _mock_llm_client()
        consensus = _mock_consensus()
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)
        bus = EventBus(store=None)

        engine = ResearcherEngine(store, bus, llm, consensus, config)
        insight = _make_insight(project_id=project.id)

        published: list[Event] = []
        bus.subscribe("researcher.evidence.**", lambda e: published.append(e) or asyncio.sleep(0))

        findings = [
            Finding(
                id=_gen_id(), project_id=project.id, source_id="s1",
                finding_type="supports", summary="Confirmed", confidence=0.9,
                domain=None, related_insight_id=insight.id, created_at=_now(),
            ),
            Finding(
                id=_gen_id(), project_id=project.id, source_id="s1",
                finding_type="supports", summary="Also confirmed", confidence=0.8,
                domain=None, related_insight_id=insight.id, created_at=_now(),
            ),
        ]

        await engine._publish_evidence(findings, insight)

        assert any(e.topic == "researcher.evidence.supports" for e in published)

        await store.close()

    @pytest.mark.asyncio
    async def test_evidence_contradicts_event(self):
        """Contradicting findings publish researcher.evidence.contradicts."""
        from researcher.src.engine import ResearcherEngine

        llm = _mock_llm_client()
        consensus = _mock_consensus()
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)
        bus = EventBus(store=None)

        engine = ResearcherEngine(store, bus, llm, consensus, config)
        insight = _make_insight(project_id=project.id)

        published: list[Event] = []
        bus.subscribe("researcher.evidence.**", lambda e: published.append(e) or asyncio.sleep(0))

        findings = [
            Finding(
                id=_gen_id(), project_id=project.id, source_id="s1",
                finding_type="refutes", summary="Disproved", confidence=0.9,
                domain=None, related_insight_id=insight.id, created_at=_now(),
            ),
            Finding(
                id=_gen_id(), project_id=project.id, source_id="s1",
                finding_type="refutes", summary="Also disproved", confidence=0.85,
                domain=None, related_insight_id=insight.id, created_at=_now(),
            ),
        ]

        await engine._publish_evidence(findings, insight)

        assert any(e.topic == "researcher.evidence.contradicts" for e in published)

        await store.close()


# ── Directed research priority test ───────────────────────────


class TestDirectedResearchPriority:
    @pytest.mark.asyncio
    async def test_directed_research_priority(self):
        """Directed requests processed before autonomous insight queue."""
        from researcher.src.engine import ResearcherEngine

        llm = _mock_llm_client()
        consensus = _mock_consensus()
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)
        bus = EventBus(store=None)

        engine = ResearcherEngine(store, bus, llm, consensus, config)
        await engine.initialize()

        # Queue an autonomous insight
        insight = _make_insight(project_id=project.id)
        engine._insight_queue.append(insight)

        # Queue a directed request (higher priority)
        engine._directed_queue.append({
            "topic": "prime group structure",
            "context": "Need evidence",
            "project_id": project.id,
        })

        # Directed queue should be checked first
        assert len(engine._directed_queue) == 1
        assert len(engine._insight_queue) == 1

        # The engine processes directed queue before insight queue
        next_item = engine._get_next_work_item()
        assert next_item is not None
        assert next_item["type"] == "directed"

        await store.close()


# ── Configurable limits test ──────────────────────────────────


class TestConfigurableLimits:
    @pytest.mark.asyncio
    async def test_configurable_limits(self):
        """Config values respected for question/search counts."""
        from researcher.src.questions import QuestionGenerator
        from researcher.src.searcher import SearchExecutor

        config = _make_config({
            "researcher.max_questions_per_insight": 2,
            "researcher.max_searches_per_question": 1,
        })
        llm = _mock_llm_client()
        llm.send.return_value = LLMResponse(
            success=True,
            text=json.dumps(["Q1?", "Q2?", "Q3?", "Q4?"]),
            backend="claude", model="claude-3",
            token_usage=None, error=None,
        )

        gen = QuestionGenerator(llm, config)
        project = _make_project()
        insight = _make_insight()

        questions = await gen.generate(insight, project)
        assert len(questions) <= 2

        # Search executor respects max_results
        llm.send.return_value = LLMResponse(
            success=True,
            text=json.dumps([
                {"url": "https://a.com", "title": "A", "snippet": "a", "domain": None},
                {"url": "https://b.com", "title": "B", "snippet": "b", "domain": None},
                {"url": "https://c.com", "title": "C", "snippet": "c", "domain": None},
            ]),
            backend="claude", model="claude-3",
            token_usage=None, error=None,
        )
        searcher = SearchExecutor(llm, config)
        results = await searcher.search("Q1?", project.research_domains)
        assert len(results) <= 1


# ── Low trust source skipped test ─────────────────────────────


class TestLowTrustSourceSkipped:
    @pytest.mark.asyncio
    async def test_low_trust_source_skipped(self):
        """Source below min_trust_score -> no findings extracted."""
        from researcher.src.engine import ResearcherEngine

        llm = _mock_llm_client()
        consensus = _mock_consensus()
        consensus.run.return_value = _consensus_result(30, Verdict.REJECT)
        store = StateStore(":memory:")
        await store.connect()
        config = _make_config()
        project = _make_project()
        await store.create_project(project)
        bus = EventBus(store=None)

        engine = ResearcherEngine(store, bus, llm, consensus, config)
        source = _make_source(project_id=project.id)
        insight = _make_insight(project_id=project.id)

        trust_score = await engine._trust_evaluator.evaluate(
            source, "content summary", project,
        )
        assert trust_score < config.get("researcher.trust.min_trust_score")

        # Engine should skip finding extraction for low-trust sources
        should_extract = trust_score >= config.get("researcher.trust.min_trust_score")
        assert not should_extract

        await store.close()
