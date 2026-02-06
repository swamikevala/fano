"""Tests for the Documenter Engine — Section 9 of Design Spec."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import uuid

import pytest

from shared.errors import LLMError
from shared.events import EventBus
from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    Confidence,
    ConsensusResult,
    EvaluationCriterion,
    Event,
    HealthStatus,
    Insight,
    InsightStatus,
    LLMResponse,
    Project,
    ProjectStatus,
    ResearchDomain,
    RoundResult,
    Section,
    SectionStatus,
    TokenUsage,
    Verdict,
    WorkItem,
)
from shared.store import StateStore


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]

# ── Helpers ──────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _project(pid: str = "proj-1") -> Project:
    now = _now()
    return Project(
        id=pid, name="Test Project", goal="Explore math.",
        context="Testing context.",
        evaluation_criteria=[EvaluationCriterion("rigor", "r", 1.0)],
        exploration_guidance="Explore.", document_guidance="Write clearly.",
        seed_modification_enabled=False,
        seed_modification_require_approval=False,
        research_domains=[ResearchDomain("math", ["algebra"], ["paper"])],
        status=ProjectStatus.ACTIVE, created_at=now, updated_at=now,
    )


def _insight(project_id: str = "proj-1", iid: str | None = None, **kw) -> Insight:
    now = _now()
    defaults = dict(
        id=iid or _gen_id(), project_id=project_id,
        text="The Fano plane has 7 points.", confidence=Confidence.HIGH,
        tags=["fano", "geometry"], source_thread_id=None,
        extraction_model="claude", status=InsightStatus.ATTESTED,
        evaluation_scores={"rigor": 0.9}, dispute_count=0,
        transient_failure_count=0, review_record=None,
        blessed_at=None, incorporated_at=None, incorporated_in_section=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Insight(**defaults)


def _section(project_id: str = "proj-1", sid: str | None = None, **kw) -> Section:
    now = _now()
    defaults = dict(
        id=sid or _gen_id(), project_id=project_id,
        title="Section", content="Body text.", status=SectionStatus.PROVISIONAL,
        order_index=0, establishes=["concept_a"], requires=[],
        source_insight_id=None, review_count=0, last_reviewed_at=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Section(**defaults)


def _annotation(project_id: str = "proj-1", aid: str | None = None, **kw) -> Annotation:
    now = _now()
    defaults = dict(
        id=aid or _gen_id(), project_id=project_id,
        type=AnnotationType.COMMENT, section_id="sec-1",
        content="Please clarify this.", status=AnnotationStatus.OPEN,
        attempt_count=0, last_attempted_at=None, created_at=now, resolved_at=None,
    )
    defaults.update(kw)
    return Annotation(**defaults)


def _make_llm_response(text: str = "Generated content") -> LLMResponse:
    return LLMResponse(
        success=True, text=text, backend="claude", model="claude-3",
        token_usage=TokenUsage(100, 50, 150, 0.01), error=None,
    )


def _make_consensus(verdict: Verdict = Verdict.ACCEPT) -> ConsensusResult:
    return ConsensusResult(
        verdict=verdict, confidence=0.9, scores={"rigor": 0.9},
        rounds_completed=1, round_history=[
            RoundResult(round_num=1, responses=[], is_converged=True, verdict=verdict)
        ], valid_vote_count=2,
    )


def _mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.send.return_value = _make_llm_response()
    llm.send_structured.return_value = _make_llm_response(
        '{"is_duplicate": false, "prerequisites": [], "concepts": ["fano_plane"]}'
    )
    llm.get_available_backends.return_value = ["claude", "gemini"]
    return llm


def _mock_consensus(verdict: Verdict = Verdict.ACCEPT) -> AsyncMock:
    consensus = AsyncMock()
    consensus.run.return_value = _make_consensus(verdict)
    return consensus


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
async def store():
    s = StateStore(":memory:")
    await s.connect()
    yield s
    await s.close()


@pytest.fixture
async def loaded_store(store: StateStore):
    """Store with project, sections, annotations, and insights."""
    await store.create_project(_project())
    return store


@pytest.fixture
def event_bus():
    return EventBus(store=None)


# ── Planner Tests ────────────────────────────────────────────


class TestPlanner:
    async def test_plan_annotations_first(self, loaded_store, event_bus):
        """Annotations are prioritized above insights."""
        from documenter.src.planner import Planner
        from shared.config import Config

        store = loaded_store
        # Create section referenced by annotation (FK constraint)
        await store.create_section(_section(sid="sec-1"))
        # Create an annotation and an insight
        ann = _annotation(aid="ann-1", section_id="sec-1")
        await store.create_annotation(ann)
        ins = _insight(iid="ins-1")
        await store.create_insight(ins)

        cfg = Config.__new__(Config)
        cfg._data = {"documenter": {"work_allocation": {"new_ratio": 0.7}}}
        cfg._project = None

        planner = Planner(store, cfg)
        items = await planner.plan_cycle("proj-1")

        assert len(items) >= 2
        annotation_items = [i for i in items if i.type == "annotation"]
        insight_items = [i for i in items if i.type == "insight"]
        assert len(annotation_items) >= 1
        assert len(insight_items) >= 1
        # Annotations come first (higher priority)
        assert annotation_items[0].priority > insight_items[0].priority

    async def test_plan_work_allocation(self, loaded_store, event_bus):
        """Planner tracks work ratios and produces balanced plan."""
        from documenter.src.planner import Planner
        from shared.config import Config

        store = loaded_store
        # Create multiple insights
        for i in range(5):
            await store.create_insight(_insight(iid=f"ins-{i}"))

        # Create sections that could be reviewed
        for i in range(3):
            await store.create_section(_section(sid=f"sec-{i}", order_index=i))

        cfg = Config.__new__(Config)
        cfg._data = {"documenter": {"work_allocation": {"new_ratio": 0.7}}}
        cfg._project = None

        planner = Planner(store, cfg)
        items = await planner.plan_cycle("proj-1")

        # Should contain both insight and review items
        types = {i.type for i in items}
        assert "insight" in types or "review" in types
        assert len(items) > 0


# ── Processor Tests ──────────────────────────────────────────


class TestProcessor:
    async def test_incorporate_insight_full_pipeline(self, loaded_store, event_bus):
        """Full pipeline: dedup -> prerequisites -> draft -> evaluate -> add."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        consensus = _mock_consensus(Verdict.ACCEPT)

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        ins = _insight(iid="ins-full")
        await store.create_insight(ins)

        await processor.incorporate_insight(ins, project)

        # Insight should be INCORPORATED
        updated = await store.get_insight("ins-full")
        assert updated.status == InsightStatus.INCORPORATED

        # A section should have been created
        sections = await store.list_sections("proj-1")
        assert len(sections) >= 1

    async def test_incorporate_duplicate_skipped(self, loaded_store, event_bus):
        """Already-represented insight marked INCORPORATED without new section."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        # Make dedup check say it IS a duplicate
        llm.send_structured.return_value = _make_llm_response(
            '{"is_duplicate": true, "prerequisites": [], "concepts": []}'
        )
        consensus = _mock_consensus()

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        ins = _insight(iid="ins-dup")
        await store.create_insight(ins)

        await processor.incorporate_insight(ins, project)

        updated = await store.get_insight("ins-dup")
        assert updated.status == InsightStatus.INCORPORATED

        # No new sections created
        sections = await store.list_sections("proj-1")
        assert len(sections) == 0

    async def test_incorporate_transient_failure_retries(self, loaded_store, event_bus):
        """API error -> transient_failure_count incremented, not dispute_count."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        # Make the draft step fail with a transient error
        llm.send.side_effect = LLMError("timeout", backend="claude", is_transient=True)
        consensus = _mock_consensus()

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        ins = _insight(iid="ins-transient")
        await store.create_insight(ins)

        await processor.incorporate_insight(ins, project)

        updated = await store.get_insight("ins-transient")
        assert updated.transient_failure_count == 1
        assert updated.dispute_count == 0
        assert updated.status == InsightStatus.TRANSIENT_FAILURE

    async def test_incorporate_dispute_shelves(self, loaded_store, event_bus):
        """Max disputes -> SHELVED status."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        consensus = _mock_consensus(Verdict.REJECT)

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_disputes_before_shelve": 2,
            }
        }
        cfg._project = None

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        # Insight already at max disputes - 1
        ins = _insight(iid="ins-dispute", dispute_count=1)
        await store.create_insight(ins)

        await processor.incorporate_insight(ins, project)

        updated = await store.get_insight("ins-dispute")
        assert updated.status == InsightStatus.SHELVED
        assert updated.dispute_count == 2

    async def test_address_comment(self, loaded_store, event_bus):
        """Annotation -> revised section -> annotation RESOLVED."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        consensus = _mock_consensus(Verdict.ACCEPT)

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_annotation_attempts": 3,
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        # Create a section to annotate
        sec = _section(sid="sec-1", content="Original content.")
        await store.create_section(sec)

        # Create annotation
        ann = _annotation(aid="ann-1", section_id="sec-1")
        await store.create_annotation(ann)

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        await processor.address_annotation(ann, project)

        updated_ann = await store.get_annotation("ann-1")
        assert updated_ann.status == AnnotationStatus.RESOLVED

    async def test_address_comment_max_attempts(self, loaded_store, event_bus):
        """3 failed attempts -> NEEDS_HUMAN_REVIEW."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        consensus = _mock_consensus(Verdict.REJECT)

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_annotation_attempts": 3,
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        sec = _section(sid="sec-1")
        await store.create_section(sec)

        # Annotation already at max_attempts - 1
        ann = _annotation(aid="ann-maxed", section_id="sec-1", attempt_count=2)
        await store.create_annotation(ann)

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        await processor.address_annotation(ann, project)

        updated = await store.get_annotation("ann-maxed")
        assert updated.status == AnnotationStatus.NEEDS_HUMAN_REVIEW

    async def test_protected_section_not_modified(self, loaded_store, event_bus):
        """Processor skips sections with PROTECTED annotation."""
        from documenter.src.annotations import AnnotationHandler
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        consensus = _mock_consensus(Verdict.ACCEPT)

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_annotation_attempts": 3,
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        sec = _section(sid="sec-protected", content="Do not touch.")
        await store.create_section(sec)

        # Create a PROTECTED annotation on the section
        prot = _annotation(
            aid="prot-1", section_id="sec-protected",
            type=AnnotationType.PROTECTED, status=AnnotationStatus.OPEN,
        )
        await store.create_annotation(prot)

        # Create a COMMENT annotation on the same section
        ann = _annotation(aid="ann-prot", section_id="sec-protected")
        await store.create_annotation(ann)

        handler = AnnotationHandler(store, event_bus)
        protected = await handler.get_protected_sections("proj-1")
        assert "sec-protected" in protected

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        await processor.address_annotation(ann, project)

        # Section content should not have changed
        updated_sec = await store.get_section("sec-protected")
        assert updated_sec.content == "Do not touch."


# ── ContextBuilder Tests ─────────────────────────────────────


class TestContextBuilder:
    async def test_context_builder_token_budget(self, loaded_store):
        """Context truncated to max_tokens."""
        from documenter.src.context import ContextBuilder
        from shared.config import Config

        store = loaded_store

        # Create sections with long content
        for i in range(10):
            await store.create_section(
                _section(sid=f"sec-{i}", order_index=i,
                         content="x " * 500, establishes=[f"concept_{i}"])
            )

        cfg = Config.__new__(Config)
        cfg._data = {"documenter": {"context": {"max_tokens": 200}}}
        cfg._project = None

        builder = ContextBuilder(store, cfg)
        ins = _insight()
        project = _project()

        ctx = await builder.build_for_insight(ins, project, max_tokens=200)
        estimated_tokens = builder.estimate_tokens(ctx)
        assert estimated_tokens <= 200

    async def test_context_builder_dedup(self, loaded_store):
        """Same section not included twice in context."""
        from documenter.src.context import ContextBuilder
        from shared.config import Config

        store = loaded_store
        # Create section that is both recent and tag-related
        sec = _section(
            sid="sec-overlap", order_index=0,
            content="Overlap content.", establishes=["fano"],
        )
        await store.create_section(sec)

        cfg = Config.__new__(Config)
        cfg._data = {"documenter": {"context": {"max_tokens": 8000}}}
        cfg._project = None

        builder = ContextBuilder(store, cfg)
        ins = _insight(tags=["fano"])
        project = _project()

        ctx = await builder.build_for_insight(ins, project)
        # The section content should appear exactly once
        assert ctx.count("Overlap content.") == 1


# ── Renderer Tests ───────────────────────────────────────────


class TestRenderer:
    async def test_render_ordered(self, loaded_store):
        """Sections rendered in order_index sequence."""
        from documenter.src.renderer import Renderer
        from shared.config import Config

        store = loaded_store
        await store.create_section(_section(sid="s3", title="Third", order_index=3))
        await store.create_section(_section(sid="s1", title="First", order_index=1))
        await store.create_section(_section(sid="s2", title="Second", order_index=2))

        cfg = Config.__new__(Config)
        cfg._data = {"documenter": {"document_dir": "/tmp/doc"}}
        cfg._project = None

        renderer = Renderer(store, cfg)
        project = _project()
        md = await renderer.render(project)

        pos_first = md.index("First")
        pos_second = md.index("Second")
        pos_third = md.index("Third")
        assert pos_first < pos_second < pos_third

    async def test_render_no_inline_annotations(self, loaded_store):
        """Annotations not baked into markdown."""
        from documenter.src.renderer import Renderer
        from shared.config import Config

        store = loaded_store
        await store.create_section(_section(sid="s1", order_index=1))
        ann = _annotation(aid="ann-1", section_id="s1", content="Please fix this")
        await store.create_annotation(ann)

        cfg = Config.__new__(Config)
        cfg._data = {"documenter": {"document_dir": "/tmp/doc"}}
        cfg._project = None

        renderer = Renderer(store, cfg)
        project = _project()
        md = await renderer.render(project)

        assert "Please fix this" not in md


# ── Canonicalization Tests ───────────────────────────────────


class TestCanonicalization:
    def test_concept_canonicalization(self):
        """'The Fano Plane' -> 'fano_plane'."""
        from documenter.src.context import canonicalize

        assert canonicalize("The Fano Plane") == "fano_plane"
        assert canonicalize("Fano plane") == "fano_plane"
        assert canonicalize("market entry strategy") == "market_entry_strategy"
        assert canonicalize("Market-Entry Strategy") == "market_entry_strategy"
        assert canonicalize("  A big idea  ") == "big_idea"
        assert canonicalize("an example") == "example"


# ── Annotation Handler Tests ────────────────────────────────


class TestAnnotationHandler:
    async def test_handle_new_annotation(self, loaded_store, event_bus):
        """AnnotationHandler creates annotation from event."""
        from documenter.src.annotations import AnnotationHandler

        store = loaded_store
        # Create the section referenced by the annotation
        await store.create_section(_section(sid="sec-1"))

        handler = AnnotationHandler(store, event_bus)
        event = Event(
            topic="user.annotation.created",
            timestamp=_now(), source="test",
            payload={
                "annotation_id": "ann-evt-1",
                "project_id": "proj-1",
                "type": "comment",
                "section_id": "sec-1",
                "content": "Needs more detail.",
            },
            correlation_id="corr-1",
        )

        await handler.handle_new_annotation(event)

        ann = await store.get_annotation("ann-evt-1")
        assert ann is not None
        assert ann.status == AnnotationStatus.OPEN
        assert ann.type == AnnotationType.COMMENT


# ── Transaction Rollback Test ────────────────────────────────


class TestTransaction:
    async def test_transaction_rollback(self, loaded_store, event_bus):
        """Failure mid-incorporate -> all changes reverted."""
        from documenter.src.context import ContextBuilder
        from documenter.src.processor import Processor
        from shared.config import Config

        store = loaded_store
        llm = _mock_llm()
        # Draft succeeds, but consensus raises an unexpected error
        consensus = AsyncMock()
        consensus.run.side_effect = RuntimeError("Unexpected crash")

        cfg = Config.__new__(Config)
        cfg._data = {
            "documenter": {
                "context": {"max_tokens": 8000},
                "max_disputes_before_shelve": 3,
            }
        }
        cfg._project = None

        ctx_builder = ContextBuilder(store, cfg)
        processor = Processor(store, llm, consensus, event_bus, ctx_builder, cfg)
        project = _project()

        ins = _insight(iid="ins-rollback")
        await store.create_insight(ins)

        with pytest.raises(RuntimeError):
            await processor.incorporate_insight(ins, project)

        # No sections should have been committed
        sections = await store.list_sections("proj-1")
        assert len(sections) == 0

        # Insight status should be unchanged (not INCORPORATED)
        updated = await store.get_insight("ins-rollback")
        assert updated.status == InsightStatus.ATTESTED
