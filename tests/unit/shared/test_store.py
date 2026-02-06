"""Tests for shared.store — StateStore (SQLite backend)."""

import asyncio
from datetime import datetime, timezone

import pytest

from shared.errors import StoreError
from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    CommentType,
    Concept,
    Confidence,
    EvaluationCriterion,
    Event,
    Exchange,
    ExchangeRole,
    Finding,
    Insight,
    InsightComment,
    InsightStatus,
    Project,
    ProjectStatus,
    ResearchDomain,
    Section,
    SectionStatus,
    Seed,
    SeedModification,
    SeedStatus,
    SeedType,
    Source,
    Thread,
    ThreadStatus,
    generate_id,
)
from shared.store import StateStore


# ── helpers ──────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _project(pid: str = "proj-1", **kw) -> Project:
    now = _now()
    defaults = dict(
        id=pid, name="P", goal="G", context="C",
        evaluation_criteria=[EvaluationCriterion("rigor", "r", 1.0)],
        exploration_guidance="EG", document_guidance="DG",
        seed_modification_enabled=True,
        seed_modification_require_approval=False,
        research_domains=[ResearchDomain("d", ["k"], ["s"])],
        status=ProjectStatus.ACTIVE,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Project(**defaults)


def _seed(project_id: str = "proj-1", sid: str | None = None, **kw) -> Seed:
    now = _now()
    defaults = dict(
        id=sid or generate_id(), project_id=project_id,
        text="seed text", type=SeedType.CONJECTURE, priority=5,
        tags=["a", "b"], confidence=Confidence.MEDIUM,
        source="test", notes=None, status=SeedStatus.ACTIVE,
        parent_seed_id=None, modification_reason=None,
        exploration_count=0, created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Seed(**defaults)


def _thread(project_id: str = "proj-1", seed_id: str = "s1",
            tid: str | None = None, **kw) -> Thread:
    now = _now()
    defaults = dict(
        id=tid or generate_id(), project_id=project_id, seed_id=seed_id,
        status=ThreadStatus.ACTIVE, priority=5, exchange_count=0,
        last_completed_sequence=0, created_at=now, updated_at=now,
        retired_at=None, retire_reason=None,
    )
    defaults.update(kw)
    return Thread(**defaults)


def _exchange(thread_id: str = "t1", eid: str | None = None, **kw) -> Exchange:
    now = _now()
    defaults = dict(
        id=eid or generate_id(), thread_id=thread_id, sequence=1,
        role=ExchangeRole.EXPLORER, model="gemini",
        prompt="p", response="r", created_at=now,
    )
    defaults.update(kw)
    return Exchange(**defaults)


def _insight(project_id: str = "proj-1", thread_id: str | None = None,
             iid: str | None = None, **kw) -> Insight:
    now = _now()
    defaults = dict(
        id=iid or generate_id(), project_id=project_id,
        text="insight text", confidence=Confidence.HIGH,
        tags=["x"], source_thread_id=thread_id, extraction_model="claude",
        status=InsightStatus.EXTRACTED, evaluation_scores={"rigor": 0.9},
        dispute_count=0, transient_failure_count=0, review_record=None,
        blessed_at=None, incorporated_at=None, incorporated_in_section=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Insight(**defaults)


def _comment(insight_id: str = "i1", cid: str | None = None, **kw) -> InsightComment:
    now = _now()
    defaults = dict(
        id=cid or generate_id(), insight_id=insight_id,
        comment_type=CommentType.ENDORSE, content="good", created_at=now,
    )
    defaults.update(kw)
    return InsightComment(**defaults)


def _section(project_id: str = "proj-1", sid: str | None = None, **kw) -> Section:
    now = _now()
    defaults = dict(
        id=sid or generate_id(), project_id=project_id,
        title="Section Title", content="body", status=SectionStatus.PROVISIONAL,
        order_index=0, establishes=["concept_a"], requires=["concept_b"],
        source_insight_id=None, review_count=0, last_reviewed_at=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Section(**defaults)


def _concept(project_id: str = "proj-1", **kw) -> Concept:
    defaults = dict(
        name="Concept A", canonical_name="concept_a",
        established_in_section=None, project_id=project_id, domain=None,
    )
    defaults.update(kw)
    return Concept(**defaults)


def _annotation(project_id: str = "proj-1", aid: str | None = None, **kw) -> Annotation:
    now = _now()
    defaults = dict(
        id=aid or generate_id(), project_id=project_id,
        type=AnnotationType.COMMENT, section_id=None, content="note",
        status=AnnotationStatus.OPEN, attempt_count=0,
        last_attempted_at=None, created_at=now, resolved_at=None,
    )
    defaults.update(kw)
    return Annotation(**defaults)


def _source(project_id: str = "proj-1", sid: str | None = None, **kw) -> Source:
    now = _now()
    defaults = dict(
        id=sid or generate_id(), project_id=project_id,
        url="https://example.com", domain="example.com",
        title="Example", trust_score=50, trust_tier=None,
        content_hash=None, evaluated_at=None, created_at=now,
    )
    defaults.update(kw)
    return Source(**defaults)


def _finding(project_id: str = "proj-1", fid: str | None = None, **kw) -> Finding:
    now = _now()
    defaults = dict(
        id=fid or generate_id(), project_id=project_id,
        source_id=None, finding_type=None, summary="found something",
        confidence=0.7, domain=None, related_insight_id=None,
        created_at=now,
    )
    defaults.update(kw)
    return Finding(**defaults)


def _seed_modification(seed_id: str = "s1", mid: str | None = None, **kw) -> SeedModification:
    now = _now()
    defaults = dict(
        id=mid or generate_id(), seed_id=seed_id,
        original_text="old", proposed_text="new", reasoning="because",
        proposing_thread_id="t1", agreement_ratio=0.8,
        status="pending", child_seed_id=None,
        created_at=now, resolved_at=None,
    )
    defaults.update(kw)
    return SeedModification(**defaults)


# ── fixtures ─────────────────────────────────────────────────

@pytest.fixture
async def store():
    s = StateStore(":memory:")
    await s.connect()
    yield s
    await s.close()


@pytest.fixture
async def loaded_store(store: StateStore):
    """Store pre-loaded with a project, seed, thread, and insight."""
    p = _project()
    await store.create_project(p)
    s = _seed(sid="s1")
    await store.create_seed(s)
    t = _thread(seed_id="s1", tid="t1")
    await store.create_thread(t)
    i = _insight(thread_id="t1", iid="i1")
    await store.create_insight(i)
    return store


# ── connection ───────────────────────────────────────────────

async def test_connect_creates_tables(store: StateStore):
    tables_sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    async with store._db.execute(tables_sql) as cur:
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    expected = {
        "projects", "seeds", "threads", "exchanges", "insights",
        "insight_dependencies", "insight_comments", "sections",
        "concepts", "annotations", "sources", "findings",
        "events", "metrics", "seed_modifications",
    }
    assert expected.issubset(names), f"Missing tables: {expected - names}"


async def test_connect_idempotent(store: StateStore):
    await store.connect()
    await store.connect()
    # Should not raise; tables already exist


# ── CRUD: projects ───────────────────────────────────────────

async def test_crud_project(store: StateStore):
    p = _project()
    await store.create_project(p)

    got = await store.get_project(p.id)
    assert got is not None
    assert got.name == p.name
    assert got.evaluation_criteria == p.evaluation_criteria
    assert got.research_domains == p.research_domains

    await store.update_project(p.id, name="Updated", status=ProjectStatus.PAUSED)
    got2 = await store.get_project(p.id)
    assert got2.name == "Updated"
    assert got2.status == ProjectStatus.PAUSED

    projects = await store.list_projects()
    assert len(projects) == 1

    projects_filtered = await store.list_projects(status=ProjectStatus.ACTIVE)
    assert len(projects_filtered) == 0


# ── CRUD: seeds ──────────────────────────────────────────────

async def test_crud_seed(store: StateStore):
    await store.create_project(_project())
    s = _seed(sid="seed-1")
    await store.create_seed(s)

    got = await store.get_seed("seed-1")
    assert got is not None
    assert got.tags == ["a", "b"]
    assert got.type == SeedType.CONJECTURE

    await store.update_seed("seed-1", priority=9, status=SeedStatus.EXPLORED)
    got2 = await store.get_seed("seed-1")
    assert got2.priority == 9
    assert got2.status == SeedStatus.EXPLORED

    seeds = await store.list_seeds("proj-1")
    assert len(seeds) == 1

    seeds_active = await store.list_seeds("proj-1", status=SeedStatus.ACTIVE)
    assert len(seeds_active) == 0


# ── CRUD: threads ────────────────────────────────────────────

async def test_crud_thread(store: StateStore):
    await store.create_project(_project())
    await store.create_seed(_seed(sid="s1"))
    t = _thread(seed_id="s1", tid="t1")
    await store.create_thread(t)

    got = await store.get_thread("t1")
    assert got is not None
    assert got.status == ThreadStatus.ACTIVE

    await store.update_thread("t1", status=ThreadStatus.STALLED, retire_reason="stuck")
    got2 = await store.get_thread("t1")
    assert got2.status == ThreadStatus.STALLED
    assert got2.retire_reason == "stuck"

    threads = await store.list_threads("proj-1")
    assert len(threads) == 1


# ── CRUD: exchanges ──────────────────────────────────────────

async def test_crud_exchange(store: StateStore):
    await store.create_project(_project())
    await store.create_seed(_seed(sid="s1"))
    await store.create_thread(_thread(seed_id="s1", tid="t1"))
    e = _exchange(thread_id="t1", eid="e1")
    await store.create_exchange(e)

    exchanges = await store.get_exchanges("t1")
    assert len(exchanges) == 1
    assert exchanges[0].role == ExchangeRole.EXPLORER


# ── CRUD: insights ───────────────────────────────────────────

async def test_crud_insight(loaded_store: StateStore):
    store = loaded_store
    got = await store.get_insight("i1")
    assert got is not None
    assert got.evaluation_scores == {"rigor": 0.9}

    await store.update_insight("i1", status=InsightStatus.ATTESTED)
    got2 = await store.get_insight("i1")
    assert got2.status == InsightStatus.ATTESTED

    insights = await store.list_insights("proj-1")
    assert len(insights) == 1

    insights_for_thread = await store.get_insights_for_thread("t1")
    assert len(insights_for_thread) == 1


# ── CRUD: insight_comments ───────────────────────────────────

async def test_crud_insight_comment(loaded_store: StateStore):
    store = loaded_store
    c = _comment(insight_id="i1", cid="c1")
    await store.create_insight_comment(c)

    comments = await store.get_insight_comments("i1")
    assert len(comments) == 1
    assert comments[0].comment_type == CommentType.ENDORSE


# ── CRUD: seed_modifications ────────────────────────────────

async def test_crud_seed_modification(loaded_store: StateStore):
    store = loaded_store
    m = _seed_modification(seed_id="s1", mid="m1")
    await store.create_seed_modification(m)

    got = await store.get_seed_modification("m1")
    assert got is not None
    assert got.status == "pending"

    await store.update_seed_modification("m1", status="approved")
    got2 = await store.get_seed_modification("m1")
    assert got2.status == "approved"

    pending = await store.list_pending_modifications("proj-1")
    assert len(pending) == 0  # it's approved now


# ── CRUD: sections ───────────────────────────────────────────

async def test_crud_section(store: StateStore):
    await store.create_project(_project())
    sec = _section(sid="sec-1")
    await store.create_section(sec)

    got = await store.get_section("sec-1")
    assert got is not None
    assert got.establishes == ["concept_a"]
    assert got.title == "Section Title"

    await store.update_section("sec-1", content="updated body", status=SectionStatus.STABLE)
    got2 = await store.get_section("sec-1")
    assert got2.content == "updated body"
    assert got2.status == SectionStatus.STABLE

    sections = await store.list_sections("proj-1")
    assert len(sections) == 1


async def test_get_recent_sections(store: StateStore):
    await store.create_project(_project())
    for i in range(5):
        await store.create_section(_section(sid=f"sec-{i}", order_index=i))
    recent = await store.get_recent_sections("proj-1", limit=2)
    assert len(recent) == 2


async def test_get_sections_establishing(store: StateStore):
    await store.create_project(_project())
    await store.create_section(_section(sid="sec-1", establishes=["concept_a", "concept_b"]))
    await store.create_section(_section(sid="sec-2", establishes=["concept_c"]))

    results = await store.get_sections_establishing(["concept_a"])
    assert len(results) >= 1
    assert any(s.id == "sec-1" for s in results)


# ── CRUD: concepts ───────────────────────────────────────────

async def test_crud_concept(store: StateStore):
    await store.create_project(_project())
    c = _concept()
    await store.create_concept(c)

    got = await store.get_concept("concept_a", "proj-1")
    assert got is not None
    assert got.name == "Concept A"

    concepts = await store.list_concepts("proj-1")
    assert len(concepts) == 1


# ── CRUD: annotations ───────────────────────────────────────

async def test_crud_annotation(store: StateStore):
    await store.create_project(_project())
    a = _annotation(aid="a1")
    await store.create_annotation(a)

    got = await store.get_annotation("a1")
    assert got is not None

    await store.update_annotation("a1", status=AnnotationStatus.RESOLVED)
    got2 = await store.get_annotation("a1")
    assert got2.status == AnnotationStatus.RESOLVED

    anns = await store.list_annotations("proj-1")
    assert len(anns) == 1

    anns_open = await store.list_annotations("proj-1", status=AnnotationStatus.OPEN)
    assert len(anns_open) == 0


# ── CRUD: sources ────────────────────────────────────────────

async def test_crud_source(store: StateStore):
    await store.create_project(_project())
    s = _source(sid="src-1")
    await store.create_source(s)

    got = await store.get_source("src-1")
    assert got is not None
    assert got.url == "https://example.com"

    got_by_url = await store.get_source_by_url("https://example.com", "proj-1")
    assert got_by_url is not None

    sources = await store.list_sources("proj-1")
    assert len(sources) == 1


# ── CRUD: findings ───────────────────────────────────────────

async def test_crud_finding(store: StateStore):
    await store.create_project(_project())
    f = _finding(fid="f1")
    await store.create_finding(f)

    got = await store.get_finding("f1")
    assert got is not None

    findings = await store.list_findings("proj-1")
    assert len(findings) == 1


# ── CRUD: events ─────────────────────────────────────────────

async def test_crud_event(store: StateStore):
    now = _now()
    e = Event(
        topic="explorer.insight.attested",
        timestamp=now, source="test",
        payload={"insight_id": "i1"}, correlation_id="corr-1",
    )
    await store.persist_event(e)

    events = await store.list_events()
    assert len(events) == 1
    assert events[0].payload == {"insight_id": "i1"}

    events_filtered = await store.list_events(topic="explorer.insight.attested")
    assert len(events_filtered) == 1

    events_empty = await store.list_events(topic="nonexistent")
    assert len(events_empty) == 0


# ── CRUD: metrics ────────────────────────────────────────────

async def test_crud_metric(store: StateStore):
    await store.create_project(_project())
    await store.record_metric("proj-1", "tokens_used", 150.0, {"model": "gemini"})

    metrics = await store.query_metrics("tokens_used")
    assert len(metrics) == 1
    assert metrics[0]["value"] == 150.0
    assert metrics[0]["labels"] == {"model": "gemini"}


# ── transactions ─────────────────────────────────────────────

async def test_transaction_commit(store: StateStore):
    async with store.transaction():
        await store.create_project(_project())
    got = await store.get_project("proj-1")
    assert got is not None


async def test_transaction_rollback(store: StateStore):
    with pytest.raises(ValueError):
        async with store.transaction():
            await store.create_project(_project())
            raise ValueError("boom")

    got = await store.get_project("proj-1")
    assert got is None


# ── concurrent reads ─────────────────────────────────────────

async def test_concurrent_reads(store: StateStore):
    await store.create_project(_project())
    await store.create_seed(_seed(sid="s1"))

    async def read_seed():
        return await store.get_seed("s1")

    results = await asyncio.gather(read_seed(), read_seed(), read_seed())
    assert all(r is not None for r in results)
    assert all(r.id == "s1" for r in results)


# ── get nonexistent returns None ─────────────────────────────

async def test_get_nonexistent_returns_none(store: StateStore):
    assert await store.get_project("nope") is None
    assert await store.get_seed("nope") is None
    assert await store.get_thread("nope") is None
    assert await store.get_insight("nope") is None
    assert await store.get_section("nope") is None
    assert await store.get_annotation("nope") is None
    assert await store.get_source("nope") is None
    assert await store.get_finding("nope") is None
    assert await store.get_seed_modification("nope") is None
    assert await store.get_concept("nope", "proj-1") is None


# ── update nonexistent raises ────────────────────────────────

async def test_update_nonexistent_raises(store: StateStore):
    with pytest.raises(StoreError):
        await store.update_project("nope", name="X")
    with pytest.raises(StoreError):
        await store.update_seed("nope", priority=1)
    with pytest.raises(StoreError):
        await store.update_thread("nope", priority=1)
    with pytest.raises(StoreError):
        await store.update_insight("nope", status=InsightStatus.ATTESTED)
    with pytest.raises(StoreError):
        await store.update_section("nope", content="X")
    with pytest.raises(StoreError):
        await store.update_annotation("nope", status=AnnotationStatus.RESOLVED)
    with pytest.raises(StoreError):
        await store.update_seed_modification("nope", status="approved")


# ── JSON roundtrip ───────────────────────────────────────────

async def test_json_roundtrip(store: StateStore):
    p = _project(
        evaluation_criteria=[
            EvaluationCriterion("rigor", "Logical soundness", 1.0),
            EvaluationCriterion("depth", "Structural depth", 0.8),
        ],
        research_domains=[
            ResearchDomain("math", ["algebra", "topology"], ["paper", "arxiv"]),
        ],
    )
    await store.create_project(p)
    got = await store.get_project(p.id)

    assert len(got.evaluation_criteria) == 2
    assert got.evaluation_criteria[0].name == "rigor"
    assert got.evaluation_criteria[1].weight == 0.8
    assert got.research_domains[0].keywords == ["algebra", "topology"]

    # Seed tags
    await store.create_seed(_seed(sid="s1", tags=["alpha", "beta", "gamma"]))
    s = await store.get_seed("s1")
    assert s.tags == ["alpha", "beta", "gamma"]

    # Insight evaluation_scores and review_record
    await store.create_seed(_seed(sid="s2"))
    await store.create_thread(_thread(seed_id="s2", tid="t1"))
    review = {"round_1": {"votes": [{"backend": "claude", "verdict": "accept"}]}}
    i = _insight(thread_id="t1", iid="i1",
                 evaluation_scores={"rigor": 0.95, "depth": 0.7},
                 review_record=review)
    await store.create_insight(i)
    got_i = await store.get_insight("i1")
    assert got_i.evaluation_scores == {"rigor": 0.95, "depth": 0.7}
    assert got_i.review_record == review

    # Section establishes/requires
    sec = _section(sid="sec-1", establishes=["c1", "c2"], requires=["c3"])
    await store.create_section(sec)
    got_sec = await store.get_section("sec-1")
    assert got_sec.establishes == ["c1", "c2"]
    assert got_sec.requires == ["c3"]

    # Event payload
    e = Event(topic="t", timestamp=_now(), source="test",
              payload={"key": [1, 2, 3]}, correlation_id="c")
    await store.persist_event(e)
    events = await store.list_events(topic="t")
    assert events[0].payload == {"key": [1, 2, 3]}


# ── list filtering ───────────────────────────────────────────

async def test_list_filtering(store: StateStore):
    await store.create_project(_project("p1"))
    await store.create_project(_project("p2"))

    await store.create_seed(_seed("p1", sid="s1", status=SeedStatus.ACTIVE))
    await store.create_seed(_seed("p1", sid="s2", status=SeedStatus.EXPLORED))
    await store.create_seed(_seed("p2", sid="s3", status=SeedStatus.ACTIVE))

    all_p1 = await store.list_seeds("p1")
    assert len(all_p1) == 2

    active_p1 = await store.list_seeds("p1", status=SeedStatus.ACTIVE)
    assert len(active_p1) == 1
    assert active_p1[0].id == "s1"

    all_p2 = await store.list_seeds("p2")
    assert len(all_p2) == 1

    # Insight filtering
    await store.create_thread(_thread("p1", "s1", tid="t1"))
    await store.create_insight(_insight("p1", "t1", iid="i1", status=InsightStatus.EXTRACTED))
    await store.create_insight(_insight("p1", "t1", iid="i2", status=InsightStatus.ATTESTED))

    all_insights = await store.list_insights("p1")
    assert len(all_insights) == 2

    attested = await store.list_insights("p1", status=InsightStatus.ATTESTED)
    assert len(attested) == 1
    assert attested[0].id == "i2"


# ── seed lineage ─────────────────────────────────────────────

async def test_seed_lineage(store: StateStore):
    await store.create_project(_project())

    grandparent = _seed(sid="sg")
    await store.create_seed(grandparent)

    parent = _seed(sid="sp", parent_seed_id="sg")
    await store.create_seed(parent)

    child = _seed(sid="sc", parent_seed_id="sp")
    await store.create_seed(child)

    lineage = await store.get_seed_lineage("sc")
    ids = [s.id for s in lineage]
    assert ids == ["sc", "sp", "sg"]


# ── foreign key enforcement ──────────────────────────────────

async def test_foreign_key_enforcement(store: StateStore):
    with pytest.raises(StoreError):
        await store.create_seed(
            _seed(project_id="nonexistent-project", sid="bad-seed")
        )
