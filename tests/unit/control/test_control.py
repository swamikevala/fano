"""Tests for control panel — v2 REST API (Layer 5).

Every user action publishes events, every response uses the standard envelope:
    {"ok": true, "data": ...}
    {"ok": false, "error": "...", "code": "..."}
"""

import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from shared.config import Config
from shared.events import EventBus
from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    CommentType,
    Confidence,
    EvaluationCriterion,
    Event,
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
    SeedStatus,
    SeedType,
    Source,
    User,
    generate_id,
)
from shared.store import StateStore

TEST_USER_ID = "test-user-001"


# ── helpers ──────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_config() -> Config:
    os.environ.setdefault("FANO_TEST_KEY", "test-key-value")
    return Config.from_dict({
        "llm": {
            "api_key_env": "FANO_TEST_KEY",
            "models": {"claude": {"model": "claude-3"}},
        },
        "consensus": {"backends": ["claude"]},
        "control": {"host": "127.0.0.1", "port": 8080},
    })


def _make_project(pid: str = "proj-1", **kw) -> Project:
    now = _now()
    defaults = dict(
        id=pid, owner_id=TEST_USER_ID, name="Test Project", goal="Explore math",
        context="Testing context",
        evaluation_criteria=[EvaluationCriterion("rigor", "r", 1.0)],
        exploration_guidance="EG", document_guidance="DG",
        seed_modification_enabled=True,
        seed_modification_require_approval=False,
        research_domains=[ResearchDomain("math", ["algebra"], ["paper"])],
        status=ProjectStatus.ACTIVE,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Project(**defaults)


def _make_test_user(uid: str = TEST_USER_ID) -> User:
    return User(
        id=uid, username="tester",
        display_name="Test User", created_at=_now(),
    )


def _make_seed(project_id: str = "proj-1", sid: str | None = None, **kw) -> Seed:
    now = _now()
    defaults = dict(
        id=sid or generate_id(), project_id=project_id,
        text="Test seed", type=SeedType.CONJECTURE, priority=5,
        tags=["math"], confidence=Confidence.MEDIUM,
        source="test", notes=None, status=SeedStatus.ACTIVE,
        parent_seed_id=None, modification_reason=None,
        exploration_count=0, created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Seed(**defaults)


def _make_insight(project_id: str = "proj-1", iid: str | None = None, **kw) -> Insight:
    now = _now()
    defaults = dict(
        id=iid or generate_id(), project_id=project_id,
        text="Test insight", confidence=Confidence.HIGH,
        tags=["x"], source_thread_id=None, extraction_model="claude",
        status=InsightStatus.EXTRACTED, evaluation_scores={"rigor": 0.9},
        dispute_count=0, transient_failure_count=0, review_record=None,
        blessed_at=None, incorporated_at=None, incorporated_in_section=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Insight(**defaults)


def _make_section(project_id: str = "proj-1", sid: str | None = None, **kw) -> Section:
    now = _now()
    defaults = dict(
        id=sid or generate_id(), project_id=project_id,
        title="Section Title", content="# Title\nBody content",
        status=SectionStatus.PROVISIONAL, order_index=0,
        establishes=["concept_a"], requires=["concept_b"],
        source_insight_id=None, review_count=0, last_reviewed_at=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Section(**defaults)


def _make_annotation(project_id: str = "proj-1", aid: str | None = None, **kw) -> Annotation:
    now = _now()
    defaults = dict(
        id=aid or generate_id(), project_id=project_id,
        type=AnnotationType.COMMENT, section_id=None, content="A note",
        status=AnnotationStatus.OPEN, attempt_count=0,
        last_attempted_at=None, created_at=now, resolved_at=None,
    )
    defaults.update(kw)
    return Annotation(**defaults)


def _make_source(project_id: str = "proj-1", sid: str | None = None, **kw) -> Source:
    now = _now()
    defaults = dict(
        id=sid or generate_id(), project_id=project_id,
        url="https://example.com", domain="example.com",
        title="Example", trust_score=50, trust_tier=None,
        content_hash=None, evaluated_at=None, created_at=now,
    )
    defaults.update(kw)
    return Source(**defaults)


def _make_finding(project_id: str = "proj-1", fid: str | None = None, **kw) -> Finding:
    now = _now()
    defaults = dict(
        id=fid or generate_id(), project_id=project_id,
        source_id=None, finding_type=None, summary="Found something",
        confidence=0.7, domain=None, related_insight_id=None,
        created_at=now,
    )
    defaults.update(kw)
    return Finding(**defaults)


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture
def loop():
    """Provide a fresh event loop for sync Flask test helpers."""
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


@pytest.fixture
def app(loop):
    """Create a Flask test app backed by an in-memory store."""
    from control.server import create_app

    store = StateStore(":memory:")
    loop.run_until_complete(store.connect())
    bus = EventBus(store=None)
    config = _make_config()
    application = create_app(store, bus, config)
    application.config["TESTING"] = True
    # Create test user for auth
    loop.run_until_complete(store.create_user(_make_test_user()))
    yield application
    loop.run_until_complete(store.close())


@pytest.fixture
def client(app):
    """Flask test client with authenticated session."""
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = TEST_USER_ID
    return c


@pytest.fixture
def store(app):
    """Get the StateStore from the app."""
    return app.config["state_store"]


@pytest.fixture
def bus(app):
    """Get the EventBus from the app."""
    return app.config["event_bus"]


@pytest.fixture
def loop_from_app(loop):
    """Event loop reference matching the app fixture."""
    return loop


@pytest.fixture
def seeded_app(app, loop):
    """App with a project and seeds pre-loaded."""
    store = app.config["state_store"]
    project = _make_project()
    loop.run_until_complete(store.create_project(project))
    for i in range(5):
        seed = _make_seed(sid=f"seed-{i}", priority=i + 1)
        loop.run_until_complete(store.create_seed(seed))
    return app


@pytest.fixture
def seeded_client(seeded_app):
    c = seeded_app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = TEST_USER_ID
    return c


# ── Project endpoints ────────────────────────────────────────


class TestProjects:
    def test_list_projects_empty(self, client):
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"] == []

    def test_create_project(self, client):
        payload = {
            "name": "New Project",
            "goal": "Research something",
            "context": "Test context",
        }
        resp = client.post("/api/projects", json=payload)
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"]["name"] == "New Project"
        assert "id" in body["data"]

    def test_list_projects_after_create(self, client):
        client.post("/api/projects", json={
            "name": "P1", "goal": "G1", "context": "C1",
        })
        resp = client.get("/api/projects")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 1

    def test_get_project(self, client, store, loop):
        project = _make_project(pid="proj-get")
        loop.run_until_complete(store.create_project(project))
        resp = client.get("/api/projects/proj-get")
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"]["id"] == "proj-get"

    def test_get_project_not_found(self, client):
        resp = client.get("/api/projects/nonexistent")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["ok"] is False
        assert body["code"] == "NOT_FOUND"

    def test_update_project(self, client, store, loop):
        project = _make_project(pid="proj-upd")
        loop.run_until_complete(store.create_project(project))
        resp = client.put("/api/projects/proj-upd", json={"name": "Updated"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True

        # Verify update
        resp2 = client.get("/api/projects/proj-upd")
        body2 = resp2.get_json()
        assert body2["data"]["name"] == "Updated"


# ── Seed endpoints ───────────────────────────────────────────


class TestSeeds:
    def test_list_seeds(self, seeded_client):
        """GET /api/seeds returns seeds for active project."""
        resp = seeded_client.get("/api/seeds?project_id=proj-1")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 5
        assert "total" in body

    def test_list_seeds_requires_project_id(self, client):
        resp = client.get("/api/seeds")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["ok"] is False
        assert body["code"] == "VALIDATION_ERROR"

    def test_create_seed_publishes_event(self, client, store, bus, loop):
        """POST /api/seeds creates seed and publishes user.seed.created event."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))

        captured_events: list[Event] = []

        async def handler(event: Event) -> None:
            captured_events.append(event)

        bus.subscribe("user.seed.created", handler)

        payload = {
            "project_id": "proj-1",
            "text": "New conjecture about primes",
            "type": "conjecture",
            "priority": 7,
            "tags": ["primes", "number_theory"],
        }
        resp = client.post("/api/seeds", json=payload)
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"]["text"] == "New conjecture about primes"

        # Event should have been published
        assert len(captured_events) == 1
        assert captured_events[0].topic == "user.seed.created"
        assert captured_events[0].payload["seed_id"] == body["data"]["id"]

    def test_get_seed(self, seeded_client):
        resp = seeded_client.get("/api/seeds/seed-0")
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"]["id"] == "seed-0"

    def test_get_seed_not_found(self, client):
        resp = client.get("/api/seeds/nonexistent")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["ok"] is False

    def test_update_seed(self, seeded_client):
        resp = seeded_client.put("/api/seeds/seed-0", json={"priority": 10})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True

    def test_delete_seed(self, seeded_client):
        resp = seeded_client.delete("/api/seeds/seed-0")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True

        # Should be retired (soft delete)
        resp2 = seeded_client.get("/api/seeds/seed-0")
        body2 = resp2.get_json()
        assert body2["data"]["status"] == "retired"

    def test_seed_approve_modification(self, client, store, loop):
        """POST approve on a seed publishes event."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        seed = _make_seed(sid="seed-mod")
        loop.run_until_complete(store.create_seed(seed))

        resp = client.post("/api/seeds/seed-mod/approve", json={
            "modification_reason": "Refined based on exploration",
        })
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True

    def test_seed_reject_modification(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        seed = _make_seed(sid="seed-rej")
        loop.run_until_complete(store.create_seed(seed))

        resp = client.post("/api/seeds/seed-rej/reject", json={
            "reason": "Not convincing",
        })
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True


# ── Insight endpoints ────────────────────────────────────────


class TestInsights:
    def test_list_insights(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        for i in range(3):
            ins = _make_insight(iid=f"ins-{i}")
            loop.run_until_complete(store.create_insight(ins))

        resp = client.get("/api/insights?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 3

    def test_get_insight(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        ins = _make_insight(iid="ins-get")
        loop.run_until_complete(store.create_insight(ins))

        resp = client.get("/api/insights/ins-get")
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"]["id"] == "ins-get"

    def test_insight_comment_publishes_event(self, client, store, bus, loop):
        """POST comment with type=endorse publishes user.insight.endorsed."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        ins = _make_insight(iid="ins-comment")
        loop.run_until_complete(store.create_insight(ins))

        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.subscribe("user.insight.**", handler)

        payload = {
            "type": "endorse",
            "content": "This is correct and important",
        }
        resp = client.post("/api/insights/ins-comment/comment", json=payload)
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["ok"] is True

        assert len(captured) == 1
        assert captured[0].topic == "user.insight.endorsed"
        assert captured[0].payload["insight_id"] == "ins-comment"

    def test_insight_comment_dismiss(self, client, store, bus, loop):
        """POST comment with type=dismiss publishes user.insight.dismissed."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        ins = _make_insight(iid="ins-dismiss")
        loop.run_until_complete(store.create_insight(ins))

        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.subscribe("user.insight.**", handler)

        resp = client.post("/api/insights/ins-dismiss/comment", json={
            "type": "dismiss",
            "content": "Not convincing",
        })
        assert resp.status_code == 201
        assert len(captured) == 1
        assert captured[0].topic == "user.insight.dismissed"

    def test_insight_comment_general(self, client, store, bus, loop):
        """POST comment with type=general publishes user.insight.commented."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        ins = _make_insight(iid="ins-gen")
        loop.run_until_complete(store.create_insight(ins))

        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.subscribe("user.insight.**", handler)

        resp = client.post("/api/insights/ins-gen/comment", json={
            "type": "general",
            "content": "Interesting observation",
        })
        assert resp.status_code == 201
        assert len(captured) == 1
        assert captured[0].topic == "user.insight.commented"

    def test_insight_not_found(self, client):
        resp = client.get("/api/insights/nonexistent")
        assert resp.status_code == 404


# ── Document endpoints ───────────────────────────────────────


class TestDocument:
    def test_document_render(self, client, store, loop):
        """GET /api/document returns rendered markdown."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        sec = _make_section(sid="sec-1", order_index=0, content="# Introduction\nHello")
        loop.run_until_complete(store.create_section(sec))

        resp = client.get("/api/document?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert "markdown" in body["data"]
        assert "Introduction" in body["data"]["markdown"]

    def test_document_sections(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        for i in range(3):
            sec = _make_section(sid=f"sec-{i}", order_index=i,
                                title=f"Section {i}")
            loop.run_until_complete(store.create_section(sec))

        resp = client.get("/api/document/sections?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 3

    def test_document_requires_project_id(self, client):
        resp = client.get("/api/document")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["ok"] is False
        assert body["code"] == "VALIDATION_ERROR"


# ── Annotation endpoints ────────────────────────────────────


class TestAnnotations:
    def test_annotation_create(self, client, store, bus, loop):
        """POST annotation creates it and publishes user.annotation.created."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))

        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.subscribe("user.annotation.**", handler)

        payload = {
            "project_id": "proj-1",
            "type": "comment",
            "content": "This section needs more detail",
            "section_id": None,
        }
        resp = client.post("/api/annotations", json=payload)
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["ok"] is True
        assert "id" in body["data"]

        # Event published
        assert len(captured) == 1
        assert captured[0].topic == "user.annotation.created"

    def test_list_annotations(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        for i in range(3):
            ann = _make_annotation(aid=f"ann-{i}")
            loop.run_until_complete(store.create_annotation(ann))

        resp = client.get("/api/annotations?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 3

    def test_get_annotation(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        ann = _make_annotation(aid="ann-get")
        loop.run_until_complete(store.create_annotation(ann))

        resp = client.get("/api/annotations/ann-get")
        body = resp.get_json()
        assert body["ok"] is True
        assert body["data"]["id"] == "ann-get"


# ── Research endpoints ───────────────────────────────────────


class TestResearch:
    def test_list_findings(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        for i in range(2):
            f = _make_finding(fid=f"find-{i}")
            loop.run_until_complete(store.create_finding(f))

        resp = client.get("/api/research/findings?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 2

    def test_list_sources(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        src = _make_source(sid="src-1")
        loop.run_until_complete(store.create_source(src))

        resp = client.get("/api/research/sources?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 1

    def test_research_request(self, client, store, bus, loop):
        """POST /api/research/request publishes user.research.requested."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))

        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.subscribe("user.research.**", handler)

        payload = {
            "project_id": "proj-1",
            "query": "Find papers on prime distribution",
            "domain": "number_theory",
        }
        resp = client.post("/api/research/request", json=payload)
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["ok"] is True

        assert len(captured) == 1
        assert captured[0].topic == "user.research.requested"


# ── Status endpoints ─────────────────────────────────────────


class TestStatus:
    def test_status_endpoint(self, client):
        """GET /api/status returns module health."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert "modules" in body["data"]

    def test_status_metrics(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        loop.run_until_complete(
            store.record_metric("proj-1", "insights.count", 42.0)
        )

        resp = client.get("/api/status/metrics?name=insights.count")
        body = resp.get_json()
        assert body["ok"] is True
        assert isinstance(body["data"], list)

    def test_status_events(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        event = Event(
            topic="test.event",
            timestamp=_now(),
            source="test",
            payload={"key": "value"},
            correlation_id=generate_id(),
        )
        loop.run_until_complete(store.persist_event(event))

        resp = client.get("/api/status/events")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) >= 1


# ── Error response format ────────────────────────────────────


class TestErrorFormat:
    def test_error_response_format(self, client):
        """Invalid request returns error envelope with code."""
        # No project_id on seed list
        resp = client.get("/api/seeds")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["ok"] is False
        assert "error" in body
        assert "code" in body

    def test_404_format(self, client):
        resp = client.get("/api/projects/doesnotexist")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["ok"] is False
        assert body["code"] == "NOT_FOUND"

    def test_missing_required_field(self, client, store, loop):
        """POST seed without text returns validation error."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))

        resp = client.post("/api/seeds", json={
            "project_id": "proj-1",
            # missing text
            "type": "conjecture",
            "priority": 5,
        })
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["ok"] is False
        assert body["code"] == "VALIDATION_ERROR"


# ── Pagination ───────────────────────────────────────────────


class TestPagination:
    def test_pagination(self, seeded_client):
        """List endpoints support offset/limit."""
        # Limit to 2
        resp = seeded_client.get("/api/seeds?project_id=proj-1&limit=2&offset=0")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 2
        assert body["total"] == 5
        assert body["offset"] == 0
        assert body["limit"] == 2

    def test_pagination_offset(self, seeded_client):
        """Offset skips items."""
        resp = seeded_client.get("/api/seeds?project_id=proj-1&limit=2&offset=3")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 2
        assert body["offset"] == 3
        assert body["total"] == 5

    def test_pagination_defaults(self, seeded_client):
        """Without offset/limit, returns all with total."""
        resp = seeded_client.get("/api/seeds?project_id=proj-1")
        body = resp.get_json()
        assert body["ok"] is True
        assert body["total"] == 5
        assert len(body["data"]) == 5


# ── Filtering ────────────────────────────────────────────────


class TestFiltering:
    def test_filtering_seeds_by_status(self, client, store, loop):
        """List endpoints support status filter."""
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        s1 = _make_seed(sid="s-act", status=SeedStatus.ACTIVE)
        s2 = _make_seed(sid="s-ret", status=SeedStatus.RETIRED)
        loop.run_until_complete(store.create_seed(s1))
        loop.run_until_complete(store.create_seed(s2))

        resp = client.get("/api/seeds?project_id=proj-1&status=active")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "s-act"

    def test_filtering_insights_by_status(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        i1 = _make_insight(iid="i-ext", status=InsightStatus.EXTRACTED)
        i2 = _make_insight(iid="i-att", status=InsightStatus.ATTESTED)
        loop.run_until_complete(store.create_insight(i1))
        loop.run_until_complete(store.create_insight(i2))

        resp = client.get("/api/insights?project_id=proj-1&status=attested")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "i-att"

    def test_filtering_projects_by_status(self, client, store, loop):
        p1 = _make_project(pid="p-active", status=ProjectStatus.ACTIVE)
        p2 = _make_project(pid="p-paused", status=ProjectStatus.PAUSED)
        loop.run_until_complete(store.create_project(p1))
        loop.run_until_complete(store.create_project(p2))

        resp = client.get("/api/projects?status=active")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "p-active"

    def test_filtering_annotations_by_status(self, client, store, loop):
        project = _make_project()
        loop.run_until_complete(store.create_project(project))
        a1 = _make_annotation(aid="a-open", status=AnnotationStatus.OPEN)
        a2 = _make_annotation(aid="a-res", status=AnnotationStatus.RESOLVED)
        loop.run_until_complete(store.create_annotation(a1))
        loop.run_until_complete(store.create_annotation(a2))

        resp = client.get("/api/annotations?project_id=proj-1&status=open")
        body = resp.get_json()
        assert body["ok"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "a-open"
