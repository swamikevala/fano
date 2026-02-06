"""Tests for UI pages — dynamic rendering from StateStore (Layer 5).

The document viewer page pulls sections, annotations, and project info
from the store and renders them via Jinja templates.
"""

import asyncio
import os
from datetime import datetime, timezone

import pytest

from shared.config import Config
from shared.events import EventBus
from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    Confidence,
    EvaluationCriterion,
    Project,
    ProjectStatus,
    ResearchDomain,
    Section,
    SectionStatus,
    generate_id,
)
from shared.store import StateStore


# ── helpers ──────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _config() -> Config:
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
        id=pid, name="Fano Plane Research", goal="Explore connections",
        context="Testing",
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


def _make_section(project_id: str = "proj-1", **kw) -> Section:
    now = _now()
    defaults = dict(
        id=generate_id(), project_id=project_id,
        title="Section Title", content="<p>Body content</p>",
        status=SectionStatus.PROVISIONAL, order_index=0,
        establishes=["concept_a"], requires=["concept_b"],
        source_insight_id=None, review_count=0, last_reviewed_at=None,
        created_at=now, updated_at=now,
    )
    defaults.update(kw)
    return Section(**defaults)


def _make_annotation(project_id: str = "proj-1", **kw) -> Annotation:
    now = _now()
    defaults = dict(
        id=generate_id(), project_id=project_id,
        type=AnnotationType.COMMENT, section_id=None,
        content="Annotation text", status=AnnotationStatus.OPEN,
        attempt_count=0, last_attempted_at=None,
        created_at=now, resolved_at=None,
    )
    defaults.update(kw)
    return Annotation(**defaults)


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


@pytest.fixture
def app(loop):
    from control.server import create_app

    store = StateStore(":memory:")
    loop.run_until_complete(store.connect())
    bus = EventBus(store=None)
    application = create_app(store, bus, _config())
    application.config["TESTING"] = True
    yield application
    loop.run_until_complete(store.close())


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def store(app):
    return app.config["state_store"]


# ── Section rendering ────────────────────────────────────────


class TestDocumentSections:
    def test_renders_section_titles(self, client, store, loop):
        """Section titles from store appear in the rendered page."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", title="Foundations of the Fano Plane")
        ))

        resp = client.get("/document?project_id=proj-1")
        assert resp.status_code == 200
        assert b"Foundations of the Fano Plane" in resp.data

    def test_renders_section_content_with_latex(self, client, store, loop):
        """Section content including LaTeX markers passes through to HTML."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", content="<p>The matrix satisfies $AA^T = I + J$.</p>")
        ))

        resp = client.get("/document?project_id=proj-1")
        assert b"AA^T = I + J" in resp.data

    def test_sections_ordered_by_index(self, client, store, loop):
        """Sections render in order_index order regardless of creation order."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s2", title="Duality and Self-Duality", order_index=1)
        ))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", title="Axiomatic Foundations", order_index=0)
        ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        assert html.index("Axiomatic Foundations") < html.index("Duality and Self-Duality")

    def test_section_status_stable_badge(self, client, store, loop):
        """A stable section gets the badge-stable CSS class."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", status=SectionStatus.STABLE)
        ))

        resp = client.get("/document?project_id=proj-1")
        assert b"badge-stable" in resp.data

    def test_section_status_needs_work_class(self, client, store, loop):
        """A needs_work section gets the needs-work CSS class (underscore → hyphen)."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", status=SectionStatus.NEEDS_WORK)
        ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        assert "badge-needs-work" in html
        assert "doc-section needs-work" in html

    def test_section_concept_tags(self, client, store, loop):
        """Section establishes/requires concepts render as tags."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", establishes=["Fano plane", "PG(2,2)"],
                          requires=["incidence structure"])
        ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        assert "Fano plane" in html
        assert "PG(2,2)" in html
        assert "incidence structure" in html

    def test_section_review_count(self, client, store, loop):
        """Section review count appears in the metadata."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", review_count=7)
        ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        # The review count should appear near "Reviews:" label
        assert "7" in html


# ── Annotation rendering ────────────────────────────────────


class TestDocumentAnnotations:
    def test_annotation_content_rendered(self, client, store, loop):
        """Annotation content from store appears in the annotation panel."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a1", content="Needs an isomorphism proof sketch")
        ))

        resp = client.get("/document?project_id=proj-1")
        assert b"Needs an isomorphism proof sketch" in resp.data

    def test_annotation_type_css_class(self, client, store, loop):
        """Each annotation type maps to the correct CSS icon class."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a1", type=AnnotationType.PROTECTED)
        ))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a2", type=AnnotationType.SUGGESTION)
        ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        assert "annotation-type-icon protected" in html
        assert "annotation-type-icon suggestion" in html

    def test_annotation_status_badge(self, client, store, loop):
        """Annotation statuses render as the correct badge classes."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a1", status=AnnotationStatus.OPEN)
        ))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a2", status=AnnotationStatus.RESOLVED)
        ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        assert "badge-open" in html
        assert "badge-resolved" in html

    def test_annotation_linked_to_section(self, client, store, loop):
        """Annotation card shows its linked section's title."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="sec-alpha", title="Collineation Symmetries", order_index=0)
        ))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a1", section_id="sec-alpha")
        ))

        resp = client.get("/document?project_id=proj-1")
        assert b"Collineation Symmetries" in resp.data

    def test_annotation_attempt_count(self, client, store, loop):
        """Annotation attempt count is displayed on the card."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_annotation(
            _make_annotation(id="a1", attempt_count=3)
        ))

        resp = client.get("/document?project_id=proj-1")
        assert b"3" in resp.data


# ── Page structure ───────────────────────────────────────────


class TestDocumentPageStructure:
    def test_project_name_as_heading(self, client, store, loop):
        """Project name appears as the document heading."""
        loop.run_until_complete(store.create_project(
            _make_project(name="The Fano Plane and Indian Traditions")
        ))

        resp = client.get("/document?project_id=proj-1")
        assert b"The Fano Plane and Indian Traditions" in resp.data

    def test_toc_has_all_sections(self, client, store, loop):
        """Table of contents lists every section title."""
        loop.run_until_complete(store.create_project(_make_project()))
        titles = ["Foundations", "Duality", "Symmetries"]
        for i, title in enumerate(titles):
            loop.run_until_complete(store.create_section(
                _make_section(id=f"s{i}", title=title, order_index=i)
            ))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        for title in titles:
            assert title in html

    def test_sidebar_stats(self, client, store, loop):
        """Sidebar stats reflect actual data counts."""
        loop.run_until_complete(store.create_project(_make_project()))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", status=SectionStatus.STABLE, order_index=0)
        ))
        loop.run_until_complete(store.create_section(
            _make_section(id="s2", status=SectionStatus.PROVISIONAL, order_index=1)
        ))
        loop.run_until_complete(store.create_annotation(_make_annotation(id="a1")))
        loop.run_until_complete(store.create_annotation(_make_annotation(id="a2")))
        loop.run_until_complete(store.create_annotation(_make_annotation(id="a3")))

        resp = client.get("/document?project_id=proj-1")
        html = resp.data.decode()
        # We can verify stat-val spans contain correct numbers
        # Sections: 2, Stable: 1, Provisional: 1, Annotations: 3
        assert 'class="stat-val">2</span>' in html  # sections count
        assert 'class="stat-val">3</span>' in html  # annotations count

    def test_empty_state_no_sections(self, client, store, loop):
        """Page renders gracefully with a project but no sections or annotations."""
        loop.run_until_complete(store.create_project(_make_project()))

        resp = client.get("/document?project_id=proj-1")
        assert resp.status_code == 200
        assert b"Document Viewer" in resp.data
        # Stats should show zeros
        html = resp.data.decode()
        assert 'class="stat-val">0</span>' in html

    def test_nonexistent_project(self, client):
        """Page renders gracefully when project_id doesn't match any project."""
        resp = client.get("/document?project_id=nonexistent")
        assert resp.status_code == 200
        assert b"Document Viewer" in resp.data

    def test_default_project_fallback(self, client, store, loop):
        """Without project_id param, falls back to first active project."""
        loop.run_until_complete(store.create_project(
            _make_project(name="Auto-Selected Project")
        ))
        loop.run_until_complete(store.create_section(
            _make_section(id="s1", title="Fallback Section")
        ))

        resp = client.get("/document")
        assert resp.status_code == 200
        assert b"Auto-Selected Project" in resp.data
        assert b"Fallback Section" in resp.data
