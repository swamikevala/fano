"""Shared test fixtures for the Research Project Assistant v2."""

from datetime import datetime, timezone

import pytest

from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    Confidence,
    EvaluationCriterion,
    Exchange,
    ExchangeRole,
    Finding,
    Insight,
    InsightComment,
    InsightStatus,
    CommentType,
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


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_project(now: datetime) -> Project:
    return Project(
        id="test-project-001",
        name="Test Research Project",
        goal="Explore test hypotheses about mathematical structure.",
        context="This is a test project for unit tests.",
        evaluation_criteria=[
            EvaluationCriterion(name="rigor", description="Logical soundness", weight=1.0),
            EvaluationCriterion(name="depth", description="Structural depth", weight=0.8),
        ],
        exploration_guidance="Follow curiosity.",
        document_guidance="Write clearly.",
        seed_modification_enabled=True,
        seed_modification_require_approval=False,
        research_domains=[
            ResearchDomain(
                name="test_domain",
                keywords=["test", "unit"],
                source_types=["academic_paper"],
            ),
        ],
        status=ProjectStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_seed(sample_project: Project, now: datetime) -> Seed:
    return Seed(
        id=generate_id(),
        project_id=sample_project.id,
        text="Test seed hypothesis about mathematical structure",
        type=SeedType.CONJECTURE,
        priority=7,
        tags=["math", "test"],
        confidence=Confidence.MEDIUM,
        source="unit test",
        notes=None,
        status=SeedStatus.ACTIVE,
        parent_seed_id=None,
        modification_reason=None,
        exploration_count=0,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_thread(sample_project: Project, sample_seed: Seed, now: datetime) -> Thread:
    return Thread(
        id=generate_id(),
        project_id=sample_project.id,
        seed_id=sample_seed.id,
        status=ThreadStatus.ACTIVE,
        priority=5,
        exchange_count=0,
        last_completed_sequence=0,
        created_at=now,
        updated_at=now,
        retired_at=None,
        retire_reason=None,
    )


@pytest.fixture
def sample_exchange(sample_thread: Thread, now: datetime) -> Exchange:
    return Exchange(
        id=generate_id(),
        thread_id=sample_thread.id,
        sequence=1,
        role=ExchangeRole.EXPLORER,
        model="gemini",
        prompt="Explore this seed hypothesis.",
        response="The mathematical structure shows interesting properties...",
        created_at=now,
    )


@pytest.fixture
def sample_insight(sample_project: Project, sample_thread: Thread, now: datetime) -> Insight:
    return Insight(
        id=generate_id(),
        project_id=sample_project.id,
        text="The structure exhibits a natural triadic symmetry.",
        confidence=Confidence.HIGH,
        tags=["symmetry", "triadic"],
        source_thread_id=sample_thread.id,
        extraction_model="claude",
        status=InsightStatus.EXTRACTED,
        evaluation_scores={},
        dispute_count=0,
        transient_failure_count=0,
        review_record=None,
        blessed_at=None,
        incorporated_at=None,
        incorporated_in_section=None,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
async def store() -> StateStore:
    """In-memory StateStore for testing."""
    s = StateStore(":memory:")
    await s.connect()
    yield s
    await s.close()


@pytest.fixture
async def store_with_project(store: StateStore, sample_project: Project) -> StateStore:
    """StateStore pre-loaded with a sample project."""
    await store.create_project(sample_project)
    return store
