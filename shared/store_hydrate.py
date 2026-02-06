"""Row-to-dataclass hydration for StateStore.

Each function takes a dict (from sqlite Row) and returns the corresponding
frozen dataclass instance, handling JSON deserialization, ISO datetime parsing,
and enum reconstruction.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

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
)


def _from_iso(v: Any) -> datetime | None:
    if v is None:
        return None
    return datetime.fromisoformat(v)


def _json_load(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return json.loads(v)
    return v


def hydrate_project(row: dict) -> Project:
    criteria_raw = _json_load(row["evaluation_criteria"])
    criteria = [EvaluationCriterion(**c) if isinstance(c, dict) else c for c in criteria_raw]
    domains_raw = _json_load(row["research_domains"])
    domains = [ResearchDomain(**d) if isinstance(d, dict) else d for d in domains_raw]
    return Project(
        id=row["id"], name=row["name"], goal=row["goal"], context=row["context"],
        evaluation_criteria=criteria, exploration_guidance=row["exploration_guidance"],
        document_guidance=row["document_guidance"],
        seed_modification_enabled=bool(row["seed_modification_enabled"]),
        seed_modification_require_approval=bool(row["seed_modification_require_approval"]),
        research_domains=domains,
        status=ProjectStatus(row["status"]),
        created_at=_from_iso(row["created_at"]),
        updated_at=_from_iso(row["updated_at"]),
    )


def hydrate_seed(row: dict) -> Seed:
    return Seed(
        id=row["id"], project_id=row["project_id"], text=row["text"],
        type=SeedType(row["type"]), priority=row["priority"],
        tags=_json_load(row["tags"]),
        confidence=Confidence(row["confidence"]) if row["confidence"] else None,
        source=row["source"], notes=row["notes"],
        status=SeedStatus(row["status"]),
        parent_seed_id=row["parent_seed_id"],
        modification_reason=row["modification_reason"],
        exploration_count=row["exploration_count"],
        created_at=_from_iso(row["created_at"]),
        updated_at=_from_iso(row["updated_at"]),
    )


def hydrate_thread(row: dict) -> Thread:
    return Thread(
        id=row["id"], project_id=row["project_id"], seed_id=row["seed_id"],
        status=ThreadStatus(row["status"]), priority=row["priority"],
        exchange_count=row["exchange_count"],
        last_completed_sequence=row["last_completed_sequence"],
        created_at=_from_iso(row["created_at"]),
        updated_at=_from_iso(row["updated_at"]),
        retired_at=_from_iso(row["retired_at"]),
        retire_reason=row["retire_reason"],
    )


def hydrate_exchange(row: dict) -> Exchange:
    return Exchange(
        id=row["id"], thread_id=row["thread_id"], sequence=row["sequence"],
        role=ExchangeRole(row["role"]), model=row["model"],
        prompt=row["prompt"], response=row["response"],
        created_at=_from_iso(row["created_at"]),
    )


def hydrate_insight(row: dict) -> Insight:
    return Insight(
        id=row["id"], project_id=row["project_id"], text=row["text"],
        confidence=Confidence(row["confidence"]), tags=_json_load(row["tags"]),
        source_thread_id=row["source_thread_id"],
        extraction_model=row["extraction_model"],
        status=InsightStatus(row["status"]),
        evaluation_scores=_json_load(row["evaluation_scores"]) or {},
        dispute_count=row["dispute_count"],
        transient_failure_count=row["transient_failure_count"],
        review_record=_json_load(row["review_record"]),
        blessed_at=_from_iso(row["blessed_at"]),
        incorporated_at=_from_iso(row["incorporated_at"]),
        incorporated_in_section=row["incorporated_in_section"],
        created_at=_from_iso(row["created_at"]),
        updated_at=_from_iso(row["updated_at"]),
    )


def hydrate_insight_comment(row: dict) -> InsightComment:
    return InsightComment(
        id=row["id"], insight_id=row["insight_id"],
        comment_type=CommentType(row["comment_type"]),
        content=row["content"],
        created_at=_from_iso(row["created_at"]),
    )


def hydrate_section(row: dict) -> Section:
    return Section(
        id=row["id"], project_id=row["project_id"],
        title=row["title"], content=row["content"],
        status=SectionStatus(row["status"]), order_index=row["order_index"],
        establishes=_json_load(row["establishes"]) or [],
        requires=_json_load(row["requires"]) or [],
        source_insight_id=row["source_insight_id"],
        review_count=row["review_count"],
        last_reviewed_at=_from_iso(row["last_reviewed_at"]),
        created_at=_from_iso(row["created_at"]),
        updated_at=_from_iso(row["updated_at"]),
    )


def hydrate_concept(row: dict) -> Concept:
    return Concept(
        name=row["name"], canonical_name=row["canonical_name"],
        established_in_section=row["established_in_section"],
        project_id=row["project_id"], domain=row["domain"],
    )


def hydrate_annotation(row: dict) -> Annotation:
    return Annotation(
        id=row["id"], project_id=row["project_id"],
        type=AnnotationType(row["type"]), section_id=row["section_id"],
        content=row["content"], status=AnnotationStatus(row["status"]),
        attempt_count=row["attempt_count"],
        last_attempted_at=_from_iso(row["last_attempted_at"]),
        created_at=_from_iso(row["created_at"]),
        resolved_at=_from_iso(row["resolved_at"]),
    )


def hydrate_source(row: dict) -> Source:
    return Source(
        id=row["id"], project_id=row["project_id"],
        url=row["url"], domain=row["domain"], title=row["title"],
        trust_score=row["trust_score"], trust_tier=row["trust_tier"],
        content_hash=row["content_hash"],
        evaluated_at=_from_iso(row["evaluated_at"]),
        created_at=_from_iso(row["created_at"]),
    )


def hydrate_finding(row: dict) -> Finding:
    return Finding(
        id=row["id"], project_id=row["project_id"],
        source_id=row["source_id"], finding_type=row["finding_type"],
        summary=row["summary"], confidence=row["confidence"],
        domain=row["domain"], related_insight_id=row["related_insight_id"],
        created_at=_from_iso(row["created_at"]),
    )


def hydrate_event(row: dict) -> Event:
    return Event(
        topic=row["topic"],
        timestamp=_from_iso(row["timestamp"]),
        source=row["source"],
        payload=_json_load(row["payload"]),
        correlation_id=row["correlation_id"],
    )


def hydrate_seed_modification(row: dict) -> SeedModification:
    return SeedModification(
        id=row["id"], seed_id=row["seed_id"],
        original_text=row["original_text"], proposed_text=row["proposed_text"],
        reasoning=row["reasoning"], proposing_thread_id=row["proposing_thread_id"],
        agreement_ratio=row["agreement_ratio"], status=row["status"],
        child_seed_id=row["child_seed_id"],
        created_at=_from_iso(row["created_at"]),
        resolved_at=_from_iso(row["resolved_at"]),
    )
