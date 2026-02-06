"""Processor — incorporates insights and addresses annotations.

All multi-step operations run inside transactions for atomicity.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from shared.errors import LLMError
from shared.models import (
    AnnotationStatus, AnnotationType, Concept, InsightStatus,
    Section, SectionStatus, Verdict, generate_id,
)
from documenter.src.annotations import AnnotationHandler
from documenter.src.context import canonicalize
from documenter.src.prompts import DEDUP_CHECK, DRAFT_SECTION, REVISE_SECTION

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import (
        Annotation, ConsensusEngineInterface, EventBusInterface,
        Insight, LLMClientInterface, Project, StateStoreInterface,
    )
    from documenter.src.context import ContextBuilder

logger = logging.getLogger(__name__)


class Processor:
    """Processes work items: incorporates insights, addresses annotations."""

    def __init__(
        self, store: StateStoreInterface, llm_client: LLMClientInterface,
        consensus: ConsensusEngineInterface, event_bus: EventBusInterface,
        context_builder: ContextBuilder, config: Config,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._consensus = consensus
        self._event_bus = event_bus
        self._ctx = context_builder
        self._config = config

    def _max_disputes(self) -> int:
        return self._config.get("documenter.max_disputes_before_shelve", 3)

    def _max_ann_attempts(self) -> int:
        return self._config.get("documenter.max_annotation_attempts",
               self._config.get("documenter.max_comment_attempts", 3))

    async def incorporate_insight(self, insight: Insight, project: Project) -> None:
        """Incorporate an attested insight within a transaction.
        CRITICAL: separates transient_failure_count from dispute_count.
        """
        async with self._store.transaction():
            try:
                await self._do_incorporate(insight, project)
            except LLMError as exc:
                if exc.is_transient:
                    await self._store.update_insight(
                        insight.id,
                        transient_failure_count=insight.transient_failure_count + 1,
                        status=InsightStatus.TRANSIENT_FAILURE,
                    )
                    return
                raise

    async def _do_incorporate(self, insight: Insight, project: Project) -> None:
        context = await self._ctx.build_for_insight(insight, project)
        sections = await self._store.list_sections(project.id)
        sec_text = "\n".join(f"### {s.title}\n{s.content}" for s in sections) or "(none)"

        # 1. DEDUP
        dedup_resp = await self._llm.send_structured(
            "claude", DEDUP_CHECK.format(insight_text=insight.text, sections_text=sec_text), schema={},
        )
        data = _parse_json(dedup_resp.text)
        if data.get("is_duplicate", False):
            await self._store.update_insight(insight.id, status=InsightStatus.INCORPORATED)
            return

        # 2. PREREQUISITES
        prereqs = data.get("prerequisites", [])
        for name in prereqs:
            canon = canonicalize(name)
            if not await self._store.get_concept(canon, project.id):
                await self._store.create_concept(Concept(
                    name=name, canonical_name=canon,
                    established_in_section=None, project_id=project.id, domain=None,
                ))

        # 3. DRAFT
        draft_resp = await self._llm.send(
            "claude", DRAFT_SECTION.format(
                goal=project.goal, insight_text=insight.text, context=context,
            ),
        )

        # 4. EVALUATE
        result = await self._consensus.run(task=None, backends=None)
        if result.verdict == Verdict.REJECT:
            new_d = insight.dispute_count + 1
            status = InsightStatus.SHELVED if new_d >= self._max_disputes() else InsightStatus.DISPUTED
            await self._store.update_insight(insight.id, dispute_count=new_d, status=status)
            return

        # 5. ADD section
        sid, now = generate_id(), datetime.now(timezone.utc)
        concepts = data.get("concepts", [])
        title = draft_resp.text.split("\n")[0].strip("# ").strip() or "New Section"
        section = Section(
            id=sid, project_id=project.id, title=title, content=draft_resp.text,
            status=SectionStatus.PROVISIONAL, order_index=len(sections),
            establishes=[canonicalize(c) for c in concepts],
            requires=[canonicalize(p) for p in prereqs],
            source_insight_id=insight.id, review_count=0,
            last_reviewed_at=None, created_at=now, updated_at=now,
        )
        await self._store.create_section(section)
        for c in concepts:
            canon = canonicalize(c)
            if not await self._store.get_concept(canon, project.id):
                await self._store.create_concept(Concept(
                    name=c, canonical_name=canon, established_in_section=sid,
                    project_id=project.id, domain=None,
                ))
        await self._store.update_insight(
            insight.id, status=InsightStatus.INCORPORATED,
            incorporated_at=now, incorporated_in_section=sid,
        )

    async def address_annotation(self, annotation: Annotation, project: Project) -> None:
        """Address a user annotation. Skips protected sections."""
        if annotation.type == AnnotationType.PROTECTED:
            return
        if annotation.section_id:
            protected = await AnnotationHandler(self._store, self._event_bus).get_protected_sections(project.id)
            if annotation.section_id in protected:
                return
        section = await self._store.get_section(annotation.section_id) if annotation.section_id else None
        if not section:
            return

        context = await self._ctx.build_for_annotation(annotation, project)
        revision_resp = await self._llm.send("claude", REVISE_SECTION.format(
            goal=project.goal, section_title=section.title,
            section_content=section.content,
            annotation_content=annotation.content, context=context,
        ))
        result = await self._consensus.run(task=None, backends=None)
        now = datetime.now(timezone.utc)

        if result.verdict == Verdict.ACCEPT:
            await self._store.update_section(section.id, content=revision_resp.text)
            await self._store.update_annotation(
                annotation.id, status=AnnotationStatus.RESOLVED,
                resolved_at=now, attempt_count=annotation.attempt_count + 1,
                last_attempted_at=now,
            )
        else:
            new_att = annotation.attempt_count + 1
            status = (AnnotationStatus.NEEDS_HUMAN_REVIEW if new_att >= self._max_ann_attempts()
                      else AnnotationStatus.ATTEMPTED)
            await self._store.update_annotation(
                annotation.id, status=status, attempt_count=new_att, last_attempted_at=now,
            )


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"is_duplicate": False, "prerequisites": [], "concepts": []}
