"""Planner — decides what work to do each documenter cycle.

Priority order:
    1. Open annotations (type=comment)
    2. Attested insights not yet incorporated
    3. Sections needing review
    4. Suggestions (type=suggestion)

Tracks actual LLM call ratio vs target (default 70% new / 30% review).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.models import (
    AnnotationStatus,
    AnnotationType,
    InsightStatus,
    WorkItem,
)

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import StateStoreInterface

# Priority levels (higher = do first)
PRIORITY_ANNOTATION = 100
PRIORITY_INSIGHT = 50
PRIORITY_REVIEW = 20
PRIORITY_SUGGESTION = 10


class Planner:
    """Plans documenter work cycles with ratio-aware allocation."""

    def __init__(self, store: StateStoreInterface, config: Config) -> None:
        self._store = store
        self._config = config
        self._new_calls: int = 0
        self._review_calls: int = 0

    def _target_new_ratio(self) -> float:
        return self._config.get("documenter.work_allocation.new_ratio", 0.7)

    def _current_new_ratio(self) -> float:
        total = self._new_calls + self._review_calls
        if total == 0:
            return self._target_new_ratio()
        return self._new_calls / total

    def record_call(self, category: str) -> None:
        """Track an LLM call for ratio balancing."""
        if category == "new":
            self._new_calls += 1
        else:
            self._review_calls += 1

    async def plan_cycle(self, project_id: str) -> list[WorkItem]:
        """Plan the next work cycle. Returns ordered list of WorkItems."""
        items: list[WorkItem] = []

        # 1. Open comment annotations (highest priority)
        open_annotations = await self._store.list_annotations(
            project_id, status=AnnotationStatus.OPEN,
        )
        for ann in open_annotations:
            if ann.type == AnnotationType.COMMENT:
                items.append(WorkItem(
                    type="annotation", entity_id=ann.id,
                    priority=PRIORITY_ANNOTATION,
                ))

        # 2. Attested insights not yet incorporated
        attested = await self._store.list_insights(
            project_id, status=InsightStatus.ATTESTED,
        )
        for ins in attested:
            items.append(WorkItem(
                type="insight", entity_id=ins.id,
                priority=PRIORITY_INSIGHT,
            ))

        # 3. Sections needing review
        sections = await self._store.list_sections(project_id)
        for sec in sections:
            items.append(WorkItem(
                type="review", entity_id=sec.id,
                priority=PRIORITY_REVIEW,
            ))

        # 4. Suggestion annotations
        for ann in open_annotations:
            if ann.type == AnnotationType.SUGGESTION:
                items.append(WorkItem(
                    type="suggestion", entity_id=ann.id,
                    priority=PRIORITY_SUGGESTION,
                ))

        # Rebalance: if drifting >10% from target, boost underrepresented
        target = self._target_new_ratio()
        current = self._current_new_ratio()
        if current > target + 0.1:
            # Too much new — boost review priorities
            items = [
                WorkItem(i.type, i.entity_id,
                         i.priority + 30 if i.type == "review" else i.priority)
                for i in items
            ]
        elif current < target - 0.1:
            # Too much review — boost insight priorities
            items = [
                WorkItem(i.type, i.entity_id,
                         i.priority + 30 if i.type == "insight" else i.priority)
                for i in items
            ]

        # Sort by priority descending
        items.sort(key=lambda w: w.priority, reverse=True)
        return items
