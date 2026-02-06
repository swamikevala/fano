"""AnnotationHandler — event handler for user annotations.

Listens for user.annotation.created events and creates annotations in the store.
Also provides utility to query protected sections.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    Event,
)

if TYPE_CHECKING:
    from shared.models import EventBusInterface, StateStoreInterface


class AnnotationHandler:
    """Handles annotation lifecycle events."""

    def __init__(self, store: StateStoreInterface, event_bus: EventBusInterface) -> None:
        self._store = store
        self._event_bus = event_bus

    async def handle_new_annotation(self, event: Event) -> None:
        """Event handler for user.annotation.created.

        Creates annotation in store. If type=COMMENT, marks as highest priority.
        """
        p = event.payload
        now = datetime.now(timezone.utc)
        annotation = Annotation(
            id=p["annotation_id"],
            project_id=p["project_id"],
            type=AnnotationType(p.get("type", "comment")),
            section_id=p.get("section_id"),
            content=p.get("content", ""),
            status=AnnotationStatus.OPEN,
            attempt_count=0,
            last_attempted_at=None,
            created_at=now,
            resolved_at=None,
        )
        await self._store.create_annotation(annotation)

    async def get_protected_sections(self, project_id: str) -> set[str]:
        """Return section_ids that have an active PROTECTED annotation."""
        annotations = await self._store.list_annotations(
            project_id, status=AnnotationStatus.OPEN,
        )
        return {
            a.section_id
            for a in annotations
            if a.type == AnnotationType.PROTECTED and a.section_id
        }
