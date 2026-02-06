"""Annotations blueprint — CRUD for annotations (v2 API)."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, request

from control.async_utils import run_async
from shared.models import (
    Annotation,
    AnnotationStatus,
    AnnotationType,
    generate_id,
)

from .helpers import (
    err,
    get_store,
    ok,
    publish_event,
    serialize,
)

bp = Blueprint("annotations_v2", __name__, url_prefix="/api")


@bp.route("/annotations", methods=["GET"])
def list_annotations():
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")

    store = get_store()
    status_filter = request.args.get("status")
    ans = None
    if status_filter:
        try:
            ans = AnnotationStatus(status_filter)
        except ValueError:
            return err(f"Invalid status: {status_filter}", "VALIDATION_ERROR")

    annotations = run_async(store.list_annotations(project_id, status=ans))
    return ok([serialize(a) for a in annotations])


@bp.route("/annotations/<annotation_id>", methods=["GET"])
def get_annotation(annotation_id: str):
    store = get_store()
    annotation = run_async(store.get_annotation(annotation_id))
    if annotation is None:
        return err("Annotation not found", "NOT_FOUND", 404)
    return ok(serialize(annotation))


@bp.route("/annotations", methods=["POST"])
def create_annotation():
    body = request.get_json(silent=True) or {}
    project_id = body.get("project_id")
    content = body.get("content", "")
    if not project_id:
        return err("project_id is required", "VALIDATION_ERROR")

    type_str = body.get("type", "comment")
    try:
        ann_type = AnnotationType(type_str)
    except ValueError:
        return err(f"Invalid annotation type: {type_str}", "VALIDATION_ERROR")

    now = datetime.now(timezone.utc)
    annotation = Annotation(
        id=generate_id(),
        project_id=project_id,
        type=ann_type,
        section_id=body.get("section_id"),
        content=content,
        status=AnnotationStatus.OPEN,
        attempt_count=0,
        last_attempted_at=None,
        created_at=now,
        resolved_at=None,
    )
    store = get_store()
    run_async(store.create_annotation(annotation))

    publish_event("user.annotation.created", {
        "annotation_id": annotation.id,
        "project_id": project_id,
        "type": type_str,
    })

    return ok(serialize(annotation), status=201)
