"""Insights blueprint — read insights, add comments (v2 API)."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, request

from control.async_utils import run_async
from shared.models import (
    CommentType,
    InsightComment,
    InsightStatus,
    generate_id,
)

from .helpers import (
    err,
    get_pagination,
    get_store,
    ok,
    ok_list,
    paginate,
    publish_event,
    serialize,
)

bp = Blueprint("insights_v2", __name__, url_prefix="/api")


# Map comment type to event topic
_COMMENT_TOPIC_MAP = {
    CommentType.ENDORSE: "user.insight.endorsed",
    CommentType.DISMISS: "user.insight.dismissed",
    CommentType.RECONSIDER: "user.insight.reconsidered",
    CommentType.GENERAL: "user.insight.commented",
}


@bp.route("/insights", methods=["GET"])
def list_insights():
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")

    store = get_store()
    status_filter = request.args.get("status")
    iss = None
    if status_filter:
        try:
            iss = InsightStatus(status_filter)
        except ValueError:
            return err(f"Invalid status: {status_filter}", "VALIDATION_ERROR")

    insights = run_async(store.list_insights(project_id, status=iss))
    total = len(insights)
    offset, limit = get_pagination()
    page = paginate(insights, offset, limit)
    return ok_list([serialize(i) for i in page], total, offset, limit)


@bp.route("/insights/<insight_id>", methods=["GET"])
def get_insight(insight_id: str):
    store = get_store()
    insight = run_async(store.get_insight(insight_id))
    if insight is None:
        return err("Insight not found", "NOT_FOUND", 404)
    return ok(serialize(insight))


@bp.route("/insights/<insight_id>/comment", methods=["POST"])
def add_comment(insight_id: str):
    store = get_store()
    insight = run_async(store.get_insight(insight_id))
    if insight is None:
        return err("Insight not found", "NOT_FOUND", 404)

    body = request.get_json(silent=True) or {}
    type_str = body.get("type", "general")
    try:
        comment_type = CommentType(type_str)
    except ValueError:
        return err(f"Invalid comment type: {type_str}", "VALIDATION_ERROR")

    content = body.get("content", "")

    comment = InsightComment(
        id=generate_id(),
        insight_id=insight_id,
        comment_type=comment_type,
        content=content or None,
        created_at=datetime.now(timezone.utc),
    )
    run_async(store.create_insight_comment(comment))

    # Publish the appropriate event
    topic = _COMMENT_TOPIC_MAP.get(comment_type, "user.insight.commented")
    publish_event(topic, {
        "insight_id": insight_id,
        "comment_id": comment.id,
        "comment_type": type_str,
        "project_id": insight.project_id,
    })

    return ok(serialize(comment), status=201)
