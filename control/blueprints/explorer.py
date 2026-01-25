"""
Explorer Blueprint - Insights management routes.
"""

import json
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from explorer.src.models.axiom import AxiomStore, SeedAphorism

from ..services import (
    EXPLORER_DATA_DIR,
    get_explorer_stats,
    get_insight_by_id,
    get_insights_by_status,
    get_review_for_insight,
    load_insight_json,
)

# Initialize AxiomStore for seeds management with thread-safe lazy initialization
_axiom_store: AxiomStore | None = None
_axiom_store_lock = threading.Lock()


def get_axiom_store() -> AxiomStore:
    """Get or create the AxiomStore instance (thread-safe)."""
    global _axiom_store
    if _axiom_store is None:
        with _axiom_store_lock:
            # Double-check after acquiring lock
            if _axiom_store is None:
                _axiom_store = AxiomStore(EXPLORER_DATA_DIR)
    return _axiom_store

bp = Blueprint("explorer", __name__, url_prefix="/api/explorer")


@bp.route("/stats")
def api_explorer_stats():
    """Get explorer insight stats."""
    return jsonify(get_explorer_stats())


@bp.route("/pipeline")
def api_explorer_pipeline():
    """Get pipeline status showing counts at each lifecycle stage."""
    from datetime import datetime

    explorations_dir = EXPLORER_DATA_DIR / "explorations"

    # Thread counts by status
    thread_counts = {
        "active": 0,
        "chunk_ready": 0,
        "archived": 0,
    }

    # Track threads needing extraction (chunk_ready but not extracted)
    awaiting_extraction = 0

    if explorations_dir.exists():
        for json_file in explorations_dir.glob("*.json"):
            try:
                data = load_insight_json(json_file)
                if data:
                    status = data.get("status", "active")
                    if status in thread_counts:
                        thread_counts[status] += 1
                    if status == "chunk_ready" and not data.get("chunks_extracted", False):
                        awaiting_extraction += 1
            except Exception:
                pass

    # Insight counts by status
    insight_counts = {
        "pending": 0,
        "blessed": 0,
        "interesting": 0,
        "rejected": 0,
        "disputed": 0,
        "reviewing": 0,
    }
    insights_dir = EXPLORER_DATA_DIR / "chunks" / "insights"
    for status in insight_counts:
        status_dir = insights_dir / status
        if status_dir.exists():
            insight_counts[status] = len(list(status_dir.glob("*.json")))

    # Also check reviews/disputed for disputed reviews
    disputed_reviews_dir = EXPLORER_DATA_DIR / "reviews" / "disputed"
    if disputed_reviews_dir.exists():
        insight_counts["disputed"] = len(list(disputed_reviews_dir.glob("*.json")))

    # Get seed count
    store = get_axiom_store()
    seed_count = len(store.get_seed_aphorisms())

    return jsonify({
        "seeds": seed_count,
        "threads": thread_counts,
        "awaiting_extraction": awaiting_extraction,
        "insights": insight_counts,
        "pipeline": [
            {"stage": "seeds", "label": "Seeds", "count": seed_count},
            {"stage": "exploring", "label": "Exploring", "count": thread_counts["active"]},
            {"stage": "synthesizing", "label": "Synthesizing", "count": awaiting_extraction},
            {"stage": "pending", "label": "Pending Review", "count": insight_counts["pending"]},
            {"stage": "blessed", "label": "Blessed", "count": insight_counts["blessed"]},
        ],
    })


@bp.route("/threads")
def api_explorer_all_threads():
    """Get all exploration threads with optional status filter."""
    from datetime import datetime

    status_filter = request.args.get("status")  # active, chunk_ready, archived, or None for all

    threads = []
    explorations_dir = EXPLORER_DATA_DIR / "explorations"

    if explorations_dir.exists():
        for json_file in explorations_dir.glob("*.json"):
            try:
                data = load_insight_json(json_file)
                if not data:
                    continue

                thread_status = data.get("status", "active")

                # Apply filter if specified
                if status_filter and thread_status != status_filter:
                    continue

                exchanges = data.get("exchanges", [])
                last_exchange = exchanges[-1] if exchanges else None

                # Determine current activity based on status
                if thread_status == "active":
                    if not exchanges:
                        current_activity = "waiting to start"
                    elif last_exchange.get("role") == "critic":
                        current_activity = "needs exploration"
                    else:
                        current_activity = "needs critique"
                elif thread_status == "chunk_ready":
                    if data.get("chunks_extracted", False):
                        current_activity = "awaiting review"
                    else:
                        current_activity = "awaiting extraction"
                else:
                    current_activity = "completed"

                # Get focus seed info
                focus_info = None
                if data.get("primary_question_id"):
                    focus_info = f"Question: {data['primary_question_id']}"
                elif data.get("related_conjecture_ids"):
                    focus_info = f"Conjecture: {data['related_conjecture_ids'][0]}"

                # Calculate time since last activity
                updated_at = data.get("updated_at", data.get("created_at"))
                time_ago = ""
                if updated_at:
                    try:
                        updated = datetime.fromisoformat(updated_at)
                        delta = datetime.now() - updated
                        if delta.total_seconds() < 60:
                            time_ago = f"{int(delta.total_seconds())}s ago"
                        elif delta.total_seconds() < 3600:
                            time_ago = f"{int(delta.total_seconds() / 60)}m ago"
                        elif delta.total_seconds() < 86400:
                            time_ago = f"{int(delta.total_seconds() / 3600)}h ago"
                        else:
                            time_ago = f"{int(delta.total_seconds() / 86400)}d ago"
                    except Exception:
                        pass

                threads.append({
                    "id": data.get("id", json_file.stem),
                    "topic": data.get("topic", "Unknown"),
                    "status": thread_status,
                    "focus": focus_info,
                    "priority": data.get("priority", 5),
                    "exchange_count": len(exchanges),
                    "current_activity": current_activity,
                    "last_model": last_exchange.get("model") if last_exchange else None,
                    "last_role": last_exchange.get("role") if last_exchange else None,
                    "time_ago": time_ago,
                    "updated_at": updated_at,
                    "created_at": data.get("created_at", ""),
                    "chunks_extracted": data.get("chunks_extracted", False),
                })
            except Exception:
                pass

    # Sort by updated_at (most recent first)
    threads.sort(key=lambda t: t.get("updated_at") or "", reverse=True)

    return jsonify({"threads": threads, "count": len(threads), "filter": status_filter})


@bp.route("/threads/active")
def api_explorer_active_threads():
    """Get active exploration threads with their current state."""
    from datetime import datetime

    threads = []
    explorations_dir = EXPLORER_DATA_DIR / "explorations"

    if explorations_dir.exists():
        for json_file in explorations_dir.glob("*.json"):
            try:
                data = load_insight_json(json_file)
                if data and data.get("status") == "active":
                    # Determine current activity
                    exchanges = data.get("exchanges", [])
                    last_exchange = exchanges[-1] if exchanges else None

                    if not exchanges:
                        current_activity = "waiting to start"
                    elif last_exchange.get("role") == "critic":
                        current_activity = "needs exploration"
                    else:
                        current_activity = "needs critique"

                    # Get focus seed info
                    focus_info = None
                    if data.get("primary_question_id"):
                        focus_info = f"Question: {data['primary_question_id']}"
                    elif data.get("related_conjecture_ids"):
                        focus_info = f"Conjecture: {data['related_conjecture_ids'][0]}"

                    # Calculate time since last activity
                    updated_at = data.get("updated_at", data.get("created_at"))
                    time_ago = ""
                    if updated_at:
                        try:
                            updated = datetime.fromisoformat(updated_at)
                            delta = datetime.now() - updated
                            if delta.total_seconds() < 60:
                                time_ago = f"{int(delta.total_seconds())}s ago"
                            elif delta.total_seconds() < 3600:
                                time_ago = f"{int(delta.total_seconds() / 60)}m ago"
                            else:
                                time_ago = f"{int(delta.total_seconds() / 3600)}h ago"
                        except Exception:
                            pass

                    threads.append({
                        "id": data.get("id", json_file.stem),
                        "topic": data.get("topic", "Unknown"),
                        "focus": focus_info,
                        "priority": data.get("priority", 5),
                        "exchange_count": len(exchanges),
                        "current_activity": current_activity,
                        "last_model": last_exchange.get("model") if last_exchange else None,
                        "last_role": last_exchange.get("role") if last_exchange else None,
                        "time_ago": time_ago,
                        "updated_at": updated_at,
                    })
            except Exception as e:
                pass

    # Sort by updated_at (most recent first)
    threads.sort(key=lambda t: t.get("updated_at") or "", reverse=True)

    return jsonify({"threads": threads, "count": len(threads)})


@bp.route("/threads/<thread_id>")
def api_explorer_thread_detail(thread_id: str):
    """Get full details of an exploration thread including all exchanges."""
    explorations_dir = EXPLORER_DATA_DIR / "explorations"
    thread_path = explorations_dir / f"{thread_id}.json"

    if not thread_path.exists():
        return jsonify({"error": "Thread not found"}), 404

    data = load_insight_json(thread_path)
    if not data:
        return jsonify({"error": "Failed to load thread"}), 500

    # Get the focus seed text if available
    focus_seed_text = None
    focus_seed_id = data.get("primary_question_id")
    if not focus_seed_id and data.get("related_conjecture_ids"):
        focus_seed_id = data["related_conjecture_ids"][0]

    if focus_seed_id:
        store = get_axiom_store()
        seed = store.get_seed_by_id(focus_seed_id)
        if seed:
            focus_seed_text = seed.text

    # Format exchanges for display
    exchanges = []
    for i, ex in enumerate(data.get("exchanges", [])):
        exchanges.append({
            "index": i,
            "id": ex.get("id", f"ex-{i}"),
            "role": ex.get("role", "unknown"),
            "model": ex.get("model", "unknown"),
            "prompt": ex.get("prompt", ""),
            "response": ex.get("response", ""),
            "timestamp": ex.get("timestamp", ""),
            "deep_mode_used": ex.get("deep_mode_used", False),
        })

    # Calculate progress toward synthesis
    exchange_count = len(exchanges)
    min_exchanges = 4  # From config
    max_exchanges = 12
    critique_count = sum(1 for ex in exchanges if ex["role"] == "critic")

    progress = {
        "exchange_count": exchange_count,
        "min_required": min_exchanges,
        "max_allowed": max_exchanges,
        "critique_count": critique_count,
        "critiques_required": 2,
        "ready_for_synthesis": exchange_count >= min_exchanges and critique_count >= 2,
        "percent": min(100, int((exchange_count / min_exchanges) * 100)),
    }

    return jsonify({
        "thread": {
            "id": data.get("id", thread_id),
            "topic": data.get("topic", "Unknown"),
            "status": data.get("status", "unknown"),
            "priority": data.get("priority", 5),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "seed_axioms": data.get("seed_axioms", []),
            "primary_question_id": data.get("primary_question_id"),
            "related_conjecture_ids": data.get("related_conjecture_ids", []),
            "focus_seed_text": focus_seed_text,
            "notes": data.get("notes", ""),
            "chunks_extracted": data.get("chunks_extracted", False),
        },
        "exchanges": exchanges,
        "progress": progress,
    })


@bp.route("/insights/<status>")
def api_explorer_insights(status: str):
    """Get insights by status."""
    valid_statuses = ["pending", "blessed", "interesting", "rejected", "reviewing", "disputed"]
    if status not in valid_statuses:
        return jsonify({"error": f"Invalid status: {status}"}), 400

    if status == "reviewing":
        # Get insights currently being reviewed
        reviewing_dir = EXPLORER_DATA_DIR / "chunks" / "reviewing"
        insights = []
        if reviewing_dir.exists():
            for json_file in reviewing_dir.glob("*.json"):
                data = load_insight_json(json_file)
                if data:
                    review = get_review_for_insight(data.get("id", ""))
                    insights.append({
                        "insight": data,
                        "review": review,
                        "rounds_completed": len(review.get("rounds", [])) if review else 0,
                    })
        return jsonify({"insights": insights, "status": status})

    elif status == "disputed":
        # Get disputed reviews
        disputed_dir = EXPLORER_DATA_DIR / "reviews" / "disputed"
        insights = []
        if disputed_dir.exists():
            for json_file in disputed_dir.glob("*.json"):
                review = load_insight_json(json_file)
                if review:
                    insight_id = review.get("chunk_id", "")
                    insight_data, _ = get_insight_by_id(insight_id)
                    if insight_data:
                        insights.append({
                            "insight": insight_data,
                            "review": review,
                        })
        return jsonify({"insights": insights, "status": status})

    else:
        # Regular status
        raw_insights = get_insights_by_status(status)
        insights = []
        for data in raw_insights:
            review = get_review_for_insight(data.get("id", ""))
            insights.append({
                "insight": data,
                "review": review,
            })
        return jsonify({"insights": insights, "status": status})


@bp.route("/insight/<insight_id>")
def api_explorer_insight(insight_id: str):
    """Get a specific insight."""
    insight, status = get_insight_by_id(insight_id)
    if not insight:
        return jsonify({"error": "Insight not found"}), 404

    review = get_review_for_insight(insight_id)
    return jsonify({
        "insight": insight,
        "review": review,
        "status": status,
    })


@bp.route("/feedback", methods=["POST"])
def api_explorer_feedback():
    """Submit feedback for an insight."""
    data = request.json
    insight_id = data.get("insight_id")
    feedback = data.get("feedback")  # "bless", "interesting", "reject"
    notes = data.get("notes", "")

    if not insight_id or not feedback:
        return jsonify({"error": "Missing insight_id or feedback"}), 400

    # Map feedback to rating and new status
    rating_map = {"bless": "⚡", "interesting": "?", "reject": "✗"}
    status_map = {"bless": "blessed", "interesting": "interesting", "reject": "rejected"}

    if feedback not in rating_map:
        return jsonify({"error": "Invalid feedback value"}), 400

    insight_data, current_status = get_insight_by_id(insight_id)
    if not insight_data:
        return jsonify({"error": "Insight not found"}), 404

    # Update the insight data
    insight_data["rating"] = rating_map[feedback]
    insight_data["status"] = status_map[feedback]
    insight_data["reviewed_at"] = datetime.now().isoformat()
    if notes:
        insight_data["review_notes"] = f"Manual review: {notes}"

    new_status = status_map[feedback]

    # Save to new location
    new_dir = EXPLORER_DATA_DIR / "chunks" / "insights" / new_status
    new_dir.mkdir(parents=True, exist_ok=True)
    new_path = new_dir / f"{insight_id}.json"

    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(insight_data, f, indent=2, default=str)

    # Remove from old location
    old_path = EXPLORER_DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.json"
    if old_path.exists():
        old_path.unlink()
    old_md = EXPLORER_DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.md"
    if old_md.exists():
        old_md.unlink()

    return jsonify({"success": True, "new_status": new_status})


@bp.route("/priority", methods=["POST"])
def api_explorer_priority():
    """Update insight priority."""
    data = request.json
    insight_id = data.get("insight_id")
    priority = data.get("priority")

    if not insight_id or priority is None:
        return jsonify({"error": "Missing insight_id or priority"}), 400

    try:
        priority = int(priority)
        if not 1 <= priority <= 10:
            return jsonify({"error": "Priority must be between 1 and 10"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid priority value"}), 400

    insight_data, status = get_insight_by_id(insight_id)
    if not insight_data:
        return jsonify({"error": "Insight not found"}), 404

    # Update priority
    insight_data["priority"] = priority

    # Save in place
    json_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(insight_data, f, indent=2, default=str)

    return jsonify({"success": True, "priority": priority})


# ============ Seeds API ============


@bp.route("/seeds")
def api_get_seeds():
    """Get all seeds/axioms."""
    store = get_axiom_store()
    type_filter = request.args.get("type")
    sort_by_priority = request.args.get("sort", "true").lower() == "true"

    seeds = store.get_seed_aphorisms(type_filter=type_filter, sort_by_priority=sort_by_priority)

    # Convert to dict for JSON serialization
    seeds_data = [asdict(seed) for seed in seeds]

    # Get counts by type
    all_seeds = store.get_seed_aphorisms(sort_by_priority=False)
    counts = {
        "total": len(all_seeds),
        "axiom": len([s for s in all_seeds if s.type == "axiom"]),
        "conjecture": len([s for s in all_seeds if s.type == "conjecture"]),
        "question": len([s for s in all_seeds if s.type == "question"]),
    }

    return jsonify({"seeds": seeds_data, "counts": counts})


@bp.route("/seed/<seed_id>")
def api_get_seed(seed_id: str):
    """Get a specific seed by ID."""
    store = get_axiom_store()
    seed = store.get_seed_by_id(seed_id)

    if not seed:
        return jsonify({"error": "Seed not found"}), 404

    return jsonify({"seed": asdict(seed)})


@bp.route("/seed", methods=["POST"])
def api_create_seed():
    """Create a new seed."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Seed text is required"}), 400

    # Generate a unique ID
    seed_id = f"seed-{uuid.uuid4().hex[:8]}"

    # Parse priority
    priority = data.get("priority", 5)
    if isinstance(priority, str):
        priority_map = {"high": 8, "medium": 5, "low": 2}
        priority = priority_map.get(priority.lower(), 5)
    try:
        priority = max(1, min(10, int(priority)))
    except (ValueError, TypeError):
        priority = 5

    # Parse tags
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    seed = SeedAphorism(
        id=seed_id,
        text=text,
        type=data.get("type", "conjecture"),
        priority=priority,
        tags=tags,
        confidence=data.get("confidence", "high"),
        source=data.get("source", "user"),
        notes=data.get("notes", ""),
    )

    store = get_axiom_store()
    store.add_seed(seed)

    return jsonify({"success": True, "seed": asdict(seed)})


@bp.route("/seed/<seed_id>", methods=["PUT"])
def api_update_seed(seed_id: str):
    """Update an existing seed."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    store = get_axiom_store()

    # Check if seed exists
    existing = store.get_seed_by_id(seed_id)
    if not existing:
        return jsonify({"error": "Seed not found"}), 404

    # Parse tags if provided as string
    if "tags" in data and isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",") if t.strip()]

    success = store.update_seed(seed_id, data)
    if success:
        updated_seed = store.get_seed_by_id(seed_id)
        return jsonify({"success": True, "seed": asdict(updated_seed)})
    else:
        return jsonify({"error": "Failed to update seed"}), 500


@bp.route("/seed/<seed_id>", methods=["DELETE"])
def api_delete_seed(seed_id: str):
    """Delete a seed."""
    store = get_axiom_store()

    success = store.delete_seed(seed_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Seed not found"}), 404


@bp.route("/seed/<seed_id>/priority", methods=["POST"])
def api_update_seed_priority(seed_id: str):
    """Update seed priority."""
    data = request.json
    priority = data.get("priority")

    if priority is None:
        return jsonify({"error": "Missing priority"}), 400

    try:
        priority = int(priority)
        if not 1 <= priority <= 10:
            return jsonify({"error": "Priority must be between 1 and 10"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid priority value"}), 400

    store = get_axiom_store()
    success = store.update_seed(seed_id, {"priority": priority})

    if success:
        return jsonify({"success": True, "priority": priority})
    else:
        return jsonify({"error": "Seed not found"}), 404


# ============ Seed Images API ============

# Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def is_allowed_image(filename: str) -> bool:
    """Check if the file extension is an allowed image type."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


@bp.route("/seed/<seed_id>/image", methods=["POST"])
def api_upload_seed_image(seed_id: str):
    """
    Upload an image to attach to a seed.

    Accepts multipart form data with an 'image' file field.
    """
    store = get_axiom_store()

    # Check if seed exists
    seed = store.get_seed_by_id(seed_id)
    if not seed:
        return jsonify({"error": "Seed not found"}), 404

    # Check if file was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    if not is_allowed_image(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        }), 400

    # Sanitize filename - keep only alphanumeric, dots, underscores, hyphens
    original_name = file.filename
    safe_name = "".join(
        c for c in original_name
        if c.isalnum() or c in "._-"
    )
    if not safe_name:
        safe_name = f"image{Path(original_name).suffix}"

    # Read file data
    image_data = file.read()

    # Add image to seed
    stored_filename = store.add_seed_image(seed_id, safe_name, image_data)

    if stored_filename:
        updated_seed = store.get_seed_by_id(seed_id)
        return jsonify({
            "success": True,
            "filename": stored_filename,
            "seed": asdict(updated_seed),
        })
    else:
        return jsonify({"error": "Failed to save image"}), 500


@bp.route("/seed/<seed_id>/image/<filename>", methods=["GET"])
def api_get_seed_image(seed_id: str, filename: str):
    """
    Serve a seed image file.

    The filename should be the stored filename (seed_id_originalname.ext).
    """
    store = get_axiom_store()

    # Check if seed exists
    seed = store.get_seed_by_id(seed_id)
    if not seed:
        return jsonify({"error": "Seed not found"}), 404

    # Verify the image belongs to this seed
    if filename not in seed.images:
        return jsonify({"error": "Image not found for this seed"}), 404

    # Get the image path
    image_path = store.images_dir / filename
    if not image_path.exists():
        return jsonify({"error": "Image file not found"}), 404

    # Determine MIME type from extension
    ext = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mimetype = mime_types.get(ext, "application/octet-stream")

    return send_file(image_path, mimetype=mimetype)


@bp.route("/seed/<seed_id>/image/<filename>", methods=["DELETE"])
def api_delete_seed_image(seed_id: str, filename: str):
    """
    Delete an image from a seed.

    The filename should be the stored filename (seed_id_originalname.ext).
    """
    store = get_axiom_store()

    # Check if seed exists
    seed = store.get_seed_by_id(seed_id)
    if not seed:
        return jsonify({"error": "Seed not found"}), 404

    # Remove the image
    success = store.remove_seed_image(seed_id, filename)

    if success:
        updated_seed = store.get_seed_by_id(seed_id)
        return jsonify({
            "success": True,
            "seed": asdict(updated_seed),
        })
    else:
        return jsonify({"error": "Image not found"}), 404
