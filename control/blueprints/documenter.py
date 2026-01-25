"""
Documenter Blueprint - Document operations, versions, and formatting routes.
"""

import json
import os
import re
import threading

import requests
from flask import Blueprint, current_app, jsonify, request

from ..services import DOC_PATH, FANO_ROOT, LOGS_DIR
from ..services.config import load_config

bp = Blueprint("documenter", __name__, url_prefix="/api")

# Default orchestrator settings
_DEFAULT_ORCHESTRATOR_HOST = "127.0.0.1"
_DEFAULT_ORCHESTRATOR_PORT = 9001


def get_orchestrator_url() -> str:
    """Get orchestrator URL from config or use defaults."""
    config = load_config()
    host = config.get("orchestrator", {}).get("host", _DEFAULT_ORCHESTRATOR_HOST)
    port = config.get("orchestrator", {}).get("port", _DEFAULT_ORCHESTRATOR_PORT)
    return f"http://{host}:{port}"


# Lock for formatting fix state thread safety
_formatting_lock = threading.Lock()


def get_process_manager():
    """Get the ProcessManager from the app context."""
    return current_app.config.get("process_manager")


def get_formatting_state():
    """Get or create formatting fix state in app context."""
    if "formatting_fix_state" not in current_app.config:
        current_app.config["formatting_fix_state"] = {
            "in_progress": False,
            "progress": None,
            "result": None,
            "sections_fixed": 0,
        }
    return current_app.config["formatting_fix_state"]


@bp.route("/document")
def api_document():
    """Get the current document content."""
    if not DOC_PATH.exists():
        return jsonify({"error": "Document not found", "content": None})

    try:
        content = DOC_PATH.read_text(encoding="utf-8")
        # Count sections (## headers)
        sections = len([l for l in content.split("\n") if l.startswith("## ")])
        return jsonify({
            "content": content,
            "sections": sections,
            "length": len(content),
            "path": str(DOC_PATH),
        })
    except Exception as e:
        return jsonify({"error": str(e), "content": None})


@bp.route("/documenter/activity")
def api_documenter_activity():
    """Get recent documenter activity."""
    log_file = LOGS_DIR / "documenter.jsonl"

    # Check if documenter is actually running
    pm = get_process_manager()
    documenter_running = pm.is_running("documenter") if pm else False

    activity = {
        "is_running": documenter_running,
        "current_task": None,
        "recent_events": [],
        "stats": {
            "sections_added": 0,
            "consensus_calls": 0,
            "opportunities_processed": 0,
        }
    }

    if not log_file.exists():
        return jsonify(activity)

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[-50:]  # Last 50 entries

        for line in reversed(lines):
            try:
                entry = json.loads(line)
                event_type = entry.get("event_type", "")

                # Get current task - only if documenter is running
                if documenter_running and activity["current_task"] is None:
                    if "opportunity.working" in event_type:
                        activity["current_task"] = {
                            "type": "processing_insight",
                            "insight_id": entry.get("insight_id"),
                            "started_at": entry.get("timestamp"),
                        }
                    elif "consensus.run_start" in event_type:
                        activity["current_task"] = {
                            "type": "consensus",
                            "started_at": entry.get("timestamp"),
                        }

                # Collect recent events
                if len(activity["recent_events"]) < 10:
                    activity["recent_events"].append({
                        "event": event_type,
                        "timestamp": entry.get("timestamp"),
                        "details": {k: v for k, v in entry.items()
                                   if k not in ["event_type", "timestamp", "level", "module", "component"]}
                    })

                # Collect stats from session summary
                if event_type == "documenter.session.summary":
                    activity["stats"]["sections_added"] = entry.get("sections_added", 0)
                    activity["stats"]["consensus_calls"] = entry.get("consensus_calls", 0)
                    activity["stats"]["opportunities_processed"] = entry.get("opportunities_processed", 0)

            except json.JSONDecodeError:
                pass

    except Exception as e:
        activity["error"] = str(e)

    return jsonify(activity)


@bp.route("/documenter/pipeline")
def api_documenter_pipeline():
    """Get documenter task pipeline grouped by insight.

    Shows all documenter tasks with their phases: Plan -> Draft -> Review.
    Tasks are grouped by insight_id for visualization.
    """
    try:
        # Fetch all documenter tasks from orchestrator
        orchestrator_url = get_orchestrator_url()
        resp = requests.get(
            f"{orchestrator_url}/tasks",
            params={"module": "documenter", "limit": "100"},
            timeout=5,
        )
        resp.raise_for_status()
        tasks = resp.json().get("tasks", [])
    except requests.RequestException as e:
        return jsonify({"error": f"Orchestrator not available: {e}", "pipelines": []}), 503

    # Group tasks by insight_id
    pipelines = {}

    for task in tasks:
        payload = task.get("payload", {})
        task_type = task.get("task_type", "")

        # Extract insight_id from payload (handle nested opportunity structure)
        insight_id = payload.get("insight_id")
        insight_text = payload.get("insight_text", "")

        # Check nested opportunity structure (legacy incorporate_insight tasks)
        opportunity = payload.get("opportunity", {})
        if not insight_id and opportunity:
            insight_id = opportunity.get("insight_id")
            insight_text = opportunity.get("text", "")

        if not insight_id:
            # Fallback: extract from key
            key = task.get("key", "")
            if ":" in key:
                insight_id = key.split(":")[-1]

        if not insight_id:
            insight_id = task.get("id", "unknown")[:8]

        # Initialize pipeline entry
        if insight_id not in pipelines:
            pipelines[insight_id] = {
                "insight_id": insight_id,
                "insight_text": insight_text[:100] if insight_text else "",
                "phases": {
                    "plan": None,
                    "draft": None,
                    "review": None,
                },
                "created_at": task.get("created_at"),
            }

        # Map task type to phase
        phase = None
        if "plan" in task_type:
            phase = "plan"
        elif "draft" in task_type:
            phase = "draft"
        elif "review" in task_type:
            phase = "review"
        elif task_type == "incorporate_insight":
            # Legacy single-phase task
            phase = "draft"

        if phase:
            pipelines[insight_id]["phases"][phase] = {
                "task_id": task.get("id"),
                "state": task.get("state", "pending"),
                "started_at": task.get("started_at"),
                "completed_at": task.get("completed_at"),
            }

    # Convert to list and sort by creation time (newest first)
    pipeline_list = list(pipelines.values())
    pipeline_list.sort(key=lambda p: p.get("created_at") or "", reverse=True)

    return jsonify({"pipelines": pipeline_list})


@bp.route("/document/versions")
def api_document_versions():
    """List document version history."""
    try:
        from documenter.versions import VersionManager

        vm = VersionManager(DOC_PATH)
        versions = vm.list_versions(limit=50)
        return jsonify({
            "versions": [v.to_dict() for v in versions],
            "total": len(vm.versions),
        })
    except Exception as e:
        return jsonify({"error": str(e), "versions": []}), 500


@bp.route("/document/versions/<version_id>")
def api_document_version(version_id):
    """Get a specific document version."""
    try:
        from documenter.versions import VersionManager

        vm = VersionManager(DOC_PATH)
        result = vm.get_version(version_id)
        if not result:
            return jsonify({"error": "Version not found"}), 404

        version, content = result
        return jsonify({
            "version": version.to_dict(),
            "content": content,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/document/versions/<version_id>/revert", methods=["POST"])
def api_document_revert(version_id):
    """Revert document to a specific version."""
    try:
        from documenter.versions import VersionManager

        vm = VersionManager(DOC_PATH)
        content = vm.revert_to(version_id)
        if content is None:
            return jsonify({"error": "Version not found"}), 404

        return jsonify({
            "status": "reverted",
            "version_id": version_id,
            "content_length": len(content),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _split_into_sections(content: str) -> list:
    """Split document into sections by ## headers. Returns [(section_name, section_content), ...]"""
    # Split on ## headers (level 2)
    pattern = r'^(## .+)$'
    parts = re.split(pattern, content, flags=re.MULTILINE)

    sections = []
    # First part is before any ## header (preamble)
    if parts[0].strip():
        sections.append(("Preamble", parts[0]))

    # Rest are header/content pairs
    for i in range(1, len(parts), 2):
        header = parts[i] if i < len(parts) else ""
        body = parts[i + 1] if i + 1 < len(parts) else ""
        section_name = header.replace("## ", "").strip()[:30]
        sections.append((section_name, header + body))

    return sections if sections else [("Document", content)]


def _run_formatting_fix(app):
    """Background worker for formatting fix."""
    with app.app_context():
        formatting_fix_state = get_formatting_state()

        try:
            from documenter.document import Document
            from documenter.formatting import detect_formatting_issues, build_formatting_fix_prompt

            # Load document
            doc = Document(DOC_PATH, enable_versioning=True)
            doc.load()

            # Check for issues first
            issues = detect_formatting_issues(doc.content)

            if not issues:
                formatting_fix_state["result"] = {"success": True, "issues_fixed": 0, "message": "No formatting issues found"}
                formatting_fix_state["in_progress"] = False
                formatting_fix_state["progress"] = None
                formatting_fix_state["sections_fixed"] = 0
                return

            # Set up API client
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                formatting_fix_state["result"] = {"success": False, "error": "ANTHROPIC_API_KEY not configured"}
                formatting_fix_state["in_progress"] = False
                formatting_fix_state["progress"] = None
                formatting_fix_state["sections_fixed"] = 0
                return

            client = anthropic.Anthropic(api_key=api_key)

            # Split document into sections
            sections = _split_into_sections(doc.content)
            total_sections = len(sections)
            fixed_sections = []
            total_issues_fixed = 0
            sections_with_fixes = 0
            formatting_fix_state["sections_fixed"] = 0

            for i, (section_name, section_content) in enumerate(sections):
                # Update progress
                formatting_fix_state["progress"] = {
                    "current": i + 1,
                    "total": total_sections,
                    "section": section_name,
                }

                # Check if this section has issues
                section_issues = detect_formatting_issues(section_content)

                if not section_issues:
                    # No issues in this section, keep as-is
                    fixed_sections.append(section_content)
                    continue

                # Fix this section
                prompt = build_formatting_fix_prompt(section_content)

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )

                fixed_content = response.content[0].text.strip()

                # Basic validation for this section
                original_word_count = len(section_content.split())
                fixed_word_count = len(fixed_content.split())

                if abs(original_word_count - fixed_word_count) > original_word_count * 0.3:
                    # Section fix seems wrong, keep original
                    print(f"Warning: Section '{section_name}' fix rejected (word count: {original_word_count} -> {fixed_word_count})")
                    fixed_sections.append(section_content)
                else:
                    fixed_sections.append(fixed_content)
                    total_issues_fixed += len(section_issues)
                    sections_with_fixes += 1

                    # Save incrementally after each fixed section
                    formatting_fix_state["sections_fixed"] = sections_with_fixes
                    doc.content = "\n".join(fixed_sections + [s for _, s in sections[i+1:]])
                    doc.save(f"Fixed formatting: section {i+1}/{total_sections}")

            formatting_fix_state["result"] = {
                "success": True,
                "issues_fixed": total_issues_fixed,
                "message": f"Fixed {sections_with_fixes} section(s) ({total_issues_fixed} issues) out of {total_sections} total",
            }
            formatting_fix_state["in_progress"] = False
            formatting_fix_state["progress"] = None
            formatting_fix_state["sections_fixed"] = 0

        except Exception as e:
            import traceback
            traceback.print_exc()
            formatting_fix_state["result"] = {"success": False, "error": str(e)}
            formatting_fix_state["in_progress"] = False
            formatting_fix_state["progress"] = None
            formatting_fix_state["sections_fixed"] = 0


@bp.route("/document/fix-formatting", methods=["POST"])
def api_document_fix_formatting():
    """Start fixing math formatting issues (runs in background)."""
    formatting_fix_state = get_formatting_state()

    # Use lock for thread-safe check-then-act
    with _formatting_lock:
        if formatting_fix_state["in_progress"]:
            return jsonify({"success": True, "status": "already_running"})

        # Start background thread
        formatting_fix_state["in_progress"] = True
        formatting_fix_state["result"] = None

    # Need to pass app to background thread for context
    app = current_app._get_current_object()
    thread = threading.Thread(target=_run_formatting_fix, args=(app,), daemon=True)
    thread.start()

    return jsonify({"success": True, "status": "started"})


@bp.route("/document/fix-formatting/status")
def api_document_fix_formatting_status():
    """Check formatting fix status (for polling)."""
    formatting_fix_state = get_formatting_state()

    # Use lock for thread-safe access
    with _formatting_lock:
        if formatting_fix_state["in_progress"]:
            response = {"status": "in_progress"}
            if formatting_fix_state["progress"]:
                response["progress"] = formatting_fix_state["progress"]
            response["sections_fixed"] = formatting_fix_state.get("sections_fixed", 0)
            return jsonify(response)
        elif formatting_fix_state["result"]:
            result = formatting_fix_state["result"]
            # Clear result after reading
            formatting_fix_state["result"] = None
            return jsonify({"status": "complete", **result})
        else:
            return jsonify({"status": "idle"})
