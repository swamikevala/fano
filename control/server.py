"""
Control Panel Server - Flask app for managing Fano components.
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress ResourceWarning about unclosed transports during shutdown
# These are harmless on Windows when child processes are being terminated
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")

# Suppress "Exception ignored in __del__" errors during interpreter shutdown on Windows
if sys.platform == "win32":
    def _quiet_unraisablehook(unraisable):
        # Suppress ValueError from closed pipes during shutdown
        if unraisable.exc_type is ValueError and "closed pipe" in str(unraisable.exc_value):
            return
        sys.__unraisablehook__(unraisable)

    sys.unraisablehook = _quiet_unraisablehook

import re
import yaml
from flask import Flask, jsonify, render_template, request, send_file
import markdown


class StatusLogFilter(logging.Filter):
    """Filter out frequent polling requests from Flask logs."""

    NOISY_ENDPOINTS = [
        "/api/status",
        "/health",
        "/api/documenter/activity",
        "/api/explorer/stats",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for endpoint in self.NOISY_ENDPOINTS:
            if endpoint in message:
                return False
        return True


class AsyncioNoiseFilter(logging.Filter):
    """Filter out noisy asyncio connection reset errors on Windows."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # Filter out Windows connection reset noise
        if "ConnectionResetError" in message or "_call_connection_lost" in message:
            return False
        return True


# Apply filters
logging.getLogger("werkzeug").addFilter(StatusLogFilter())
logging.getLogger("asyncio").addFilter(AsyncioNoiseFilter())

# Paths
FANO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = FANO_ROOT / "logs"
EXPLORER_DATA_DIR = FANO_ROOT / "explorer" / "data"

# Add explorer to path for imports
sys.path.insert(0, str(FANO_ROOT / "explorer" / "src"))


def render_markdown_content(text: str) -> str:
    """Convert markdown to HTML while preserving LaTeX math."""
    if not text:
        return ""
    # Protect LaTeX blocks from markdown processing
    math_blocks = []

    def save_math(match):
        math_blocks.append(match.group(0))
        return f"%%MATH_{len(math_blocks)-1}%%"

    # Protect display and inline math
    text = re.sub(r'\\\[[\s\S]*?\\\]', save_math, text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', save_math, text)
    text = re.sub(r'\\\(.*?\\\)', save_math, text)
    text = re.sub(r'\$[^\$\n]+\$', save_math, text)

    # Process markdown
    html = markdown.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])

    # Restore math blocks
    for i, math in enumerate(math_blocks):
        html = html.replace(f"%%MATH_{i}%%", math)

    return html


def get_explorer_stats() -> dict:
    """Get insight counts by status."""
    stats = {}
    for status in ["pending", "blessed", "interesting", "rejected"]:
        dir_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status
        stats[status] = len(list(dir_path.glob("*.json"))) if dir_path.exists() else 0

    # Count reviewing
    reviewing_dir = EXPLORER_DATA_DIR / "chunks" / "reviewing"
    stats["reviewing"] = len(list(reviewing_dir.glob("*.json"))) if reviewing_dir.exists() else 0

    # Count disputed
    disputed_dir = EXPLORER_DATA_DIR / "reviews" / "disputed"
    stats["disputed"] = len(list(disputed_dir.glob("*.json"))) if disputed_dir.exists() else 0

    return stats


def load_insight_json(json_path: Path) -> dict:
    """Load an insight from JSON file."""
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_insights_by_status(status: str) -> list:
    """Get all insights with a given status."""
    insights_dir = EXPLORER_DATA_DIR / "chunks" / "insights" / status
    insights = []

    if not insights_dir.exists():
        return insights

    for json_file in insights_dir.glob("*.json"):
        data = load_insight_json(json_file)
        if data:
            insights.append(data)

    # Sort by most recent
    insights.sort(key=lambda x: x.get("reviewed_at") or x.get("extracted_at", ""), reverse=True)
    return insights


def get_insight_by_id(insight_id: str) -> tuple:
    """Get a specific insight by ID, return (data, status)."""
    for status in ["pending", "blessed", "interesting", "rejected"]:
        json_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
        if json_path.exists():
            return load_insight_json(json_path), status
    return None, None


def get_review_for_insight(insight_id: str) -> dict:
    """Get review data for an insight."""
    for subdir in ["completed", "disputed"]:
        review_path = EXPLORER_DATA_DIR / "reviews" / subdir / f"{insight_id}.json"
        if review_path.exists():
            return load_insight_json(review_path)
    return None


# Global state
_processes = {
    "pool": None,
    "explorer": None,
    "documenter": None,
}
_process_logs = {
    "pool": [],
    "explorer": [],
    "documenter": [],
}


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def check_pool_health(host: str = "127.0.0.1", port: int = 9000) -> bool:
    """Check if pool is responding to health checks."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def register_pool_process(proc: subprocess.Popen):
    """Register an externally started pool process."""
    global _processes
    _processes["pool"] = proc


def load_config() -> dict:
    """Load the unified configuration."""
    config_path = FANO_ROOT / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


def create_app() -> Flask:
    """Create the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(FANO_ROOT / "control" / "templates"),
        static_folder=str(FANO_ROOT / "control" / "static"),
    )

    @app.route("/")
    def index():
        """Main dashboard."""
        config = load_config()
        return render_template("index.html", config=config)

    @app.route("/api/status")
    def api_status():
        """Get status of all components."""
        config = load_config()
        pool_host = config.get("llm", {}).get("pool", {}).get("host", "127.0.0.1")
        pool_port = config.get("llm", {}).get("pool", {}).get("port", 9000)

        # Check pool status - either via subprocess OR by checking if port is responding
        pool_proc_running = _processes.get("pool") is not None and _processes["pool"].poll() is None
        pool_responding = check_pool_health(pool_host, pool_port)
        pool_running = pool_proc_running or pool_responding

        # Check explorer status
        explorer_running = _processes.get("explorer") is not None and _processes["explorer"].poll() is None

        # Check documenter status
        documenter_running = _processes.get("documenter") is not None and _processes["documenter"].poll() is None

        # Get backend status
        backends = {}
        llm_config = config.get("llm", {}).get("backends", {})
        for name, backend_config in llm_config.items():
            backends[name] = {
                "enabled": backend_config.get("enabled", False),
                "type": backend_config.get("type", "unknown"),
                "available": False,  # Will be updated by pool status check
            }

        # Check pool API if running
        if pool_running:
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://{pool_host}:{pool_port}/status", timeout=2) as resp:
                    pool_status = json.loads(resp.read().decode())
                    # Update backend availability from pool status
                    if pool_status.get("gemini"):
                        backends.get("gemini", {})["available"] = pool_status["gemini"].get("available", False)
                        backends.get("gemini", {})["authenticated"] = pool_status["gemini"].get("authenticated", False)
                    if pool_status.get("chatgpt"):
                        backends.get("chatgpt", {})["available"] = pool_status["chatgpt"].get("available", False)
                        backends.get("chatgpt", {})["authenticated"] = pool_status["chatgpt"].get("authenticated", False)
                    if pool_status.get("claude"):
                        backends.get("claude", {})["available"] = pool_status["claude"].get("available", False)
                        backends.get("claude", {})["authenticated"] = pool_status["claude"].get("authenticated", False)
            except Exception:
                pass

        # Check API backends (Claude/OpenRouter can be available without pool)
        if os.environ.get("ANTHROPIC_API_KEY"):
            if "claude" in backends:
                backends["claude"]["available"] = True
        if os.environ.get("OPENROUTER_API_KEY"):
            if "openrouter" in backends:
                backends["openrouter"]["available"] = True

        # Get stats from logs
        stats = get_stats()

        return jsonify({
            "pool": {
                "running": pool_running,
                "pid": _processes["pool"].pid if pool_proc_running else None,
                "external": pool_responding and not pool_proc_running,
            },
            "explorer": {
                "running": explorer_running,
                "pid": _processes["explorer"].pid if explorer_running else None,
            },
            "documenter": {
                "running": documenter_running,
                "pid": _processes["documenter"].pid if documenter_running else None,
            },
            "backends": backends,
            "stats": stats,
        })

    @app.route("/api/start/<component>", methods=["POST"])
    def api_start(component: str):
        """Start a component."""
        if component not in ["pool", "explorer", "documenter"]:
            return jsonify({"error": f"Unknown component: {component}"}), 400

        config = load_config()

        # Check if already running
        if component == "pool":
            pool_host = config.get("llm", {}).get("pool", {}).get("host", "127.0.0.1")
            pool_port = config.get("llm", {}).get("pool", {}).get("port", 9000)
            if check_pool_health(pool_host, pool_port):
                return jsonify({"error": "Pool is already running (detected on port)", "already_running": True}), 400
        elif _processes.get(component) is not None and _processes[component].poll() is None:
            return jsonify({"error": f"{component} is already running"}), 400

        try:
            if component == "pool":
                _processes["pool"] = start_pool()
            elif component == "explorer":
                options = request.json or {}
                _processes["explorer"] = start_explorer(options.get("mode", "start"))
            elif component == "documenter":
                _processes["documenter"] = start_documenter()

            return jsonify({"status": "started", "component": component})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stop/<component>", methods=["POST"])
    def api_stop(component: str):
        """Stop a component."""
        if component not in ["pool", "explorer", "documenter"]:
            return jsonify({"error": f"Unknown component: {component}"}), 400

        # Special handling for pool - might be external
        if component == "pool":
            config = load_config()
            pool_host = config.get("llm", {}).get("pool", {}).get("host", "127.0.0.1")
            pool_port = config.get("llm", {}).get("pool", {}).get("port", 9000)

            proc = _processes.get("pool")
            proc_running = proc is not None and proc.poll() is None

            # If we have the process, terminate it directly
            if proc_running:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    _processes["pool"] = None
                    return jsonify({"status": "stopped", "component": "pool"})
                except Exception as e:
                    proc.kill()
                    _processes["pool"] = None
                    return jsonify({"status": "killed", "component": "pool", "error": str(e)})

            # Otherwise, try to call the shutdown endpoint (for external pool)
            if check_pool_health(pool_host, pool_port):
                try:
                    import urllib.request
                    req = urllib.request.Request(
                        f"http://{pool_host}:{pool_port}/shutdown",
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        result = json.loads(resp.read().decode())
                    return jsonify({"status": "shutdown_requested", "component": "pool", "result": result})
                except Exception as e:
                    return jsonify({"error": f"Failed to shutdown external pool: {e}"}), 500

            return jsonify({"error": "Pool is not running"}), 400

        # For other components, use normal process termination
        proc = _processes.get(component)
        if proc is None or proc.poll() is not None:
            return jsonify({"error": f"{component} is not running"}), 400

        try:
            proc.terminate()
            proc.wait(timeout=5)
            _processes[component] = None
            return jsonify({"status": "stopped", "component": component})
        except Exception as e:
            proc.kill()
            _processes[component] = None
            return jsonify({"status": "killed", "component": component, "error": str(e)})

    @app.route("/api/logs/<component>")
    def api_logs(component: str):
        """Get recent logs for a component."""
        if component not in ["pool", "explorer", "documenter", "llm"]:
            return jsonify({"error": f"Unknown component: {component}"}), 400

        log_file = LOGS_DIR / f"{component}.jsonl"
        if not log_file.exists():
            return jsonify({"logs": []})

        # Get last N lines
        limit = request.args.get("limit", 100, type=int)
        lines = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                for line in all_lines[-limit:]:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({"logs": lines})

    @app.route("/api/config")
    def api_config():
        """Get current configuration."""
        return jsonify(load_config())

    @app.route("/api/config", methods=["POST"])
    def api_update_config():
        """Update configuration."""
        try:
            new_config = request.json
            config_path = FANO_ROOT / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
            return jsonify({"status": "updated"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/documenter/activity")
    def api_documenter_activity():
        """Get recent documenter activity."""
        log_file = LOGS_DIR / "documenter.jsonl"

        # Check if documenter is actually running
        documenter_running = _processes.get("documenter") is not None and _processes["documenter"].poll() is None

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

    @app.route("/api/document")
    def api_document():
        """Get the current document content."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        if not doc_path.exists():
            return jsonify({"error": "Document not found", "content": None})

        try:
            content = doc_path.read_text(encoding="utf-8")
            # Count sections (## headers)
            sections = len([l for l in content.split("\n") if l.startswith("## ")])
            return jsonify({
                "content": content,
                "sections": sections,
                "length": len(content),
                "path": str(doc_path),
            })
        except Exception as e:
            return jsonify({"error": str(e), "content": None})

    @app.route("/api/document/versions")
    def api_document_versions():
        """List document version history."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            # Import here to avoid circular imports
            sys.path.insert(0, str(FANO_ROOT))
            from documenter.versions import VersionManager

            vm = VersionManager(doc_path)
            versions = vm.list_versions(limit=50)
            return jsonify({
                "versions": [v.to_dict() for v in versions],
                "total": len(vm.versions),
            })
        except Exception as e:
            return jsonify({"error": str(e), "versions": []}), 500

    @app.route("/api/document/versions/<version_id>")
    def api_document_version(version_id):
        """Get a specific document version."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            sys.path.insert(0, str(FANO_ROOT))
            from documenter.versions import VersionManager

            vm = VersionManager(doc_path)
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

    @app.route("/api/document/versions/<version_id>/revert", methods=["POST"])
    def api_document_revert(version_id):
        """Revert document to a specific version."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            sys.path.insert(0, str(FANO_ROOT))
            from documenter.versions import VersionManager

            vm = VersionManager(doc_path)
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

    # Track formatting fix state (survives page refresh)
    formatting_fix_state = {
        "in_progress": False,
        "progress": None,
        "result": None,
        "sections_fixed": 0,
    }

    def _finish_formatting(result: dict):
        """Helper to complete formatting operation and reset state."""
        formatting_fix_state["result"] = result
        formatting_fix_state["in_progress"] = False
        formatting_fix_state["progress"] = None
        formatting_fix_state["sections_fixed"] = 0

    def _split_into_sections(content: str) -> list[tuple[str, str]]:
        """Split document into sections by ## headers. Returns [(section_name, section_content), ...]"""
        import re
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

    def _run_formatting_fix():
        """Background worker for formatting fix."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            sys.path.insert(0, str(FANO_ROOT))
            from documenter.document import Document
            from documenter.formatting import detect_formatting_issues, build_formatting_fix_prompt

            # Load document
            doc = Document(doc_path, enable_versioning=True)
            doc.load()

            # Check for issues first
            issues = detect_formatting_issues(doc.content)

            if not issues:
                return _finish_formatting({"success": True, "issues_fixed": 0, "message": "No formatting issues found"})

            # Set up API client
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return _finish_formatting({"success": False, "error": "ANTHROPIC_API_KEY not configured"})

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

            _finish_formatting({
                "success": True,
                "issues_fixed": total_issues_fixed,
                "message": f"Fixed {sections_with_fixes} section(s) ({total_issues_fixed} issues) out of {total_sections} total",
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            _finish_formatting({"success": False, "error": str(e)})

    @app.route("/api/document/fix-formatting", methods=["POST"])
    def api_document_fix_formatting():
        """Start fixing math formatting issues (runs in background)."""
        if formatting_fix_state["in_progress"]:
            return jsonify({"success": True, "status": "already_running"})

        # Start background thread
        formatting_fix_state["in_progress"] = True
        formatting_fix_state["result"] = None
        thread = threading.Thread(target=_run_formatting_fix, daemon=True)
        thread.start()

        return jsonify({"success": True, "status": "started"})

    @app.route("/api/document/fix-formatting/status")
    def api_document_fix_formatting_status():
        """Check formatting fix status (for polling)."""
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

    # ==================== Annotation API ====================
    #
    # Annotations use inline markers in the markdown document.
    # Markers look like: <!-- @ann:c001 --> for comments, <!-- @ann:p001 --> for protected
    # The annotation content is stored in annotations.json

    @app.route("/api/annotations")
    def api_annotations():
        """Get all annotations."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            sys.path.insert(0, str(FANO_ROOT))
            from documenter.annotations import AnnotationManager
            from documenter.document import Document

            am = AnnotationManager(doc_path)
            doc = Document(doc_path)
            doc.load()

            # Get markers present in document
            markers_in_doc = set(doc.find_all_markers())

            # Build response with linked/unlinked info
            annotations = {}
            unlinked = []

            for ann_id, ann in am.annotations.items():
                ann_dict = ann.to_dict()
                if ann_id in markers_in_doc:
                    ann_dict["linked"] = True
                    annotations[ann_id] = ann_dict
                else:
                    ann_dict["linked"] = False
                    unlinked.append(ann_dict)

            return jsonify({
                "annotations": annotations,
                "unlinked": unlinked,
            })
        except Exception as e:
            return jsonify({"error": str(e), "annotations": {}, "unlinked": []}), 500

    @app.route("/api/annotations", methods=["POST"])
    def api_add_annotation():
        """Add a new annotation (comment or protected)."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            data = request.json
            ann_type = data.get("type", "comment")  # "comment" or "protected"
            content = data.get("content", "")
            char_offset = data.get("char_offset", 0)
            text_preview = data.get("text_preview", "")
            # Full selected text for searching in markdown
            search_text = data.get("search_text", text_preview)

            if ann_type == "comment" and not content:
                return jsonify({"error": "Comment content is required"}), 400

            sys.path.insert(0, str(FANO_ROOT))
            from documenter.annotations import AnnotationManager
            from documenter.document import Document

            # Create annotation (generates ID)
            am = AnnotationManager(doc_path)
            annotation = am.add(ann_type, content, text_preview)

            # Insert marker into document - use search_text to find correct position
            doc = Document(doc_path, enable_versioning=True)
            doc.load()
            doc.insert_marker(annotation.id, char_offset, search_text)
            doc.save(f"Added annotation {annotation.id}")

            return jsonify({
                "success": True,
                "annotation": annotation.to_dict(),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/annotations/<ann_id>", methods=["DELETE"])
    def api_delete_annotation(ann_id):
        """Delete an annotation and its marker."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            sys.path.insert(0, str(FANO_ROOT))
            from documenter.annotations import AnnotationManager
            from documenter.document import Document

            # Delete from annotations
            am = AnnotationManager(doc_path)
            if not am.delete(ann_id):
                return jsonify({"error": "Annotation not found"}), 404

            # Remove marker from document
            doc = Document(doc_path, enable_versioning=True)
            doc.load()
            doc.remove_marker(ann_id)
            doc.save(f"Removed annotation {ann_id}")

            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Legacy endpoints for backwards compatibility
    @app.route("/api/comments", methods=["POST"])
    def api_add_comment():
        """Add a new comment (legacy endpoint)."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            data = request.json or {}
            content = data.get("content", "")
            char_offset = data.get("char_offset", 0)
            selected_text = data.get("selected_text", "")
            text_preview = data.get("text_preview", selected_text[:50])
            search_text = data.get("search_text", selected_text)

            if not content:
                return jsonify({"error": "Comment content is required"}), 400

            sys.path.insert(0, str(FANO_ROOT))
            from documenter.annotations import AnnotationManager
            from documenter.document import Document

            am = AnnotationManager(doc_path)
            annotation = am.add("comment", content, text_preview)

            doc = Document(doc_path, enable_versioning=True)
            doc.load()
            doc.insert_marker(annotation.id, char_offset, search_text)
            doc.save(f"Added comment {annotation.id}")

            return jsonify({"success": True, "comment": annotation.to_dict()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/comments/<comment_id>", methods=["DELETE"])
    def api_delete_comment(comment_id):
        """Delete a comment (legacy endpoint)."""
        return api_delete_annotation(comment_id)

    @app.route("/api/protected", methods=["POST"])
    def api_add_protected():
        """Add a new protected region (legacy endpoint)."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        try:
            data = request.json or {}
            char_offset = data.get("char_offset", 0)
            selected_text = data.get("selected_text", "")
            text_preview = data.get("text_preview", selected_text[:50])
            search_text = data.get("search_text", selected_text)

            sys.path.insert(0, str(FANO_ROOT))
            from documenter.annotations import AnnotationManager
            from documenter.document import Document

            am = AnnotationManager(doc_path)
            annotation = am.add("protected", "", text_preview)

            doc = Document(doc_path, enable_versioning=True)
            doc.load()
            doc.insert_marker(annotation.id, char_offset, search_text)
            doc.save(f"Added protected region {annotation.id}")

            return jsonify({"success": True, "protected": annotation.to_dict()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/protected/<protected_id>", methods=["DELETE"])
    def api_delete_protected(protected_id):
        """Delete a protected region (legacy endpoint)."""
        return api_delete_annotation(protected_id)

    @app.route("/document")
    def view_document():
        """View the document in browser."""
        doc_path = FANO_ROOT / "documenter" / "document" / "main.md"
        if not doc_path.exists():
            return "<h1>Document not found</h1><p>The document hasn't been created yet. Start the documenter to begin.</p>"

        try:
            import markdown
            content = doc_path.read_text(encoding="utf-8")
            html_content = markdown.markdown(content, extensions=['tables', 'fenced_code', 'toc'])
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fano Document</title>
                <style>
                    body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #333; }}
                    h1 {{ border-bottom: 2px solid #8b5cf6; padding-bottom: 10px; }}
                    h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                    pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                    blockquote {{ border-left: 4px solid #8b5cf6; margin: 0; padding-left: 20px; color: #666; }}
                    a {{ color: #8b5cf6; }}
                    .nav {{ background: #333; color: white; padding: 10px 20px; margin: -20px -20px 20px -20px; }}
                    .nav a {{ color: #8b5cf6; margin-right: 20px; }}
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/">← Dashboard</a>
                    <span>Fano Living Document</span>
                </div>
                {html_content}
            </body>
            </html>
            """
        except ImportError:
            # Fallback if markdown not installed
            content = doc_path.read_text(encoding="utf-8")
            return f"<pre>{content}</pre>"
        except Exception as e:
            return f"<h1>Error</h1><p>{e}</p>"

    # ==================== Explorer API ====================

    @app.route("/api/explorer/stats")
    def api_explorer_stats():
        """Get explorer insight stats."""
        return jsonify(get_explorer_stats())

    @app.route("/api/explorer/insights/<status>")
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

    @app.route("/api/explorer/insight/<insight_id>")
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

    @app.route("/api/explorer/feedback", methods=["POST"])
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

    @app.route("/api/explorer/priority", methods=["POST"])
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

    # ==================== Server Control ====================

    @app.route("/api/server/restart", methods=["POST"])
    def api_server_restart():
        """Restart the server."""

        def delayed_restart():
            import time
            time.sleep(0.5)  # Let response send first

            # Terminate child processes cleanly before restart
            # Suppress stderr during shutdown to hide harmless asyncio transport warnings
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            try:
                for name, proc in list(_processes.items()):
                    if proc is not None and proc.poll() is None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr

            # Start new server process (detached on Windows)
            server_script = Path(__file__).resolve()
            if sys.platform == "win32":
                # Windows: use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                subprocess.Popen(
                    [sys.executable, str(server_script)],
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    close_fds=True,
                )
            else:
                # Unix: use nohup-like behavior
                subprocess.Popen(
                    [sys.executable, str(server_script)],
                    start_new_session=True,
                    close_fds=True,
                )

            # Exit current process
            os._exit(0)

        # Start restart in background thread
        thread = threading.Thread(target=delayed_restart, daemon=False)
        thread.start()

        return jsonify({"success": True, "message": "Server restarting..."})

    @app.route("/health")
    def health():
        """Health check endpoint for restart polling."""
        return jsonify({"status": "ok"})

    return app


def start_pool() -> subprocess.Popen:
    """Start the pool service."""
    pool_script = FANO_ROOT / "pool" / "run_pool.py"
    # Don't capture stdout to avoid buffer blocking
    return subprocess.Popen(
        [sys.executable, str(pool_script)],
        cwd=str(FANO_ROOT),
    )


def start_explorer(mode: str = "start") -> subprocess.Popen:
    """Start the explorer."""
    explorer_script = FANO_ROOT / "explorer" / "fano_explorer.py"
    # Don't capture stdout to avoid buffer blocking
    return subprocess.Popen(
        [sys.executable, str(explorer_script), mode],
        cwd=str(FANO_ROOT / "explorer"),
    )


def start_documenter() -> subprocess.Popen:
    """Start the documenter."""
    documenter_script = FANO_ROOT / "documenter" / "fano_documenter.py"
    # Don't capture stdout to avoid buffer blocking
    return subprocess.Popen(
        [sys.executable, str(documenter_script), "start"],
        cwd=str(FANO_ROOT / "documenter"),
    )


def get_stats() -> dict:
    """Get statistics from logs."""
    stats = {
        "documenter": {
            "sections": 0,
            "consensus_calls": 0,
        },
        "explorer": {
            "threads": 0,
            "blessed": 0,
        },
    }

    # Get documenter stats from most recent session summary
    doc_log = LOGS_DIR / "documenter.jsonl"
    if doc_log.exists():
        try:
            with open(doc_log, "r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    try:
                        entry = json.loads(line)
                        if entry.get("event_type") == "documenter.session.summary":
                            stats["documenter"]["sections"] = entry.get("sections", 0)
                            stats["documenter"]["consensus_calls"] = entry.get("consensus_calls", 0)
                            break
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

    # Get explorer stats
    blessed_dir = FANO_ROOT / "explorer" / "data" / "chunks" / "insights" / "blessed"
    if blessed_dir.exists():
        stats["explorer"]["blessed"] = len(list(blessed_dir.glob("*.json")))

    return stats


def cleanup_all_processes():
    """Kill all managed processes (pool, explorer, documenter)."""
    for name, proc in _processes.items():
        if proc is not None and proc.poll() is None:
            print(f"  Stopping {name}...")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def start_server(host: str = "127.0.0.1", port: int = 8080, debug: bool = True, pool_process: Optional[subprocess.Popen] = None):
    """Start the control panel server."""
    # Register externally-started pool process if provided
    if pool_process is not None:
        register_pool_process(pool_process)

    app = create_app()
    print(f"\n  Fano Control Panel")
    print(f"  ==================")
    print(f"  Open http://{host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    start_server()
