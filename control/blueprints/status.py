"""
Status Blueprint - Status, config, and logs routes.
"""

import json
import os
import socket

from flask import Blueprint, current_app, jsonify, request

from ..services import FANO_ROOT, LOGS_DIR, load_config, save_config

bp = Blueprint("status", __name__, url_prefix="/api")


def get_process_manager():
    """Get the ProcessManager from the app context."""
    return current_app.config.get("process_manager")


def check_pool_health(host: str = "127.0.0.1", port: int = 9000) -> bool:
    """Check if pool is responding to health checks."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_stats() -> dict:
    """Get statistics from logs and data directories."""
    stats = {
        "documenter": {
            "sections": 0,
            "consensus_calls": 0,
        },
        "explorer": {
            "threads": 0,
            "blessed": 0,
        },
        "researcher": {
            "sources": 0,
            "findings": 0,
            "patterns": 0,
            "top_numbers": {},
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

    # Count active exploration threads
    explorations_dir = FANO_ROOT / "explorer" / "data" / "explorations"
    if explorations_dir.exists():
        active_count = 0
        for f in explorations_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if data.get("status") == "active":
                        active_count += 1
            except Exception:
                pass
        stats["explorer"]["threads"] = active_count

    # Get researcher stats
    try:
        from researcher.src import ResearcherAPI
        api = ResearcherAPI(FANO_ROOT / "researcher" / "data")
        researcher_stats = api.get_statistics()
        stats["researcher"] = {
            "sources": researcher_stats.get("sources", 0),
            "findings": researcher_stats.get("findings", 0),
            "patterns": researcher_stats.get("cross_domain_numbers", 0),
            "top_numbers": researcher_stats.get("top_numbers", {}),
        }
    except Exception:
        pass  # Keep default zeros

    return stats


@bp.route("/status")
def api_status():
    """Get status of all components."""
    config = load_config()
    pool_host = config.get("llm", {}).get("pool", {}).get("host", "127.0.0.1")
    pool_port = config.get("llm", {}).get("pool", {}).get("port", 9000)

    pm = get_process_manager()

    # Check pool status - either via subprocess OR by checking if port is responding
    pool_proc_running = pm.is_running("pool") if pm else False
    pool_responding = check_pool_health(pool_host, pool_port)
    pool_running = pool_proc_running or pool_responding

    # Check explorer status
    explorer_running = pm.is_running("explorer") if pm else False

    # Check documenter status
    documenter_running = pm.is_running("documenter") if pm else False

    # Check researcher status
    researcher_running = pm.is_running("researcher") if pm else False

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
            "pid": pm.get_pid("pool") if pm else None,
            "external": pool_responding and not pool_proc_running,
        },
        "explorer": {
            "running": explorer_running,
            "pid": pm.get_pid("explorer") if pm else None,
        },
        "documenter": {
            "running": documenter_running,
            "pid": pm.get_pid("documenter") if pm else None,
        },
        "researcher": {
            "running": researcher_running,
            "pid": pm.get_pid("researcher") if pm else None,
        },
        "backends": backends,
        "stats": stats,
    })


@bp.route("/logs/<component>")
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


@bp.route("/config")
def api_config():
    """Get current configuration."""
    return jsonify(load_config())


@bp.route("/config", methods=["POST"])
def api_update_config():
    """Update configuration."""
    try:
        new_config = request.json
        save_config(new_config)
        return jsonify({"status": "updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
