"""State management for the Browser Pool Service."""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from threading import Lock

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

log = get_logger("pool", "state")

# Map backend to its deep/pro mode key prefix
BACKEND_MODE_KEY = {
    "gemini": "deep_mode",
    "chatgpt": "pro_mode",
}

# Maximum age (in seconds) for active work to be considered valid for recovery
# Work older than this is considered stale and will be auto-cleared
MAX_ACTIVE_WORK_AGE_SECONDS = 7200  # 2 hours


class StateManager:
    """
    Manages persistent state for the pool service.

    Tracks:
    - Rate limit status per backend
    - Deep/Pro mode daily usage counters
    - Authentication status
    """

    def __init__(self, state_file: Path, config: dict):
        self.state_file = state_file
        self.config = config
        self._lock = Lock()
        self._state = self._load()

    def _load(self) -> dict:
        """Load state from file or create default."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning("pool.state.load_failed", error=str(e), using_defaults=True)

        return {
            "gemini": {
                "authenticated": False,
                "rate_limited": False,
                "rate_limit_resets_at": None,
                "deep_mode_uses_today": 0,
                "deep_mode_reset_date": None,
                "active_work": None,  # Tracks in-progress request
            },
            "chatgpt": {
                "authenticated": False,
                "rate_limited": False,
                "rate_limit_resets_at": None,
                "pro_mode_uses_today": 0,
                "pro_mode_reset_date": None,
                "active_work": None,  # Tracks in-progress request
            },
            "claude": {
                "authenticated": True,  # API-based, always "authenticated" if key exists
                "rate_limited": False,
                "rate_limit_resets_at": None,
            },
            # Recovered responses waiting to be picked up by clients
            "recovered_responses": [],
        }

    def _save(self):
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, default=str)

    def _check_daily_reset(self, backend: str, mode_key: str):
        """Check if daily counter should reset."""
        reset_date_key = f"{mode_key}_reset_date"
        today = datetime.now().date().isoformat()

        if self._state[backend].get(reset_date_key) != today:
            self._state[backend][f"{mode_key}_uses_today"] = 0
            self._state[backend][reset_date_key] = today
            self._save()

    def get_backend_state(self, backend: str) -> dict:
        """Get state for a backend."""
        with self._lock:
            if backend not in self._state:
                return {}

            # Check rate limit expiry
            backend_state = self._state[backend]
            if backend_state.get("rate_limited") and backend_state.get("rate_limit_resets_at"):
                reset_at = datetime.fromisoformat(backend_state["rate_limit_resets_at"])
                if datetime.now() >= reset_at:
                    backend_state["rate_limited"] = False
                    backend_state["rate_limit_resets_at"] = None
                    self._save()

            # Check daily reset for deep/pro mode
            if backend in BACKEND_MODE_KEY:
                self._check_daily_reset(backend, BACKEND_MODE_KEY[backend])

            return backend_state.copy()

    def mark_authenticated(self, backend: str, authenticated: bool = True):
        """Mark a backend as authenticated or not."""
        with self._lock:
            self._state[backend]["authenticated"] = authenticated
            self._save()
            log.info("pool.state.auth_changed", backend=backend, authenticated=authenticated)

    def mark_rate_limited(self, backend: str, retry_after_seconds: int = 3600):
        """Mark a backend as rate limited."""
        with self._lock:
            reset_at = datetime.now() + timedelta(seconds=retry_after_seconds)
            self._state[backend]["rate_limited"] = True
            self._state[backend]["rate_limit_resets_at"] = reset_at.isoformat()
            self._save()
            log.warning("pool.rate_limit.activated", backend=backend, resets_at=reset_at.isoformat())

    def clear_rate_limit(self, backend: str):
        """Clear rate limit for a backend."""
        with self._lock:
            self._state[backend]["rate_limited"] = False
            self._state[backend]["rate_limit_resets_at"] = None
            self._save()
            log.info("pool.rate_limit.cleared", backend=backend)

    def increment_deep_mode_usage(self, backend: str):
        """Increment deep/pro mode usage counter."""
        with self._lock:
            if backend in BACKEND_MODE_KEY:
                mode_key = BACKEND_MODE_KEY[backend]
                self._check_daily_reset(backend, mode_key)
                self._state[backend][f"{mode_key}_uses_today"] += 1
                self._save()

    def can_use_deep_mode(self, backend: str) -> bool:
        """Check if deep/pro mode can be used (under daily limit)."""
        with self._lock:
            if backend not in BACKEND_MODE_KEY:
                return True
            mode_key = BACKEND_MODE_KEY[backend]
            self._check_daily_reset(backend, mode_key)
            limit = self.config.get("backends", {}).get(backend, {}).get(mode_key, {}).get("daily_limit", 20)
            uses = self._state[backend].get(f"{mode_key}_uses_today", 0)
            return uses < limit

    def is_available(self, backend: str) -> bool:
        """Check if a backend is available (authenticated and not rate limited)."""
        state = self.get_backend_state(backend)
        return state.get("authenticated", False) and not state.get("rate_limited", False)

    # --- Active Work Tracking ---

    def set_active_work(
        self,
        backend: str,
        request_id: str,
        prompt: str,
        chat_url: str,
        thread_id: Optional[str] = None,
        options: Optional[dict] = None,
    ):
        """Record that a backend is working on a request."""
        with self._lock:
            if backend not in self._state:
                return
            self._state[backend]["active_work"] = {
                "request_id": request_id,
                "prompt": prompt,
                "chat_url": chat_url,
                "thread_id": thread_id,
                "started_at": time.time(),
                "options": options or {},
            }
            self._save()
            log.info("pool.state.active_work_set",
                     backend=backend,
                     request_id=request_id,
                     chat_url=chat_url)

    def clear_active_work(self, backend: str):
        """Clear active work for a backend (request completed)."""
        with self._lock:
            if backend not in self._state:
                return
            had_work = self._state[backend].get("active_work") is not None
            self._state[backend]["active_work"] = None
            self._save()
            if had_work:
                log.info("pool.state.active_work_cleared", backend=backend)

    def get_active_work(self, backend: str, check_staleness: bool = True) -> Optional[dict]:
        """
        Get active work for a backend, if any.

        Args:
            backend: Backend name
            check_staleness: If True, auto-clear and return None if work is stale

        Returns:
            Active work dict or None
        """
        with self._lock:
            if backend not in self._state:
                return None

            active_work = self._state[backend].get("active_work")
            if not active_work:
                return None

            # Check staleness if enabled
            if check_staleness:
                started_at = active_work.get("started_at", 0)
                age_seconds = time.time() - started_at

                if age_seconds > MAX_ACTIVE_WORK_AGE_SECONDS:
                    log.warning("pool.state.active_work_stale",
                               backend=backend,
                               request_id=active_work.get("request_id"),
                               age_seconds=round(age_seconds),
                               max_age_seconds=MAX_ACTIVE_WORK_AGE_SECONDS)
                    # Auto-clear stale work
                    self._state[backend]["active_work"] = None
                    self._save()
                    return None

            return active_work

    # --- Recovered Responses ---

    def add_recovered_response(
        self,
        backend: str,
        request_id: str,
        prompt: str,
        response: str,
        thread_id: Optional[str] = None,
        options: Optional[dict] = None,
    ):
        """Store a recovered response for later pickup."""
        with self._lock:
            # Initialize if not present
            if "recovered_responses" not in self._state:
                self._state["recovered_responses"] = []

            self._state["recovered_responses"].append({
                "request_id": request_id,
                "backend": backend,
                "prompt": prompt,
                "response": response,
                "thread_id": thread_id,
                "options": options or {},
                "recovered_at": time.time(),
            })
            self._save()
            log.info("pool.state.response_recovered",
                     backend=backend,
                     request_id=request_id,
                     response_length=len(response))

    def get_recovered_responses(self) -> list[dict]:
        """Get all recovered responses pending pickup."""
        with self._lock:
            return list(self._state.get("recovered_responses", []))

    def clear_recovered_response(self, request_id: str) -> bool:
        """Remove a recovered response (after pickup). Returns True if found."""
        with self._lock:
            if "recovered_responses" not in self._state:
                return False
            original_len = len(self._state["recovered_responses"])
            self._state["recovered_responses"] = [
                r for r in self._state["recovered_responses"]
                if r.get("request_id") != request_id
            ]
            removed = len(self._state["recovered_responses"]) < original_len
            if removed:
                self._save()
                log.info("pool.state.recovered_response_cleared", request_id=request_id)
            return removed
