"""State management for the Browser Pool Service."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from threading import Lock

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

log = get_logger("pool", "state")


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
            },
            "chatgpt": {
                "authenticated": False,
                "rate_limited": False,
                "rate_limit_resets_at": None,
                "pro_mode_uses_today": 0,
                "pro_mode_reset_date": None,
            },
            "claude": {
                "authenticated": True,  # API-based, always "authenticated" if key exists
                "rate_limited": False,
                "rate_limit_resets_at": None,
            },
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
            state = self._state.get(backend, {}).copy()

            # Check rate limit expiry
            if state.get("rate_limited") and state.get("rate_limit_resets_at"):
                reset_at = datetime.fromisoformat(state["rate_limit_resets_at"])
                if datetime.now() >= reset_at:
                    self._state[backend]["rate_limited"] = False
                    self._state[backend]["rate_limit_resets_at"] = None
                    self._save()
                    state["rate_limited"] = False

            # Check daily reset for deep/pro mode
            if backend == "gemini":
                self._check_daily_reset(backend, "deep_mode")
            elif backend == "chatgpt":
                self._check_daily_reset(backend, "pro_mode")

            return self._state.get(backend, {}).copy()

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
            if backend == "gemini":
                self._check_daily_reset(backend, "deep_mode")
                self._state[backend]["deep_mode_uses_today"] += 1
            elif backend == "chatgpt":
                self._check_daily_reset(backend, "pro_mode")
                self._state[backend]["pro_mode_uses_today"] += 1
            self._save()

    def can_use_deep_mode(self, backend: str) -> bool:
        """Check if deep/pro mode can be used (under daily limit)."""
        with self._lock:
            if backend == "gemini":
                self._check_daily_reset(backend, "deep_mode")
                limit = self.config.get("backends", {}).get("gemini", {}).get("deep_mode", {}).get("daily_limit", 20)
                uses = self._state[backend].get("deep_mode_uses_today", 0)
                return uses < limit
            elif backend == "chatgpt":
                self._check_daily_reset(backend, "pro_mode")
                limit = self.config.get("backends", {}).get("chatgpt", {}).get("pro_mode", {}).get("daily_limit", 100)
                uses = self._state[backend].get("pro_mode_uses_today", 0)
                return uses < limit
            return True

    def is_available(self, backend: str) -> bool:
        """Check if a backend is available (authenticated and not rate limited)."""
        state = self.get_backend_state(backend)
        return state.get("authenticated", False) and not state.get("rate_limited", False)
