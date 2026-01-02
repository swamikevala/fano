"""
Model selection and deep mode tracking for Fano Explorer.

Handles:
- Weighted random model selection based on phase (exploration/critique/synthesis)
- Daily limits for deep modes (Gemini Deep Think, ChatGPT Pro)
- Strategic deep mode usage (synthesis and breakthrough moments)
"""

import json
import random
from datetime import datetime, date
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.thread import ExplorationThread

from .base import CONFIG, rate_tracker

# State file for tracking deep mode usage
STATE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEEP_MODE_STATE_FILE = STATE_DIR / "deep_mode_state.json"


class DeepModeTracker:
    """Tracks daily usage of deep/pro modes."""

    def __init__(self):
        self.state = self._load()
        self._check_daily_reset()

    def _load(self) -> dict:
        """Load state from file."""
        if DEEP_MODE_STATE_FILE.exists():
            try:
                with open(DEEP_MODE_STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default_state()

    def _default_state(self) -> dict:
        return {
            "gemini_deep_think": {
                "used_today": 0,
                "last_reset": date.today().isoformat(),
            },
            "chatgpt_pro": {
                "used_today": 0,
                "last_reset": date.today().isoformat(),
            },
        }

    def _save(self):
        """Save state to file."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEEP_MODE_STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def _check_daily_reset(self):
        """Reset counters if it's a new day."""
        today = date.today().isoformat()
        for mode in ["gemini_deep_think", "chatgpt_pro"]:
            if self.state.get(mode, {}).get("last_reset") != today:
                self.state[mode] = {
                    "used_today": 0,
                    "last_reset": today,
                }
        self._save()

    def get_remaining(self, mode: str) -> int:
        """Get remaining deep mode uses for today."""
        config_limits = CONFIG.get("deep_modes", {})
        limit = config_limits.get(mode, {}).get("daily_limit", 0)
        used = self.state.get(mode, {}).get("used_today", 0)
        return max(0, limit - used)

    def can_use_deep_mode(self, mode: str) -> bool:
        """Check if deep mode can be used."""
        return self.get_remaining(mode) > 0

    def record_usage(self, mode: str):
        """Record a deep mode usage."""
        self._check_daily_reset()
        if mode not in self.state:
            self.state[mode] = {"used_today": 0, "last_reset": date.today().isoformat()}
        self.state[mode]["used_today"] += 1
        self._save()

    def get_status(self) -> dict:
        """Get current status of all deep modes."""
        self._check_daily_reset()
        config_limits = CONFIG.get("deep_modes", {})
        status = {}
        for mode in ["gemini_deep_think", "chatgpt_pro"]:
            limit = config_limits.get(mode, {}).get("daily_limit", 0)
            used = self.state.get(mode, {}).get("used_today", 0)
            status[mode] = {
                "limit": limit,
                "used": used,
                "remaining": max(0, limit - used),
            }
        return status


# Global tracker instance
deep_mode_tracker = DeepModeTracker()


def select_model(phase: str, available_models: dict) -> Optional[str]:
    """
    Select a model based on weighted random selection for the given phase.

    Args:
        phase: 'exploration', 'critique', or 'synthesis'
        available_models: dict of {model_name: interface} for available models

    Returns:
        Selected model name, or None if no models available
    """
    weights = CONFIG.get("model_weights", {}).get(phase, {})

    # Filter to only available and non-rate-limited models
    candidates = []
    for model_name in available_models.keys():
        if rate_tracker.is_available(model_name):
            weight = weights.get(model_name, 50)  # Default weight 50
            candidates.append((model_name, weight))

    if not candidates:
        return None

    # Weighted random selection
    total_weight = sum(w for _, w in candidates)
    if total_weight == 0:
        return candidates[0][0] if candidates else None

    r = random.uniform(0, total_weight)
    cumulative = 0
    for model_name, weight in candidates:
        cumulative += weight
        if r <= cumulative:
            return model_name

    # Fallback to first candidate
    return candidates[0][0]


def should_use_deep_mode(
    model: str,
    thread: "ExplorationThread",
    phase: str,
) -> bool:
    """
    Determine if deep mode should be used for this request.

    Deep mode is used when:
    - Phase is 'synthesis' (always worth using deep mode), OR
    - Thread has >= 4 exchanges AND recent responses contain profundity signals

    AND the daily limit hasn't been exceeded.

    Args:
        model: 'gemini' or 'chatgpt'
        thread: The exploration thread
        phase: 'exploration', 'critique', or 'synthesis'

    Returns:
        True if deep mode should be used
    """
    # Map model to deep mode name
    mode_name = "gemini_deep_think" if model == "gemini" else "chatgpt_pro"

    # Check if we have remaining quota
    if not deep_mode_tracker.can_use_deep_mode(mode_name):
        return False

    # Always use deep mode for synthesis
    if phase == "synthesis":
        return True

    # For exploration/critique, only use deep mode for breakthrough moments
    if thread.exchange_count < 4:
        return False

    # Check for profundity signals in recent exchanges
    profundity_signals = CONFIG.get("synthesis", {}).get("profundity_signals", [])
    if not profundity_signals:
        return False

    # Check last 2 exchanges for profundity signals
    recent_exchanges = thread.exchanges[-2:] if thread.exchanges else []
    for exchange in recent_exchanges:
        response_lower = exchange.response.lower()
        for signal in profundity_signals:
            if signal.lower() in response_lower:
                return True

    return False


def get_deep_mode_status() -> dict:
    """Get current deep mode usage status."""
    return deep_mode_tracker.get_status()
