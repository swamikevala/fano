"""
Browser automation utilities for Fano Explorer.

Handles Playwright setup, session persistence, and rate limit detection.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml

from playwright.async_api import async_playwright, Browser, BrowserContext, Page


# Load config - use absolute path resolution
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

BROWSER_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "browser_data"
RATE_LIMIT_FILE = BROWSER_DATA_DIR / "rate_limits.json"
CHAT_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chat_logs"


class ChatLogger:
    """Logs chat sessions to local files for reference."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.log_dir = CHAT_LOGS_DIR / model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[Path] = None
        self.session_start: Optional[datetime] = None

    def start_session(self) -> str:
        """Start a new chat session, returns the session ID (datetime string)."""
        self.session_start = datetime.now()
        session_id = self.session_start.strftime("%Y-%m-%d_%H-%M-%S")
        self.current_session = self.log_dir / f"{session_id}.md"

        # Write session header
        with open(self.current_session, "w", encoding="utf-8") as f:
            f.write(f"# {self.model_name.upper()} Chat Session\n")
            f.write(f"**Started:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.model_name}\n")
            f.write("\n---\n\n")

        print(f"[{self.model_name}] Chat log: {self.current_session}")
        return session_id

    def log_exchange(self, prompt: str, response: str):
        """Log a prompt/response exchange to the current session."""
        if not self.current_session:
            self.start_session()

        timestamp = datetime.now().strftime("%H:%M:%S")

        with open(self.current_session, "a", encoding="utf-8") as f:
            f.write(f"## User ({timestamp})\n\n")
            f.write(f"{prompt}\n\n")
            f.write(f"## {self.model_name.upper()} ({timestamp})\n\n")
            f.write(f"{response}\n\n")
            f.write("---\n\n")

    def get_session_id(self) -> Optional[str]:
        """Get current session ID for chat naming."""
        if self.session_start:
            return self.session_start.strftime("%Y-%m-%d_%H-%M-%S")
        return None


class RateLimitTracker:
    """Tracks rate limit status across sessions."""
    
    def __init__(self):
        self.limits = self._load()
    
    def _load(self) -> dict:
        if RATE_LIMIT_FILE.exists():
            with open(RATE_LIMIT_FILE) as f:
                return json.load(f)
        return {
            "chatgpt": {"limited": False, "retry_at": None},
            "gemini": {"limited": False, "retry_at": None},
        }
    
    def _save(self):
        RATE_LIMIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RATE_LIMIT_FILE, "w") as f:
            json.dump(self.limits, f, indent=2, default=str)
    
    def mark_limited(self, model: str, retry_after_seconds: int = 3600):
        """Mark a model as rate-limited."""
        retry_at = datetime.now() + timedelta(seconds=retry_after_seconds)
        self.limits[model] = {
            "limited": True,
            "retry_at": retry_at.isoformat(),
        }
        self._save()
    
    def mark_available(self, model: str):
        """Mark a model as available."""
        self.limits[model] = {"limited": False, "retry_at": None}
        self._save()
    
    def is_available(self, model: str) -> bool:
        """Check if a model is available."""
        info = self.limits.get(model, {"limited": False})
        if not info["limited"]:
            return True
        if info["retry_at"]:
            retry_at = datetime.fromisoformat(info["retry_at"])
            if datetime.now() >= retry_at:
                self.mark_available(model)
                return True
        return False
    
    def get_status(self) -> dict:
        """Get status of all models."""
        return self.limits.copy()


# Global tracker
rate_tracker = RateLimitTracker()


def get_rate_limit_status() -> dict:
    """Get current rate limit status for all models."""
    return rate_tracker.get_status()


async def get_browser_context(model: str, playwright_instance=None):
    """
    Get a browser context with saved session for the given model.
    Sessions are stored separately per model.
    
    If playwright_instance is provided, uses that instead of creating a new one.
    """
    storage_dir = BROWSER_DATA_DIR / model
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DEBUG] Using browser data dir: {storage_dir}")
    
    if playwright_instance is None:
        playwright_instance = await async_playwright().start()
    
    # Get viewport size from config (with smaller defaults for Windows)
    viewport_width = CONFIG["browser"].get("viewport_width", 1280)
    viewport_height = CONFIG["browser"].get("viewport_height", 720)

    browser = await playwright_instance.chromium.launch_persistent_context(
        user_data_dir=str(storage_dir),
        headless=CONFIG["browser"].get("headless", False),
        slow_mo=CONFIG["browser"].get("slow_mo", 100),
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--force-device-scale-factor=1",  # Fix Windows DPI scaling issues
        ],
        viewport={"width": viewport_width, "height": viewport_height},
    )
    
    return browser, playwright_instance


async def authenticate_all():
    """
    Open browsers for manual authentication to all services.
    User logs in manually, sessions are saved automatically.
    """
    models_to_auth = [
        ("chatgpt", CONFIG["models"]["chatgpt"]["url"]),
        ("gemini", CONFIG["models"]["gemini"]["url"]),
    ]
    
    for model, url in models_to_auth:
        print(f"\n{'='*50}")
        print(f"Authenticating: {model.upper()}")
        print(f"{'='*50}")
        print(f"Opening {url}")
        print(f"Browser data will be saved to: {BROWSER_DATA_DIR / model}")
        print()
        print("Instructions:")
        print("  1. Log in with your account")
        print("  2. Make sure you're fully logged in (can see the chat interface)")
        print("  3. CLOSE THE BROWSER WINDOW (not just the tab)")
        print()
        
        playwright = await async_playwright().start()
        
        try:
            context, _ = await get_browser_context(model, playwright)
            page = await context.new_page()
            await page.goto(url)
            
            print("Waiting for you to close the browser...")
            
            # Wait for the context to close (browser window closed)
            await context.wait_for_event("close", timeout=0)
            
            print(f"✓ {model} session saved!")
            
        except Exception as e:
            print(f"Error during {model} auth: {e}")
        finally:
            await playwright.stop()
        
        # Small delay between authentications
        await asyncio.sleep(1)
    
    print()
    print("="*50)
    print("Authentication complete!")
    print("="*50)


class BaseLLMInterface:
    """Base class for LLM browser interfaces."""

    model_name: str = "base"

    def __init__(self):
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.config = CONFIG["models"].get(self.model_name, {})
        self.chat_logger = ChatLogger(self.model_name)
    
    async def connect(self):
        """Establish browser connection."""
        self.context, self.playwright = await get_browser_context(self.model_name)
        self.page = await self.context.new_page()
        
        print(f"[{self.model_name}] Navigating to {self.config['url']}...")
        await self.page.goto(self.config["url"])
        await asyncio.sleep(3)  # Let page settle
        
        # Log current URL (helps debug redirects)
        print(f"[{self.model_name}] Current URL: {self.page.url}")
    
    async def disconnect(self):
        """Close browser connection."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def send_message(self, message: str) -> str:
        """Send a message and get response. Override in subclasses."""
        raise NotImplementedError
    
    def _check_rate_limit(self, response_text: str) -> bool:
        """Check if response indicates rate limiting."""
        patterns = self.config.get("selectors", {}).get("rate_limit_patterns", [])
        for pattern in patterns:
            if pattern.lower() in response_text.lower():
                return True
        return False
    
    async def _wait_for_response(self, timeout: int = None) -> str:
        """Wait for and extract response. Override in subclasses."""
        raise NotImplementedError
