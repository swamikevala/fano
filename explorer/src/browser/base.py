"""
Browser automation utilities for Fano Explorer.

Handles Playwright setup, session persistence, and rate limit detection.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
import yaml

from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from shared.logging import get_logger

# Load environment variables
load_dotenv()

log = get_logger("explorer", "browser")


class AuthenticationRequired(Exception):
    """Raised when browser session needs authentication."""
    pass

# Firewall authentication settings
FIREWALL_DOMAIN = os.environ.get("FIREWALL_DOMAIN", "")
FIREWALL_USERNAME = os.environ.get("FIREWALL_USERNAME", "")
FIREWALL_PASSWORD = os.environ.get("FIREWALL_PASSWORD", "")


# Load config - use absolute path resolution
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
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

        log.debug("browser.chat.session_started", model=self.model_name, log_file=str(self.current_session))
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
            with open(RATE_LIMIT_FILE, encoding="utf-8") as f:
                return json.load(f)
        return {
            "chatgpt": {"limited": False, "retry_at": None},
            "gemini": {"limited": False, "retry_at": None},
        }
    
    def _save(self):
        RATE_LIMIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RATE_LIMIT_FILE, "w", encoding="utf-8") as f:
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
    
    log.debug("browser.context.init", model=model, storage_dir=str(storage_dir))
    
    if playwright_instance is None:
        playwright_instance = await async_playwright().start()
    
    # Get viewport size from config (with smaller defaults for Windows)
    viewport_width = CONFIG["browser"].get("viewport_width", 1280)
    viewport_height = CONFIG["browser"].get("viewport_height", 720)

    # Check for SSL bypass flag (for corporate proxies)
    disable_ssl = os.environ.get("BROWSER_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")

    chrome_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--force-device-scale-factor=1",  # Fix Windows DPI scaling issues
    ]

    if disable_ssl:
        log.warning("browser.ssl.disabled", model=model, reason="corporate_proxy_workaround")
        chrome_args.extend([
            "--ignore-certificate-errors",
            "--ignore-ssl-errors",
            "--allow-insecure-localhost",
        ])

    browser = await playwright_instance.chromium.launch_persistent_context(
        user_data_dir=str(storage_dir),
        headless=CONFIG["browser"].get("headless", False),
        slow_mo=CONFIG["browser"].get("slow_mo", 100),
        args=chrome_args,
        viewport={"width": viewport_width, "height": viewport_height},
        ignore_https_errors=disable_ssl,  # Also ignore at Playwright level
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
        ("claude", CONFIG["models"]["claude"]["url"]),
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
        self._url_update_callback: Optional[Callable[[str], None]] = None

    def set_url_update_callback(self, callback: Optional[Callable[[str], None]]):
        """Set callback to be called when chat URL changes."""
        self._url_update_callback = callback

    def _notify_url_change(self, url: str):
        """Notify that the chat URL has changed."""
        if self._url_update_callback:
            self._url_update_callback(url)
    
    async def connect(self):
        """Establish browser connection."""
        self.context, self.playwright = await get_browser_context(self.model_name)
        self.page = await self.context.new_page()

        log.info("browser.navigation.start", model=self.model_name, url=self.config['url'])

        # Use firewall-aware navigation
        success = await self._navigate_with_firewall_check(self.config["url"])
        if not success:
            log.warning("browser.navigation.firewall_issue", model=self.model_name)

        await asyncio.sleep(1)  # Let page settle

        # Log current URL (helps debug redirects)
        log.debug("browser.navigation.complete", model=self.model_name, current_url=self.page.url)
    
    async def disconnect(self):
        """Close browser connection."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()

    async def check_selector_health(self, selectors: dict[str, list[str]]) -> dict[str, bool]:
        """
        Check if critical selectors are present on the page.

        Args:
            selectors: Dict mapping selector names to lists of possible selectors
                       e.g. {"input": ["#prompt-textarea", "textarea"], "send": ["button[type=submit]"]}

        Returns:
            Dict mapping selector names to whether any selector was found
        """
        results = {}
        missing = []

        for name, selector_list in selectors.items():
            found = False
            for selector in selector_list:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        found = True
                        break
                except Exception:
                    continue

            results[name] = found
            if not found:
                missing.append(name)

        if missing:
            log.warning(
                "browser.selector.health_check_failed",
                model=self.model_name,
                missing_selectors=missing,
                hint=f"UI may have changed. Check src/browser/{self.model_name}.py"
            )

        return results
    
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

    def _is_firewall_redirect(self) -> bool:
        """Check if current page is a firewall authentication portal."""
        if not FIREWALL_DOMAIN or not self.page:
            return False

        current_url = self.page.url
        # Check if the URL contains the firewall domain
        if FIREWALL_DOMAIN in current_url:
            log.info(f"[{self.model_name}] Firewall redirect detected: {current_url}")
            return True
        return False

    async def _handle_firewall_auth(self) -> bool:
        """
        Handle firewall authentication if we're redirected to the portal.

        Returns True if authentication was successful (or not needed),
        False if authentication failed or credentials are missing.
        """
        if not self._is_firewall_redirect():
            return True  # Not a firewall redirect, nothing to do

        if not FIREWALL_USERNAME or not FIREWALL_PASSWORD:
            print(f"\n{'='*60}")
            print(f"[{self.model_name}] FIREWALL AUTHENTICATION REQUIRED")
            print(f"{'='*60}")
            print(f"Your network firewall is blocking access.")
            print(f"Please add your credentials to .env:")
            print(f"  FIREWALL_USERNAME=your_username")
            print(f"  FIREWALL_PASSWORD=your_password")
            print(f"{'='*60}\n")
            return False

        print(f"[{self.model_name}] Firewall detected - authenticating...")
        log.info(f"[{self.model_name}] Attempting firewall authentication")

        try:
            # Wait for page to fully load
            await asyncio.sleep(1)

            # Look for common username/password input fields
            # Try multiple selector patterns for username field
            username_selectors = [
                "input[name='username']",
                "input[name='user']",
                "input[name='uid']",
                "input[type='text'][name*='user']",
                "input#username",
                "input#user",
                "input[placeholder*='user' i]",
                "input[placeholder*='name' i]",
            ]

            password_selectors = [
                "input[name='password']",
                "input[name='passwd']",
                "input[name='pwd']",
                "input[type='password']",
                "input#password",
            ]

            submit_selectors = [
                "input[type='submit']",
                "button[type='submit']",
                "button:has-text('Login')",
                "button:has-text('Sign in')",
                "button:has-text('Submit')",
                "input[value='Login']",
                "input[value='Submit']",
            ]

            # Find and fill username
            username_field = None
            for selector in username_selectors:
                try:
                    username_field = await self.page.query_selector(selector)
                    if username_field and await username_field.is_visible():
                        log.info(f"[{self.model_name}] Found username field: {selector}")
                        break
                    username_field = None
                except Exception:
                    continue

            if not username_field:
                log.error(f"[{self.model_name}] Could not find username field")
                return False

            # Find and fill password
            password_field = None
            for selector in password_selectors:
                try:
                    password_field = await self.page.query_selector(selector)
                    if password_field and await password_field.is_visible():
                        log.info(f"[{self.model_name}] Found password field: {selector}")
                        break
                    password_field = None
                except Exception:
                    continue

            if not password_field:
                log.error(f"[{self.model_name}] Could not find password field")
                return False

            # Enter credentials
            await username_field.click()
            await username_field.fill(FIREWALL_USERNAME)
            await asyncio.sleep(0.3)

            await password_field.click()
            await password_field.fill(FIREWALL_PASSWORD)
            await asyncio.sleep(0.3)

            # Find and click submit button
            submit_button = None
            for selector in submit_selectors:
                try:
                    submit_button = await self.page.query_selector(selector)
                    if submit_button and await submit_button.is_visible():
                        log.info(f"[{self.model_name}] Found submit button: {selector}")
                        break
                    submit_button = None
                except Exception:
                    continue

            if submit_button:
                await submit_button.click()
            else:
                # Fallback: press Enter
                log.info(f"[{self.model_name}] No submit button found, pressing Enter")
                await self.page.keyboard.press("Enter")

            # Wait for authentication to complete
            print(f"[{self.model_name}] Waiting for firewall authentication...")
            await asyncio.sleep(3)

            # Check if we're still on the firewall page
            if self._is_firewall_redirect():
                log.error(f"[{self.model_name}] Still on firewall page after auth attempt")
                print(f"[{self.model_name}] Firewall authentication may have failed - still on portal")
                return False

            print(f"[{self.model_name}] Firewall authentication successful!")
            log.info(f"[{self.model_name}] Firewall authentication completed")
            return True

        except Exception as e:
            log.error(f"[{self.model_name}] Firewall auth error: {e}")
            print(f"[{self.model_name}] Firewall authentication error: {e}")
            return False

    async def _navigate_with_firewall_check(self, url: str, max_retries: int = 2) -> bool:
        """
        Navigate to a URL with automatic firewall authentication handling.

        Returns True if navigation was successful, False otherwise.
        """
        for attempt in range(max_retries + 1):
            await self.page.goto(url)
            await asyncio.sleep(2)  # Let page settle

            # Check for firewall redirect
            if self._is_firewall_redirect():
                success = await self._handle_firewall_auth()
                if success:
                    # After auth, navigate to original URL
                    await self.page.goto(url)
                    await asyncio.sleep(2)

                    # Verify we're not still redirected
                    if not self._is_firewall_redirect():
                        return True
                    elif attempt < max_retries:
                        print(f"[{self.model_name}] Retrying navigation (attempt {attempt + 2}/{max_retries + 1})")
                        continue
                else:
                    return False
            else:
                # No firewall, navigation successful
                return True

        return False

    async def _wait_for_response(self, timeout: int = None) -> str:
        """Wait for and extract response. Override in subclasses."""
        raise NotImplementedError

    async def try_get_response(self) -> Optional[str]:
        """
        Try to get the last response if already generated, without waiting.

        Used for recovery after restart - checks if a response exists and
        is not currently being generated.

        Returns:
            Response text if ready, None if still generating or no response.
        """
        raise NotImplementedError

    async def is_generating(self) -> bool:
        """Check if the LLM is currently generating a response."""
        raise NotImplementedError
