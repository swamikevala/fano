"""
Claude browser automation interface.

Handles interaction with Claude web interface, including:
- Message sending
- Response extraction
- Extended Thinking mode toggle
- Rate limit detection
"""

import asyncio
from datetime import datetime
from typing import Optional

from shared.logging import get_logger

from .base import BaseLLMInterface, rate_tracker, AuthenticationRequired

log = get_logger("explorer", "browser.claude")


class ClaudeInterface(BaseLLMInterface):
    """Interface for Claude web UI automation."""

    model_name = "claude"

    def __init__(self):
        super().__init__()
        self.extended_thinking_enabled = False
        self.last_deep_mode_used = False
        self._response_in_progress = False

    async def connect(self):
        """Connect to Claude."""
        await super().connect()
        await self._wait_for_ready()
        is_logged_in = await self._check_login_status()
        if not is_logged_in:
            raise AuthenticationRequired("Claude login required")
        await self._check_selectors()

    async def _wait_for_ready(self, timeout: int = 30):
        """Wait for Claude interface to be ready."""
        print(f"[claude] Waiting for interface to be ready...")

        # Multiple possible selectors for the input (Claude uses contenteditable)
        input_selectors = [
            "div.ProseMirror[contenteditable='true']",
            "div[contenteditable='true'][data-placeholder]",
            "div[contenteditable='true']",
            "textarea",
        ]

        for selector in input_selectors:
            try:
                element = await self.page.wait_for_selector(selector, timeout=5000)
                if element:
                    print(f"[claude] Found input with selector: {selector}")
                    self._input_selector = selector
                    return
            except Exception:
                continue

        print(f"[claude] WARNING: Could not find input element. Current URL: {self.page.url}")
        print(f"[claude] Page title: {await self.page.title()}")
        self._input_selector = None

    async def _check_selectors(self):
        """Check critical selectors are present."""
        await self.check_selector_health({
            "input": [
                "div.ProseMirror[contenteditable='true']",
                "div[contenteditable='true']",
            ],
            "send_button": [
                "button[aria-label='Send message']",
                "button[aria-label='Send Message']",
                "button[data-testid='send-button']",
                "fieldset button:not([disabled])",
            ],
            "response_area": [
                "[data-testid='message-content']",
                "div[data-is-streaming]",
                ".font-claude-message",
            ],
        })

    async def _check_login_status(self) -> bool:
        """
        Check if we're logged in to Claude.

        Returns:
            True if logged in, False if login page detected
        """
        try:
            # Check 1: If input element was found, likely logged in
            if getattr(self, '_input_selector', None):
                print(f"[claude] Appears to be logged in (input found)")
                return True

            # Check 2: Look for login page indicators
            login_indicators = [
                "button:has-text('Log in')",
                "button:has-text('Sign in')",
                "a[href*='/login']",
                "a[href*='accounts.google.com']",
                "[data-testid='login-button']",
            ]

            for selector in login_indicators:
                try:
                    element = await self.page.query_selector(selector)
                    if element and await element.is_visible():
                        print(f"[claude] WARNING: Not logged in! Found: {selector}")
                        print(f"[claude] Please run auth setup for Claude")
                        return False
                except Exception:
                    continue

            # Check 3: URL-based detection
            try:
                current_url = self.page.url
                if '/login' in current_url.lower() or 'auth' in current_url.lower():
                    print(f"[claude] WARNING: On login page: {current_url}")
                    return False
            except Exception:
                pass

            # Check 4: Look for logged-in indicators
            logged_in_indicators = [
                "[data-testid='user-menu']",
                "button[aria-label*='menu']",
                "[data-testid='new-chat-button']",
                "nav",  # Sidebar presence usually means logged in
            ]

            for selector in logged_in_indicators:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        print(f"[claude] Confirmed logged in (found: {selector})")
                        return True
                except Exception:
                    continue

            # Uncertain - assume logged in if no login indicators found
            print(f"[claude] Login status uncertain, assuming logged in")
            return True

        except Exception as e:
            print(f"[claude] Login check error: {e}, assuming logged in")
            return True

    async def start_new_chat(self):
        """Start a new conversation."""
        if self._response_in_progress:
            log.warning("[claude] start_new_chat blocked: response in progress!")
            return

        try:
            log.info("[claude] Starting new chat")

            # Reset mode tracking
            self.extended_thinking_enabled = False

            # Navigate to new chat
            await self._navigate_with_firewall_check("https://claude.ai/new")

            # Start a new logging session
            session_id = self.chat_log.start_session()
            print(f"[claude] Started new chat (session: {session_id})")

        except Exception as e:
            print(f"[claude] Could not start new chat: {e}")

    async def enable_extended_thinking(self) -> bool:
        """
        Enable Extended Thinking mode if available.
        Returns True if successfully enabled.
        """
        if self._response_in_progress:
            log.warning("[claude] enable_extended_thinking blocked: response in progress!")
            return self.extended_thinking_enabled

        if self.extended_thinking_enabled:
            log.info("[claude] Extended Thinking already enabled, skipping")
            return True

        log.info("[claude] Enabling Extended Thinking mode...")

        try:
            # Look for model selector or settings that might have Extended Thinking toggle
            # Claude's Extended Thinking is typically in the model dropdown or settings

            # Method 1: Look for model selector dropdown
            model_btn_selectors = [
                "button[data-testid='model-selector']",
                "[aria-label*='model']",
                "button:has-text('Claude')",
                "[data-testid='model-selector-dropdown']",
            ]

            model_btn = None
            for selector in model_btn_selectors:
                try:
                    model_btn = await self.page.query_selector(selector)
                    if model_btn and await model_btn.is_visible():
                        log.info(f"[claude] Found model selector: {selector}")
                        break
                    model_btn = None
                except Exception:
                    continue

            if model_btn:
                await model_btn.click()
                await asyncio.sleep(0.5)

                # Look for Extended Thinking option in dropdown
                thinking_options = [
                    "[role='menuitem']:has-text('Extended')",
                    "[role='option']:has-text('Extended')",
                    "[role='menuitem']:has-text('Thinking')",
                    "button:has-text('Extended thinking')",
                ]

                for selector in thinking_options:
                    try:
                        option = await self.page.query_selector(selector)
                        if option and await option.is_visible():
                            text = await option.inner_text()
                            log.info(f"[claude] Selecting Extended Thinking: {text[:30]}")
                            await option.click()
                            await asyncio.sleep(1)
                            self.extended_thinking_enabled = True
                            return True
                    except Exception:
                        continue

                # Close dropdown if no option found
                await self.page.keyboard.press("Escape")

            # Method 2: Look for toggle switch
            toggle_selectors = [
                "[data-testid='extended-thinking-toggle']",
                "button[aria-label*='thinking']",
                "input[type='checkbox'][aria-label*='thinking']",
                "[role='switch'][aria-label*='Extended']",
            ]

            for selector in toggle_selectors:
                try:
                    toggle = await self.page.query_selector(selector)
                    if toggle and await toggle.is_visible():
                        log.info(f"[claude] Found thinking toggle: {selector}")
                        await toggle.click()
                        await asyncio.sleep(0.5)
                        self.extended_thinking_enabled = True
                        return True
                except Exception:
                    continue

            log.warning("[claude] Extended Thinking toggle not found")
            return False

        except Exception as e:
            log.error(f"[claude] Could not enable Extended Thinking: {e}")
            return False

    async def send_message(self, message: str, use_extended_thinking: bool = False) -> str:
        """
        Send a message to Claude and wait for response.

        Args:
            message: The message to send
            use_extended_thinking: Whether to use Extended Thinking mode

        Returns the response text.
        Sets self.last_deep_mode_used to indicate if extended thinking was used.
        """
        if not rate_tracker.is_available(self.model_name):
            raise RateLimitError("Claude is rate-limited")

        # Track whether extended thinking was used for this message
        self.last_deep_mode_used = False

        # Try to enable Extended Thinking if requested
        if use_extended_thinking:
            if not self.extended_thinking_enabled:
                success = await self.enable_extended_thinking()
                if success:
                    self.last_deep_mode_used = True
            if self.extended_thinking_enabled:
                self.last_deep_mode_used = True

        mode_str = "Extended Thinking" if self.extended_thinking_enabled else "standard"
        log.info(f"[claude] Sending message in {mode_str} mode ({len(message)} chars)")
        print(f"[claude] Sending message ({len(message)} chars)...")

        try:
            # Find the input area
            input_selectors = [
                "div.ProseMirror[contenteditable='true']",
                "div[contenteditable='true'][data-placeholder]",
                "div[contenteditable='true']",
            ]

            input_elem = None
            for selector in input_selectors:
                input_elem = await self.page.query_selector(selector)
                if input_elem:
                    is_visible = await input_elem.is_visible()
                    if is_visible:
                        print(f"[claude] Using input selector: {selector}")
                        break
                    input_elem = None

            if not input_elem:
                raise Exception("Could not find Claude input element")

            # Click and input message
            await input_elem.click()
            await asyncio.sleep(0.3)

            # Use clipboard paste for contenteditable
            await self._paste_text(input_elem, message)
            await asyncio.sleep(0.5)

            # Find and click send button
            send_selectors = [
                "button[aria-label='Send message']",
                "button[aria-label='Send Message']",
                "button[data-testid='send-button']",
                "fieldset button:not([disabled])",
                "button:has(svg)",  # Send button often has an icon
            ]

            sent = False
            for selector in send_selectors:
                send_btn = await self.page.query_selector(selector)
                if send_btn:
                    is_visible = await send_btn.is_visible()
                    is_disabled = await send_btn.is_disabled() if is_visible else True
                    if is_visible and not is_disabled:
                        await send_btn.click()
                        sent = True
                        print(f"[claude] Clicked send button: {selector}")
                        break

            if not sent:
                # Fallback: press Enter
                print(f"[claude] No send button found, pressing Enter...")
                await self.page.keyboard.press("Enter")

            # Start URL monitoring in background
            asyncio.create_task(self._monitor_and_notify_url())

            # Set guard to prevent concurrent operations
            self._response_in_progress = True
            log.info(f"[claude] Response guard ON - waiting for response...")

            try:
                response = await self._wait_for_response()
            finally:
                self._response_in_progress = False
                log.info(f"[claude] Response guard OFF")

            # Check for rate limiting
            if self._check_rate_limit(response):
                rate_tracker.mark_limited(self.model_name)
                raise RateLimitError("Claude rate limit detected")

            # Log the exchange locally
            self.chat_log.log_exchange(message, response)

            print(f"[claude] Got response ({len(response)} chars)")
            return response

        except Exception as e:
            print(f"[claude] Error: {e}")
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                rate_tracker.mark_limited(self.model_name)
            raise

    async def _wait_for_response(self, timeout: int = None) -> str:
        """Wait for Claude to finish responding and extract text."""
        if timeout is None:
            timeout = self.config.get("response_timeout", 3600)

        mode_str = "Extended Thinking" if self.extended_thinking_enabled else "standard"
        log.info(f"[claude] Waiting for response in {mode_str} mode (timeout: {timeout}s)...")

        start_time = datetime.now()
        last_response = ""
        stable_count = 0
        last_log_time = start_time
        response_started = False

        # Selectors for assistant responses
        response_selectors = [
            "[data-testid='message-content']",
            "div[data-is-streaming]",
            ".font-claude-message",
            "[data-message-author='assistant']",
            "div.prose",
        ]

        # Selectors indicating Claude is still generating
        still_processing_selectors = [
            "[data-is-streaming='true']",
            "button[aria-label='Stop']",
            "button:has-text('Stop')",
            ".animate-pulse",
            "[class*='loading']",
            "[class*='typing']",
        ]

        # For Extended Thinking, wait longer before checking stability
        initial_wait = 30 if self.extended_thinking_enabled else 5
        log.info(f"[claude] Initial wait: {initial_wait}s before stability checks")
        await asyncio.sleep(initial_wait)

        while (datetime.now() - start_time).seconds < timeout:
            elapsed = (datetime.now() - start_time).seconds

            # Log progress every 30 seconds
            if (datetime.now() - last_log_time).seconds >= 30:
                log.info(f"[claude] Still waiting... ({elapsed}s elapsed, stable_count={stable_count})")
                last_log_time = datetime.now()

            # Check if still generating
            is_still_processing = False
            for selector in still_processing_selectors:
                try:
                    elem = await self.page.query_selector(selector)
                    if elem:
                        is_visible = await elem.is_visible()
                        if is_visible:
                            is_still_processing = True
                            break
                except Exception:
                    continue

            if is_still_processing:
                stable_count = 0
                await asyncio.sleep(3)
                continue

            # Try to get the response content
            current_response = ""
            for selector in response_selectors:
                try:
                    messages = await self.page.query_selector_all(selector)
                    if messages:
                        last_msg = messages[-1]
                        current_response = await last_msg.inner_text()
                        if current_response and len(current_response) > 10:
                            response_started = True
                            break
                except Exception:
                    continue

            if current_response:
                if current_response == last_response:
                    stable_count += 1
                    # For Extended Thinking, require more stability
                    required_stable = 15 if self.extended_thinking_enabled else 5

                    if stable_count >= required_stable:
                        # Double-check no processing indicators
                        final_check = False
                        for selector in still_processing_selectors[:3]:
                            try:
                                elem = await self.page.query_selector(selector)
                                if elem and await elem.is_visible():
                                    final_check = True
                                    break
                            except Exception:
                                continue

                        if not final_check:
                            elapsed = (datetime.now() - start_time).seconds
                            log.info(f"[claude] Response complete ({elapsed}s, {len(current_response)} chars)")
                            return current_response.strip()
                        else:
                            stable_count = required_stable // 2
                else:
                    stable_count = 0
                    last_response = current_response

            await asyncio.sleep(3)

        if last_response:
            elapsed = (datetime.now() - start_time).seconds
            log.warning(f"[claude] Timeout reached after {elapsed}s, returning partial response")
            return last_response.strip()
        raise TimeoutError("Claude response timeout")

    async def _paste_text(self, element, text: str):
        """
        Paste text into an element using DOM manipulation.
        Works with ProseMirror and contenteditable elements.
        """
        await element.evaluate("""(el, text) => {
            // Clear existing content
            el.textContent = '';

            // Focus the element
            el.focus();

            // Use DOM manipulation to add text with line breaks
            const lines = text.split('\\n');
            lines.forEach((line, index) => {
                if (index > 0) {
                    el.appendChild(document.createElement('br'));
                }
                if (line) {
                    el.appendChild(document.createTextNode(line));
                }
            });

            // Dispatch input event to notify any listeners
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }""", text)

    async def _monitor_and_notify_url(self):
        """Monitor URL and notify when it changes to include conversation ID."""
        base_url = "https://claude.ai/new"
        for _ in range(60):
            try:
                current_url = self.page.url
                # Claude conversation URLs typically have /chat/ followed by an ID
                if "/chat/" in current_url and current_url != base_url:
                    log.info(f"[claude] URL changed to conversation URL: {current_url}")
                    self._notify_url_change(current_url)
                    return
            except Exception as e:
                log.debug(f"[claude] URL monitor error: {e}")
            await asyncio.sleep(1)
        log.warning("[claude] URL monitor: never saw URL change to conversation URL")

    async def is_generating(self) -> bool:
        """Check if Claude is currently generating a response."""
        still_processing_selectors = [
            "[data-is-streaming='true']",
            "button[aria-label='Stop']",
            "button:has-text('Stop')",
            ".animate-pulse",
            "[class*='loading']",
        ]

        for selector in still_processing_selectors:
            try:
                elem = await self.page.query_selector(selector)
                if elem and await elem.is_visible():
                    return True
            except Exception:
                continue
        return False

    async def try_get_response(self) -> Optional[str]:
        """
        Try to get the last response if already generated, without waiting.

        Used for recovery after restart.

        Returns:
            Response text if ready, None if still generating or no response.
        """
        if await self.is_generating():
            log.info("[claude] try_get_response: still generating")
            return None

        response_selectors = [
            "[data-testid='message-content']",
            "div[data-is-streaming]",
            ".font-claude-message",
            "[data-message-author='assistant']",
            "div.prose",
        ]

        for selector in response_selectors:
            try:
                messages = await self.page.query_selector_all(selector)
                if messages:
                    last_msg = messages[-1]
                    response = await last_msg.inner_text()
                    if response and len(response) > 10:
                        log.info(f"[claude] try_get_response: found response ({len(response)} chars)")
                        return response.strip()
            except Exception:
                continue

        log.info("[claude] try_get_response: no response found")
        return None

    async def get_conversation_history(self) -> list[dict]:
        """Extract full conversation history from current chat."""
        history = []

        # This needs to be adjusted based on Claude's actual DOM structure
        user_msgs = await self.page.query_selector_all("[data-message-author='user']")
        model_msgs = await self.page.query_selector_all("[data-message-author='assistant']")

        for user, model in zip(user_msgs, model_msgs):
            user_text = await user.inner_text()
            model_text = await model.inner_text()
            history.append({"role": "user", "content": user_text.strip()})
            history.append({"role": "assistant", "content": model_text.strip()})

        return history


class RateLimitError(Exception):
    """Raised when rate limit is detected."""
    pass
