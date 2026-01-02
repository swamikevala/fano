"""
Gemini browser automation interface.

Handles interaction with Gemini web interface, including:
- Message sending
- Response extraction  
- Deep Think mode toggle
- Rate limit detection
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .base import BaseLLMInterface, rate_tracker
from .model_selector import deep_mode_tracker

logger = logging.getLogger(__name__)


class GeminiInterface(BaseLLMInterface):
    """Interface for Gemini web UI automation."""

    model_name = "gemini"

    def __init__(self):
        super().__init__()
        self.deep_think_enabled = False
        self.last_deep_mode_used = False
        self._response_in_progress = False  # Guard against concurrent operations
        self._deep_think_confirmation_done = False  # Track if we've handled confirmation
    
    async def connect(self):
        """Connect to Gemini."""
        await super().connect()
        await self._wait_for_ready()
        await self._check_login_status()
    
    async def _check_login_status(self):
        """Check if we're logged in to Gemini."""
        # Look for sign-in button (means we're NOT logged in)
        sign_in_selectors = [
            "a[href*='accounts.google.com']",
            "button:has-text('Sign in')",
            "[data-test-id='sign-in-button']",
        ]
        
        for selector in sign_in_selectors:
            sign_in = await self.page.query_selector(selector)
            if sign_in:
                is_visible = await sign_in.is_visible()
                if is_visible:
                    print(f"[gemini] WARNING: Not logged in! Sign-in button visible.")
                    print(f"[gemini] Please run 'python fano_explorer.py auth' and log in to Gemini.")
                    return False
        
        print(f"[gemini] Appears to be logged in")
        return True
    
    async def _wait_for_ready(self, timeout: int = 30):
        """Wait for Gemini interface to be ready."""
        print(f"[gemini] Waiting for interface to be ready...")
        
        input_selectors = [
            "div.ql-editor",  # Quill editor
            "rich-textarea",
            ".input-area textarea",
            "div[contenteditable='true']",
            "textarea",
        ]
        
        for selector in input_selectors:
            try:
                element = await self.page.wait_for_selector(selector, timeout=5000)
                if element:
                    print(f"[gemini] Found input with selector: {selector}")
                    self._input_selector = selector
                    return
            except:
                continue
        
        print(f"[gemini] WARNING: Could not find input element")
        print(f"[gemini] Current URL: {self.page.url}")
        self._input_selector = None
    
    async def enable_deep_think(self) -> bool:
        """
        Enable Deep Think mode if available.
        Returns True if successfully enabled.

        Gemini's thinking mode is accessed via:
        1. Model selector dropdown (top of chat) -> Select "Thinking" model variant
        2. Or "Think deeper" toggle if available
        """
        # CRITICAL: Never try to enable while waiting for a response
        if self._response_in_progress:
            logger.warning("[gemini] enable_deep_think blocked: response in progress!")
            return self.deep_think_enabled

        # Prevent re-enabling if already enabled (avoids double "new chat" clicks)
        if self.deep_think_enabled:
            logger.info("[gemini] Deep Think already enabled, skipping")
            return True

        logger.info("[gemini] Enabling Deep Think mode...")

        try:
            # Step 1: Click the "Tools" button to open the tools panel
            tools_opened = False
            buttons = await self.page.query_selector_all("button")
            for btn in buttons:
                try:
                    is_visible = await btn.is_visible()
                    if not is_visible:
                        continue
                    text = (await btn.inner_text() or "").strip().lower()
                    aria = (await btn.get_attribute("aria-label") or "").lower()

                    if 'tools' in text or 'tools' in aria:
                        logger.info(f"[gemini] Clicking Tools button: '{text}'")
                        await btn.click()
                        await asyncio.sleep(1.0)
                        tools_opened = True
                        break
                except Exception:
                    continue

            if not tools_opened:
                logger.warning("[gemini] Tools button not found")
                return False

            # Step 2: Click Deep Think option
            deep_think_clicked = False
            try:
                deep_think_elem = await self.page.query_selector("text='Deep Think'")
                if deep_think_elem:
                    is_visible = await deep_think_elem.is_visible()
                    if is_visible:
                        logger.info("[gemini] Clicking Deep Think option")
                        await deep_think_elem.click()
                        deep_think_clicked = True
            except Exception as e:
                logger.debug(f"[gemini] First Deep Think selector failed: {e}")

            # Fallback: search elements for exact "Deep Think" text
            if not deep_think_clicked:
                all_elements = await self.page.query_selector_all("button, span, div, li, [role='menuitem'], [role='option']")
                for elem in all_elements:
                    try:
                        is_visible = await elem.is_visible()
                        if not is_visible:
                            continue
                        text = (await elem.inner_text() or "").strip()

                        if text.lower() == 'deep think':
                            logger.info("[gemini] Clicking Deep Think option (fallback)")
                            await elem.click()
                            deep_think_clicked = True
                            break
                    except Exception:
                        continue

            if not deep_think_clicked:
                logger.warning("[gemini] Deep Think option not found")
                return False

            # Step 3: Handle confirmation dialog ONCE
            # Mark that we're about to handle confirmation to prevent double-handling
            if not self._deep_think_confirmation_done:
                logger.info("[gemini] Waiting for confirmation dialog...")
                await asyncio.sleep(1)
                await self._handle_mode_change_confirmation()
                self._deep_think_confirmation_done = True
                logger.info("[gemini] Confirmation dialog handled")

            # Step 4: Wait for new chat to fully load
            logger.info("[gemini] Waiting for new chat to load...")
            await asyncio.sleep(3)  # Increased from 2 to 3 seconds

            # Mark as enabled
            self.deep_think_enabled = True
            logger.info("[gemini] Deep Think enabled successfully")
            return True

        except Exception as e:
            logger.error(f"[gemini] Could not enable Deep Think: {e}")
            return False

    async def _handle_mode_change_confirmation(self):
        """Handle any confirmation dialogs when switching modes."""
        await asyncio.sleep(0.5)  # Wait for dialog to appear

        # First, look for any visible dialog/modal
        dialog_selectors = [
            "[role='dialog']",
            "[role='alertdialog']",
            ".modal",
            ".dialog",
            "mat-dialog-container",
        ]

        dialog_found = False
        for selector in dialog_selectors:
            try:
                dialog = await self.page.query_selector(selector)
                if dialog and await dialog.is_visible():
                    dialog_found = True
                    logger.info(f"[gemini] Found confirmation dialog: {selector}")
                    break
            except Exception:
                continue

        # Button text patterns to look for (case insensitive search)
        confirm_patterns = [
            'start new chat', 'new chat', 'confirm', 'continue',
            'ok', 'start', 'yes', 'proceed', 'got it'
        ]

        # Find and click any matching button
        buttons = await self.page.query_selector_all("button")
        for btn in buttons:
            try:
                if not await btn.is_visible():
                    continue
                text = (await btn.inner_text() or "").strip().lower()
                for pattern in confirm_patterns:
                    if pattern in text:
                        logger.info(f"[gemini] Clicking confirmation button: '{text}'")
                        await btn.click()
                        await asyncio.sleep(1)
                        return
            except Exception:
                continue

        if dialog_found:
            logger.warning("[gemini] Dialog found but couldn't find confirm button")
    
    async def start_new_chat(self):
        """Start a new conversation."""
        # CRITICAL: Never navigate away while waiting for a response
        if self._response_in_progress:
            logger.warning("[gemini] start_new_chat blocked: response in progress!")
            return

        try:
            logger.info("[gemini] Starting new chat, resetting Deep Think state")

            # Reset Deep Think state - new chat starts in standard mode
            self.deep_think_enabled = False
            self._deep_think_confirmation_done = False

            # Just navigate to app root - more reliable than trying to click buttons
            await self.page.goto("https://gemini.google.com/app")
            await asyncio.sleep(2)

            # Start a new logging session
            session_id = self.chat_logger.start_session()
            print(f"[gemini] Started new chat (session: {session_id})")

        except Exception as e:
            print(f"[gemini] Could not start new chat: {e}")
    
    async def send_message(self, message: str, use_deep_think: bool = False) -> str:
        """
        Send a message to Gemini and wait for response.

        Args:
            message: The message to send
            use_deep_think: Whether to use Deep Think mode (default False, controlled by orchestrator)

        Returns the response text.
        Sets self.last_deep_mode_used to indicate if deep mode was actually used.
        """
        if not rate_tracker.is_available(self.model_name):
            raise RateLimitError("Gemini is rate-limited")

        # Track whether deep mode was used for this message
        self.last_deep_mode_used = False

        # Try to enable Deep Think if requested
        if use_deep_think:
            if not self.deep_think_enabled:
                success = await self.enable_deep_think()
                if success:
                    self.last_deep_mode_used = True
            # If already enabled from previous message, it's still active
            if self.deep_think_enabled:
                self.last_deep_mode_used = True

        # Log the mode we're using
        mode_str = "Deep Think" if self.deep_think_enabled else "standard"
        logger.info(f"[gemini] Sending message in {mode_str} mode ({len(message)} chars)")

        try:
            # Find the input area
            input_selectors = [
                "div.ql-editor",
                "rich-textarea div[contenteditable='true']",
                "div[contenteditable='true'][aria-label*='prompt']",
                "textarea",
            ]
            
            input_elem = None
            for selector in input_selectors:
                input_elem = await self.page.query_selector(selector)
                if input_elem:
                    is_visible = await input_elem.is_visible()
                    if is_visible:
                        print(f"[gemini] Using input selector: {selector}")
                        break
                    input_elem = None
            
            if not input_elem:
                raise Exception("Could not find Gemini input element")
            
            # Click and input message
            await input_elem.click()
            await asyncio.sleep(0.3)

            # Use clipboard paste to avoid newline triggering premature submission
            # (keyboard.type() is too slow and newlines cause issues)
            await self._paste_text(input_elem, message)
            await asyncio.sleep(0.5)
            
            # Find and click send button
            send_selectors = [
                "button[aria-label='Send message']",
                "button[aria-label='Submit']",
                "button.send-button",
                "button[mattooltip='Send message']",
                "button:has(mat-icon:has-text('send'))",
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
                        print(f"[gemini] Clicked send button: {selector}")
                        break
            
            if not sent:
                # Fallback: press Enter
                print(f"[gemini] No send button found, pressing Enter...")
                await self.page.keyboard.press("Enter")

            # CRITICAL: Set guard to prevent any concurrent operations during response wait
            self._response_in_progress = True
            logger.info(f"[gemini] Response guard ON - waiting for response...")

            try:
                # Wait for response
                response = await self._wait_for_response()
            finally:
                # Always clear the guard, even on error
                self._response_in_progress = False
                logger.info(f"[gemini] Response guard OFF")

            # Check for rate limiting
            if self._check_rate_limit(response):
                rate_tracker.mark_limited(self.model_name)
                raise RateLimitError("Gemini rate limit detected")

            # Log the exchange locally
            self.chat_logger.log_exchange(message, response)

            # Try to rename the chat with datetime (best effort)
            await self._try_rename_chat()

            print(f"[gemini] Got response ({len(response)} chars)")
            return response

        except Exception as e:
            print(f"[gemini] Error: {e}")
            if "rate" in str(e).lower() or "limit" in str(e).lower() or "quota" in str(e).lower():
                rate_tracker.mark_limited(self.model_name)
            raise
    
    async def _wait_for_response(self, timeout: int = None) -> str:
        """Wait for Gemini to finish responding and extract text."""
        if timeout is None:
            timeout = self.config.get("response_timeout", 600)  # Deep Think can be slow

        mode_str = "Deep Think" if self.deep_think_enabled else "standard"
        logger.info(f"[gemini] Waiting for response in {mode_str} mode (timeout: {timeout}s)...")

        start_time = datetime.now()
        last_response = ""
        stable_count = 0
        last_log_time = start_time

        # Selectors for model responses
        response_selectors = [
            "message-content.model-response-text",
            "div.model-response-text",
            "div[data-message-id] .markdown-content",
            ".response-container .markdown",
        ]

        # Extended loading selectors for Deep Think mode
        loading_selectors = [
            ".loading",
            ".thinking",
            "[aria-busy='true']",
            "mat-spinner",
            # Deep Think specific selectors
            "[data-thinking='true']",
            ".deep-think-indicator",
            "div[class*='thinking']",
            "div[class*='loading']",
        ]

        while (datetime.now() - start_time).seconds < timeout:
            # Log progress every 30 seconds
            elapsed = (datetime.now() - start_time).seconds
            if (datetime.now() - last_log_time).seconds >= 30:
                logger.info(f"[gemini] Still waiting... ({elapsed}s elapsed, stable_count={stable_count})")
                last_log_time = datetime.now()

            for selector in response_selectors:
                messages = await self.page.query_selector_all(selector)
                if messages:
                    last_msg = messages[-1]
                    current_response = await last_msg.inner_text()

                    if current_response == last_response and current_response:
                        stable_count += 1
                        # For Deep Think, require more stability checks (10 instead of 5)
                        required_stable = 10 if self.deep_think_enabled else 5
                        if stable_count >= required_stable:
                            # Check for loading indicator
                            is_loading = False
                            for ls in loading_selectors:
                                try:
                                    loading = await self.page.query_selector(ls)
                                    if loading:
                                        is_visible = await loading.is_visible()
                                        if is_visible:
                                            is_loading = True
                                            logger.debug(f"[gemini] Loading indicator found: {ls}")
                                            break
                                except Exception:
                                    continue

                            if not is_loading:
                                elapsed = (datetime.now() - start_time).seconds
                                logger.info(f"[gemini] Response complete ({elapsed}s, {len(current_response)} chars)")
                                return current_response.strip()
                    else:
                        if current_response != last_response:
                            logger.debug(f"[gemini] Response changed, resetting stable count")
                        stable_count = 0
                        last_response = current_response
                    break

            await asyncio.sleep(2)  # Longer interval for Deep Think

        if last_response:
            elapsed = (datetime.now() - start_time).seconds
            logger.warning(f"[gemini] Timeout reached after {elapsed}s, returning partial response ({len(last_response)} chars)")
            return last_response.strip()
        raise TimeoutError("Gemini response timeout")
    
    async def get_conversation_history(self) -> list[dict]:
        """Extract conversation history from current chat."""
        history = []
        
        # This will need adjustment based on actual Gemini DOM structure
        user_msgs = await self.page.query_selector_all(".user-message, div[data-message-author='user']")
        model_msgs = await self.page.query_selector_all(".model-response-text, div[data-message-author='model']")
        
        # Interleave user and model messages
        for i, (user, model) in enumerate(zip(user_msgs, model_msgs)):
            user_text = await user.inner_text()
            model_text = await model.inner_text()
            history.append({"role": "user", "content": user_text.strip()})
            history.append({"role": "assistant", "content": model_text.strip()})
        
        return history


    async def _try_rename_chat(self):
        """Try to rename the current chat with session datetime (best effort)."""
        session_id = self.chat_logger.get_session_id()
        if not session_id:
            return

        try:
            # Gemini doesn't have easy rename UI, but we try to find it
            # This is fragile - that's okay, logs are saved locally
            chat_title = f"Fano_{session_id}"

            # Look for menu/options button on current chat
            menu_btn = await self.page.query_selector(
                "button[aria-label*='Options'], button[aria-label*='Menu'], button[aria-label*='More']"
            )
            if menu_btn:
                await menu_btn.click()
                await asyncio.sleep(0.5)

                # Look for rename option
                rename_btn = await self.page.query_selector(
                    "button:has-text('Rename'), [role='menuitem']:has-text('Rename')"
                )
                if rename_btn:
                    await rename_btn.click()
                    await asyncio.sleep(0.5)

                    # Find input and type new name
                    input_elem = await self.page.query_selector("input[type='text']")
                    if input_elem:
                        await input_elem.fill(chat_title)
                        await self.page.keyboard.press("Enter")
                        print(f"[gemini] Renamed chat to: {chat_title}")
        except Exception as e:
            # Renaming is best-effort, don't fail if it doesn't work
            print(f"[gemini] Could not rename chat (non-critical): {e}")

    async def _paste_text(self, element, text: str):
        """
        Paste text into an element using DOM manipulation.
        This is faster than keyboard.type() and doesn't trigger form submission on newlines.
        Works with Trusted Types security policies.
        """
        # Use DOM manipulation instead of innerHTML (blocked by Trusted Types)
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


class RateLimitError(Exception):
    """Raised when rate limit is detected."""
    pass
