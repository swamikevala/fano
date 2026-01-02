"""
ChatGPT browser automation interface.

Handles interaction with ChatGPT web interface, including:
- Message sending
- Response extraction
- Rate limit detection
- Pro mode detection
"""

import asyncio
import re
from datetime import datetime
from typing import Optional

from .base import BaseLLMInterface, rate_tracker
from .model_selector import deep_mode_tracker


class ChatGPTInterface(BaseLLMInterface):
    """Interface for ChatGPT web UI automation."""

    model_name = "chatgpt"

    def __init__(self):
        super().__init__()
        self.pro_mode_enabled = False
        self.thinking_mode_enabled = False
        self.last_deep_mode_used = False
        self._current_mode = "default"  # "default", "thinking", "pro"

    async def connect(self):
        """Connect to ChatGPT."""
        await super().connect()
        # Wait for the chat interface to load
        await self._wait_for_ready()
        # Run selector health check
        await self._check_selectors()
    
    async def _wait_for_ready(self, timeout: int = 30):
        """Wait for ChatGPT interface to be ready."""
        print(f"[chatgpt] Waiting for interface to be ready...")
        
        # Multiple possible selectors for the input (ChatGPT changes these often)
        input_selectors = [
            "#prompt-textarea",  # Current as of late 2024
            "textarea[placeholder*='Message']",
            "textarea[data-id='root']",
            "div[contenteditable='true'][data-placeholder]",
            "textarea",
        ]
        
        for selector in input_selectors:
            try:
                element = await self.page.wait_for_selector(selector, timeout=5000)
                if element:
                    print(f"[chatgpt] Found input with selector: {selector}")
                    self._input_selector = selector
                    return
            except:
                continue
        
        print(f"[chatgpt] WARNING: Could not find input element. Current URL: {self.page.url}")
        print(f"[chatgpt] Page title: {await self.page.title()}")
        self._input_selector = None

    async def _check_selectors(self):
        """Check critical selectors are present."""
        await self.check_selector_health({
            "input": [
                "#prompt-textarea",
                "textarea[placeholder*='Message']",
                "textarea[data-id='root']",
                "div[contenteditable='true']",
            ],
            "send_button": [
                "[data-testid='send-button']",
                "button[aria-label*='Send']",
                "button[type='submit']",
            ],
            "model_switcher": [
                "[data-testid='model-switcher-dropdown-button']",
            ],
            "response_area": [
                "[data-message-author-role='assistant']",
                ".markdown",
            ],
        })

    async def start_new_chat(self):
        """Start a new conversation."""
        try:
            # Just navigate to root - more reliable than trying to click buttons
            # (sidebar elements often intercept clicks)
            await self.page.goto("https://chatgpt.com/")
            await asyncio.sleep(2)

            # Start a new logging session
            session_id = self.chat_logger.start_session()
            print(f"[chatgpt] Started new chat (session: {session_id})")

        except Exception as e:
            print(f"[chatgpt] Could not start new chat: {e}")
    
    async def enable_pro_mode(self) -> bool:
        """
        Try to enable Pro/Plus mode if available.
        Returns True if successfully enabled.
        """
        try:
            print(f"[chatgpt] Checking Pro mode status...")

            # Find the model switcher button (data-testid="model-switcher-dropdown-button")
            model_btn = await self.page.query_selector(
                "[data-testid='model-switcher-dropdown-button']"
            )

            if model_btn:
                is_visible = await model_btn.is_visible()
                if is_visible:
                    # Check aria-label or text for "Pro"
                    aria_label = await model_btn.get_attribute("aria-label") or ""
                    text = await model_btn.inner_text()
                    print(f"[chatgpt] Model selector: '{text}' (aria: {aria_label[:50]})")

                    # Check if already on Pro mode
                    if "pro" in aria_label.lower() or "pro" in text.lower():
                        print(f"[chatgpt] Already on Pro mode")
                        self.pro_mode_enabled = True
                        return True

                    # Not on Pro - click to open dropdown and select Pro
                    print(f"[chatgpt] Opening model selector to switch to Pro...")
                    await model_btn.click()
                    await asyncio.sleep(0.5)

                    # Look for Pro option in dropdown
                    pro_options = [
                        "[role='menuitem']:has-text('Pro')",
                        "[role='option']:has-text('Pro')",
                        "div:has-text('Pro'):not(:has(div))",  # Leaf element with Pro
                    ]

                    for selector in pro_options:
                        try:
                            option = await self.page.query_selector(selector)
                            if option:
                                opt_visible = await option.is_visible()
                                if opt_visible:
                                    opt_text = await option.inner_text()
                                    print(f"[chatgpt] Selecting Pro option: {opt_text[:30]}")
                                    await option.click()
                                    await asyncio.sleep(1)
                                    self.pro_mode_enabled = True
                                    return True
                        except Exception:
                            continue

                    # Close dropdown
                    await self.page.keyboard.press("Escape")
                    print(f"[chatgpt] Pro option not found in dropdown")

            # Fallback: assume Pro if we can't find selector (might be default)
            print(f"[chatgpt] Model selector not found, assuming Pro is active")
            self.pro_mode_enabled = True
            return True

        except Exception as e:
            print(f"[chatgpt] Could not enable Pro mode: {e}")
            return False

    async def enable_thinking_mode(self) -> bool:
        """
        Enable GPT-5.2 Thinking mode if available.
        This is the preferred mode for standard queries (not Pro).
        Returns True if successfully enabled.
        """
        if self._current_mode == "thinking":
            print(f"[chatgpt] Already in Thinking mode")
            return True

        try:
            print(f"[chatgpt] Enabling Thinking mode...")

            # Find the model switcher button
            model_btn = await self.page.query_selector(
                "[data-testid='model-switcher-dropdown-button']"
            )

            if model_btn:
                is_visible = await model_btn.is_visible()
                if is_visible:
                    # Check current mode
                    text = await model_btn.inner_text()
                    text_lower = text.lower()

                    # Check if already on Thinking mode
                    if "thinking" in text_lower or "5.2" in text_lower:
                        print(f"[chatgpt] Already on Thinking mode: {text}")
                        self.thinking_mode_enabled = True
                        self._current_mode = "thinking"
                        return True

                    # Click to open dropdown
                    print(f"[chatgpt] Opening model selector (current: {text})...")
                    await model_btn.click()
                    await asyncio.sleep(0.5)

                    # Look for Thinking/5.2 option in dropdown
                    thinking_options = [
                        "[role='menuitem']:has-text('Thinking')",
                        "[role='menuitem']:has-text('5.2')",
                        "[role='option']:has-text('Thinking')",
                        "[role='option']:has-text('5.2')",
                        "div:has-text('Thinking'):not(:has(div))",
                        "div:has-text('5.2'):not(:has(div))",
                    ]

                    for selector in thinking_options:
                        try:
                            option = await self.page.query_selector(selector)
                            if option:
                                opt_visible = await option.is_visible()
                                if opt_visible:
                                    opt_text = await option.inner_text()
                                    # Make sure it's not Pro or other variants
                                    if "pro" not in opt_text.lower():
                                        print(f"[chatgpt] Selecting Thinking option: {opt_text[:30]}")
                                        await option.click()
                                        await asyncio.sleep(1)
                                        self.thinking_mode_enabled = True
                                        self._current_mode = "thinking"
                                        return True
                        except Exception:
                            continue

                    # Close dropdown
                    await self.page.keyboard.press("Escape")
                    print(f"[chatgpt] Thinking option not found in dropdown")

            # If we can't find the selector, continue without error
            print(f"[chatgpt] Model selector not found, using default mode")
            return False

        except Exception as e:
            print(f"[chatgpt] Could not enable Thinking mode: {e}")
            return False

    async def send_message(self, message: str, use_pro_mode: bool = False, use_thinking_mode: bool = False) -> str:
        """
        Send a message to ChatGPT and wait for response.

        Args:
            message: The message to send
            use_pro_mode: Whether to use Pro mode (for Round 2 deep analysis)
            use_thinking_mode: Whether to use Thinking mode (for Round 1 standard)

        Returns the response text, or raises exception on error.
        Sets self.last_deep_mode_used to indicate if pro mode was used.
        """
        if not rate_tracker.is_available(self.model_name):
            raise RateLimitError("ChatGPT is rate-limited")

        # Track whether pro mode was used for this message
        self.last_deep_mode_used = False

        # Mode selection: Pro takes precedence over Thinking
        if use_pro_mode:
            if self._current_mode != "pro":
                success = await self.enable_pro_mode()
                if success:
                    self.last_deep_mode_used = True
                    self._current_mode = "pro"
                    print(f"[chatgpt] Pro mode ENABLED for this message")
            else:
                self.last_deep_mode_used = True
        elif use_thinking_mode:
            if self._current_mode != "thinking":
                success = await self.enable_thinking_mode()
                if success:
                    self._current_mode = "thinking"
                    print(f"[chatgpt] Thinking mode ENABLED for this message")

        mode_str = f" [{self._current_mode.upper()}]" if self._current_mode != "default" else ""
        print(f"[chatgpt]{mode_str} Sending message ({len(message)} chars)...")
        
        try:
            # Find the input element
            input_elem = None
            input_selectors = [
                "#prompt-textarea",
                "textarea[placeholder*='Message']",
                "textarea[data-id='root']",
                "div[contenteditable='true']",
            ]
            
            for selector in input_selectors:
                input_elem = await self.page.query_selector(selector)
                if input_elem:
                    print(f"[chatgpt] Using input selector: {selector}")
                    break
            
            if not input_elem:
                raise Exception("Could not find input element")
            
            # Clear and type message
            await input_elem.click()
            await asyncio.sleep(0.3)

            # Use fill for textarea, or clipboard paste for contenteditable
            # (keyboard.type() is too slow and newlines trigger premature submission)
            tag = await input_elem.evaluate("el => el.tagName.toLowerCase()")
            if tag == "textarea":
                await input_elem.fill(message)
            else:
                # For contenteditable, use clipboard paste to avoid newline issues
                await self._paste_text(input_elem, message)

            await asyncio.sleep(0.5)
            
            # Find and click send button
            send_selectors = [
                "button[data-testid='send-button']",
                "button[aria-label='Send message']",
                "button[aria-label='Send prompt']",
                "form button[type='submit']",
                "button:has(svg path[d*='M15.192'])",  # Send icon path
            ]
            
            sent = False
            for selector in send_selectors:
                send_btn = await self.page.query_selector(selector)
                if send_btn:
                    is_disabled = await send_btn.is_disabled()
                    if not is_disabled:
                        await send_btn.click()
                        sent = True
                        print(f"[chatgpt] Clicked send button: {selector}")
                        break
            
            if not sent:
                # Try pressing Enter as fallback
                print(f"[chatgpt] No send button found, pressing Enter...")
                await self.page.keyboard.press("Enter")
            
            # Wait for response
            response = await self._wait_for_response()

            # Check for rate limiting
            if self._check_rate_limit(response):
                rate_tracker.mark_limited(self.model_name)
                raise RateLimitError("ChatGPT rate limit detected")

            # Log the exchange locally
            self.chat_logger.log_exchange(message, response)

            # Try to rename the chat with datetime (best effort)
            await self._try_rename_chat()

            print(f"[chatgpt] Got response ({len(response)} chars)")
            return response
            
        except Exception as e:
            print(f"[chatgpt] Error: {e}")
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                rate_tracker.mark_limited(self.model_name)
            raise
    
    async def _wait_for_response(self, timeout: int = None) -> str:
        """Wait for ChatGPT to finish responding and extract text."""
        if timeout is None:
            timeout = self.config.get("response_timeout", 600)  # Pro mode can be slow

        print(f"[chatgpt] Waiting for response (timeout: {timeout}s)...")
        
        start_time = datetime.now()
        last_response = ""
        stable_count = 0
        
        # Selectors for assistant messages
        response_selectors = [
            "div[data-message-author-role='assistant']",
            "[data-testid='conversation-turn']:last-child div.markdown",
            "div.agent-turn div.markdown",
        ]
        
        while (datetime.now() - start_time).seconds < timeout:
            for selector in response_selectors:
                messages = await self.page.query_selector_all(selector)
                
                if messages:
                    # Get the last message
                    last_msg = messages[-1]
                    current_response = await last_msg.inner_text()
                    
                    # Check if response is stable (hasn't changed)
                    if current_response == last_response and current_response:
                        stable_count += 1
                        if stable_count >= 3:  # Stable for 3 checks
                            # Also check if streaming indicator is gone
                            streaming = await self.page.query_selector(
                                "button[aria-label='Stop generating'], button[aria-label='Stop streaming']"
                            )
                            if not streaming:
                                return current_response.strip()
                    else:
                        stable_count = 0
                        last_response = current_response
                    break
            
            await asyncio.sleep(1)
        
        # Timeout - return whatever we have
        if last_response:
            print(f"[chatgpt] Timeout reached, returning partial response")
            return last_response.strip()
        raise TimeoutError("ChatGPT response timeout")
    
    async def get_conversation_history(self) -> list[dict]:
        """Extract full conversation history from current chat."""
        history = []
        
        messages = await self.page.query_selector_all(
            "div[data-message-author-role]"
        )
        
        for msg in messages:
            role = await msg.get_attribute("data-message-author-role")
            text = await msg.inner_text()
            history.append({"role": role, "content": text.strip()})
        
        return history


    async def _try_rename_chat(self):
        """Try to rename the current chat with session datetime (best effort)."""
        session_id = self.chat_logger.get_session_id()
        if not session_id:
            return

        try:
            # ChatGPT auto-generates titles, but we can try to find and click the edit button
            # This is fragile and may not work if UI changes - that's okay, logs are saved locally
            chat_title = f"Fano_{session_id}"

            # Look for the current chat title in sidebar and try to rename
            # Find the active/current conversation item
            active_chat = await self.page.query_selector(
                "nav li.bg-token-sidebar-surface-secondary, nav [class*='active'], nav a[class*='bg-']"
            )
            if active_chat:
                # Try to find edit/rename button
                edit_btn = await active_chat.query_selector(
                    "button[aria-label*='Rename'], button[aria-label*='Edit']"
                )
                if edit_btn:
                    await edit_btn.click()
                    await asyncio.sleep(0.5)
                    # Find the input and type the new name
                    input_elem = await self.page.query_selector("input[type='text']")
                    if input_elem:
                        await input_elem.fill(chat_title)
                        await self.page.keyboard.press("Enter")
                        print(f"[chatgpt] Renamed chat to: {chat_title}")
        except Exception as e:
            # Renaming is best-effort, don't fail if it doesn't work
            print(f"[chatgpt] Could not rename chat (non-critical): {e}")

    async def _paste_text(self, element, text: str):
        """
        Paste text into an element using DOM manipulation.
        This is faster than keyboard.type() and doesn't trigger form submission on newlines.
        Works with Trusted Types security policies.
        """
        # Use DOM manipulation instead of innerHTML (may be blocked by Trusted Types)
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
