"""
Gemini browser automation interface.

Handles interaction with Gemini web interface, including:
- Message sending
- Response extraction  
- Deep Think mode toggle
- Rate limit detection
"""

import asyncio
from datetime import datetime
from typing import Optional

from .base import BaseLLMInterface, rate_tracker


class GeminiInterface(BaseLLMInterface):
    """Interface for Gemini web UI automation."""
    
    model_name = "gemini"
    
    def __init__(self):
        super().__init__()
        self.deep_think_enabled = False
    
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
        """
        try:
            print(f"[gemini] Attempting to enable Deep Think...")
            
            # Look for Deep Think toggle/button
            deep_think_selectors = [
                "button[aria-label*='Deep']",
                "button:has-text('Deep Think')",
                "mat-button-toggle:has-text('Deep')",
                "[data-test-id='deep-think-toggle']",
            ]
            
            for selector in deep_think_selectors:
                btn = await self.page.query_selector(selector)
                if btn:
                    is_visible = await btn.is_visible()
                    if is_visible:
                        # Check if already enabled
                        aria_pressed = await btn.get_attribute("aria-pressed")
                        classes = await btn.get_attribute("class") or ""
                        
                        if aria_pressed != "true" and "selected" not in classes:
                            await btn.click()
                            await asyncio.sleep(1)
                            print(f"[gemini] Enabled Deep Think")
                        else:
                            print(f"[gemini] Deep Think already enabled")
                        
                        self.deep_think_enabled = True
                        return True
            
            print(f"[gemini] Deep Think toggle not found")
            return False
            
        except Exception as e:
            print(f"[gemini] Could not enable Deep Think: {e}")
            return False
    
    async def start_new_chat(self):
        """Start a new conversation."""
        try:
            new_chat_selectors = [
                "button[aria-label='New chat']",
                "a[href*='/new']",
                "button:has-text('New chat')",
                "[data-test-id='new-chat']",
            ]
            
            for selector in new_chat_selectors:
                btn = await self.page.query_selector(selector)
                if btn:
                    is_visible = await btn.is_visible()
                    if is_visible:
                        await btn.click()
                        await asyncio.sleep(2)
                        print(f"[gemini] Started new chat")
                        return
            
            # Alternative: navigate to app root
            await self.page.goto("https://gemini.google.com/app")
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"[gemini] Could not start new chat: {e}")
    
    async def send_message(self, message: str, use_deep_think: bool = True) -> str:
        """
        Send a message to Gemini and wait for response.
        
        Args:
            message: The message to send
            use_deep_think: Whether to try enabling Deep Think mode
            
        Returns the response text.
        """
        if not rate_tracker.is_available(self.model_name):
            raise RateLimitError("Gemini is rate-limited")
        
        # Try to enable Deep Think if requested
        if use_deep_think and not self.deep_think_enabled:
            await self.enable_deep_think()
        
        print(f"[gemini] Sending message ({len(message)} chars)...")
        
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
            
            # Click and type
            await input_elem.click()
            await asyncio.sleep(0.3)
            
            # Clear any existing content
            await self.page.keyboard.press("Control+a")
            await asyncio.sleep(0.1)
            
            # Type the message
            await self.page.keyboard.type(message, delay=5)
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
            
            # Wait for response
            response = await self._wait_for_response()
            
            # Check for rate limiting
            if self._check_rate_limit(response):
                rate_tracker.mark_limited(self.model_name)
                raise RateLimitError("Gemini rate limit detected")
            
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
        
        print(f"[gemini] Waiting for response (timeout: {timeout}s)...")
        
        start_time = datetime.now()
        last_response = ""
        stable_count = 0
        
        # Selectors for model responses
        response_selectors = [
            "message-content.model-response-text",
            "div.model-response-text",
            "div[data-message-id] .markdown-content",
            ".response-container .markdown",
        ]
        
        while (datetime.now() - start_time).seconds < timeout:
            for selector in response_selectors:
                messages = await self.page.query_selector_all(selector)
                if messages:
                    last_msg = messages[-1]
                    current_response = await last_msg.inner_text()
                    
                    if current_response == last_response and current_response:
                        stable_count += 1
                        if stable_count >= 5:  # More checks for Deep Think
                            # Check for loading indicator
                            loading_selectors = [
                                ".loading",
                                ".thinking",
                                "[aria-busy='true']",
                                "mat-spinner",
                            ]
                            is_loading = False
                            for ls in loading_selectors:
                                loading = await self.page.query_selector(ls)
                                if loading:
                                    is_visible = await loading.is_visible()
                                    if is_visible:
                                        is_loading = True
                                        break
                            
                            if not is_loading:
                                return current_response.strip()
                    else:
                        stable_count = 0
                        last_response = current_response
                    break
            
            await asyncio.sleep(2)  # Longer interval for Deep Think
        
        if last_response:
            print(f"[gemini] Timeout reached, returning partial response")
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


class RateLimitError(Exception):
    """Raised when rate limit is detected."""
    pass
