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


class ChatGPTInterface(BaseLLMInterface):
    """Interface for ChatGPT web UI automation."""
    
    model_name = "chatgpt"
    
    async def connect(self):
        """Connect to ChatGPT."""
        await super().connect()
        # Wait for the chat interface to load
        await self._wait_for_ready()
    
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
    
    async def start_new_chat(self):
        """Start a new conversation."""
        try:
            # Look for "New chat" button/link
            new_chat_selectors = [
                "a[href='/']",
                "nav a:has-text('New chat')",
                "button:has-text('New chat')",
                "[data-testid='new-chat-button']",
            ]
            
            for selector in new_chat_selectors:
                btn = await self.page.query_selector(selector)
                if btn:
                    await btn.click()
                    await asyncio.sleep(2)
                    print(f"[chatgpt] Started new chat")
                    return
            
            # Alternative: just navigate to root
            await self.page.goto("https://chat.openai.com/")
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"[chatgpt] Could not start new chat: {e}")
    
    async def send_message(self, message: str) -> str:
        """
        Send a message to ChatGPT and wait for response.
        
        Returns the response text, or raises exception on error.
        """
        if not rate_tracker.is_available(self.model_name):
            raise RateLimitError("ChatGPT is rate-limited")
        
        print(f"[chatgpt] Sending message ({len(message)} chars)...")
        
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
            
            # Use fill for textarea, or keyboard for contenteditable
            tag = await input_elem.evaluate("el => el.tagName.toLowerCase()")
            if tag == "textarea":
                await input_elem.fill(message)
            else:
                await self.page.keyboard.type(message, delay=5)
            
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
            timeout = self.config.get("response_timeout", 300)
        
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


class RateLimitError(Exception):
    """Raised when rate limit is detected."""
    pass
