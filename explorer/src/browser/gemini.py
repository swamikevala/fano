"""
Gemini browser automation interface.

Handles interaction with Gemini web interface, including:
- Message sending
- Response extraction  
- Deep Think mode toggle
- Rate limit detection
"""

import asyncio
import re
from datetime import datetime
from typing import Optional

from shared.logging import get_logger

from .base import BaseLLMInterface, rate_tracker
from .model_selector import deep_mode_tracker

log = get_logger("explorer", "browser.gemini")


class GeminiInterface(BaseLLMInterface):
    """Interface for Gemini web UI automation."""

    model_name = "gemini"

    def __init__(self):
        super().__init__()
        self.deep_think_enabled = False
        self.last_deep_mode_used = False
        self._response_in_progress = False  # Guard against concurrent operations
        self._deep_think_confirmation_done = False  # Track if we've handled confirmation

    def get_copy_button_selectors(self) -> list[str]:
        """Gemini copy button selectors - must target response copy, not code block copy."""
        return [
            # Response action buttons are in a specific container
            '.response-actions button[aria-label="Copy"]',
            'message-actions button[aria-label="Copy"]',
            '.model-response-actions button[aria-label="Copy"]',
            # Generic fallbacks - use last to get most recent response's button
            'button[aria-label="Copy response"]',
            'button[aria-label="Copy to clipboard"]',
            'button[mattooltip="Copy"]',
        ]

    async def _extract_response_text(self, response_element) -> str:
        """
        Extract response text with special handling for Gemini's math rendering.

        Gemini renders math using special elements (MathML, KaTeX, etc.) that
        clipboard doesn't capture properly. This method tries:
        1. DOM-based extraction with math element handling (best for math)
        2. Clipboard extraction (fallback)
        3. inner_text (last resort)
        """
        # Try DOM-based extraction with math handling FIRST
        # Clipboard often loses math symbols from rendered elements
        try:
            extracted = await self._extract_with_math_handling(response_element)
            if extracted and len(extracted.strip()) > 10:
                log.info("browser.gemini.dom_extraction_success", chars=len(extracted))
                return extracted
        except Exception as e:
            log.warning("browser.gemini.dom_extraction_failed", error=str(e))

        log.info("browser.gemini.dom_failed_trying_clipboard")

        # Try clipboard as fallback
        clipboard_text = await self._extract_via_clipboard(response_element)
        if clipboard_text:
            log.info("browser.gemini.clipboard_fallback_success", chars=len(clipboard_text))
            return clipboard_text

        # Final fallback
        log.warning("browser.gemini.using_inner_text_fallback")
        return await response_element.inner_text()

    async def _extract_with_math_handling(self, response_element) -> str:
        """
        Extract text from response element with special handling for math elements.

        Gemini may render math as:
        - MathML elements (<math>)
        - KaTeX spans with data-* attributes
        - Elements with aria-label containing the formula
        - Annotation elements with LaTeX source
        """
        # Use JavaScript to walk the DOM and extract text with math preservation
        extracted = await response_element.evaluate("""(el) => {
            function extractWithMath(node) {
                let result = '';

                for (const child of node.childNodes) {
                    // Text nodes - just add the text
                    if (child.nodeType === Node.TEXT_NODE) {
                        result += child.textContent;
                        continue;
                    }

                    // Element nodes - check for math
                    if (child.nodeType === Node.ELEMENT_NODE) {
                        const tagName = child.tagName.toLowerCase();

                        // MathML: look for annotation with LaTeX
                        if (tagName === 'math') {
                            // Try to find annotation-xml or annotation with LaTeX
                            const annotation = child.querySelector('annotation[encoding*="tex"], annotation[encoding*="latex"], annotation-xml[encoding*="tex"]');
                            if (annotation) {
                                result += ' $' + annotation.textContent.trim() + '$ ';
                                continue;
                            }
                            // Fallback: check alttext attribute
                            const alttext = child.getAttribute('alttext');
                            if (alttext) {
                                result += ' $' + alttext.trim() + '$ ';
                                continue;
                            }
                            // Last resort: get aria-label
                            const ariaLabel = child.getAttribute('aria-label');
                            if (ariaLabel) {
                                result += ' $' + ariaLabel.trim() + '$ ';
                                continue;
                            }
                        }

                        // KaTeX elements - look for source in data attributes
                        if (child.classList && (child.classList.contains('katex') || child.classList.contains('katex-display'))) {
                            // KaTeX stores source in data attribute or annotation
                            const annotation = child.querySelector('annotation[encoding*="tex"]');
                            if (annotation) {
                                const isDisplay = child.classList.contains('katex-display');
                                const delim = isDisplay ? '$$' : '$';
                                result += ' ' + delim + annotation.textContent.trim() + delim + ' ';
                                continue;
                            }
                        }

                        // Check for aria-label on math containers
                        if (child.getAttribute('role') === 'math' || child.classList.contains('math-container')) {
                            const ariaLabel = child.getAttribute('aria-label');
                            if (ariaLabel) {
                                result += ' $' + ariaLabel.trim() + '$ ';
                                continue;
                            }
                        }

                        // Check data-formula attribute (some renderers use this)
                        const dataFormula = child.getAttribute('data-formula') || child.getAttribute('data-latex') || child.getAttribute('data-tex');
                        if (dataFormula) {
                            result += ' $' + dataFormula.trim() + '$ ';
                            continue;
                        }

                        // Skip script/style elements
                        if (tagName === 'script' || tagName === 'style') {
                            continue;
                        }

                        // Code blocks - preserve with backticks
                        if (tagName === 'code') {
                            const parentTag = child.parentElement ? child.parentElement.tagName.toLowerCase() : '';
                            if (parentTag === 'pre') {
                                result += '\\n```\\n' + child.textContent + '\\n```\\n';
                            } else {
                                result += '`' + child.textContent + '`';
                            }
                            continue;
                        }

                        // Line breaks
                        if (tagName === 'br') {
                            result += '\\n';
                            continue;
                        }

                        // Block elements - add newlines
                        if (['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr'].includes(tagName)) {
                            result += '\\n' + extractWithMath(child) + '\\n';
                            continue;
                        }

                        // Recurse into other elements
                        result += extractWithMath(child);
                    }
                }

                return result;
            }

            return extractWithMath(el);
        }""")

        # Clean up extra whitespace
        if extracted:
            # Normalize multiple newlines
            extracted = re.sub(r'\n{3,}', '\n\n', extracted)
            # Normalize spaces around math delimiters
            extracted = re.sub(r'\s*\$\s*', ' $', extracted)
            extracted = re.sub(r'\s*\$\$\s*', '\n$$', extracted)

        return extracted.strip() if extracted else ""
    
    async def connect(self):
        """Connect to Gemini."""
        from .base import AuthenticationRequired
        await super().connect()
        await self._wait_for_ready()
        is_logged_in = await self._check_login_status()
        if not is_logged_in:
            raise AuthenticationRequired("Gemini login required")
        await self._check_selectors()
    
    async def _check_login_status(self):
        """Check if we're logged in to Gemini."""
        # More reliable check: look for the chat input (only present when logged in)
        # If input is found, we're logged in
        if hasattr(self, '_input_selector') and self._input_selector:
            # Input was found during _wait_for_ready, we're logged in
            print(f"[gemini] Appears to be logged in (input found)")
            return True

        # Fallback: check for prominent sign-in button
        # Only check for actual sign-in buttons, not any Google link
        sign_in_selectors = [
            "button:has-text('Sign in')",
            "[data-test-id='sign-in-button']",
            "a[href*='accounts.google.com/ServiceLogin']",  # More specific
        ]

        for selector in sign_in_selectors:
            try:
                sign_in = await self.page.query_selector(selector)
                if sign_in:
                    is_visible = await sign_in.is_visible()
                    # Also check if it's prominent (not just a small link)
                    if is_visible:
                        box = await sign_in.bounding_box()
                        # If sign-in button is large/prominent, we're not logged in
                        if box and box['width'] > 50 and box['height'] > 20:
                            print(f"[gemini] WARNING: Not logged in! Sign-in button visible.")
                            print(f"[gemini] Please run 'python fano_explorer.py auth' and log in to Gemini.")
                            return False
            except Exception:
                continue

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
            except Exception:
                continue
        
        print(f"[gemini] WARNING: Could not find input element")
        print(f"[gemini] Current URL: {self.page.url}")
        self._input_selector = None

    async def _check_selectors(self):
        """Check critical selectors are present."""
        await self.check_selector_health({
            "input": [
                "div.ql-editor",
                "rich-textarea",
                "div[contenteditable='true']",
            ],
            "send_button": [
                "button.send-button",
                "[aria-label*='Send']",
                "button[mattooltip='Send message']",
            ],
            "response_area": [
                "message-content",
                ".model-response",
                "[data-message-id]",
            ],
            "tools_menu": [
                "button[aria-label*='Select tools']",
                "button:has-text('Tools')",
            ],
        })

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
            log.warning("[gemini] enable_deep_think blocked: response in progress!")
            return self.deep_think_enabled

        # Prevent re-enabling if already enabled (avoids double "new chat" clicks)
        if self.deep_think_enabled:
            log.info("[gemini] Deep Think already enabled, skipping")
            return True

        log.info("[gemini] Enabling Deep Think mode...")

        try:
            # Step 1: Click the "Tools" button or model selector to open the menu
            menu_opened = False

            # Search ALL clickable elements, not just buttons
            all_clickable = await self.page.query_selector_all("button, [role='button'], a, div[tabindex], span[tabindex]")
            element_texts = []  # Collect for debugging

            # Priority 1: Look for "Tools" (exact match first)
            for elem in all_clickable:
                try:
                    is_visible = await elem.is_visible()
                    if not is_visible:
                        continue
                    text = (await elem.inner_text() or "").strip()
                    aria = (await elem.get_attribute("aria-label") or "")
                    if text or aria:
                        element_texts.append(f"'{text}' (aria: '{aria}')")

                    if text.lower() == 'tools' or 'tools' in aria.lower():
                        log.info(f"[gemini] Clicking Tools: '{text}' (aria: '{aria}')")
                        await elem.click()
                        await asyncio.sleep(1.0)
                        menu_opened = True
                        break
                except Exception:
                    continue

            # Priority 2: Look for model selectors if Tools not found
            if not menu_opened:
                model_patterns = ['ultra', 'pro', 'flash', 'model', 'gemini']
                for elem in all_clickable:
                    try:
                        is_visible = await elem.is_visible()
                        if not is_visible:
                            continue
                        text = (await elem.inner_text() or "").strip().lower()
                        aria = (await elem.get_attribute("aria-label") or "").lower()

                        for pattern in model_patterns:
                            if pattern in text or pattern in aria:
                                log.info(f"[gemini] Clicking model selector: '{text}' (aria: '{aria}')")
                                await elem.click()
                                await asyncio.sleep(1.0)
                                menu_opened = True
                                break
                        if menu_opened:
                            break
                    except Exception:
                        continue

            if not menu_opened:
                log.warning(f"[gemini] Tools/model selector not found. Visible elements: {element_texts[:15]}")
                return False

            # Step 2: Click Deep Think option (NOT Deep Research - that's different)
            deep_think_clicked = False
            deep_think_patterns = ['deep think']  # Only Deep Think, not Deep Research

            # Wait a moment for menu to fully render
            await asyncio.sleep(0.5)

            # Use JavaScript to find and click the correct menu item
            # This bypasses overlay issues and finds the exact element
            deep_think_clicked = await self.page.evaluate("""() => {
                // Look for menu items with specific text - only "deep think"
                const patterns = ['deep think'];

                // Get all potential clickable elements
                const elements = document.querySelectorAll('button, [role="menuitem"], [role="option"], mat-option, li, a, span');

                for (const elem of elements) {
                    // Skip invisible elements
                    const rect = elem.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;

                    // Get direct text content (not nested children's full text)
                    let text = '';
                    for (const node of elem.childNodes) {
                        if (node.nodeType === Node.TEXT_NODE) {
                            text += node.textContent;
                        }
                    }
                    // Also check textContent if no direct text found
                    if (!text.trim()) {
                        text = elem.textContent || '';
                    }

                    const textLower = text.toLowerCase().trim();

                    // Skip if this is a container with too much text (contains multiple items)
                    if (textLower.length > 50) continue;

                    // Check for exact or close match
                    for (const pattern of patterns) {
                        if (textLower.includes(pattern)) {
                            console.log('[gemini] Clicking:', text.trim());
                            elem.click();
                            return true;
                        }
                    }
                }
                return false;
            }""")

            if deep_think_clicked:
                log.info(f"[gemini] Deep Think clicked via JavaScript")
            else:
                # Fallback: try to find by aria-label or specific attributes
                log.info(f"[gemini] JS click failed, trying selector fallback...")

                # Search all visible elements for Deep Think options
                all_elements = await self.page.query_selector_all("[role='menuitem'], [role='option'], mat-option, button.menu-item")
                menu_texts = []  # Collect for debugging

                for elem in all_elements:
                    try:
                        is_visible = await elem.is_visible()
                        if not is_visible:
                            continue
                        # Get text and normalize
                        raw_text = (await elem.inner_text() or "")
                        text = ' '.join(raw_text.split()).strip()

                        # Skip long texts (containers)
                        if len(text) > 50:
                            continue

                        if text:
                            menu_texts.append(text)

                        text_lower = text.lower()

                        for pattern in deep_think_patterns:
                            if pattern in text_lower:
                                log.info(f"[gemini] Found Deep Think option: '{text}'")
                                await elem.click(force=True)
                                deep_think_clicked = True
                                break
                        if deep_think_clicked:
                            break
                    except Exception:
                        continue

            if not deep_think_clicked:
                log.warning(f"[gemini] Deep Think option not found. Menu items: {menu_texts[:20] if 'menu_texts' in dir() else 'N/A'}")
                # Take screenshot for debugging
                try:
                    screenshot_path = self.chat_logger.log_dir / "deep_think_menu_debug.png"
                    await self.page.screenshot(path=str(screenshot_path))
                    log.info(f"[gemini] Menu screenshot saved to: {screenshot_path}")
                except Exception:
                    pass
                return False

            # Step 3: Handle confirmation dialog ONCE
            # Mark that we're about to handle confirmation to prevent double-handling
            if not self._deep_think_confirmation_done:
                log.info("[gemini] Waiting for confirmation dialog...")
                await asyncio.sleep(1)
                await self._handle_mode_change_confirmation()
                self._deep_think_confirmation_done = True
                log.info("[gemini] Confirmation dialog handled")

            # Step 4: Wait for new chat to fully load
            log.info("[gemini] Waiting for new chat to load...")
            await asyncio.sleep(3)  # Increased from 2 to 3 seconds

            # Step 5: Verify Deep Think is actually enabled by checking page state
            verified = await self._verify_deep_think_active()
            if verified:
                self.deep_think_enabled = True
                log.info("[gemini] Deep Think enabled and VERIFIED")
                return True
            else:
                log.warning("[gemini] Deep Think clicked but NOT verified as active!")
                # Try to take a screenshot for debugging
                try:
                    screenshot_path = self.chat_logger.log_dir / "deep_think_debug.png"
                    await self.page.screenshot(path=str(screenshot_path))
                    log.info(f"[gemini] Debug screenshot saved to: {screenshot_path}")
                except Exception:
                    pass
                return False

        except GeminiQuotaExhausted:
            # Propagate quota exhaustion - this is a critical error
            raise
        except Exception as e:
            log.error(f"[gemini] Could not enable Deep Think: {e}")
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
                    log.info(f"[gemini] Found confirmation dialog: {selector}")
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
                        log.info(f"[gemini] Clicking confirmation button: '{text}'")
                        await btn.click()
                        await asyncio.sleep(1)
                        return
            except Exception:
                continue

        if dialog_found:
            log.warning("[gemini] Dialog found but couldn't find confirm button")

    async def _verify_deep_think_active(self) -> bool:
        """
        Verify that Deep Think mode is actually active by checking page indicators.

        Returns True if we can confirm Deep Think is active.
        Raises GeminiQuotaExhausted if quota limit message is detected.
        """
        try:
            # Look for indicators that Deep Think is active
            page_text = await self.page.inner_text("body")
            page_text_lower = page_text.lower()

            # FIRST: Check for quota exhaustion message
            # Pattern: "You've reached your limit for chats with Deep Think until Jan 10, 2:56 PM"
            quota_patterns = [
                "you've reached your limit",
                "reached your limit for chats with deep think",
                "limit for deep think until",
            ]

            for pattern in quota_patterns:
                if pattern in page_text_lower:
                    # Try to extract the resume time
                    resume_time = self._extract_resume_time(page_text)
                    log.error(f"[gemini] Deep Think quota exhausted! Resume time: {resume_time}")
                    raise GeminiQuotaExhausted(
                        f"Gemini Deep Think quota exhausted until {resume_time}",
                        resume_time=resume_time
                    )

            # Check for Deep Think indicators in the page
            deep_think_indicators = [
                "deep think",
                "thinking mode",
                "deep thinking",
            ]

            for indicator in deep_think_indicators:
                if indicator in page_text_lower:
                    log.info(f"[gemini] Found Deep Think indicator: '{indicator}'")
                    return True

            # Also check for model selector showing Deep Think
            model_selectors = await self.page.query_selector_all("[aria-label*='model'], [data-model], .model-selector, button:has-text('Deep')")
            for selector in model_selectors:
                try:
                    text = await selector.inner_text()
                    if text and "deep" in text.lower():
                        log.info(f"[gemini] Found Deep Think in model selector: '{text}'")
                        return True
                except Exception:
                    continue

            # Log what we see on the page for debugging
            log.warning(f"[gemini] Page text sample (first 500 chars): {page_text[:500]}")
            return False

        except GeminiQuotaExhausted:
            # Re-raise quota exhaustion - don't catch it here
            raise
        except Exception as e:
            log.error(f"[gemini] Error verifying Deep Think: {e}")
            return False

    def _extract_resume_time(self, page_text: str) -> str:
        """Extract the resume time from a quota exhaustion message."""
        import re

        # Pattern: "until Jan 10, 2:56 PM" or similar
        patterns = [
            r"until\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{1,2}:\d{2}\s*[AP]M)",
            r"until\s+(\d{1,2}:\d{2}\s*[AP]M)",
            r"until\s+(tomorrow)",
        ]

        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1)

        return "unknown time"

    async def start_new_chat(self):
        """Start a new conversation."""
        # CRITICAL: Never navigate away while waiting for a response
        if self._response_in_progress:
            log.warning("[gemini] start_new_chat blocked: response in progress!")
            return

        try:
            log.info("[gemini] Starting new chat, resetting Deep Think state")

            # Reset Deep Think state - new chat starts in standard mode
            self.deep_think_enabled = False
            self._deep_think_confirmation_done = False

            # Use firewall-aware navigation
            await self._navigate_with_firewall_check("https://gemini.google.com/app")

            # Start a new logging session
            session_id = self.chat_logger.start_session()
            print(f"[gemini] Started new chat (session: {session_id})")

        except Exception as e:
            print(f"[gemini] Could not start new chat: {e}")
    
    def _check_deep_think_overload(self, response_text: str) -> bool:
        """Check if response indicates Deep Think is temporarily overloaded."""
        patterns = self.config.get("selectors", {}).get("deep_think_overload_patterns", [])
        for pattern in patterns:
            if pattern.lower() in response_text.lower():
                return True
        return False

    async def _upload_images(self, images: list) -> bool:
        """
        Upload images to the current Gemini chat.

        Args:
            images: List of image attachments with 'filename', 'data' (base64), and 'media_type'

        Returns:
            True if upload succeeded, False otherwise
        """
        if not images:
            return True

        import base64
        import tempfile
        import os

        log.info("browser.gemini.image_upload.started", image_count=len(images))

        # Prepare temp files first
        temp_files = []
        filenames = []
        try:
            for img in images:
                # Decode base64 data and write to temp file
                image_data = base64.b64decode(img.data if hasattr(img, 'data') else img['data'])
                filename = img.filename if hasattr(img, 'filename') else img['filename']
                filenames.append(filename)

                # Determine extension from filename or media type
                ext = os.path.splitext(filename)[1]
                if not ext:
                    media_type = img.media_type if hasattr(img, 'media_type') else img.get('media_type', 'image/png')
                    ext_map = {
                        'image/png': '.png',
                        'image/jpeg': '.jpg',
                        'image/gif': '.gif',
                        'image/webp': '.webp',
                    }
                    ext = ext_map.get(media_type, '.png')

                # Create temp file
                fd, temp_path = tempfile.mkstemp(suffix=ext)
                os.write(fd, image_data)
                os.close(fd)
                temp_files.append(temp_path)

            log.debug("browser.gemini.image_upload.temp_files_created", count=len(temp_files))

            # Try Method 1: Use file_chooser event (handles native file dialogs)
            upload_success = await self._upload_via_file_chooser(temp_files, filenames)
            if upload_success:
                return True

            # Try Method 2: Direct file input (fallback)
            upload_success = await self._upload_via_file_input(temp_files, filenames)
            if upload_success:
                return True

            log.warning("browser.gemini.image_upload.all_methods_failed", image_count=len(images))
            return False

        except Exception as e:
            log.error("browser.gemini.image_upload.failed",
                     image_count=len(images),
                     error_type=type(e).__name__,
                     error=str(e))
            return False
        finally:
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    async def _upload_via_file_chooser(self, temp_files: list, filenames: list) -> bool:
        """
        Upload files using Playwright's file_chooser event.
        This handles native file dialogs that don't expose file input elements.

        Gemini's UI flow:
        1. Click "+" button to open attachment menu
        2. Click "Upload files" from menu - THIS triggers the file dialog
        """
        try:
            # Step 1: Find the "+" add button using JavaScript for more precision
            # Look for buttons with add/plus icons near the input area
            add_button = await self.page.evaluate_handle("""() => {
                // Find buttons with mat-icon containing 'add' text
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const icon = btn.querySelector('mat-icon');
                    if (icon) {
                        const iconText = icon.textContent.trim().toLowerCase();
                        if (iconText === 'add' || iconText === '+' || iconText === 'add_circle') {
                            if (btn.offsetParent !== null) { // visible
                                return btn;
                            }
                        }
                    }
                    // Also check aria-label
                    const label = btn.getAttribute('aria-label') || '';
                    if (label.toLowerCase().includes('add') && btn.offsetParent !== null) {
                        return btn;
                    }
                }
                return null;
            }""")

            if not add_button or await add_button.evaluate("el => el === null"):
                # Fallback to CSS selectors
                add_button_selectors = [
                    "button[aria-label='Add']",
                    "button[aria-label='+']",
                    "button[data-test-id='add-button']",
                    ".input-area button:first-of-type",
                ]

                add_button = None
                used_selector = None
                for selector in add_button_selectors:
                    try:
                        btn = await self.page.query_selector(selector)
                        if btn and await btn.is_visible():
                            add_button = btn
                            used_selector = selector
                            break
                    except Exception:
                        continue

                if not add_button:
                    log.debug("browser.gemini.image_upload.no_add_button_found")
                    return False
                log.info("browser.gemini.image_upload.found_add_button", selector=used_selector)
            else:
                log.info("browser.gemini.image_upload.found_add_button", selector="javascript:mat-icon=add")

            # Click to open the menu
            await add_button.click()
            await asyncio.sleep(0.8)  # Wait for menu to appear

            # Step 2: Find "Upload file" menu item using JavaScript
            upload_item = await self.page.evaluate_handle("""() => {
                // Look for menu items with "Upload" text
                const menuItems = document.querySelectorAll('[role="menuitem"], [role="option"], .mat-menu-item, .menu-item');
                for (const item of menuItems) {
                    const text = item.textContent.toLowerCase();
                    if (text.includes('upload') && item.offsetParent !== null) {
                        return item;
                    }
                }
                // Also check for any visible element with upload text
                const allElements = document.querySelectorAll('button, div[role="button"], span');
                for (const el of allElements) {
                    const text = el.textContent.trim().toLowerCase();
                    if (text === 'upload file' || text === 'upload files' || text === 'upload') {
                        if (el.offsetParent !== null) {
                            return el;
                        }
                    }
                }
                return null;
            }""")

            if not upload_item or await upload_item.evaluate("el => el === null"):
                # Fallback to CSS selectors
                upload_menu_selectors = [
                    "[role='menuitem']:has-text('Upload')",
                    "button:has-text('Upload file')",
                    "[aria-label*='Upload file']",
                    "span:has-text('Upload file')",
                ]

                upload_item = None
                for selector in upload_menu_selectors:
                    try:
                        item = await self.page.query_selector(selector)
                        if item and await item.is_visible():
                            upload_item = item
                            log.info("browser.gemini.image_upload.found_upload_menu", selector=selector)
                            break
                    except Exception:
                        continue

                if not upload_item:
                    log.debug("browser.gemini.image_upload.no_upload_menu_found")
                    # Try pressing Escape to close any open menu
                    await self.page.keyboard.press("Escape")
                    return False
            else:
                log.info("browser.gemini.image_upload.found_upload_menu", selector="javascript:upload-text")

            # Now set up file_chooser and click the upload menu item
            async with self.page.expect_file_chooser(timeout=5000) as fc_info:
                await upload_item.click()

            file_chooser = await fc_info.value
            await file_chooser.set_files(temp_files)
            log.info("browser.gemini.image_upload.file_chooser_success", file_count=len(temp_files))

            # Wait for upload processing
            await asyncio.sleep(1.5)

            # Verify upload by checking for preview
            if await self._verify_image_preview():
                log.info("browser.gemini.image_upload.verified", filenames=filenames)
                return True

            # Even if we can't verify preview, the upload might have worked
            log.info("browser.gemini.image_upload.completed_no_preview", filenames=filenames)
            return True

        except Exception as e:
            # TimeoutError means no file dialog appeared - try fallback
            if "timeout" in str(e).lower():
                log.debug("browser.gemini.image_upload.no_file_dialog", error=str(e))
            else:
                log.warning("browser.gemini.image_upload.file_chooser_error",
                          error_type=type(e).__name__, error=str(e))

            # Debug: log what buttons/menus are visible
            try:
                debug_info = await self.page.evaluate("""() => {
                    const info = { buttons: [], menuItems: [], matIcons: [] };
                    // Get all visible buttons with aria-labels
                    document.querySelectorAll('button').forEach(btn => {
                        if (btn.offsetParent !== null) {
                            info.buttons.push({
                                label: btn.getAttribute('aria-label') || '',
                                text: btn.textContent.trim().substring(0, 50)
                            });
                        }
                    });
                    // Get all menu items
                    document.querySelectorAll('[role="menuitem"]').forEach(item => {
                        if (item.offsetParent !== null) {
                            info.menuItems.push(item.textContent.trim().substring(0, 50));
                        }
                    });
                    // Get mat-icon contents
                    document.querySelectorAll('mat-icon').forEach(icon => {
                        if (icon.offsetParent !== null) {
                            info.matIcons.push(icon.textContent.trim());
                        }
                    });
                    return info;
                }""")
                log.debug("browser.gemini.image_upload.debug_info",
                         buttons=debug_info.get('buttons', [])[:5],
                         menu_items=debug_info.get('menuItems', [])[:5],
                         mat_icons=debug_info.get('matIcons', [])[:10])
            except Exception:
                pass

            # Try pressing Escape to close any open menu
            try:
                await self.page.keyboard.press("Escape")
            except Exception:
                pass
            return False

    async def _upload_via_file_input(self, temp_files: list, filenames: list) -> bool:
        """
        Upload files by finding and setting file input elements directly.
        Fallback method when file_chooser doesn't work.
        """
        try:
            # Try to find file input elements directly
            file_input_selectors = [
                "input[type='file'][accept*='image']",
                "input[type='file'][multiple]",
                "input[type='file']",
            ]

            file_input = None
            for selector in file_input_selectors:
                file_input = await self.page.query_selector(selector)
                if file_input:
                    log.debug("browser.gemini.image_upload.input_found", selector=selector)
                    break

            if not file_input:
                # Try clicking menu buttons to reveal file input
                # Gemini may use a two-step process: Add files -> Upload files
                add_button_selectors = [
                    "button[aria-label='Add files']",
                    "button[aria-label*='Add files']",
                    "button[aria-label='+']",
                    "button[aria-label='Add']",
                    "[aria-label='Add files']",
                ]

                for selector in add_button_selectors:
                    try:
                        btn = await self.page.query_selector(selector)
                        if btn and await btn.is_visible():
                            log.info("browser.gemini.image_upload.clicking_add_for_menu", selector=selector)
                            await btn.click()
                            await asyncio.sleep(0.8)  # Wait for menu

                            # Look for "Upload files" menu item
                            upload_menu_selectors = [
                                "text='Upload files'",
                                "[role='menuitem']:has-text('Upload files')",
                                "[role='menuitem']:has-text('Upload')",
                                "button:has-text('Upload files')",
                                "[aria-label='Upload files']",
                                "div:has-text('Upload files')",
                            ]

                            for menu_selector in upload_menu_selectors:
                                try:
                                    menu_item = await self.page.query_selector(menu_selector)
                                    if menu_item and await menu_item.is_visible():
                                        log.info("browser.gemini.image_upload.clicking_upload_menu", selector=menu_selector)
                                        await menu_item.click()
                                        await asyncio.sleep(0.5)
                                        break
                                except Exception:
                                    continue

                            # Check for file input again
                            for input_sel in file_input_selectors:
                                file_input = await self.page.query_selector(input_sel)
                                if file_input:
                                    log.info("browser.gemini.image_upload.input_found_after_menu", selector=input_sel)
                                    break

                            if file_input:
                                break
                    except Exception:
                        continue

            if not file_input:
                log.debug("browser.gemini.image_upload.no_file_input_found")
                return False

            # Upload files
            await file_input.set_input_files(temp_files)
            log.info("browser.gemini.image_upload.files_set", file_count=len(temp_files))

            # Wait for upload processing
            await asyncio.sleep(1.5)

            # Verify upload
            if await self._verify_image_preview():
                log.info("browser.gemini.image_upload.verified_via_input", filenames=filenames)
                return True

            log.info("browser.gemini.image_upload.completed_via_input", filenames=filenames)
            return True

        except Exception as e:
            log.warning("browser.gemini.image_upload.file_input_error",
                       error_type=type(e).__name__, error=str(e))
            return False

    async def _verify_image_preview(self) -> bool:
        """Check if image preview is visible to confirm upload."""
        preview_selectors = [
            "img[src*='blob:']",
            "img[alt*='Upload']",
            "img[alt*='preview']",
            ".uploaded-image",
            "[data-image-preview]",
            ".image-preview",
            "img[src*='data:']",
            # Gemini-specific preview containers
            ".attachment-preview",
            "[data-attachment]",
            ".file-preview",
        ]

        for selector in preview_selectors:
            try:
                preview = await self.page.query_selector(selector)
                if preview and await preview.is_visible():
                    log.debug("browser.gemini.image_upload.preview_found", selector=selector)
                    return True
            except Exception:
                continue

        return False

    async def send_message(self, message: str, use_deep_think: bool = False, images: list = None) -> str:
        """
        Send a message to Gemini and wait for response.

        Args:
            message: The message to send
            use_deep_think: Whether to use Deep Think mode (default False, controlled by orchestrator)
            images: Optional list of image attachments to include

        Returns the response text.
        Sets self.last_deep_mode_used to indicate if deep mode was actually used.
        """
        if not rate_tracker.is_available(self.model_name):
            raise RateLimitError("Gemini is rate-limited")

        # Upload images if provided
        if images:
            upload_success = await self._upload_images(images)
            if not upload_success:
                log.warning("browser.gemini.image_upload.skipped", reason="upload_failed")

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
        log.info(f"[gemini] Sending message in {mode_str} mode ({len(message)} chars)")

        # Get retry settings for Deep Think overload
        retry_config = self.config.get("deep_think_retry", {})
        retry_enabled = retry_config.get("enabled", True)
        retry_wait = retry_config.get("wait_seconds", 600)  # 10 min default
        max_retries = retry_config.get("max_retries", 3)

        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._send_message_once(message)

                # Check for Deep Think overload (temporary, retryable)
                if self.deep_think_enabled and self._check_deep_think_overload(response):
                    if retry_enabled and attempt <= max_retries:
                        log.warning(
                            f"[gemini] Deep Think overloaded (attempt {attempt}/{max_retries}). "
                            f"Waiting {retry_wait}s before retry..."
                        )
                        print(
                            f"[gemini] Deep Think busy - waiting {retry_wait // 60} minutes "
                            f"before retry (attempt {attempt}/{max_retries})..."
                        )
                        await asyncio.sleep(retry_wait)
                        # Start a new chat for the retry
                        await self.start_new_chat()
                        continue
                    else:
                        log.warning("[gemini] Deep Think overloaded, max retries reached")
                        raise RateLimitError("Deep Think overloaded after max retries")

                return response

            except RateLimitError:
                raise
            except Exception as e:
                if attempt <= max_retries and "overload" in str(e).lower():
                    log.warning(f"[gemini] Retrying after error: {e}")
                    await asyncio.sleep(retry_wait)
                    await self.start_new_chat()
                    continue
                raise

    async def _send_message_once(self, message: str) -> str:
        """Send a message once (without retry logic)."""
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

            # Start URL monitoring in background - will notify callback when URL changes
            asyncio.create_task(self._monitor_and_notify_url())

            # CRITICAL: Set guard to prevent any concurrent operations during response wait
            self._response_in_progress = True
            log.info(f"[gemini] Response guard ON - waiting for response...")

            try:
                # Wait for response
                response = await self._wait_for_response()
            finally:
                # Always clear the guard, even on error
                self._response_in_progress = False
                log.info(f"[gemini] Response guard OFF")

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
            timeout = self.config.get("response_timeout", 3600)  # Deep Think can take very long

        mode_str = "Deep Think" if self.deep_think_enabled else "standard"
        log.info(f"[gemini] Waiting for response in {mode_str} mode (timeout: {timeout}s)...")

        start_time = datetime.now()
        last_response = ""
        stable_count = 0
        last_log_time = start_time
        response_started = False

        # Selectors for model responses (the final answer, not thinking)
        response_selectors = [
            "message-content.model-response-text",
            "div.model-response-text",
            "div[data-message-id] .markdown-content",
            ".response-container .markdown",
        ]

        # Selectors that indicate Gemini is still thinking/processing
        # These must ALL be absent for response to be considered complete in Deep Think
        still_processing_selectors = [
            # Generic loading
            ".loading",
            "[aria-busy='true']",
            "mat-spinner",
            # Deep Think specific - look for animated/active thinking indicators
            "[data-thinking='true']",
            ".thinking-indicator",
            "div[class*='thinking'][class*='active']",
            "div[class*='thinking'][class*='progress']",
            # Animated elements that indicate processing
            ".animate-pulse",
            ".animate-spin",
            "[class*='loading']",
            "[class*='spinner']",
            # Stop/cancel button indicates still processing
            "button[aria-label*='Stop']",
            "button[aria-label*='Cancel']",
            "button:has-text('Stop')",
        ]

        # In Deep Think mode, wait longer before checking stability
        # because thinking output stabilizes before the actual response
        initial_wait = 30 if self.deep_think_enabled else 5

        log.info(f"[gemini] Initial wait: {initial_wait}s before stability checks")
        await asyncio.sleep(initial_wait)

        while (datetime.now() - start_time).seconds < timeout:
            elapsed = (datetime.now() - start_time).seconds

            # Log progress every 30 seconds
            if (datetime.now() - last_log_time).seconds >= 30:
                log.info(f"[gemini] Still waiting... ({elapsed}s elapsed, stable_count={stable_count}, response_started={response_started})")
                last_log_time = datetime.now()

            # First, check if Gemini is still actively processing
            is_still_processing = False
            for selector in still_processing_selectors:
                try:
                    elem = await self.page.query_selector(selector)
                    if elem:
                        is_visible = await elem.is_visible()
                        if is_visible:
                            is_still_processing = True
                            if elapsed > 60:  # Only log after 1 minute to reduce noise
                                log.debug(f"[gemini] Still processing (indicator: {selector})")
                            break
                except Exception:
                    continue

            if is_still_processing:
                # Reset stability if still processing
                stable_count = 0
                await asyncio.sleep(3)
                continue

            # Now try to get the response content
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
                    # For Deep Think, require much more stability (20 checks @ 3s = 60s stable)
                    # This ensures we don't capture thinking content as final response
                    required_stable = 20 if self.deep_think_enabled else 5

                    if stable_count >= required_stable:
                        # Double-check no processing indicators
                        final_check_processing = False
                        for selector in still_processing_selectors[:5]:  # Quick check of main ones
                            try:
                                elem = await self.page.query_selector(selector)
                                if elem and await elem.is_visible():
                                    final_check_processing = True
                                    break
                            except Exception:
                                continue

                        if not final_check_processing:
                            elapsed = (datetime.now() - start_time).seconds
                            # Use clipboard extraction for final response (preserves LaTeX)
                            for selector in response_selectors:
                                try:
                                    messages = await self.page.query_selector_all(selector)
                                    if messages:
                                        last_msg = messages[-1]
                                        final_response = await self._extract_response_text(last_msg)
                                        log.info(f"[gemini] Response complete ({elapsed}s, {len(final_response)} chars)")
                                        return final_response
                                except Exception:
                                    continue
                            # Fallback if extraction fails
                            log.info(f"[gemini] Response complete ({elapsed}s, {len(current_response)} chars)")
                            return current_response.strip()
                        else:
                            log.info(f"[gemini] Stable but still processing, continuing to wait...")
                            stable_count = required_stable // 2  # Partial reset
                else:
                    if last_response and current_response != last_response:
                        log.debug(f"[gemini] Response changed ({len(last_response)} -> {len(current_response)} chars)")
                    stable_count = 0
                    last_response = current_response

            await asyncio.sleep(3)  # Check every 3 seconds

        if last_response:
            elapsed = (datetime.now() - start_time).seconds
            log.warning(f"[gemini] Timeout reached after {elapsed}s, returning partial response ({len(last_response)} chars)")
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

    async def _monitor_and_notify_url(self):
        """Monitor URL and notify when it changes to include conversation ID."""
        base_url = "https://gemini.google.com/app"
        for _ in range(60):  # Check for 60 seconds
            try:
                current_url = self.page.url
                # Gemini conversation URLs have /app/ followed by a UUID
                if current_url != base_url and current_url.startswith(base_url) and len(current_url) > len(base_url) + 1:
                    log.info(f"[gemini] URL changed to conversation URL: {current_url}")
                    self._notify_url_change(current_url)
                    return
            except Exception as e:
                log.debug(f"[gemini] URL monitor error: {e}")
            await asyncio.sleep(1)
        log.warning("[gemini] URL monitor: never saw URL change to conversation URL")

    async def is_generating(self) -> bool:
        """Check if Gemini is currently generating a response."""
        # Selectors that indicate Gemini is still thinking/processing
        still_processing_selectors = [
            ".loading",
            "[aria-busy='true']",
            "mat-spinner",
            "[data-thinking='true']",
            ".thinking-indicator",
            ".animate-pulse",
            ".animate-spin",
            "[class*='loading']",
            "[class*='spinner']",
            "button[aria-label*='Stop']",
            "button[aria-label*='Cancel']",
            "button:has-text('Stop')",
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

        Used for recovery after restart - checks if a response exists and
        is not currently being generated.

        Returns:
            Response text if ready, None if still generating or no response.
        """
        # If still generating, return None
        if await self.is_generating():
            log.info("[gemini] try_get_response: still generating")
            return None

        # Selectors for model responses
        response_selectors = [
            "message-content.model-response-text",
            "div.model-response-text",
            "div[data-message-id] .markdown-content",
            ".response-container .markdown",
        ]

        for selector in response_selectors:
            try:
                messages = await self.page.query_selector_all(selector)
                if messages:
                    last_msg = messages[-1]
                    # Use clipboard extraction for better LaTeX preservation
                    response = await self._extract_response_text(last_msg)
                    if response and len(response) > 10:
                        log.info(f"[gemini] try_get_response: found response ({len(response)} chars)")
                        return response.strip()
            except Exception:
                continue

        log.info("[gemini] try_get_response: no response found")
        return None


class RateLimitError(Exception):
    """Raised when rate limit is detected."""
    pass


class GeminiQuotaExhausted(Exception):
    """
    Raised when Gemini Deep Think quota is exhausted.

    This is different from RateLimitError - quota exhaustion means
    Deep Think is unavailable until a specific time, but standard
    mode may still work.
    """
    def __init__(self, message: str, resume_time: str = None):
        super().__init__(message)
        self.resume_time = resume_time  # e.g., "Jan 10, 2:56 PM"
