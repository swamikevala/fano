"""
Browser automation for researcher module.

Uses Playwright to fetch web pages and perform searches.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from shared.logging import get_logger

log = get_logger("researcher", "browser")

# Browser data directory (persistent sessions)
BROWSER_DATA_DIR = Path(__file__).parent.parent.parent / "browser_data"


class ResearcherBrowser:
    """
    Browser automation for fetching web content.

    Uses Playwright to:
    - Fetch pages from any URL
    - Perform Google searches
    - Handle JavaScript-rendered content
    """

    def __init__(self, headless: bool = True):
        """
        Initialize browser.

        Args:
            headless: Run browser in headless mode (no visible window)
        """
        self.headless = headless
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._connected = False

    async def connect(self):
        """Start browser and create page."""
        if self._connected:
            return

        BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        storage_dir = BROWSER_DATA_DIR / "researcher"
        storage_dir.mkdir(parents=True, exist_ok=True)

        self.playwright = await async_playwright().start()

        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(storage_dir),
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
            viewport={"width": 1280, "height": 720},
        )

        self.page = await self.context.new_page()
        self._connected = True
        log.info("browser.connected", headless=self.headless)

    async def disconnect(self):
        """Close browser."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
        self._connected = False
        log.info("browser.disconnected")

    async def fetch_page(self, url: str, wait_for: str = "domcontentloaded") -> Optional[dict]:
        """
        Fetch a page and extract content.

        Args:
            url: URL to fetch
            wait_for: Wait condition ('domcontentloaded', 'load', 'networkidle')

        Returns:
            Dict with url, title, text, html, or None if failed
        """
        if not self._connected:
            await self.connect()

        try:
            log.debug("browser.fetching", url=url[:60])

            response = await self.page.goto(url, wait_until=wait_for, timeout=30000)

            if not response or response.status >= 400:
                log.warning("browser.fetch_failed", url=url[:60], status=response.status if response else None)
                return None

            # Wait a bit for JS rendering
            await asyncio.sleep(1)

            # Extract content
            title = await self.page.title()

            # Get text content (without scripts/styles)
            text = await self.page.evaluate("""
                () => {
                    // Remove script and style elements
                    const elements = document.querySelectorAll('script, style, nav, footer, header, aside');
                    elements.forEach(el => el.remove());

                    // Get main content or body
                    const main = document.querySelector('main, article, .content, #content') || document.body;
                    return main ? main.innerText : '';
                }
            """)

            # Get full HTML for potential later parsing
            html = await self.page.content()

            return {
                "url": url,
                "domain": urlparse(url).netloc,
                "title": title,
                "text": text,
                "html": html,
                "fetched_at": datetime.now().isoformat(),
            }

        except Exception as e:
            log.error("browser.fetch_error", url=url[:60], error=str(e))
            return None

    async def google_search(self, query: str, num_results: int = 10) -> list[dict]:
        """
        Perform a Google search and extract results.

        Args:
            query: Search query
            num_results: Number of results to extract

        Returns:
            List of dicts with url, title, snippet
        """
        if not self._connected:
            await self.connect()

        try:
            # Navigate to Google
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={num_results}"
            log.debug("browser.google_search", query=query[:50])

            await self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)  # Wait for results to load

            # Extract search results
            results = await self.page.evaluate("""
                (maxResults) => {
                    const results = [];

                    // Google search result selectors (may need updates as Google changes)
                    const resultElements = document.querySelectorAll('div.g, div[data-hveid]');

                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {
                        const el = resultElements[i];

                        // Find link
                        const linkEl = el.querySelector('a[href^="http"]');
                        if (!linkEl) continue;

                        const url = linkEl.href;

                        // Skip Google's own pages
                        if (url.includes('google.com')) continue;

                        // Find title
                        const titleEl = el.querySelector('h3');
                        const title = titleEl ? titleEl.innerText : '';

                        // Find snippet
                        const snippetEl = el.querySelector('[data-sncf], .VwiC3b, span.st');
                        const snippet = snippetEl ? snippetEl.innerText : '';

                        if (url && title) {
                            results.push({ url, title, snippet });
                        }
                    }

                    return results;
                }
            """, num_results)

            log.info("browser.google_search.complete", query=query[:50], results=len(results))
            return results

        except Exception as e:
            log.error("browser.google_search.error", query=query[:50], error=str(e))
            return []

    async def search_site(self, site: str, query: str) -> list[dict]:
        """
        Search within a specific site using Google site: operator.

        Args:
            site: Domain to search (e.g., 'wisdomlib.org')
            query: Search query

        Returns:
            List of search results
        """
        full_query = f"site:{site} {query}"
        return await self.google_search(full_query)


class BrowserPool:
    """
    Pool of browser instances for parallel fetching.

    Manages multiple browser contexts for efficient fetching.
    """

    def __init__(self, pool_size: int = 2, headless: bool = True):
        """
        Initialize browser pool.

        Args:
            pool_size: Number of browser instances
            headless: Run in headless mode
        """
        self.pool_size = pool_size
        self.headless = headless
        self._browsers: list[ResearcherBrowser] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._initialized = False

    async def initialize(self):
        """Initialize all browsers in the pool."""
        if self._initialized:
            return

        for i in range(self.pool_size):
            browser = ResearcherBrowser(headless=self.headless)
            await browser.connect()
            self._browsers.append(browser)
            await self._available.put(browser)

        self._initialized = True
        log.info("browser_pool.initialized", pool_size=self.pool_size)

    async def acquire(self) -> ResearcherBrowser:
        """Get an available browser from the pool."""
        if not self._initialized:
            await self.initialize()
        return await self._available.get()

    async def release(self, browser: ResearcherBrowser):
        """Return a browser to the pool."""
        await self._available.put(browser)

    async def shutdown(self):
        """Shut down all browsers."""
        for browser in self._browsers:
            await browser.disconnect()
        self._browsers.clear()
        self._initialized = False
        log.info("browser_pool.shutdown")

    async def fetch_page(self, url: str) -> Optional[dict]:
        """Fetch a page using an available browser."""
        browser = await self.acquire()
        try:
            return await browser.fetch_page(url)
        finally:
            await self.release(browser)

    async def google_search(self, query: str, num_results: int = 10) -> list[dict]:
        """Perform Google search using an available browser."""
        browser = await self.acquire()
        try:
            return await browser.google_search(query, num_results)
        finally:
            await self.release(browser)
