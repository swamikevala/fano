"""
HTTP-based content fetching for researcher module.

Uses aiohttp for fetching web pages. No browser automation required.
"""

import asyncio
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, quote_plus

import aiohttp

from shared.logging import get_logger

log = get_logger("researcher", "http_fetcher")


class ResearcherBrowser:
    """
    HTTP-based web content fetcher.

    Replaces Playwright browser automation with simple HTTP requests.
    For most research purposes, direct HTTP is sufficient.
    """

    def __init__(self, headless: bool = True):
        """
        Initialize fetcher.

        Args:
            headless: Ignored (kept for API compatibility)
        """
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False

    async def connect(self):
        """Initialize HTTP session."""
        if self._connected:
            return

        self._session = aiohttp.ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._connected = True
        log.info("http_fetcher.connected")

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        log.info("http_fetcher.disconnected")

    async def fetch_page(self, url: str, wait_for: str = "domcontentloaded") -> Optional[dict]:
        """
        Fetch a page and extract content.

        Args:
            url: URL to fetch
            wait_for: Ignored (kept for API compatibility)

        Returns:
            Dict with url, title, text, html, or None if failed
        """
        if not self._connected:
            await self.connect()

        try:
            log.debug("http_fetcher.fetching", url=url[:60])

            async with self._session.get(url) as response:
                if response.status >= 400:
                    log.warning("http_fetcher.fetch_failed", url=url[:60], status=response.status)
                    return None

                html = await response.text()

                # Extract title
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else ""

                # Extract text content (simple approach - strip HTML tags)
                text = self._extract_text(html)

                return {
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "title": title,
                    "text": text,
                    "html": html,
                    "fetched_at": datetime.now().isoformat(),
                }

        except Exception as e:
            log.error("http_fetcher.fetch_error", url=url[:60], error=str(e))
            return None

    def _extract_text(self, html: str) -> str:
        """Extract text from HTML, removing scripts, styles, and tags."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Decode HTML entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def google_search(self, query: str, num_results: int = 10) -> list[dict]:
        """
        Search using DuckDuckGo HTML (Google blocks automated requests).

        Args:
            query: Search query
            num_results: Number of results to extract

        Returns:
            List of dicts with url, title, snippet
        """
        if not self._connected:
            await self.connect()

        try:
            # Use DuckDuckGo HTML which is more tolerant of automated requests
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            log.debug("http_fetcher.search", query=query[:50])

            async with self._session.get(search_url) as response:
                if response.status != 200:
                    log.warning("http_fetcher.search_failed", query=query[:50], status=response.status)
                    return []

                html = await response.text()

            # Parse DuckDuckGo results
            results = []
            # DuckDuckGo result pattern
            result_pattern = re.compile(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>.*?'
                r'<a[^>]+class="result__snippet"[^>]*>([^<]*)</a>',
                re.DOTALL
            )

            for match in result_pattern.finditer(html):
                if len(results) >= num_results:
                    break

                url = match.group(1)
                title = match.group(2).strip()
                snippet = match.group(3).strip()

                # Skip DuckDuckGo internal URLs
                if 'duckduckgo.com' in url:
                    continue

                results.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                })

            log.info("http_fetcher.search.complete", query=query[:50], results=len(results))
            return results

        except Exception as e:
            log.error("http_fetcher.search.error", query=query[:50], error=str(e))
            return []

    async def search_site(self, site: str, query: str) -> list[dict]:
        """
        Search within a specific site using site: operator.

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
    Pool of HTTP fetcher instances for parallel fetching.

    Simplified from browser pool - just manages HTTP sessions.
    """

    def __init__(self, pool_size: int = 2, headless: bool = True):
        """
        Initialize pool.

        Args:
            pool_size: Number of fetcher instances
            headless: Ignored (kept for API compatibility)
        """
        self.pool_size = pool_size
        self._fetchers: list[ResearcherBrowser] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._initialized = False

    async def initialize(self):
        """Initialize all fetchers in the pool."""
        if self._initialized:
            return

        for i in range(self.pool_size):
            fetcher = ResearcherBrowser()
            await fetcher.connect()
            self._fetchers.append(fetcher)
            await self._available.put(fetcher)

        self._initialized = True
        log.info("http_pool.initialized", pool_size=self.pool_size)

    async def acquire(self) -> ResearcherBrowser:
        """Get an available fetcher from the pool."""
        if not self._initialized:
            await self.initialize()
        return await self._available.get()

    async def release(self, fetcher: ResearcherBrowser):
        """Return a fetcher to the pool."""
        await self._available.put(fetcher)

    async def shutdown(self):
        """Shut down all fetchers."""
        for fetcher in self._fetchers:
            await fetcher.disconnect()
        self._fetchers.clear()
        self._initialized = False
        log.info("http_pool.shutdown")

    async def fetch_page(self, url: str) -> Optional[dict]:
        """Fetch a page using an available fetcher."""
        fetcher = await self.acquire()
        try:
            return await fetcher.fetch_page(url)
        finally:
            await self.release(fetcher)

    async def google_search(self, query: str, num_results: int = 10) -> list[dict]:
        """Perform search using an available fetcher."""
        fetcher = await self.acquire()
        try:
            return await fetcher.google_search(query, num_results)
        finally:
            await self.release(fetcher)
