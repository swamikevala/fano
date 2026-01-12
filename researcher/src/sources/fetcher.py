"""
Content fetcher using Playwright browser automation.

Handles page fetching, content extraction, and caching.
"""

import asyncio
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import yaml

from shared.logging import get_logger
from .browser import ResearcherBrowser, BrowserPool

log = get_logger("researcher", "fetcher")


class ContentFetcher:
    """
    Fetches web content using Playwright browser automation.

    Features:
    - JavaScript rendering support
    - Rate limiting per domain
    - Content caching
    - Clean text extraction
    """

    def __init__(
        self,
        config_path: Path,
        cache_dir: Path,
        browser: Optional[ResearcherBrowser] = None,
        browser_pool: Optional[BrowserPool] = None
    ):
        """
        Initialize fetcher.

        Args:
            config_path: Path to settings.yaml
            cache_dir: Path to cache directory
            browser: Optional shared browser instance
            browser_pool: Optional browser pool for parallel fetching
        """
        self.config = self._load_config(config_path)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Browser - use provided or create own
        self._browser = browser
        self._browser_pool = browser_pool
        self._owns_browser = False

        # Rate limiting per domain
        self._domain_last_fetch: dict[str, datetime] = {}
        self._min_delay_seconds = 2

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    async def _get_browser(self) -> ResearcherBrowser:
        """Get browser instance."""
        if self._browser_pool:
            return await self._browser_pool.acquire()

        if self._browser is None:
            self._browser = ResearcherBrowser(headless=True)
            await self._browser.connect()
            self._owns_browser = True

        return self._browser

    async def _release_browser(self, browser: ResearcherBrowser):
        """Release browser back to pool if using pool."""
        if self._browser_pool:
            await self._browser_pool.release(browser)

    async def close(self):
        """Close owned browser resources."""
        if self._owns_browser and self._browser:
            await self._browser.disconnect()
            self._browser = None

    async def fetch(self, url: str, use_cache: bool = True) -> Optional[dict]:
        """
        Fetch content from URL using browser.

        Args:
            url: URL to fetch
            use_cache: Whether to use cached content

        Returns:
            Dict with content, title, text, hash, or None if failed
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                log.debug("fetcher.cache_hit", url=url[:50])
                return cached

        # Check rate limit
        domain = urlparse(url).netloc
        await self._wait_for_rate_limit(domain)

        # Fetch using browser
        browser = await self._get_browser()
        try:
            raw_content = await browser.fetch_page(url)

            if raw_content:
                # Process and enhance content
                result = self._process_content(raw_content)

                # Cache the result
                if result:
                    self._cache_content(url, result)
                    log.info("fetcher.fetch.success", url=url[:50], words=result.get("word_count", 0))

                return result

        finally:
            await self._release_browser(browser)

        return None

    async def fetch_multiple(self, urls: list[str]) -> list[dict]:
        """
        Fetch multiple URLs (uses pool if available).

        Args:
            urls: List of URLs to fetch

        Returns:
            List of successfully fetched content dicts
        """
        results = []
        for url in urls:
            content = await self.fetch(url)
            if content:
                results.append(content)
        return results

    def _process_content(self, raw: dict) -> dict:
        """
        Process raw browser content into clean result.

        Args:
            raw: Raw content from browser

        Returns:
            Processed content dict
        """
        text = raw.get("text", "")
        html = raw.get("html", "")

        # Clean text
        text = self._clean_text(text)

        # Compute content hash
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Detect features
        has_sanskrit = bool(re.search(r'[\u0900-\u097F]', text))  # Devanagari
        has_iast = bool(re.search(r'[āīūṛṝḷḹṃḥśṣṇṭḍ]', text))  # IAST diacritics
        has_verse_refs = bool(re.search(
            r'\b\d+\.\d+|\bverse\s+\d+|\bśloka\s+\d+|\bsūtra\s+\d+',
            text,
            re.IGNORECASE
        ))

        return {
            "url": raw["url"],
            "domain": raw["domain"],
            "title": raw.get("title", "").strip(),
            "text": text,
            "content_hash": content_hash,
            "content_type": "html",
            "has_sanskrit": has_sanskrit,
            "has_iast": has_iast,
            "has_verse_references": has_verse_refs,
            "fetched_at": raw.get("fetched_at", datetime.now().isoformat()),
            "word_count": len(text.split()),
        }

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    async def _wait_for_rate_limit(self, domain: str):
        """Wait if necessary to respect rate limits."""
        if domain in self._domain_last_fetch:
            elapsed = (datetime.now() - self._domain_last_fetch[domain]).total_seconds()
            if elapsed < self._min_delay_seconds:
                wait_time = self._min_delay_seconds - elapsed
                log.debug("fetcher.rate_limit_wait", domain=domain, wait=wait_time)
                await asyncio.sleep(wait_time)

        self._domain_last_fetch[domain] = datetime.now()

    def _get_cached(self, url: str) -> Optional[dict]:
        """Get cached content for URL."""
        import json

        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        return None

    def _cache_content(self, url: str, content: dict) -> None:
        """Cache content for URL."""
        import json

        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("fetcher.cache_write_failed", url=url[:50], error=str(e))
