"""
Web search using Playwright browser automation.

Performs Google searches and discovers sources from known sites.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import yaml

from shared.logging import get_logger
from .browser import ResearcherBrowser, BrowserPool

log = get_logger("researcher", "searcher")


class WebSearcher:
    """
    Web search using Playwright browser automation.

    Features:
    - Google search via browser
    - Site-specific searches
    - Direct URLs from known authoritative sites
    - Rate limiting
    """

    def __init__(
        self,
        config_path: Path,
        trusted_sources_path: Path,
        browser: Optional[ResearcherBrowser] = None,
        browser_pool: Optional[BrowserPool] = None
    ):
        """
        Initialize searcher.

        Args:
            config_path: Path to settings.yaml
            trusted_sources_path: Path to trusted_sources.yaml
            browser: Optional shared browser instance
            browser_pool: Optional browser pool
        """
        self.config = self._load_config(config_path)
        self.trusted_sources = self._load_trusted_sources(trusted_sources_path)

        # Browser
        self._browser = browser
        self._browser_pool = browser_pool
        self._owns_browser = False

        # Rate limiting
        self._search_count = 0
        self._search_reset_time = datetime.now()
        self._max_searches_per_hour = self.config.get("search", {}).get(
            "max_searches_per_hour", 30
        )

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def _load_trusted_sources(self, path: Path) -> dict:
        """Load trusted sources config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {"seed_sources": {"tier_1": [], "tier_2": []}}

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
        """Release browser back to pool."""
        if self._browser_pool:
            await self._browser_pool.release(browser)

    async def close(self):
        """Close owned browser resources."""
        if self._owns_browser and self._browser:
            await self._browser.disconnect()
            self._browser = None

    async def search(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> list[dict]:
        """
        Search for relevant sources.

        Args:
            query: Search query
            domain: Optional research domain to contextualize search
            limit: Maximum results to return

        Returns:
            List of search results with url, title, snippet
        """
        # Check rate limit
        if not self._check_rate_limit():
            log.warning("searcher.rate_limited", query=query[:50])
            return []

        results = []

        # First, get direct URLs from known authoritative sites
        direct_urls = self._get_direct_urls(query, domain)
        for url in direct_urls[:limit // 2]:
            results.append({
                "url": url,
                "title": f"Direct: {urlparse(url).netloc}",
                "snippet": "",
                "source_type": "direct",
            })

        # Then do Google search for remaining slots
        if len(results) < limit:
            browser = await self._get_browser()
            try:
                # Search trusted sites first
                trusted_results = await self._search_trusted_sites(browser, query, limit // 2)
                results.extend(trusted_results)

                # General search if still need more
                if len(results) < limit:
                    general_results = await browser.google_search(
                        query + " Hindu Sanskrit scripture",
                        num_results=limit - len(results)
                    )
                    for r in general_results:
                        r["source_type"] = "search"
                    results.extend(general_results)

            finally:
                await self._release_browser(browser)

        self._search_count += 1
        log.info(
            "searcher.search.complete",
            query=query[:50],
            result_count=len(results)
        )

        return results[:limit]

    async def _search_trusted_sites(
        self,
        browser: ResearcherBrowser,
        query: str,
        limit: int
    ) -> list[dict]:
        """Search within trusted sites using site: operator."""
        results = []

        # Get tier 1 domains
        tier1 = self.trusted_sources.get("seed_sources", {}).get("tier_1", [])
        tier1_domains = [s.get("domain", "") for s in tier1 if s.get("domain")]

        # Search each trusted domain
        for domain in tier1_domains[:3]:  # Limit to top 3 trusted sites
            if len(results) >= limit:
                break

            site_results = await browser.search_site(domain, query)
            for r in site_results[:2]:  # Max 2 per site
                r["source_type"] = "trusted_site"
                r["trusted_domain"] = domain
                results.append(r)

        return results[:limit]

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()

        # Reset counter every hour
        if (now - self._search_reset_time).total_seconds() > 3600:
            self._search_count = 0
            self._search_reset_time = now

        return self._search_count < self._max_searches_per_hour

    def _get_direct_urls(self, query: str, domain: Optional[str]) -> list[str]:
        """
        Get direct URLs from known authoritative sites.

        These are URLs we know have relevant content without searching.
        """
        urls = []

        # Add domain-specific known pages
        domain_urls = self._get_domain_specific_urls(query, domain)
        urls.extend(domain_urls)

        return urls

    def _get_domain_specific_urls(
        self,
        query: str,
        domain: Optional[str]
    ) -> list[str]:
        """Get URLs specific to a research domain."""
        urls = []

        # Map domains to known resource URLs
        domain_urls = {
            "kashmiri_shaivism": [
                "https://www.wisdomlib.org/hinduism/essay/kashmir-shaivism",
                "https://www.wisdomlib.org/hinduism/book/shiva-sutras",
            ],
            "tantraloka": [
                "https://www.wisdomlib.org/hinduism/book/tantraloka",
            ],
            "soundarya_lahari": [
                "https://www.wisdomlib.org/hinduism/book/soundarya-lahari",
            ],
            "hatha_yoga": [
                "https://www.wisdomlib.org/hinduism/book/hatha-yoga-pradipika",
            ],
            "vedas": [
                "https://www.sacred-texts.com/hin/rigveda/index.htm",
                "https://www.wisdomlib.org/hinduism/book/rig-veda",
            ],
            "indian_music": [
                "https://www.wisdomlib.org/hinduism/book/sangita-ratnakara",
                "https://www.wisdomlib.org/hinduism/book/natyashastra",
            ],
            "yantras": [
                "https://www.wisdomlib.org/definition/yantra",
            ],
            "sanskrit_grammar": [
                "https://www.wisdomlib.org/hinduism/book/ashtadhyayi",
            ],
            "bhagavata_purana": [
                "https://www.wisdomlib.org/hinduism/book/the-bhagavata-purana",
            ],
            "shiva_purana": [
                "https://www.wisdomlib.org/hinduism/book/shiva-purana",
            ],
        }

        if domain and domain in domain_urls:
            urls.extend(domain_urls[domain])

        return urls

    def get_search_suggestions(self, context: dict) -> list[str]:
        """
        Generate search suggestions based on research context.

        Args:
            context: Research context with concepts, numbers, domains

        Returns:
            List of suggested search queries
        """
        suggestions = []

        # Concept-based suggestions
        for concept in context.get("active_concepts", [])[:5]:
            suggestions.append(f"{concept} Hindu scripture")
            suggestions.append(f"{concept} Sanskrit meaning")

        # Number-based suggestions
        for number in context.get("active_numbers", [])[:5]:
            suggestions.append(f"{number} significance Hindu tradition")
            suggestions.append(f"{number} sacred number Sanskrit")

        # Domain-based suggestions
        for domain in context.get("active_domains", [])[:3]:
            domain_clean = domain.replace("_", " ")
            suggestions.append(f"{domain_clean} texts")

        return suggestions[:20]
