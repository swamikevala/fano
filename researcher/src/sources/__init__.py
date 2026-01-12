"""
Source discovery, fetching, and caching modules.

Uses Playwright browser automation for web access.
"""

from .browser import ResearcherBrowser, BrowserPool
from .searcher import WebSearcher
from .fetcher import ContentFetcher
from .cache import ContentCache

__all__ = [
    "ResearcherBrowser",
    "BrowserPool",
    "WebSearcher",
    "ContentFetcher",
    "ContentCache",
]
