"""Polite asynchronous web crawler package for findex (Lab 6)."""

from findex.crawler.crawl import CrawlStats, Page, canonicalize_url, crawl
from findex.crawler.fetch import DEFAULT_USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

__all__ = [
    "DEFAULT_USER_AGENT",
    "CrawlStats",
    "FetchResult",
    "Page",
    "RobotsCache",
    "canonicalize_url",
    "crawl",
    "fetch",
]
