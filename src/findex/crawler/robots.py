"""Async per-host robots.txt cache backed by urllib.robotparser."""

import asyncio
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT

log = logging.getLogger(__name__)


class RobotsCache:
    """Fetches, parses, and caches robots.txt rules per (scheme, host) asynchronously."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, *, timeout: float = 5.0) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _load_for_origin(self, client: httpx.AsyncClient, origin: str) -> None:
        robots_url = f"{origin}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            async with asyncio.timeout(self._timeout):
                resp = await client.get(robots_url)
        except (TimeoutError, httpx.HTTPError) as exc:
            log.warning("Could not fetch %s (%s); defaulting to allow", robots_url, exc)
            self._parsers[origin] = None
            return

        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            self._parsers[origin] = rp
            log.info("Loaded robots.txt from %s", robots_url)
        elif 400 <= resp.status_code < 500:
            # RFC 9309: 4xx on robots.txt means no restrictions apply
            self._parsers[origin] = None
            log.debug("No robots.txt at %s (HTTP %d); allowing crawl", robots_url, resp.status_code)
        else:
            # RFC 9309: 5xx on robots.txt -> treat as temporarily disallowed
            rp.parse(["User-agent: *", "Disallow: /"])
            self._parsers[origin] = rp
            log.warning(
                "HTTP %d fetching %s; disallowing host conservatively",
                resp.status_code,
                robots_url,
            )

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        """Return True if robots.txt permits fetching url for our User-Agent."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

        if origin not in self._parsers:
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                if origin not in self._parsers:
                    await self._load_for_origin(client, origin)

        rp = self._parsers[origin]
        if rp is None:
            return True
        return rp.can_fetch(self._user_agent, url)
