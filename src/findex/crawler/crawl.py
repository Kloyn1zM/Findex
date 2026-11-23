"""Structured-concurrency async web crawler (Queue frontier + TaskGroup workers)."""

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TextIO
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)

_SKIP_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".bz2",
        ".xz",
        ".css",
        ".js",
        ".json",
        ".xml",
        ".rss",
        ".mp3",
        ".mp4",
        ".woff",
        ".woff2",
        ".ttf",
    }
)


@dataclass(frozen=True, slots=True)
class Page:
    """Extracted web page yielded by the crawler async generator."""

    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    """Live telemetry updated cooperatively by crawler workers for UI display."""

    pages_fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    skipped_robots: int = 0
    started_at: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        if self.started_at <= 0.0:
            return 0.0
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_sec(self) -> float:
        elapsed = self.elapsed_sec
        if elapsed <= 0.0:
            return 0.0
        return self.pages_fetched / elapsed


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """Resolve and normalize a URL (lowercase scheme/host, no fragment, sorted query)."""
    resolved = urljoin(base_url, url.strip()) if base_url else url.strip()
    parsed = urlparse(resolved)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        return None

    host = (parsed.hostname or "").lower()
    if not host:
        return None

    try:
        port = parsed.port
    except ValueError:
        return None

    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443) or port is None:
        netloc = host
    else:
        netloc = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    lower_path = path.lower()
    if any(lower_path.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return None

    query_pairs = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    sorted_query = urlencode(query_pairs)

    return urlunparse((scheme, netloc, path, "", sorted_query, ""))


class _HTMLExtractor(HTMLParser):
    """Lightweight HTML parser extracting <title>, visible text, and <a href> links."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if lower == "title":
            self._in_title = True
        elif lower == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    self.links.append(attr_val)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._title_parts)).strip()

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._text_parts)).strip()


def extract_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Extract (title, plain_text, canonical_links) from an HTML document."""
    parser = _HTMLExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — tolerate severely malformed HTML pages
        log.debug("Recovered from malformed HTML while parsing %s", base_url)

    links: list[str] = []
    for raw_href in parser.links:
        canon = canonicalize_url(raw_href, base_url=base_url)
        if canon is not None:
            links.append(canon)
    return parser.title, parser.text, links


def _domain_allowed(url: str, allowed_domains: set[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == dom or host.endswith(f".{dom}") for dom in allowed_domains)


def _write_crawl_log(log_fh: TextIO | None, result: FetchResult) -> None:
    if log_fh is None:
        return
    entry = {
        "url": result.url,
        "status": result.status,
        "bytes": result.bytes_len,
        "elapsed_ms": round(result.elapsed_sec * 1000.0, 2),
        "attempts": result.attempts,
        "error": result.error,
    }
    log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.2,
    timeout: float = 10.0,
    max_retries: int = 3,
    allowed_domains: Iterable[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    crawl_log_path: Path | None = None,
    stats: CrawlStats | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> AsyncIterator[Page]:
    """Asynchronously crawl pages from seed URLs, yielding Page objects as they arrive.

    Uses an asyncio.Queue frontier, TaskGroup workers, global + per-host semaphores,
    per-host politeness delay, robots.txt enforcement, and URL canonicalization.
    """
    if max_pages <= 0:
        return

    seed_list: list[str] = []
    for raw_seed in seeds:
        canon = canonicalize_url(raw_seed)
        if canon is not None:
            seed_list.append(canon)
    if not seed_list:
        return

    if allowed_domains is not None:
        domain_set = {d.lower().lstrip(".") for d in allowed_domains if d.strip()}
    else:
        domain_set = {
            (urlparse(u).hostname or "").lower() for u in seed_list if urlparse(u).hostname
        }

    effective_concurrency = max(1, concurrency)
    effective_per_host = max(1, per_host)

    frontier: asyncio.Queue[str] = asyncio.Queue()
    results_q: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()

    for url in seed_list:
        if url not in seen and _domain_allowed(url, domain_set):
            seen.add(url)
            frontier.put_nowait(url)

    if frontier.empty():
        return

    if stats is not None:
        stats.started_at = time.perf_counter()
        stats.queue_depth = frontier.qsize()

    global_sem = asyncio.Semaphore(effective_concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_next_slot: dict[str, float] = {}
    robots = RobotsCache(user_agent=user_agent, timeout=min(timeout, 5.0))
    stop_event = asyncio.Event()
    pages_fetched = 0

    async def _respect_host_delay(host: str) -> None:
        if delay <= 0.0:
            return
        now = time.monotonic()
        scheduled = max(now, host_next_slot.get(host, now))
        host_next_slot[host] = scheduled + delay
        wait_sec = scheduled - now
        if wait_sec > 0.0:
            await asyncio.sleep(wait_sec)

    async def _worker(client: httpx.AsyncClient, log_fh: TextIO | None) -> None:
        nonlocal pages_fetched
        while not stop_event.is_set():
            url = await frontier.get()
            if stats is not None:
                stats.queue_depth = frontier.qsize()
            try:
                if stop_event.is_set():
                    continue

                if not await robots.allowed(client, url):
                    if stats is not None:
                        stats.skipped_robots += 1
                    log.debug("Disallowed by robots.txt: %s", url)
                    continue

                host = (urlparse(url).hostname or "").lower()
                host_sem = host_sems.setdefault(host, asyncio.Semaphore(effective_per_host))

                async with global_sem, host_sem:
                    if stop_event.is_set():
                        continue
                    await _respect_host_delay(host)
                    if stop_event.is_set():
                        continue

                    if stats is not None:
                        stats.in_flight += 1
                    try:
                        res = await fetch(
                            client,
                            url,
                            timeout=timeout,
                            max_retries=max_retries,
                        )
                    finally:
                        if stats is not None:
                            stats.in_flight -= 1

                _write_crawl_log(log_fh, res)

                if not res.ok:
                    if stats is not None:
                        stats.errors += 1
                    continue

                if "html" not in res.content_type.lower() and res.content_type:
                    continue

                title, text, links = extract_html(res.text, base_url=res.final_url)
                if not text:
                    continue

                if pages_fetched >= max_pages:
                    stop_event.set()
                    continue

                pages_fetched += 1
                if stats is not None:
                    stats.pages_fetched = pages_fetched

                page = Page(
                    url=res.final_url,
                    title=title or res.final_url,
                    text=text,
                    fetched_at=datetime.now(tz=UTC).isoformat(),
                )
                await results_q.put(page)

                if pages_fetched >= max_pages:
                    stop_event.set()
                    continue

                # Mark canonical final URL as seen too (handles redirects cleanly)
                canon_final = canonicalize_url(res.final_url)
                if canon_final is not None:
                    seen.add(canon_final)

                for link in links:
                    if link not in seen and _domain_allowed(link, domain_set):
                        seen.add(link)
                        frontier.put_nowait(link)

                if stats is not None:
                    stats.queue_depth = frontier.qsize()
            finally:
                frontier.task_done()

    async def _run_crawler() -> None:
        log_fh: TextIO | None = None
        if crawl_log_path is not None:
            crawl_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = crawl_log_path.open("w", encoding="utf-8")

        limits = httpx.Limits(
            max_connections=max(effective_concurrency * 2, 20),
            max_keepalive_connections=max(effective_concurrency, 10),
        )
        headers = {"User-Agent": user_agent}

        try:
            async with (
                httpx.AsyncClient(
                    headers=headers,
                    follow_redirects=True,
                    max_redirects=5,
                    limits=limits,
                    transport=transport,
                ) as client,
                asyncio.TaskGroup() as tg,
            ):
                worker_tasks = [
                    tg.create_task(_worker(client, log_fh)) for _ in range(effective_concurrency)
                ]
                join_task = asyncio.create_task(frontier.join())
                stop_task = asyncio.create_task(stop_event.wait())
                try:
                    _, pending = await asyncio.wait(
                        {join_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for pending_task in pending:
                        pending_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await pending_task
                finally:
                    for wt in worker_tasks:
                        wt.cancel()
        finally:
            if log_fh is not None:
                log_fh.close()
            await results_q.put(None)

    crawler_task = asyncio.create_task(_run_crawler())
    try:
        while True:
            item = await results_q.get()
            if item is None:
                break
            yield item
        await crawler_task
    finally:
        if not crawler_task.done():
            crawler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await crawler_task
