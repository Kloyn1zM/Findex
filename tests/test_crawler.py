"""Tests for Lab 6: async fetcher, robots.txt cache, URL canonicalization, and crawler."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler.crawl import CrawlStats, Page, canonicalize_url, crawl
from findex.crawler.fetch import DEFAULT_USER_AGENT, fetch, parse_retry_after
from findex.crawler.robots import RobotsCache
from findex.index import build_index


def test_canonicalize_url_normalizes_and_filters() -> None:
    assert (
        canonicalize_url("HTTPS://Example.COM:443/docs/?b=2&a=1#section")
        == "https://example.com/docs?a=1&b=2"
    )
    assert canonicalize_url("guide.html#top", base_url="https://example.com/v1/") == (
        "https://example.com/v1/guide.html"
    )
    assert canonicalize_url("https://example.com/") == "https://example.com/"
    assert canonicalize_url("mailto:test@example.com") is None
    assert canonicalize_url("https://example.com/image.png") is None


def test_parse_retry_after_numeric_and_invalid() -> None:
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after("0.5") == 0.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("not-a-date") is None


def test_fetch_retries_429_and_503_but_not_404() -> None:
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls[path] = calls.get(path, 0) + 1
        assert request.headers.get("user-agent") == DEFAULT_USER_AGENT

        if path == "/404":
            return httpx.Response(404, text="not found")
        if path == "/flaky503":
            if calls[path] < 3:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, text="recovered", headers={"content-type": "text/html"})
        if path == "/rate429":
            if calls[path] == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, text="ok after 429", headers={"content-type": "text/html"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/html"})

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            r404 = await fetch(client, "https://example.test/404", max_retries=3, base_backoff=0.0)
            assert not r404.ok
            assert r404.status == 404
            assert r404.attempts == 1  # never retry 404

            r503 = await fetch(
                client,
                "https://example.test/flaky503",
                max_retries=3,
                base_backoff=0.0,
                max_backoff=0.01,
            )
            assert r503.ok
            assert r503.status == 200
            assert r503.attempts == 3

            r429 = await fetch(
                client,
                "https://example.test/rate429",
                max_retries=2,
                base_backoff=0.0,
            )
            assert r429.ok
            assert r429.attempts == 2

    asyncio.run(_run(), debug=True)


def test_fetch_timeout_exhausts_retries() -> None:
    def slow_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async def _run() -> None:
        transport = httpx.MockTransport(slow_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.test/slow",
                timeout=0.5,
                max_retries=1,
                base_backoff=0.0,
                max_backoff=0.01,
            )
            assert not res.ok
            assert res.attempts == 2
            assert res.error is not None
            assert "Timeout" in res.error

    asyncio.run(_run(), debug=True)


def test_robots_cache_fetches_once_and_enforces_rules() -> None:
    robots_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_hits
        if request.url.path == "/robots.txt":
            robots_hits += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\nAllow: /\n")
        return httpx.Response(200, text="ok")

    async def _run() -> None:
        cache = RobotsCache(user_agent=DEFAULT_USER_AGENT)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            results = await asyncio.gather(
                cache.allowed(client, "https://site.test/public/page"),
                cache.allowed(client, "https://site.test/private/secret"),
                cache.allowed(client, "https://site.test/about"),
            )
            assert results == [True, False, True]
            assert robots_hits == 1

    asyncio.run(_run(), debug=True)


def test_crawl_respects_robots_domains_dedup_max_pages_and_feeds_index(tmp_path: Path) -> None:
    fetched_urls: list[str] = []

    pages_html: dict[str, str] = {
        "/": (
            "<html><head><title>Home</title></head><body>"
            "<script>var ignored = 1;</script>"
            "<p>Welcome to async search engine docs.</p>"
            '<a href="/p1">P1</a>'
            '<a href="/p1#duplicate">P1 dup</a>'
            '<a href="/p2">P2</a>'
            '<a href="/blocked/secret">Secret</a>'
            '<a href="https://other.test/external">External</a>'
            '<a href="/missing">Broken</a>'
            "</body></html>"
        ),
        "/p1": (
            "<html><head><title>Page One</title></head><body>"
            "<p>Coroutines and TaskGroup structured concurrency.</p>"
            '<a href="/p3">P3</a>'
            '<a href="/">Back home</a>'
            "</body></html>"
        ),
        "/p2": (
            "<html><head><title>Page Two</title></head><body>"
            "<p>Semaphores and polite rate limiting.</p>"
            '<a href="/p4">P4</a>'
            "</body></html>"
        ),
        "/p3": (
            "<html><head><title>Page Three</title></head><body><p>Extra page 3.</p></body></html>"
        ),
        "/p4": (
            "<html><head><title>Page Four</title></head><body><p>Extra page 4.</p></body></html>"
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked/\n")
        fetched_urls.append(str(request.url))
        if path in pages_html:
            return httpx.Response(200, text=pages_html[path], headers={"content-type": "text/html"})
        return httpx.Response(404, text="missing")

    async def _run() -> list[Page]:
        log_path = tmp_path / "crawl.log"
        stats = CrawlStats()
        collected: list[Page] = []
        async for page in crawl(
            ["https://docs.test/"],
            max_pages=3,
            concurrency=4,
            per_host=2,
            delay=0.0,
            transport=httpx.MockTransport(handler),
            crawl_log_path=log_path,
            stats=stats,
        ):
            collected.append(page)

        assert log_path.exists()
        assert len(log_path.read_text(encoding="utf-8").splitlines()) >= 3
        return collected

    pages = asyncio.run(_run(), debug=True)

    # Exact max_pages cap, no duplicates, no external or robots-disallowed URLs
    assert len(pages) == 3
    urls = [p.url for p in pages]
    assert len(urls) == len(set(urls))
    assert all("blocked" not in u for u in fetched_urls)
    assert all("other.test" not in u for u in fetched_urls)
    assert all("ignored" not in p.text for p in pages)

    # Verify crawled JSONL streams directly into Lab 1 iter_documents + Lab 2/5 build_index
    jsonl_path = tmp_path / "crawl.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(
                json.dumps(
                    {"url": p.url, "title": p.title, "text": p.text, "fetched_at": p.fetched_at}
                )
                + "\n"
            )

    streamed_docs = list(iter_documents(jsonl_path))
    assert len(streamed_docs) == 3
    ix = build_index(jsonl_path, include_positions=True)
    assert ix.num_docs == 3
    assert "async" in ix or "coroutines" in ix or "semaphores" in ix


def test_user_agent_includes_contact_and_debug_mode_does_not_block_event_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert "kam09086@gmail.com" in DEFAULT_USER_AGENT

    big_paragraphs = "".join(
        f"<p>Paragraph {i}: asyncio event loop coroutine non-blocking parsing test.</p>"
        for i in range(4_000)
    )
    big_html_bytes = (
        f"<html><head><title>Huge Page</title></head><body>{big_paragraphs}</body></html>"
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=big_html_bytes, headers={"content-type": "text/html"})

    async def _run() -> list[Page]:
        pages: list[Page] = []
        async for page in crawl(
            ["https://huge.test/"],
            max_pages=1,
            concurrency=2,
            per_host=2,
            delay=0.0,
            transport=httpx.MockTransport(handler),
        ):
            pages.append(page)
        return pages

    with caplog.at_level("WARNING", logger="asyncio"):
        pages = asyncio.run(_run(), debug=True)

    assert len(pages) == 1
    assert pages[0].title == "Huge Page"
    slow_warnings = [rec.getMessage() for rec in caplog.records if "took" in rec.getMessage()]
    assert slow_warnings == []
