"""Polite async HTTP fetcher with timeouts, retries, exponential backoff, and Retry-After."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "findex-crawler/0.6.0 (+https://github.com/Kloyn1zM/Findex)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of fetching a single URL (used for both crawling and crawl.log)."""

    url: str
    final_url: str
    status: int | None
    text: str
    content_type: str
    bytes_len: int
    elapsed_sec: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300 and self.error is None


def parse_retry_after(header_value: str | None) -> float | None:
    """Parse an HTTP Retry-After header (delta-seconds or HTTP-date) into seconds."""
    if not header_value:
        return None
    raw = header_value.strip()
    try:
        seconds = float(raw)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)

    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = (dt - datetime.now(tz=UTC)).total_seconds()
    return max(0.0, delta)


def _should_retry_status(status_code: int) -> bool:
    """Retry 429 Too Many Requests and 5xx server errors; never retry other 4xx."""
    return status_code == 429 or 500 <= status_code < 600


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 0.5,
    max_backoff: float = 15.0,
) -> FetchResult:
    """Fetch a URL with bounded retries, exponential backoff + jitter, and Retry-After."""
    t0 = time.perf_counter()
    last_status: int | None = None
    last_error: str | None = None
    max_attempts = max(1, max_retries + 1)

    for attempt in range(max_attempts):
        retry_delay: float | None = None
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)

            last_status = resp.status_code
            content_type = resp.headers.get("content-type", "")
            raw_bytes = resp.content
            elapsed = time.perf_counter() - t0

            if _should_retry_status(resp.status_code) and attempt < max_attempts - 1:
                if resp.status_code == 429:
                    retry_after = parse_retry_after(resp.headers.get("retry-after"))
                    if retry_after is not None:
                        retry_delay = min(max_backoff, retry_after)
                if retry_delay is None:
                    retry_delay = min(max_backoff, base_backoff * (2**attempt)) + random.uniform(
                        0.0, 1.0
                    )
                log.warning(
                    "HTTP %d for %s (attempt %d/%d); retrying in %.2f s",
                    resp.status_code,
                    url,
                    attempt + 1,
                    max_attempts,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue

            error_msg = None if 200 <= resp.status_code < 300 else f"HTTP {resp.status_code}"
            return FetchResult(
                url=url,
                final_url=str(resp.url),
                status=resp.status_code,
                text=resp.text if 200 <= resp.status_code < 300 else "",
                content_type=content_type,
                bytes_len=len(raw_bytes),
                elapsed_sec=elapsed,
                attempts=attempt + 1,
                error=error_msg,
            )

        except (TimeoutError, httpx.TimeoutException) as exc:
            last_error = f"Timeout: {exc or 'request timed out'}"
        except httpx.TooManyRedirects as exc:
            elapsed = time.perf_counter() - t0
            log.warning("Redirect loop for %s: %s", url, exc)
            return FetchResult(
                url=url,
                final_url=url,
                status=last_status,
                text="",
                content_type="",
                bytes_len=0,
                elapsed_sec=elapsed,
                attempts=attempt + 1,
                error=f"TooManyRedirects: {exc}",
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_attempts - 1:
            retry_delay = min(max_backoff, base_backoff * (2**attempt)) + random.uniform(0.0, 1.0)
            log.warning(
                "Transient error for %s (%s, attempt %d/%d); retrying in %.2f s",
                url,
                last_error,
                attempt + 1,
                max_attempts,
                retry_delay,
            )
            await asyncio.sleep(retry_delay)

    elapsed = time.perf_counter() - t0
    return FetchResult(
        url=url,
        final_url=url,
        status=last_status,
        text="",
        content_type="",
        bytes_len=0,
        elapsed_sec=elapsed,
        attempts=max_attempts,
        error=last_error or "Max retries exceeded",
    )
