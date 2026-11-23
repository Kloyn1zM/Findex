"""Tests for Lab 7: FastAPI web package, Settings, Depends overrides, pagination, 500, Jinja2 UI."""

import json
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.corpus import Document
from findex.index import Index, build_index_from_docs
from findex.web import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    SearchResponse,
    Settings,
    StatsResponse,
    create_app,
    download_index_file,
    get_index,
    get_optional_index,
)
from findex.web import app as web_app_module

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _in_memory_index(*, include_positions: bool = True) -> Index:
    """Build a small deterministic in-memory Index without touching disk."""
    docs = [
        Document(
            doc_id=0,
            path=Path("00_python.txt"),
            title="Python Async Guide",
            text=(
                "Python programming language emphasizes readability. "
                "Python event loop powers async FastAPI services."
            ),
        ),
        Document(
            doc_id=1,
            path=Path("01_search.txt"),
            title="Search Engine Basics",
            text="Python full text search engine ranks documents with BM25 and TF-IDF.",
        ),
        Document(
            doc_id=2,
            path=Path("02_web.txt"),
            title="FastAPI Web Apps",
            text="Python web apps validate HTTP schemas with Pydantic and Jinja2 templates.",
        ),
        Document(
            doc_id=3,
            path=Path("03_xss.txt"),
            title="Security & Escaping",
            text="Python HTML <script>alert(1)</script> escaping keeps Jinja2 snippets safe.",
        ),
        Document(
            doc_id=4,
            path=Path("04_rust.txt"),
            title="Rust Systems",
            text="Rust systems programming language provides memory safety without GC.",
        ),
    ]
    return build_index_from_docs(docs, include_positions=include_positions)


@contextmanager
def _client_with_override(index: Index | None) -> Generator[httpx.Client]:
    """Create a TestClient backed purely by dependency_overrides (no index file on disk)."""
    app = create_app(Settings(index_path=None))
    app.dependency_overrides[get_optional_index] = lambda: index
    if index is not None:
        app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as raw_client:
        yield cast("httpx.Client", raw_client)


def _json_object(resp: httpx.Response) -> dict[str, object]:
    raw: object = json.loads(resp.text)
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def test_health_returns_503_when_index_not_loaded() -> None:
    with _client_with_override(None) as client:
        resp_health = client.get("/health")
        assert resp_health.status_code == 503
        assert "X-Request-ID" in resp_health.headers
        assert "X-Process-Time-Ms" in resp_health.headers

        resp_search = client.get("/search", params={"q": "python"})
        assert resp_search.status_code == 503

        resp_home = client.get("/")
        assert resp_home.status_code == 200
        assert "not loaded" in resp_home.text.lower()


def test_search_pagination_scorers_and_phrases_via_dependency_overrides() -> None:
    ix = _in_memory_index(include_positions=True)
    with _client_with_override(ix) as client:
        r_p1 = client.get("/search", params={"q": "python", "k": 2, "page": 1, "scorer": "bm25"})
        assert r_p1.status_code == 200
        p1 = SearchResponse.model_validate_json(r_p1.text)
        assert p1.scorer == "bm25"
        assert p1.page == 1
        assert p1.k == 2
        assert p1.total == 4
        assert p1.total_pages == 2
        assert p1.has_prev is False
        assert p1.has_next is True
        assert p1.count == 2
        assert [h.rank for h in p1.hits] == [1, 2]

        r_p2 = client.get("/search", params={"q": "python", "k": 2, "page": 2, "scorer": "bm25"})
        assert r_p2.status_code == 200
        p2 = SearchResponse.model_validate_json(r_p2.text)
        assert p2.page == 2
        assert p2.total == 4
        assert p2.total_pages == 2
        assert p2.has_prev is True
        assert p2.has_next is False
        assert p2.count == 2
        assert [h.rank for h in p2.hits] == [3, 4]
        assert {h.doc_id for h in p1.hits}.isdisjoint({h.doc_id for h in p2.hits})

        r_p3 = client.get("/search", params={"q": "python", "k": 2, "page": 3})
        assert r_p3.status_code == 200
        p3 = SearchResponse.model_validate_json(r_p3.text)
        assert p3.count == 0
        assert p3.total == 4

        r_tfidf = client.get("/search", params={"q": "python", "k": 3, "scorer": "tfidf"})
        assert r_tfidf.status_code == 200
        assert SearchResponse.model_validate_json(r_tfidf.text).scorer == "tfidf"

        r_phrase = client.get("/search", params={"q": '"event loop"'})
        assert r_phrase.status_code == 200
        phrase_payload = SearchResponse.model_validate_json(r_phrase.text)
        assert phrase_payload.total == 1
        assert phrase_payload.hits[0].title == "Python Async Guide"


def test_docs_404_stats_health_and_validation_errors() -> None:
    ix_no_pos = _in_memory_index(include_positions=False)
    with _client_with_override(ix_no_pos) as client:
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        health = HealthResponse.model_validate_json(resp_health.text)
        assert health.status == "ok"
        assert health.num_docs == 5
        assert health.vocab_size > 0

        resp_stats = client.get("/stats", params={"top": 5})
        assert resp_stats.status_code == 200
        stats = StatsResponse.model_validate_json(resp_stats.text)
        assert stats.num_docs == 5
        assert len(stats.top_terms) == 5

        ok_doc = client.get("/docs/0")
        assert ok_doc.status_code == 200
        doc = DocumentResponse.model_validate_json(ok_doc.text)
        assert doc.doc_id == 0
        assert doc.title == "Python Async Guide"
        assert "event loop" in doc.text

        missing_doc = client.get("/docs/99999")
        assert missing_doc.status_code == 404
        assert "99999" in str(_json_object(missing_doc)["detail"])

        # Query & validation errors
        assert client.get("/search", params={"q": "   "}).status_code == 400
        assert client.get("/search", params={"q": "(python"}).status_code == 400
        r_phrase = client.get("/search", params={"q": '"event loop"'})
        assert r_phrase.status_code == 400
        assert "positions" in str(_json_object(r_phrase)["detail"])

        assert client.get("/search").status_code == 422
        assert client.get("/search", params={"q": "python", "k": 0}).status_code == 422
        assert client.get("/search", params={"q": "python", "page": 0}).status_code == 422
        assert client.get("/search", params={"q": "python", "scorer": "invalid"}).status_code == 422

        schema = _json_object(client.get("/openapi.json"))
        paths = cast("dict[str, object]", schema["paths"])
        assert {"/search", "/docs/{doc_id}", "/stats", "/health"}.issubset(paths)


def test_jinja2_ui_search_pagination_document_view_and_xss_escaping() -> None:
    ix = _in_memory_index(include_positions=True)
    with _client_with_override(ix) as client:
        r_home = client.get("/")
        assert r_home.status_code == 200
        assert "text/html" in r_home.headers["content-type"]
        assert "findex" in r_home.text
        assert "5" in r_home.text

        r_results = client.get("/", params={"q": "python", "k": 2, "page": 1})
        assert r_results.status_code == 200
        assert "<mark>Python</mark>" in r_results.text
        assert "Page 1 / 2" in r_results.text
        assert "page=2" in r_results.text

        # Ensure raw <script> tags in document text are HTML-escaped in rendered snippets
        r_xss = client.get("/", params={"q": "escaping"})
        assert r_xss.status_code == 200
        assert "<script>alert(1)</script>" not in r_xss.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r_xss.text
        assert "<mark>escaping</mark>" in r_xss.text

        r_view = client.get("/view/0", params={"q": "python", "page": 1})
        assert r_view.status_code == 200
        assert "Python Async Guide" in r_view.text
        assert "Back to search" in r_view.text

        r_missing_view = client.get("/view/9999")
        assert r_missing_view.status_code == 404
        assert "not found" in r_missing_view.text.lower()


def test_unexpected_exception_returns_500_with_request_id_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app: FastAPI = create_app(Settings(index_path=None))

    def _boom() -> Index:
        raise RuntimeError("simulated database explosion")

    app.dependency_overrides[get_index] = _boom

    with (
        caplog.at_level("ERROR", logger="findex.web.app"),
        TestClient(app, raise_server_exceptions=False) as raw_client,
    ):
        client = cast("httpx.Client", raw_client)
        resp = client.get("/search", params={"q": "python"}, headers={"X-Request-ID": "req-abc123"})

    assert resp.status_code == 500
    assert resp.headers["X-Request-ID"] == "req-abc123"
    err = ErrorResponse.model_validate_json(resp.text)
    assert err.request_id == "req-abc123"
    assert err.detail == "Internal server error"
    assert "Traceback" not in resp.text
    assert "simulated database explosion" not in resp.text
    assert any("req-abc123" in rec.getMessage() for rec in caplog.records)


def test_settings_from_env_and_lifespan_loads_and_closes_index(
    saved_index_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INDEX_PATH", str(saved_index_path))
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("WORKERS", "4")

    cfg = Settings()
    assert cfg.index_path == saved_index_path
    assert cfg.port == 9090
    assert cfg.workers == 4

    app = create_app(cfg)
    with TestClient(app) as raw_client:
        client = cast("httpx.Client", raw_client)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert HealthResponse.model_validate_json(resp.text).num_docs == 5


def test_lifespan_downloads_missing_index_from_index_url_with_redirect_and_skips_if_local_exists(
    saved_index_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_bytes = saved_index_path.read_bytes()
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        if request.url.path == "/releases/download/v1.0.0/index.bin":
            return httpx.Response(
                302,
                headers={"Location": "https://objects.example.test/assets/index.bin"},
            )
        if request.url.path == "/assets/index.bin":
            return httpx.Response(200, content=index_bytes)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    real_download = download_index_file

    def _mock_download(url: str, dest: Path) -> Path:
        return real_download(url, dest, transport=transport)

    monkeypatch.setattr(web_app_module, "download_index_file", _mock_download)

    nested_target = tmp_path / "nested" / "dir" / "downloaded.bin"
    monkeypatch.setenv("INDEX_PATH", str(nested_target))
    monkeypatch.setenv(
        "INDEX_URL", "https://github.example.test/releases/download/v1.0.0/index.bin"
    )

    cfg = Settings()
    assert cfg.index_path == nested_target
    assert cfg.index_url == "https://github.example.test/releases/download/v1.0.0/index.bin"

    # 1) File absent -> lifespan downloads via 302 redirect, creates parents, atomically renames
    with TestClient(create_app(cfg)) as raw_client:
        client = cast("httpx.Client", raw_client)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert HealthResponse.model_validate_json(resp.text).num_docs == 5

    assert nested_target.exists()
    assert nested_target.read_bytes() == index_bytes
    assert list(nested_target.parent.glob("*.tmp")) == []
    assert requests_seen == [
        "https://github.example.test/releases/download/v1.0.0/index.bin",
        "https://objects.example.test/assets/index.bin",
    ]

    # 2) File already exists -> zero network calls (offline mode)
    requests_seen.clear()
    with TestClient(create_app(cfg)) as raw_client:
        client = cast("httpx.Client", raw_client)
        assert client.get("/health").status_code == 200
    assert requests_seen == []

    # 3) Failed download cleans up temporary file and raises HTTPStatusError
    bad_target = tmp_path / "failed" / "missing.bin"
    with pytest.raises(httpx.HTTPStatusError):
        download_index_file(
            "https://github.example.test/missing.bin",
            bad_target,
            transport=transport,
        )
    assert not bad_target.exists()
    assert list(bad_target.parent.glob("*.tmp")) == []


def test_cli_serve_errors_and_workers_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INDEX_PATH", raising=False)
    monkeypatch.delenv("INDEX_URL", raising=False)
    res_no_path = runner.invoke(cli_app, ["serve"])
    assert res_no_path.exit_code == 1
    assert "Traceback" not in res_no_path.output

    missing = tmp_path / "nope.bin"
    res_missing = runner.invoke(cli_app, ["serve", str(missing), "--workers", "2"])
    assert res_missing.exit_code == 1
    assert "Traceback" not in res_missing.output

    res_help = runner.invoke(cli_app, ["serve", "--help"], env={"NO_COLOR": "1", "TERM": "dumb"})
    assert res_help.exit_code == 0
    plain_help = _ANSI_RE.sub("", res_help.output)
    assert "--workers" in plain_help
