"""FastAPI web application factory, Jinja2 views, and hybrid REST endpoints (Lab 7–8)."""

import contextlib
import heapq
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex import __version__
from findex.index import Index
from findex.query import QuerySyntaxError
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.search import search_paginated
from findex.semantic import (
    SemanticIndex,
    SentenceTransformerEmbedder,
    TextEmbedder,
    hybrid_search,
    semantic_search,
)
from findex.store import open_index
from findex.web.deps import AppState, get_app_state, get_index, get_optional_index
from findex.web.schemas import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    ScorerName,
    SearchHit,
    SearchMode,
    SearchResponse,
    StatsResponse,
    TermStat,
)
from findex.web.settings import Settings

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_BOLD_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")


def _term_tf_key(item: tuple[str, int, int]) -> int:
    return item[1]


def render_highlighted_html(snippet: str) -> Markup:
    """Escape raw snippet text and wrap **matched** terms in <mark> safely."""
    escaped = str(escape(snippet))
    return Markup(_BOLD_MARKER_RE.sub(r"<mark>\1</mark>", escaped))


def _build_query_url(base: str, **params: object) -> str:
    filtered = {k: str(v) for k, v in params.items() if v is not None and str(v) != ""}
    if not filtered:
        return base
    return f"{base}?{urlencode(filtered)}"


def download_index_file(
    url: str,
    dest: Path,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 120.0,
) -> Path:
    """Stream-download an index file (following redirects) atomically to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.tmp")
    log.info("Downloading index from %s -> %s", url, dest)
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=timeout, transport=transport) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65_536):
                    fh.write(chunk)
        tmp_path.replace(dest)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    log.info("Downloaded index to %s (%.2f MiB)", dest, dest.stat().st_size / (1024 * 1024))
    return dest


def _execute_paginated_query(
    ix: Index,
    state: AppState,
    query: str,
    *,
    mode: SearchMode,
    scorer: Scorer,
    page: int,
    page_size: int,
) -> tuple[list[SearchResult], int]:
    """Run keyword, semantic, or hybrid search and return (page_results, total_hits)."""
    if mode == "keyword":
        return search_paginated(ix, query, scorer=scorer, page=page, page_size=page_size)

    if state.semantic is None or state.embedder is None:
        raise ValueError(
            "Semantic embeddings not loaded; start server with --embeddings or EMBEDDINGS_PATH"
        )

    offset = (page - 1) * page_size
    fetch_k = offset + page_size

    if mode == "semantic":
        total_docs = len({int(d) for d in state.semantic.chunk_doc_ids.tolist()})
        if offset >= total_docs:
            return [], total_docs
        all_up_to = semantic_search(
            ix,
            state.semantic,
            state.embedder,
            query,
            k=min(total_docs, fetch_k),
        )
        return all_up_to[offset : offset + page_size], total_docs

    pool_limit = max(fetch_k, ix.num_docs)
    all_hybrid = hybrid_search(
        ix,
        state.semantic,
        state.embedder,
        query,
        scorer=scorer,
        k=pool_limit,
    )
    total = len(all_hybrid)
    return all_hybrid[offset : offset + page_size], total


def create_app(
    settings: Settings | None = None,
    *,
    preloaded_index: Index | None = None,
    preloaded_semantic: SemanticIndex | None = None,
    embedder: TextEmbedder | None = None,
) -> FastAPI:
    """Create a configured FastAPI app loading index and optional embeddings from Settings."""
    cfg = settings if settings is not None else Settings()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _init_semantic(state: AppState) -> None:
        if preloaded_semantic is not None:
            state.semantic = preloaded_semantic
            state.embedder = embedder or SentenceTransformerEmbedder(preloaded_semantic.model_name)
        elif cfg.embeddings_path is not None:
            loaded_sem = SemanticIndex.load(cfg.embeddings_path)
            state.semantic = loaded_sem
            state.embedder = embedder or SentenceTransformerEmbedder(loaded_sem.model_name)
            log.info(
                "Loaded semantic index (%d chunks, dim=%d, model=%s)",
                loaded_sem.num_chunks,
                loaded_sem.dim,
                loaded_sem.model_name,
            )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None]:
        state = AppState(started_at=time.perf_counter())
        app_instance.state.findex = state
        _init_semantic(state)

        if preloaded_index is not None:
            state.index = preloaded_index
            log.info("Web app started with preloaded %r", preloaded_index)
            try:
                yield
            finally:
                state.index = None
                state.semantic = None
                state.embedder = None
            return

        if cfg.index_path is None:
            log.warning("INDEX_PATH is not configured; service running without a loaded index")
            yield
            return

        if not cfg.index_path.exists() and cfg.index_url:
            download_index_file(cfg.index_url, cfg.index_path)

        log.info("Loading index from %s", cfg.index_path)
        with open_index(cfg.index_path) as loaded:
            state.index = loaded
            log.info("Web app ready with %r", loaded)
            try:
                yield
            finally:
                state.index = None
                state.semantic = None
                state.embedder = None
        log.info("Index closed cleanly on shutdown")

    app = FastAPI(
        title="findex Search API",
        version=__version__,
        description=(
            "Full-text and hybrid semantic search service (BM25, TF-IDF, RRF, pagination, Jinja2)."
        ),
        lifespan=lifespan,
        responses={500: {"model": ErrorResponse}},
    )
    app.state.findex = AppState(started_at=time.perf_counter())

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        t0 = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.exception(
                "Unhandled exception [request_id=%s] %s %s (%.2f ms)",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            payload = ErrorResponse(detail="Internal server error", request_id=request_id)
            response = JSONResponse(status_code=500, content=payload.model_dump())
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
            return response

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        log.info(
            "[%s] %s %s -> %d (%.2f ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def search_page(
        request: Request,
        ix: Annotated[Index | None, Depends(get_optional_index)],
        state: Annotated[AppState, Depends(get_app_state)],
        q: Annotated[str, Query()] = "",
        mode: Annotated[SearchMode, Query()] = "keyword",
        scorer: Annotated[ScorerName, Query()] = "bm25",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        hits_view: list[dict[str, object]] = []
        error_msg: str | None = None
        total = 0
        total_pages = 0
        elapsed_ms = 0.0
        query_clean = q.strip()

        if ix is None:
            error_msg = "Search index is not loaded yet (service starting up)."
        elif query_clean:
            active_scorer: Scorer = BM25() if scorer == "bm25" else TfIdf()
            t0 = time.perf_counter()
            try:
                results, total = _execute_paginated_query(
                    ix,
                    state,
                    query_clean,
                    mode=mode,
                    scorer=active_scorer,
                    page=page,
                    page_size=k,
                )
            except (QuerySyntaxError, ValueError, RuntimeError) as exc:
                error_msg = str(exc)
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                total_pages = math.ceil(total / k) if total > 0 else 0
                offset = (page - 1) * k
                back_qs = urlencode(
                    {"q": query_clean, "mode": mode, "scorer": scorer, "k": k, "page": page}
                )
                for idx, res in enumerate(results, start=1):
                    hits_view.append(
                        {
                            "rank": offset + idx,
                            "doc_id": res.doc_id,
                            "score": res.score,
                            "title": res.title,
                            "snippet_html": render_highlighted_html(res.snippet),
                            "view_url": f"/view/{res.doc_id}?{back_qs}",
                        }
                    )

        prev_url = (
            _build_query_url("/", q=query_clean, mode=mode, scorer=scorer, k=k, page=page - 1)
            if page > 1 and total_pages > 0
            else None
        )
        next_url = (
            _build_query_url("/", q=query_clean, mode=mode, scorer=scorer, k=k, page=page + 1)
            if page < total_pages
            else None
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "index_ready": ix is not None,
                "semantic_ready": state.semantic is not None,
                "num_docs": ix.num_docs if ix is not None else 0,
                "vocab_size": len(ix) if ix is not None else 0,
                "q": q,
                "mode": mode,
                "scorer": scorer,
                "k": k,
                "page": page,
                "total": total,
                "total_pages": total_pages,
                "elapsed_ms": elapsed_ms,
                "hits": hits_view,
                "error_msg": error_msg,
                "prev_url": prev_url,
                "next_url": next_url,
            },
        )

    @app.get("/view/{doc_id:int}", response_class=HTMLResponse, include_in_schema=False)
    def view_document_page(
        request: Request,
        doc_id: int,
        ix: Annotated[Index, Depends(get_index)],
        q: Annotated[str, Query()] = "",
        mode: Annotated[SearchMode, Query()] = "keyword",
        scorer: Annotated[ScorerName, Query()] = "bm25",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        back_url = (
            _build_query_url("/", q=q.strip(), mode=mode, scorer=scorer, k=k, page=page)
            if q.strip()
            else "/"
        )
        meta = ix.doc_meta.get(doc_id)
        if meta is None:
            return templates.TemplateResponse(
                request=request,
                name="not_found.html",
                context={"version": __version__, "doc_id": doc_id, "back_url": back_url},
                status_code=404,
            )

        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={
                "version": __version__,
                "doc_id": meta.doc_id,
                "title": meta.title,
                "path": meta.path,
                "length": ix.doc_length(doc_id),
                "text": ix.doc_text(doc_id),
                "back_url": back_url,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(
        ix: Annotated[Index, Depends(get_index)],
        state: Annotated[AppState, Depends(get_app_state)],
    ) -> HealthResponse:
        uptime = max(0.0, time.perf_counter() - state.started_at)
        return HealthResponse(
            status="ok",
            version=__version__,
            num_docs=ix.num_docs,
            vocab_size=len(ix),
            uptime_sec=round(uptime, 3),
            semantic_ready=state.semantic is not None,
        )

    @app.get("/search", response_model=SearchResponse)
    def search_endpoint(
        ix: Annotated[Index, Depends(get_index)],
        state: Annotated[AppState, Depends(get_app_state)],
        q: Annotated[str, Query(min_length=1, description="Boolean, phrase, or natural query")],
        k: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 10,
        scorer: Annotated[ScorerName, Query(description="Keyword ranking algorithm")] = "bm25",
        mode: Annotated[SearchMode, Query(description="Search mode")] = "keyword",
        page: Annotated[int, Query(ge=1, le=1000, description="1-based page number")] = 1,
    ) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query must not be blank")

        active_scorer: Scorer = BM25() if scorer == "bm25" else TfIdf()
        t0 = time.perf_counter()
        try:
            results, total = _execute_paginated_query(
                ix,
                state,
                q,
                mode=mode,
                scorer=active_scorer,
                page=page,
                page_size=k,
            )
        except (QuerySyntaxError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        total_pages = math.ceil(total / k) if total > 0 else 0
        offset = (page - 1) * k
        hits = [
            SearchHit(
                rank=offset + idx,
                doc_id=res.doc_id,
                score=round(res.score, 6),
                title=res.title,
                snippet=res.snippet,
            )
            for idx, res in enumerate(results, start=1)
        ]
        return SearchResponse(
            query=q,
            mode=mode,
            scorer=scorer,
            k=k,
            page=page,
            total=total,
            total_pages=total_pages,
            has_prev=page > 1 and total_pages > 0,
            has_next=page < total_pages,
            count=len(hits),
            elapsed_ms=round(elapsed_ms, 3),
            hits=hits,
        )

    @app.get("/docs/{doc_id:int}", response_model=DocumentResponse)
    def get_document(
        doc_id: int,
        ix: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        meta = ix.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return DocumentResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            path=meta.path,
            length=ix.doc_length(doc_id),
            text=ix.doc_text(doc_id),
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats_endpoint(
        ix: Annotated[Index, Depends(get_index)],
        top: Annotated[int, Query(ge=1, le=100, description="Number of top terms")] = 20,
    ) -> StatsResponse:
        total_tokens = sum(ix.doc_lengths.values())
        total_postings = sum(len(plist) for plist in ix.values())
        term_triples = ((term, sum(p.tf for p in plist), len(plist)) for term, plist in ix.items())
        top_rows = heapq.nlargest(top, term_triples, key=_term_tf_key)

        return StatsResponse(
            num_docs=ix.num_docs,
            total_tokens=total_tokens,
            vocab_size=len(ix),
            total_postings=total_postings,
            avg_doc_length=round(ix.avg_doc_length, 2),
            has_positions=ix.has_positions,
            has_texts=bool(ix.doc_texts),
            top_terms=[
                TermStat(term=term, total_tf=tf_sum, df=df) for term, tf_sum, df in top_rows
            ],
        )

    return app
