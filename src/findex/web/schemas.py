"""Pydantic v2 HTTP boundary schemas for the findex web API (Lab 7)."""

from typing import Literal

from pydantic import BaseModel, Field

ScorerName = Literal["bm25", "tfidf"]


class HealthResponse(BaseModel):
    """Service liveness and loaded-index readiness snapshot."""

    status: Literal["ok"]
    version: str
    num_docs: int
    vocab_size: int
    uptime_sec: float


class SearchHit(BaseModel):
    """Single ranked search result item."""

    rank: int = Field(ge=1)
    doc_id: int = Field(ge=0)
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    """Paginated ranked search API payload."""

    query: str
    scorer: ScorerName
    k: int = Field(ge=1)
    page: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_prev: bool
    has_next: bool
    count: int = Field(ge=0)
    elapsed_ms: float
    hits: list[SearchHit]


class DocumentResponse(BaseModel):
    """Full document record returned by GET /docs/{doc_id}."""

    doc_id: int
    title: str
    path: str
    length: int
    text: str


class TermStat(BaseModel):
    """Aggregate corpus frequency for a single indexed term."""

    term: str
    total_tf: int
    df: int


class StatsResponse(BaseModel):
    """Summary statistics for the currently loaded index."""

    num_docs: int
    total_tokens: int
    vocab_size: int
    total_postings: int
    avg_doc_length: float
    has_positions: bool
    has_texts: bool
    top_terms: list[TermStat]


class ErrorResponse(BaseModel):
    """Structured error payload returned on unexpected 500 server failures."""

    detail: str
    request_id: str
