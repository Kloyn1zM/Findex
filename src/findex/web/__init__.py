"""Web service package for findex (FastAPI, Pydantic v2, pydantic-settings, Jinja2)."""

from findex.web.app import create_app, download_index_file, render_highlighted_html
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

__all__ = [
    "AppState",
    "DocumentResponse",
    "ErrorResponse",
    "HealthResponse",
    "ScorerName",
    "SearchHit",
    "SearchMode",
    "SearchResponse",
    "Settings",
    "StatsResponse",
    "TermStat",
    "create_app",
    "download_index_file",
    "get_app_state",
    "get_index",
    "get_optional_index",
    "render_highlighted_html",
]
