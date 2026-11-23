"""FastAPI dependency providers for the findex web application (Lab 7–8)."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.index import Index
from findex.semantic import SemanticIndex, TextEmbedder


@dataclass(slots=True)
class AppState:
    """Mutable runtime container attached to app.state during lifespan."""

    index: Index | None = None
    semantic: SemanticIndex | None = None
    embedder: TextEmbedder | None = None
    started_at: float = 0.0


def get_app_state(request: Request) -> AppState:
    """Return the AppState container bound to the FastAPI application."""
    state: object = getattr(request.app.state, "findex", None)
    if isinstance(state, AppState):
        return state
    return AppState()


def get_optional_index(state: Annotated[AppState, Depends(get_app_state)]) -> Index | None:
    """Return the loaded Index if available, or None when not yet initialized."""
    return state.index


def get_index(index: Annotated[Index | None, Depends(get_optional_index)]) -> Index:
    """Require a loaded Index, raising HTTP 503 when the service is not ready."""
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return index
