"""Tests for Lab 8 M3: chunking, .npy embeddings, E @ q search, RRF, and hybrid API."""

import math
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from findex.index import build_index
from findex.semantic import (
    SemanticIndex,
    build_semantic_index,
    chunk_text,
    hybrid_search,
    reciprocal_rank_fusion,
    semantic_search,
)
from findex.store import open_index
from findex.web import HealthResponse, SearchResponse, Settings, create_app


@contextmanager
def _open_client(app: FastAPI) -> Generator[httpx.Client]:
    with TestClient(app) as raw_client:
        yield cast("httpx.Client", raw_client)


class _ConceptMockEmbedder:
    """Deterministic lightweight embedder mapping synonyms to shared axes (no PyTorch in CI)."""

    _AXES: tuple[tuple[str, ...], ...] = (
        ("python", "interpreter", "programming", "scripting", "asyncio", "coroutine"),
        ("rust", "systems", "ownership", "memory", "borrow"),
        ("feline", "cat", "kitten", "rug", "mat"),
        ("search", "index", "bm25", "ranking", "retrieval"),
    )

    @property
    def model_name(self) -> str:
        return "mock-concept-v1"

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> NDArray[np.float32]:
        _ = batch_size
        mat: NDArray[np.float32] = np.zeros((len(texts), len(self._AXES)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            lower = text.lower()
            scores = [float(sum(1 for kw in keywords if kw in lower)) for keywords in self._AXES]
            norm = math.sqrt(sum(s * s for s in scores))
            if norm > 0.0:
                scores = [s / norm for s in scores]
            for axis_idx, val in enumerate(scores):
                mat[row_idx, axis_idx] = val
        return mat


def test_chunk_text_splits_with_overlap() -> None:
    words = [f"w{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words), chunk_words=250, overlap_words=50)
    assert len(chunks) == 3
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert chunks[-1].endswith("w599")
    assert chunk_text("") == []


def test_reciprocal_rank_fusion_combines_rankings() -> None:
    # Docs 2 and 3 appear in both lists -> must beat single-list #1 docs (1 and 4)
    fused = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 3]], rrf_k=60, limit=3)
    assert fused[0][0] == 2
    assert fused[1][0] == 3
    assert fused[2][0] == 1
    expected_doc2 = (1.0 / 62.0) + (1.0 / 62.0)
    assert fused[0][1] == pytest.approx(expected_doc2, rel=1e-12)


def test_semantic_index_roundtrip_and_paraphrase_retrieval(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "01_cat.txt").write_text("The cat sat quietly on the mat.", encoding="utf-8")
    (corpus / "02_rust.txt").write_text("Rust systems memory ownership model.", encoding="utf-8")
    (corpus / "03_py.txt").write_text("Python asyncio coroutine runtime.", encoding="utf-8")

    ix = build_index(corpus, include_positions=False)
    embedder = _ConceptMockEmbedder()
    sem = build_semantic_index(ix, embedder, chunk_words=50, overlap_words=10)

    saved = sem.save(tmp_path / "emb.npy")
    loaded = SemanticIndex.load(saved)
    assert loaded.num_chunks == 3
    assert loaded.dim == 4
    assert loaded.model_name == "mock-concept-v1"

    # Paraphrase query shares zero words with 01_cat.txt ("feline" + "rug" vs "cat" + "mat")
    sem_hits = semantic_search(ix, loaded, embedder, "a feline rested on a rug", k=3)
    assert sem_hits[0].title == "01_cat"

    # Hybrid search fuses keyword + semantic without crashing on zero keyword hits
    hyb_hits = hybrid_search(ix, loaded, embedder, "feline rug", k=3)
    assert hyb_hits[0].title == "01_cat"


def test_api_semantic_and_hybrid_modes(saved_index_path: Path, tmp_path: Path) -> None:
    embedder = _ConceptMockEmbedder()
    with open_index(saved_index_path) as ix:
        sem = build_semantic_index(ix, embedder)
        emb_path = sem.save(tmp_path / "api_emb.npy")

    # Without embeddings -> semantic mode returns 400 cleanly
    with _open_client(create_app(Settings(index_path=saved_index_path))) as client_no_sem:
        health0 = HealthResponse.model_validate_json(client_no_sem.get("/health").text)
        assert health0.semantic_ready is False
        resp_err = client_no_sem.get("/search", params={"q": "python", "mode": "semantic"})
        assert resp_err.status_code == 400

    # With embeddings -> semantic & hybrid modes work end-to-end
    app = create_app(
        Settings(index_path=saved_index_path, embeddings_path=emb_path),
        embedder=embedder,
    )
    with _open_client(app) as client:
        health = HealthResponse.model_validate_json(client.get("/health").text)
        assert health.semantic_ready is True

        r_sem = client.get(
            "/search",
            params={"q": "scripting interpreter", "mode": "semantic", "k": 3},
        )
        assert r_sem.status_code == 200
        sem_payload = SearchResponse.model_validate_json(r_sem.text)
        assert sem_payload.mode == "semantic"
        assert sem_payload.count >= 1
        assert sem_payload.hits[0].title == "01_python"

        r_hyb = client.get("/search", params={"q": "python", "mode": "hybrid", "k": 3})
        assert r_hyb.status_code == 200
        hyb_payload = SearchResponse.model_validate_json(r_hyb.text)
        assert hyb_payload.mode == "hybrid"
        assert hyb_payload.hits[0].title == "01_python"
