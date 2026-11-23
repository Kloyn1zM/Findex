"""Semantic vector search (NumPy E @ q) and hybrid Reciprocal Rank Fusion (Lab 8 M3)."""

import importlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.query import QuerySyntaxError
from findex.scoring import BM25, Scorer, SearchResult, make_snippet
from findex.search import search as keyword_search
from findex.tokenize import tokenize

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class TextEmbedder(Protocol):
    """Protocol for models that map texts to L2-normalized float32 embeddings."""

    @property
    def model_name(self) -> str: ...

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> NDArray[np.float32]: ...


class _STModelProtocol(Protocol):
    """Minimal structural typing for sentence_transformers.SentenceTransformer."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object: ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers (optional [semantic] extra)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model: _STModelProtocol | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> _STModelProtocol:
        if self._model is not None:
            return self._model
        try:
            st_module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            msg = (
                "Semantic search requires optional extra 'semantic'. "
                "Install with: uv sync --extra semantic"
            )
            raise RuntimeError(msg) from exc

        ctor = cast("Callable[[str], _STModelProtocol]", st_module.SentenceTransformer)
        self._model = ctor(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._load()
        raw = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(raw, dtype=np.float32)


def chunk_text(
    text: str,
    *,
    chunk_words: int = 250,
    overlap_words: int = 40,
) -> list[str]:
    """Split text into overlapping word chunks suitable for sentence-embedding models."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def _rank_key(item: tuple[int, float]) -> tuple[float, int]:
    return (-item[1], item[0])


def _sidecar_paths(path: Path) -> tuple[Path, Path, Path]:
    """Derive (embeddings.npy, doc_ids.npy, meta.json) paths from a base .npy path."""
    emb_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
    doc_ids_path = emb_path.with_suffix(".doc_ids.npy")
    meta_path = emb_path.with_suffix(".meta.json")
    return emb_path, doc_ids_path, meta_path


@dataclass(slots=True)
class SemanticIndex:
    """Normalized float32 chunk embedding matrix E with chunk -> doc_id mapping."""

    embeddings: NDArray[np.float32]
    chunk_doc_ids: NDArray[np.int32]
    model_name: str

    @property
    def num_chunks(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def save(self, path: Path) -> Path:
        """Persist embeddings E and chunk_doc_ids as .npy files plus a small JSON manifest."""
        emb_path, doc_ids_path, meta_path = _sidecar_paths(path)
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(emb_path, np.asarray(self.embeddings, dtype=np.float32))
        np.save(doc_ids_path, np.asarray(self.chunk_doc_ids, dtype=np.int32))
        manifest = {
            "model_name": self.model_name,
            "num_chunks": self.num_chunks,
            "dim": self.dim,
        }
        meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return emb_path

    @classmethod
    def load(cls, path: Path) -> "SemanticIndex":
        """Load a persisted SemanticIndex from .npy files and its JSON manifest."""
        emb_path, doc_ids_path, meta_path = _sidecar_paths(path)
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not doc_ids_path.exists():
            raise FileNotFoundError(f"Chunk map file not found: {doc_ids_path}")

        embeddings: NDArray[np.float32] = np.asarray(np.load(emb_path), dtype=np.float32)
        chunk_doc_ids: NDArray[np.int32] = np.asarray(np.load(doc_ids_path), dtype=np.int32)

        model_name = DEFAULT_EMBEDDING_MODEL
        if meta_path.exists():
            raw: object = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = cast("dict[str, object]", raw)
                if isinstance(meta.get("model_name"), str):
                    model_name = str(meta["model_name"])

        return cls(embeddings=embeddings, chunk_doc_ids=chunk_doc_ids, model_name=model_name)

    def search_vector(
        self,
        query_vec: NDArray[np.float32],
        *,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Score all chunks via E @ q, aggregate best chunk per doc, and return top-k."""
        if k <= 0 or self.num_chunks == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0.0:
            q = q / q_norm

        sims: NDArray[np.float32] = self.embeddings @ q
        max_doc_id = int(self.chunk_doc_ids.max())
        doc_scores: NDArray[np.float32] = np.full(max_doc_id + 1, -np.inf, dtype=np.float32)
        np.maximum.at(doc_scores, self.chunk_doc_ids, sims)

        valid_ids: NDArray[np.int32] = np.flatnonzero(np.isfinite(doc_scores)).astype(np.int32)
        valid_scores: NDArray[np.float32] = doc_scores[valid_ids]
        n_valid = valid_scores.size
        if n_valid == 0:
            return []

        if k < n_valid:
            part_idx: NDArray[np.intp] = np.argpartition(valid_scores, -k)[-k:]
            top_ids = valid_ids[part_idx]
            top_scores = valid_scores[part_idx]
        else:
            top_ids = valid_ids
            top_scores = valid_scores

        pairs = [(int(top_ids[i]), float(top_scores[i])) for i in range(top_ids.size)]
        pairs.sort(key=_rank_key)
        return pairs


def build_semantic_index(
    index: Index,
    embedder: TextEmbedder,
    *,
    chunk_words: int = 250,
    overlap_words: int = 40,
    batch_size: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
) -> SemanticIndex:
    """Chunk every document in index and encode all chunks into a normalized SemanticIndex."""
    chunk_texts: list[str] = []
    chunk_doc_ids: list[int] = []

    for doc_id in sorted(index.doc_meta):
        meta = index.doc_meta[doc_id]
        raw_text = index.doc_text(doc_id).strip()
        combined = f"{meta.title}. {raw_text}" if raw_text else meta.title
        pieces = chunk_text(combined, chunk_words=chunk_words, overlap_words=overlap_words)
        for piece in pieces:
            chunk_texts.append(piece)
            chunk_doc_ids.append(doc_id)

    total = len(chunk_texts)
    if total == 0:
        return SemanticIndex(
            embeddings=np.empty((0, 0), dtype=np.float32),
            chunk_doc_ids=np.empty(0, dtype=np.int32),
            model_name=embedder.model_name,
        )

    blocks: list[NDArray[np.float32]] = []
    for start in range(0, total, batch_size):
        batch = chunk_texts[start : start + batch_size]
        vecs = np.asarray(embedder.encode(batch, batch_size=batch_size), dtype=np.float32)
        norms: NDArray[np.float32] = np.linalg.norm(vecs, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        blocks.append((vecs / safe_norms).astype(np.float32, copy=False))
        if on_progress is not None:
            on_progress(min(start + len(batch), total), total)

    matrix: NDArray[np.float32] = np.vstack(blocks).astype(np.float32, copy=False)
    doc_id_arr: NDArray[np.int32] = np.asarray(chunk_doc_ids, dtype=np.int32)
    return SemanticIndex(
        embeddings=matrix,
        chunk_doc_ids=doc_id_arr,
        model_name=embedder.model_name,
    )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    rrf_k: int = 60,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked doc_id lists using RRF: sum(1 / (rrf_k + rank))."""
    if limit <= 0:
        return []
    fused: defaultdict[int, float] = defaultdict(float)
    for doc_ids in ranked_lists:
        for rank, doc_id in enumerate(doc_ids, start=1):
            fused[doc_id] += 1.0 / (rrf_k + rank)

    pairs = list(fused.items())
    pairs.sort(key=_rank_key)
    return pairs[:limit]


def _build_results(
    index: Index,
    pairs: Sequence[tuple[int, float]],
    highlight_terms: Sequence[str],
    *,
    highlight_start: str,
    highlight_end: str,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for doc_id, score in pairs:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            continue
        snippet = make_snippet(
            index.doc_text(doc_id),
            highlight_terms,
            highlight_start=highlight_start,
            highlight_end=highlight_end,
        )
        results.append(SearchResult(doc_id=doc_id, score=score, title=meta.title, snippet=snippet))
    return results


def semantic_search(
    index: Index,
    semantic_index: SemanticIndex,
    embedder: TextEmbedder,
    query: str,
    *,
    k: int = 10,
    highlight_start: str = "**",
    highlight_end: str = "**",
) -> list[SearchResult]:
    """Run dense vector search (E @ q) and return top-k SearchResult objects."""
    if k <= 0 or not query.strip():
        return []
    q_mat = embedder.encode([query])
    if q_mat.size == 0:
        return []
    pairs = semantic_index.search_vector(q_mat[0], k=k)
    terms = list(tokenize(query))
    return _build_results(
        index,
        pairs,
        terms,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
    )


def hybrid_search(
    index: Index,
    semantic_index: SemanticIndex,
    embedder: TextEmbedder,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    rrf_k: int = 60,
    highlight_start: str = "**",
    highlight_end: str = "**",
) -> list[SearchResult]:
    """Run keyword + semantic search and fuse rankings with Reciprocal Rank Fusion."""
    if k <= 0 or not query.strip():
        return []

    active_scorer: Scorer = scorer if scorer is not None else BM25()
    pool_k = max(k * 5, 50)

    try:
        kw_hits = keyword_search(index, query, scorer=active_scorer, k=pool_k)
    except (QuerySyntaxError, ValueError):
        fallback_q = " OR ".join(tokenize(query))
        kw_hits = (
            keyword_search(index, fallback_q, scorer=active_scorer, k=pool_k) if fallback_q else []
        )
    kw_ids = [h.doc_id for h in kw_hits]

    q_mat = embedder.encode([query])
    sem_pairs = semantic_index.search_vector(q_mat[0], k=pool_k) if q_mat.size > 0 else []
    sem_ids = [doc_id for doc_id, _ in sem_pairs]

    fused_pairs = reciprocal_rank_fusion([kw_ids, sem_ids], rrf_k=rrf_k, limit=k)
    terms = list(tokenize(query))
    return _build_results(
        index,
        fused_pairs,
        terms,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
    )
