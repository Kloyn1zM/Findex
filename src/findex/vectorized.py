"""NumPy-vectorized BM25 and TF-IDF scoring over int32 postings arrays (Lab 8 M2)."""

import math
import weakref
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.scoring import BM25, TfIdf

_VECTOR_CACHE: dict[int, tuple[weakref.ref[Index], "VectorizedIndex"]] = {}


def _rank_sort_key(item: tuple[int, float]) -> tuple[float, int]:
    return (-item[1], item[0])


class VectorizedIndex:
    """Columnar NumPy int32 view over an Index for vectorized scoring."""

    def __init__(self, index: Index) -> None:
        self._index = index
        self.num_docs: int = index.num_docs
        self.avg_doc_length: float = index.avg_doc_length
        self.size: int = (max(index.doc_lengths) + 1) if index.doc_lengths else 0

        lengths: NDArray[np.int32] = np.zeros(self.size, dtype=np.int32)
        for doc_id, length in index.doc_lengths.items():
            lengths[doc_id] = length
        self.doc_lengths: NDArray[np.int32] = lengths
        self._postings: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}

    def term_arrays(self, term: str) -> tuple[NDArray[np.int32], NDArray[np.int32]] | None:
        """Return contiguous (doc_ids: int32, tfs: int32) arrays for term, cached on first use."""
        cached = self._postings.get(term)
        if cached is not None:
            return cached
        if term not in self._index:
            return None

        plist = self._index[term]
        count = len(plist)
        doc_ids: NDArray[np.int32] = np.fromiter(
            (p.doc_id for p in plist),
            dtype=np.int32,
            count=count,
        )
        tfs: NDArray[np.int32] = np.fromiter(
            (p.tf for p in plist),
            dtype=np.int32,
            count=count,
        )
        pair = (doc_ids, tfs)
        self._postings[term] = pair
        return pair

    def preload_all(self) -> None:
        """Materialize int32 arrays for every term in the vocabulary upfront."""
        for term in self._index:
            self.term_arrays(term)


def get_vectorized_index(index: Index) -> VectorizedIndex:
    """Return a cached VectorizedIndex tied to the lifetime of index."""
    entry = _VECTOR_CACHE.get(index.cache_token)
    if entry is not None:
        ref, vindex = entry
        if ref() is index:
            return vindex
    vindex = VectorizedIndex(index)
    _VECTOR_CACHE[index.cache_token] = (weakref.ref(index), vindex)
    return vindex


def _term_scores_numpy(
    vindex: VectorizedIndex,
    doc_ids: NDArray[np.int32],
    tfs: NDArray[np.int32],
    scorer: BM25 | TfIdf,
) -> NDArray[np.float64]:
    """Compute vectorized BM25 or TF-IDF contributions for one term's postings."""
    n = vindex.num_docs
    df = doc_ids.size
    if df == 0 or n == 0:
        return np.empty(0, dtype=np.float64)

    tfs_f: NDArray[np.float64] = tfs.astype(np.float64)
    if isinstance(scorer, BM25):
        avg_len = vindex.avg_doc_length
        if avg_len <= 0.0:
            return np.zeros(df, dtype=np.float64)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        doc_lens_f: NDArray[np.float64] = vindex.doc_lengths[doc_ids].astype(np.float64)
        norm: NDArray[np.float64] = scorer.k1 * (1.0 - scorer.b + scorer.b * (doc_lens_f / avg_len))
        scores: NDArray[np.float64] = idf * (tfs_f * (scorer.k1 + 1.0)) / (tfs_f + norm)
        return scores

    idf = math.log(1.0 + n / df)
    tfidf_scores: NDArray[np.float64] = (1.0 + np.log(tfs_f)) * idf
    return tfidf_scores


def _select_top_k(
    doc_ids: NDArray[np.int32],
    scores: NDArray[np.float64],
    k: int,
) -> tuple[tuple[int, float], ...]:
    """Select top-k (doc_id, score) pairs in O(n) via np.argpartition + deterministic tie-break."""
    n_valid = scores.size
    if n_valid == 0 or k <= 0:
        return ()

    if k < n_valid:
        part_idx: NDArray[np.intp] = np.argpartition(scores, -k)[-k:]
        threshold = float(scores[part_idx].min())
        gt_mask = scores > threshold
        gt_ids = doc_ids[gt_mask]
        gt_scores = scores[gt_mask]
        remaining = k - gt_ids.size

        if remaining > 0:
            eq_mask = scores == threshold
            eq_ids = doc_ids[eq_mask]
            eq_order = np.argsort(eq_ids, kind="mergesort")[:remaining]
            chosen_eq_ids = eq_ids[eq_order]
            chosen_eq_scores = np.full(chosen_eq_ids.size, threshold, dtype=np.float64)
            top_ids: NDArray[np.int32] = np.concatenate((gt_ids, chosen_eq_ids))
            top_scores: NDArray[np.float64] = np.concatenate((gt_scores, chosen_eq_scores))
        else:
            top_ids = gt_ids
            top_scores = gt_scores
    else:
        top_ids = doc_ids
        top_scores = scores

    pairs = [(int(top_ids[i]), float(top_scores[i])) for i in range(top_ids.size)]
    pairs.sort(key=_rank_sort_key)
    return tuple(pairs)


def score_vectorized(
    index: Index,
    pos_terms: Sequence[str],
    matched_ids: Sequence[int],
    scorer: BM25 | TfIdf,
    k: int,
) -> tuple[tuple[int, float], ...]:
    """Vectorized ranking over positive query terms constrained to matched_ids."""
    if k <= 0 or not matched_ids or not pos_terms:
        return ()

    vindex = get_vectorized_index(index)
    if vindex.size == 0 or vindex.num_docs == 0:
        return ()

    # Fast path for single-term queries without boolean narrowing (avoids O(N) zero buffer)
    if len(pos_terms) == 1:
        arrays = vindex.term_arrays(pos_terms[0])
        if arrays is None:
            return ()
        doc_ids, tfs = arrays
        if doc_ids.size == len(matched_ids):
            term_scores = _term_scores_numpy(vindex, doc_ids, tfs, scorer)
            mask = term_scores > 0.0
            return _select_top_k(doc_ids[mask], term_scores[mask], k)

    scores: NDArray[np.float64] = np.zeros(vindex.size, dtype=np.float64)
    for term in pos_terms:
        arrays = vindex.term_arrays(term)
        if arrays is None:
            continue
        doc_ids, tfs = arrays
        if doc_ids.size == 0:
            continue
        scores[doc_ids] += _term_scores_numpy(vindex, doc_ids, tfs, scorer)

    allowed_arr: NDArray[np.int32] = np.asarray(matched_ids, dtype=np.int32)
    candidate_scores: NDArray[np.float64] = scores[allowed_arr]
    mask = candidate_scores > 0.0
    return _select_top_k(allowed_arr[mask], candidate_scores[mask], k)
