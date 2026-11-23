"""Boolean and ranked search with LRU query caching and snippets (Lab 2–3)."""

import argparse
import functools
import gc
import heapq
import logging
import sys
import time
import tracemalloc
import weakref
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.index import Index
from findex.query import merge_and, merge_and_not, merge_or, parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf, make_snippet
from findex.store import open_index
from findex.timing import timed
from findex.tokenize import tokenize
from findex.vectorized import score_vectorized

log = logging.getLogger(__name__)

SearchEngine = Literal["merge", "set"]
ScoringBackend = Literal["numpy", "python"]

_INDEX_REGISTRY: dict[int, weakref.ref[Index]] = {}


def _score_key(item: tuple[int, float]) -> float:
    return item[1]


def _term_doc_ids(index: Index, term: str) -> list[int]:
    if term not in index:
        return []
    return [p.doc_id for p in index[term]]


def _combine_and(a: list[int], b: list[int], engine: SearchEngine) -> list[int]:
    if engine == "merge":
        return merge_and(a, b)
    return sorted(set(a) & set(b))


def _combine_or(a: list[int], b: list[int], engine: SearchEngine) -> list[int]:
    if engine == "merge":
        return merge_or(a, b)
    return sorted(set(a) | set(b))


def _combine_and_not(a: list[int], b: list[int], engine: SearchEngine) -> list[int]:
    if engine == "merge":
        return merge_and_not(a, b)
    return sorted(set(a) - set(b))


def boolean_search(
    index: Index,
    query: str,
    *,
    engine: SearchEngine = "merge",
) -> list[int]:
    """Evaluate a left-to-right Boolean query with implicit AND, OR, and NOT."""
    raw_parts = query.split()
    if not raw_parts:
        return []

    all_docs: list[int] | None = None
    result: list[int] | None = None
    pending_op: Literal["AND", "OR"] = "AND"
    negate_next = False

    for raw in raw_parts:
        if raw in {"AND", "OR"}:
            pending_op = "OR" if raw == "OR" else "AND"
            continue
        if raw == "NOT":
            negate_next = not negate_next
            continue

        normalized_terms = list(tokenize(raw))
        if not normalized_terms:
            continue

        for term in normalized_terms:
            term_docs = _term_doc_ids(index, term)

            if result is None:
                if negate_next:
                    if all_docs is None:
                        all_docs = sorted(index.doc_meta)
                    result = _combine_and_not(all_docs, term_docs, engine)
                else:
                    result = term_docs
            elif negate_next and pending_op == "AND":
                result = _combine_and_not(result, term_docs, engine)
            elif negate_next and pending_op == "OR":
                if all_docs is None:
                    all_docs = sorted(index.doc_meta)
                complement = _combine_and_not(all_docs, term_docs, engine)
                result = _combine_or(result, complement, engine)
            elif pending_op == "OR":
                result = _combine_or(result, term_docs, engine)
            else:
                result = _combine_and(result, term_docs, engine)

            pending_op = "AND"
            negate_next = False

    return result if result is not None else []


def _compute_ranked_ids(
    index: Index,
    query: str,
    scorer: Scorer,
    k: int,
    backend: ScoringBackend = "numpy",
) -> tuple[tuple[int, float], ...]:
    """Parse query, evaluate matched documents, score them, and return top-k (doc_id, score)."""
    if k <= 0 or not query.strip():
        return ()

    tree = parse_query(query)
    matched_ids = tree.evaluate(index)
    if not matched_ids:
        return ()

    pos_terms = tree.positive_terms()
    if not pos_terms:
        return tuple((doc_id, 0.0) for doc_id in matched_ids[:k])

    if backend == "numpy" and isinstance(scorer, (BM25, TfIdf)):
        return score_vectorized(index, pos_terms, matched_ids, scorer, k)

    allowed = set(matched_ids)
    scores: defaultdict[int, float] = defaultdict(float)

    for term in pos_terms:
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id in allowed:
                scores[posting.doc_id] += scorer.score(term, posting, index)

    top_pairs = heapq.nlargest(k, scores.items(), key=_score_key)
    return tuple(top_pairs)


@functools.lru_cache(maxsize=256)
def _cached_ranked_ids(
    cache_token: int,
    query: str,
    scorer: Scorer,
    k: int,
    backend: ScoringBackend = "numpy",
) -> tuple[tuple[int, float], ...]:
    ref = _INDEX_REGISTRY.get(cache_token)
    index = ref() if ref is not None else None
    if index is None:
        return ()
    return _compute_ranked_ids(index, query, scorer, k, backend=backend)


@dataclass(frozen=True, slots=True)
class SearchCacheInfo:
    """Public snapshot of LRU cache statistics for ranked search."""

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


def search_cache_info() -> SearchCacheInfo:
    """Return LRU cache statistics for the ranked search path."""
    info = _cached_ranked_ids.cache_info()
    return SearchCacheInfo(
        hits=info.hits,
        misses=info.misses,
        maxsize=info.maxsize,
        currsize=info.currsize,
    )


def clear_search_cache() -> None:
    """Clear cached ranked query results."""
    _cached_ranked_ids.cache_clear()


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    highlight_start: str = "**",
    highlight_end: str = "**",
    backend: ScoringBackend = "numpy",
) -> list[SearchResult]:
    """Run a parsed, ranked search returning top-k SearchResult objects with snippets."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    if k <= 0 or not query.strip():
        return []

    try:
        hash(active_scorer)
    except TypeError:
        is_hashable = False
    else:
        is_hashable = True

    if is_hashable:
        _INDEX_REGISTRY[index.cache_token] = weakref.ref(index)
        before_hits = _cached_ranked_ids.cache_info().hits
        ranked_pairs = _cached_ranked_ids(index.cache_token, query, active_scorer, k, backend)
        after_hits = _cached_ranked_ids.cache_info().hits
        if after_hits > before_hits:
            log.debug("search cache HIT for query=%r (%s)", query, _cached_ranked_ids.cache_info())
    else:
        ranked_pairs = _compute_ranked_ids(index, query, active_scorer, k, backend=backend)

    pos_terms = parse_query(query).positive_terms()
    results: list[SearchResult] = []
    for doc_id, score in ranked_pairs:
        meta = index.doc_meta[doc_id]
        snippet = make_snippet(
            index.doc_text(doc_id),
            pos_terms,
            highlight_start=highlight_start,
            highlight_end=highlight_end,
        )
        results.append(
            SearchResult(
                doc_id=doc_id,
                score=score,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results


def search_paginated(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    page: int = 1,
    page_size: int = 10,
    highlight_start: str = "**",
    highlight_end: str = "**",
    backend: ScoringBackend = "numpy",
) -> tuple[list[SearchResult], int]:
    """Run a ranked search returning (page_results, total_matching_docs)."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    if page < 1 or page_size <= 0 or not query.strip():
        return [], 0

    tree = parse_query(query)
    matched_ids = tree.evaluate(index)
    total = len(matched_ids)
    if total == 0:
        return [], 0

    offset = (page - 1) * page_size
    if offset >= total:
        return [], total

    fetch_k = min(total, offset + page_size)
    try:
        hash(active_scorer)
    except TypeError:
        is_hashable = False
    else:
        is_hashable = True

    if is_hashable:
        _INDEX_REGISTRY[index.cache_token] = weakref.ref(index)
        ranked_pairs = _cached_ranked_ids(index.cache_token, query, active_scorer, fetch_k, backend)
    else:
        ranked_pairs = _compute_ranked_ids(index, query, active_scorer, fetch_k, backend=backend)

    page_pairs = ranked_pairs[offset : offset + page_size]
    pos_terms = tree.positive_terms()
    results: list[SearchResult] = []
    for doc_id, score in page_pairs:
        meta = index.doc_meta[doc_id]
        snippet = make_snippet(
            index.doc_text(doc_id),
            pos_terms,
            highlight_start=highlight_start,
            highlight_end=highlight_end,
        )
        results.append(
            SearchResult(
                doc_id=doc_id,
                score=score,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.search",
        description="Run ranked or Boolean queries against a persisted index.",
    )
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", type=str, help="Query string (supports AND, OR, NOT, (), quotes)")
    parser.add_argument(
        "--mode",
        choices=("ranked", "boolean"),
        default="ranked",
        help="Search mode (default: ranked)",
    )
    parser.add_argument(
        "--scorer",
        choices=("bm25", "tfidf"),
        default="bm25",
        help="Ranking algorithm in ranked mode (default: bm25)",
    )
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="Postings backend in boolean mode (default: merge)",
    )
    parser.add_argument(
        "--limit",
        "-k",
        type=int,
        default=10,
        help="Maximum number of results to display (default: 10)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (shows @timed durations and cache hits)",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )

    index_path: Path = args.index_path
    query: str = args.query
    mode: Literal["ranked", "boolean"] = args.mode
    scorer_name: Literal["bm25", "tfidf"] = args.scorer
    engine: SearchEngine = args.engine
    limit: int = args.limit

    gc.collect()
    tracemalloc.start()

    with open_index(index_path) as index:
        t_search = time.perf_counter()
        if mode == "boolean":
            doc_ids = boolean_search(index, query, engine=engine)
            search_ms = (time.perf_counter() - t_search) * 1000.0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Query:  {query!r} (mode=boolean, engine={engine})")
            print(f"Hits:   {len(doc_ids):,} documents | search={search_ms:.3f} ms")
            print(f"Peak:   {peak_bytes / (1024 * 1024):.2f} MiB\n")
            for rank, doc_id in enumerate(doc_ids[:limit], start=1):
                print(f"  {rank:>2}. [{doc_id}] {index.doc_meta[doc_id].title}")
        else:
            scorer: Scorer = BM25() if scorer_name == "bm25" else TfIdf()
            results = search(index, query, scorer=scorer, k=limit)
            search_ms = (time.perf_counter() - t_search) * 1000.0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Query:  {query!r} (mode=ranked, scorer={scorer_name})")
            print(f"Top:    {len(results)} results | search={search_ms:.3f} ms")
            print(f"Peak:   {peak_bytes / (1024 * 1024):.2f} MiB\n")
            for rank, res in enumerate(results, start=1):
                print(f"  {rank:>2}. {res}")
                if res.snippet:
                    print(f"      {res.snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
