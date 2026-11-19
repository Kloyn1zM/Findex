"""Boolean search using two-pointer sorted merge or Python sets (Lab 2, M2)."""

import argparse
import gc
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from findex.index import InvertedIndex
from findex.store import load
from findex.tokenize import tokenize

SearchEngine = Literal["merge", "set"]


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Two-pointer intersection of two sorted doc_id sequences in O(len(a) + len(b))."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        da = a[i]
        db = b[j]
        if da == db:
            result.append(da)
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Two-pointer union of two sorted doc_id sequences in O(len(a) + len(b))."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        da = a[i]
        db = b[j]
        if da == db:
            result.append(da)
            i += 1
            j += 1
        elif da < db:
            result.append(da)
            i += 1
        else:
            result.append(db)
            j += 1
    if i < len_a:
        result.extend(a[i:])
    if j < len_b:
        result.extend(b[j:])
    return result


def merge_and_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Two-pointer difference (elements in sorted a that are absent from sorted b)."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        da = a[i]
        db = b[j]
        if da == db:
            i += 1
            j += 1
        elif da < db:
            result.append(da)
            i += 1
        else:
            j += 1
    if i < len_a:
        result.extend(a[i:])
    return result


def _term_doc_ids(index: InvertedIndex, term: str) -> list[int]:
    plist = index.postings.get(term)
    if not plist:
        return []
    return [p.doc_id for p in plist]


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
    index: InvertedIndex,
    query: str,
    *,
    engine: SearchEngine = "merge",
) -> list[int]:
    """Evaluate a left-to-right Boolean query with implicit AND, OR, and NOT.

    Operators must be uppercase (`AND`, `OR`, `NOT`). Terms are normalized
    through the same streaming tokenizer used during indexing.
    """
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.search",
        description="Run Boolean queries against a persisted inverted index.",
    )
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", type=str, help="Boolean query (implicit AND, OR, NOT)")
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="Postings combination backend (default: merge)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of matching titles to display (default: 10)",
    )
    args = parser.parse_args(argv)

    index_path: Path = args.index_path
    query: str = args.query
    engine: SearchEngine = args.engine
    limit: int = args.limit

    gc.collect()
    tracemalloc.start()

    t_load = time.perf_counter()
    index = load(index_path)
    load_ms = (time.perf_counter() - t_load) * 1000

    t_search = time.perf_counter()
    doc_ids = boolean_search(index, query, engine=engine)
    search_ms = (time.perf_counter() - t_search) * 1000

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query:  {query!r} (engine={engine})")
    print(f"Hits:   {len(doc_ids):,} documents")
    print(f"Timing: load={load_ms:.1f} ms | search={search_ms:.3f} ms")
    print(f"Peak:   {peak_bytes / (1024 * 1024):.2f} MiB\n")

    for rank, doc_id in enumerate(doc_ids[:limit], start=1):
        meta = index.doc_meta[doc_id]
        print(f"  {rank:>2}. [{doc_id}] {meta.title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
