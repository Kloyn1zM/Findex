"""Inverted index data models and single-pass streaming builder (Lab 2, M1)."""

import argparse
import gc
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    """Single occurrence record of a term inside a document."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Metadata needed to display search results."""

    doc_id: int
    title: str
    path: str


@dataclass(slots=True)
class InvertedIndex:
    """In-memory inverted index built from a streaming corpus."""

    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    has_positions: bool = False


def build_index(
    root: Path,
    *,
    include_positions: bool = False,
    limit: int | None = None,
) -> InvertedIndex:
    """Build an inverted index in a single lazy pass over the corpus.

    Because iter_documents yields strictly increasing doc_ids, every term's
    postings list is naturally sorted by doc_id without a post-sort pass.
    """
    postings_map: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    doc_stream = iter_documents(root)
    if limit is not None:
        doc_stream = islice(doc_stream, limit)

    for doc in doc_stream:
        if include_positions:
            term_positions: defaultdict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokenize(doc.text)):
                term_positions[token].append(pos)
            doc_len = sum(len(pos_list) for pos_list in term_positions.values())
            for term, pos_list in term_positions.items():
                postings_map[term].append(
                    Posting(doc_id=doc.doc_id, tf=len(pos_list), positions=tuple(pos_list))
                )
        else:
            term_counts = Counter(tokenize(doc.text))
            doc_len = term_counts.total()
            for term, tf in term_counts.items():
                postings_map[term].append(Posting(doc_id=doc.doc_id, tf=tf))

        doc_lengths[doc.doc_id] = doc_len
        doc_meta[doc.doc_id] = DocMeta(
            doc_id=doc.doc_id,
            title=doc.title,
            path=str(doc.path),
        )

    return InvertedIndex(
        postings=dict(postings_map),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=include_positions,
    )


def main(argv: list[str] | None = None) -> int:
    from findex.store import save

    parser = argparse.ArgumentParser(
        prog="python -m findex.index",
        description="Build and persist an inverted index from a corpus.",
    )
    parser.add_argument("corpus", type=Path, help="Path to corpus directory or .jsonl file")
    parser.add_argument("--out", type=Path, required=True, help="Output index file path")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("binary", "pickle"),
        default="binary",
        help="Serialization format (default: binary)",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Store token positions per posting (enables phrase queries in Lab 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N documents",
    )
    args = parser.parse_args(argv)

    corpus_path: Path = args.corpus
    out_path: Path = args.out
    fmt: Literal["binary", "pickle"] = args.fmt
    include_positions: bool = args.positions
    limit: int | None = args.limit

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    index = build_index(corpus_path, include_positions=include_positions, limit=limit)
    build_sec = time.perf_counter() - t0

    t_save = time.perf_counter()
    save(index, out_path, fmt=fmt)
    save_sec = time.perf_counter() - t_save

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_mib = out_path.stat().st_size / (1024 * 1024)
    peak_mib = peak_bytes / (1024 * 1024)

    print(f"Indexed {len(index.doc_meta):,} documents ({len(index.postings):,} unique terms)")
    print(f"Positions:   {'yes' if index.has_positions else 'no'}")
    print(f"Format:      {fmt} -> {out_path} ({file_mib:.2f} MiB)")
    print(f"Build time:  {build_sec:.3f} s | Save time: {save_sec:.3f} s")
    print(f"Peak memory: {peak_mib:.2f} MiB ({peak_bytes:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
