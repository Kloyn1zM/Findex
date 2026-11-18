"""Corpus statistics pipeline (M3) and eager-vs-lazy memory benchmark (M4)."""

import argparse
import gc
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Summary metrics and resource measurements for a corpus pass."""

    mode: str
    num_docs: int
    total_tokens: int
    vocab_size: int
    top_terms: list[tuple[str, int]]
    elapsed_sec: float
    peak_memory_bytes: int


def compute_stats_lazy(
    root: Path,
    limit: int | None = None,
    top_k: int = 50,
) -> CorpusStats:
    """Compute corpus statistics in a single streaming pass."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_stream = iter_documents(root)
    if limit is not None:
        doc_stream = islice(doc_stream, limit)

    counts: Counter[str] = Counter()
    num_docs = 0

    for doc in doc_stream:
        num_docs += 1
        counts.update(tokenize(doc.text))

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CorpusStats(
        mode="lazy (generators)",
        num_docs=num_docs,
        total_tokens=counts.total(),
        vocab_size=len(counts),
        top_terms=counts.most_common(top_k),
        elapsed_sec=elapsed,
        peak_memory_bytes=peak_bytes,
    )


def compute_stats_eager(
    root: Path,
    limit: int | None = None,
    top_k: int = 50,
) -> CorpusStats:
    """Deliberately eager baseline: materialize all documents and token lists."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_stream = iter_documents(root)
    if limit is not None:
        doc_stream = islice(doc_stream, limit)

    all_docs: list[Document] = list(doc_stream)
    tokenized_docs: list[list[str]] = [list(tokenize(doc.text)) for doc in all_docs]
    flat_tokens: list[str] = [tok for doc_tokens in tokenized_docs for tok in doc_tokens]

    counts: Counter[str] = Counter(flat_tokens)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CorpusStats(
        mode="eager (lists)",
        num_docs=len(all_docs),
        total_tokens=counts.total(),
        vocab_size=len(counts),
        top_terms=counts.most_common(top_k),
        elapsed_sec=elapsed,
        peak_memory_bytes=peak_bytes,
    )


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _print_stats_report(stats: CorpusStats) -> None:
    print(f"=== Corpus Statistics [{stats.mode}] ===")
    print(f"Documents:       {stats.num_docs:,}")
    print(f"Total tokens:    {stats.total_tokens:,}")
    print(f"Vocabulary size: {stats.vocab_size:,}")
    print(f"Elapsed time:    {stats.elapsed_sec:.3f} s")
    peak_mib = _format_mib(stats.peak_memory_bytes)
    print(f"Peak memory:     {peak_mib} ({stats.peak_memory_bytes:,} B)")
    print(f"\nTop-{len(stats.top_terms)} terms:")
    for rank, (term, count) in enumerate(stats.top_terms, start=1):
        print(f"  {rank:>2}. {term:<20} {count:,}")


def _print_comparison_table(eager: CorpusStats, lazy: CorpusStats) -> None:
    print("\nMarkdown table for README (M4):\n")
    print("| Version | Documents | Peak memory | Elapsed |")
    print("|---|---|---|---|")
    for row in (eager, lazy):
        print(
            f"| {row.mode} | {row.num_docs:,} | "
            f"{_format_mib(row.peak_memory_bytes)} | {row.elapsed_sec:.3f} s |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.stats",
        description="Compute streaming corpus statistics and peak memory.",
    )
    parser.add_argument("corpus", type=Path, help="Path to corpus directory or .jsonl file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents via itertools.islice",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top terms to display (default: 50)",
    )
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="Run both eager and lazy pipelines and print comparison table",
    )
    args = parser.parse_args(argv)

    corpus_path: Path = args.corpus
    limit: int | None = args.limit
    top_k: int = args.top
    compare_eager: bool = args.compare_eager

    if compare_eager:
        eager_stats = compute_stats_eager(corpus_path, limit=limit, top_k=top_k)
        lazy_stats = compute_stats_lazy(corpus_path, limit=limit, top_k=top_k)
        _print_stats_report(lazy_stats)
        _print_comparison_table(eager_stats, lazy_stats)
    else:
        lazy_stats = compute_stats_lazy(corpus_path, limit=limit, top_k=top_k)
        _print_stats_report(lazy_stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
