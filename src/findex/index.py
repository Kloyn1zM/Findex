"""Inverted index object model and single-pass streaming builder (Lab 2–4)."""

import argparse
import functools
import gc
import itertools
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from findex.corpus import Document, iter_documents
from findex.timing import timed
from findex.tokenize import tokenize

type DocId = int

_INDEX_TOKEN_SEQ = itertools.count(1)


@dataclass(frozen=True, slots=True)
class Posting:
    """Single occurrence record of a term inside a document."""

    doc_id: DocId
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Metadata needed to display search results."""

    doc_id: DocId
    title: str
    path: str


class Index(Mapping[str, list[Posting]]):
    """Inverted index implementing the Python Mapping protocol.

    Supports len(index), `term in index`, index[term], iteration over terms,
    repr(index), and cached corpus statistics needed for BM25/TF-IDF ranking.
    """

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[DocId, int],
        doc_meta: dict[DocId, DocMeta],
        *,
        has_positions: bool = False,
        doc_texts: dict[DocId, str] | None = None,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self._doc_texts: dict[DocId, str] = doc_texts if doc_texts is not None else {}
        self.has_positions = has_positions
        self.cache_token: int = next(_INDEX_TOKEN_SEQ)
        self._closed = False

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @property
    def doc_lengths(self) -> dict[DocId, int]:
        return self._doc_lengths

    @property
    def doc_meta(self) -> dict[DocId, DocMeta]:
        return self._doc_meta

    @property
    def doc_texts(self) -> dict[DocId, str]:
        return self._doc_texts

    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def closed(self) -> bool:
        return self._closed

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def doc_length(self, doc_id: DocId) -> int:
        return self._doc_lengths[doc_id]

    def doc_text(self, doc_id: DocId) -> str:
        return self._doc_texts.get(doc_id, "")

    def df(self, term: str) -> int:
        plist = self._postings.get(term)
        return len(plist) if plist is not None else 0

    def close(self) -> None:
        self._closed = True

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._postings

    def __getitem__(self, key: str) -> list[Posting]:
        return self._postings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):,}, docs={self.num_docs:,})"


# Backward-compatible alias for Lab 2 callers
InvertedIndex = Index


def build_index_from_docs(
    docs: Iterable[Document],
    *,
    include_positions: bool = False,
    store_text: bool = True,
    on_doc: Callable[[int], None] | None = None,
) -> Index:
    """Build an Index in a single pass from any Iterable of Document objects."""
    postings_map: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}
    doc_texts: dict[DocId, str] = {}

    for processed, doc in enumerate(docs, start=1):
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
        if store_text:
            doc_texts[doc.doc_id] = doc.text

        if on_doc is not None:
            on_doc(processed)

    return Index(
        postings=dict(postings_map),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=include_positions,
        doc_texts=doc_texts,
    )


@timed
def build_index(
    root: Path,
    *,
    include_positions: bool = False,
    store_text: bool = True,
    limit: int | None = None,
    executor: Literal["serial", "threads", "processes"] = "serial",
    workers: int | None = None,
    on_doc: Callable[[int], None] | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> Index:
    """Build an Index from a corpus path using serial, threaded, or process execution."""
    if limit is None and on_doc is None:
        from findex.parallel import build_index_parallel

        index, _ = build_index_parallel(
            root,
            executor=executor,
            workers=workers,
            include_positions=include_positions,
            store_text=store_text,
            on_chunk=on_chunk,
        )
        return index

    doc_stream: Iterable[Document] = iter_documents(root)
    if limit is not None:
        doc_stream = islice(doc_stream, limit)

    return build_index_from_docs(
        doc_stream,
        include_positions=include_positions,
        store_text=store_text,
        on_doc=on_doc,
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
        help="Store token positions per posting (enables phrase queries)",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Do not embed raw document text inside the index (smaller file, no snippets)",
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
    store_text: bool = not args.no_text
    limit: int | None = args.limit

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    index = build_index(
        corpus_path,
        include_positions=include_positions,
        store_text=store_text,
        limit=limit,
    )
    build_sec = time.perf_counter() - t0

    t_save = time.perf_counter()
    save(index, out_path, fmt=fmt)
    save_sec = time.perf_counter() - t_save

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_mib = out_path.stat().st_size / (1024 * 1024)
    peak_mib = peak_bytes / (1024 * 1024)

    print(f"Built {index!r}")
    print(f"Positions:   {'yes' if index.has_positions else 'no'}")
    print(f"Format:      {fmt} -> {out_path} ({file_mib:.2f} MiB)")
    print(f"Build time:  {build_sec:.3f} s | Save time: {save_sec:.3f} s")
    print(f"Peak memory: {peak_mib:.2f} MiB ({peak_bytes:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
