"""Map-reduce parallel index construction across serial, thread, and process executors (Lab 5)."""

import os
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.corpus import (
    collect_corpus_files,
    count_documents_in_file,
    iter_documents_from_paths,
)
from findex.index import DocId, DocMeta, Index, Posting
from findex.tokenize import tokenize

ExecutorKind = Literal["serial", "threads", "processes"]


@dataclass(slots=True)
class PartialIndex:
    """Compact, picklable partial index storing postings as flat uint32 arrays.

    Packing doc_ids, tfs, and positions into contiguous array('I') buffers avoids
    pickling millions of small Posting objects across ProcessPoolExecutor pipes.
    """

    chunk_index: int
    start_doc_id: DocId
    num_docs: int
    terms: list[str]
    counts: array[int]
    doc_ids: array[int]
    tfs: array[int]
    positions: array[int]
    doc_lengths: dict[DocId, int]
    doc_meta: dict[DocId, DocMeta]
    doc_texts: dict[DocId, str]
    has_positions: bool
    cpu_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class BuildMetrics:
    """Timing breakdown for parallel index construction (used in Lab 5 benchmarks)."""

    executor: ExecutorKind
    workers: int
    chunks: int
    wall_sec: float
    cpu_sec: float
    merge_sec: float


def build_partial(
    doc_paths: Sequence[Path],
    *,
    chunk_index: int = 0,
    start_doc_id: DocId = 0,
    include_positions: bool = False,
    store_text: bool = True,
) -> PartialIndex:
    """Module-level, picklable worker: stream files at doc_paths into a compact PartialIndex."""
    cpu_t0 = time.process_time()

    doc_ids_map: defaultdict[str, list[int]] = defaultdict(list)
    tfs_map: defaultdict[str, list[int]] = defaultdict(list)
    pos_map: defaultdict[str, list[int]] = defaultdict(list)
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}
    doc_texts: dict[DocId, str] = {}
    num_docs = 0

    for doc in iter_documents_from_paths(doc_paths, start_id=start_doc_id):
        num_docs += 1
        if include_positions:
            term_positions: defaultdict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokenize(doc.text)):
                term_positions[token].append(pos)
            doc_len = sum(len(pos_list) for pos_list in term_positions.values())
            for term, pos_list in term_positions.items():
                doc_ids_map[term].append(doc.doc_id)
                tfs_map[term].append(len(pos_list))
                pos_map[term].extend(pos_list)
        else:
            term_counts = Counter(tokenize(doc.text))
            doc_len = term_counts.total()
            for term, tf in term_counts.items():
                doc_ids_map[term].append(doc.doc_id)
                tfs_map[term].append(tf)

        doc_lengths[doc.doc_id] = doc_len
        doc_meta[doc.doc_id] = DocMeta(
            doc_id=doc.doc_id,
            title=doc.title,
            path=str(doc.path),
        )
        if store_text:
            doc_texts[doc.doc_id] = doc.text

    terms: list[str] = []
    counts: array[int] = array("I")
    flat_doc_ids: array[int] = array("I")
    flat_tfs: array[int] = array("I")
    flat_positions: array[int] = array("I")

    for term, dids in doc_ids_map.items():
        terms.append(term)
        counts.append(len(dids))
        flat_doc_ids.extend(dids)
        flat_tfs.extend(tfs_map[term])
        if include_positions:
            flat_positions.extend(pos_map[term])

    cpu_elapsed = time.process_time() - cpu_t0
    return PartialIndex(
        chunk_index=chunk_index,
        start_doc_id=start_doc_id,
        num_docs=num_docs,
        terms=terms,
        counts=counts,
        doc_ids=flat_doc_ids,
        tfs=flat_tfs,
        positions=flat_positions,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        doc_texts=doc_texts,
        has_positions=include_positions,
        cpu_sec=cpu_elapsed,
    )


def _merge_sorted_postings(a: list[Posting], b: list[Posting]) -> list[Posting]:
    """Two-pointer sorted merge of two posting lists ordered by doc_id."""
    merged: list[Posting] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        if a[i].doc_id <= b[j].doc_id:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    if i < len_a:
        merged.extend(a[i:])
    if j < len_b:
        merged.extend(b[j:])
    return merged


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Merge compact PartialIndex chunks into a single Index with sorted postings."""
    ordered = sorted(partials, key=lambda p: p.chunk_index)
    if not ordered:
        return Index(postings={}, doc_lengths={}, doc_meta={}, has_positions=False, doc_texts={})

    has_positions = ordered[0].has_positions
    merged_postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}
    doc_texts: dict[DocId, str] = {}

    contiguous_chunks = all(p.chunk_index == idx for idx, p in enumerate(ordered))
    expected_doc_id = 0

    for partial in ordered:
        delta = (expected_doc_id - partial.start_doc_id) if contiguous_chunks else 0
        p_dids = partial.doc_ids
        p_tfs = partial.tfs
        p_pos = partial.positions
        cursor = 0
        pos_cursor = 0

        for term, count in zip(partial.terms, partial.counts, strict=True):
            end = cursor + count
            if has_positions:
                chunk_postings: list[Posting] = []
                for idx in range(cursor, end):
                    tf = p_tfs[idx]
                    pos_end = pos_cursor + tf
                    chunk_postings.append(
                        Posting(
                            doc_id=p_dids[idx] + delta,
                            tf=tf,
                            positions=tuple(p_pos[pos_cursor:pos_end]),
                        )
                    )
                    pos_cursor = pos_end
            else:
                chunk_postings = [
                    Posting(doc_id=p_dids[idx] + delta, tf=p_tfs[idx]) for idx in range(cursor, end)
                ]
            cursor = end

            existing = merged_postings.get(term)
            if existing is None:
                merged_postings[term] = chunk_postings
            elif not chunk_postings:
                continue
            elif not existing or existing[-1].doc_id < chunk_postings[0].doc_id:
                existing.extend(chunk_postings)
            else:
                merged_postings[term] = _merge_sorted_postings(existing, chunk_postings)

        if delta == 0:
            doc_lengths.update(partial.doc_lengths)
            doc_meta.update(partial.doc_meta)
            doc_texts.update(partial.doc_texts)
        else:
            for old_id, length in partial.doc_lengths.items():
                doc_lengths[old_id + delta] = length
            for old_id, meta in partial.doc_meta.items():
                new_id = old_id + delta
                doc_meta[new_id] = DocMeta(doc_id=new_id, title=meta.title, path=meta.path)
            for old_id, text in partial.doc_texts.items():
                doc_texts[old_id + delta] = text

        expected_doc_id += partial.num_docs

    return Index(
        postings=merged_postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=has_positions,
        doc_texts=doc_texts,
    )


def plan_chunks(root: Path, num_chunks: int) -> list[tuple[int, list[Path], DocId]]:
    """Partition corpus file paths into contiguous chunks with upfront start_doc_id ranges."""
    files = collect_corpus_files(root)
    if not files:
        return []

    effective_chunks = max(1, min(num_chunks, len(files)))
    if effective_chunks == 1:
        return [(0, files, 0)]

    file_counts = [count_documents_in_file(p) for p in files]
    total_files = len(files)

    base, rem = divmod(total_files, effective_chunks)
    chunks: list[tuple[int, list[Path], DocId]] = []
    file_cursor = 0
    doc_cursor = 0

    for chunk_idx in range(effective_chunks):
        chunk_len = base + (1 if chunk_idx < rem else 0)
        chunk_paths = files[file_cursor : file_cursor + chunk_len]
        chunk_doc_total = sum(file_counts[file_cursor : file_cursor + chunk_len])
        chunks.append((chunk_idx, chunk_paths, doc_cursor))
        file_cursor += chunk_len
        doc_cursor += chunk_doc_total

    return chunks


def _run_chunk(
    args: tuple[int, list[Path], DocId, bool, bool],
) -> PartialIndex:
    """Top-level helper unpacking chunk tuple for clean pool.submit calls."""
    chunk_idx, paths, start_id, include_positions, store_text = args
    return build_partial(
        paths,
        chunk_index=chunk_idx,
        start_doc_id=start_id,
        include_positions=include_positions,
        store_text=store_text,
    )


def build_index_parallel(
    root: Path,
    *,
    executor: ExecutorKind = "processes",
    workers: int | None = None,
    include_positions: bool = False,
    store_text: bool = True,
    on_chunk: Callable[[int, int], None] | None = None,
) -> tuple[Index, BuildMetrics]:
    """Build an Index using serial, threaded, or process-based map-reduce execution."""
    resolved_workers = 1 if executor == "serial" else max(1, workers or (os.cpu_count() or 1))
    num_chunks = 1 if executor == "serial" else resolved_workers

    wall_t0 = time.perf_counter()
    parent_cpu_t0 = time.process_time()

    chunks = plan_chunks(root, num_chunks=num_chunks)
    total_chunks = len(chunks)
    partials: list[PartialIndex] = []

    if total_chunks == 0:
        empty = Index(postings={}, doc_lengths={}, doc_meta={}, has_positions=include_positions)
        metrics = BuildMetrics(
            executor=executor,
            workers=resolved_workers,
            chunks=0,
            wall_sec=time.perf_counter() - wall_t0,
            cpu_sec=time.process_time() - parent_cpu_t0,
            merge_sec=0.0,
        )
        return empty, metrics

    chunk_args = [
        (chunk_idx, paths, start_id, include_positions, store_text)
        for chunk_idx, paths, start_id in chunks
    ]

    if executor == "serial":
        for completed, arg in enumerate(chunk_args, start=1):
            partials.append(_run_chunk(arg))
            if on_chunk is not None:
                on_chunk(completed, total_chunks)
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=resolved_workers) as pool:
            futures: list[Future[PartialIndex]] = [
                pool.submit(_run_chunk, arg) for arg in chunk_args
            ]
            for completed, fut in enumerate(as_completed(futures), start=1):
                partials.append(fut.result())
                if on_chunk is not None:
                    on_chunk(completed, total_chunks)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as pool:
            futures = [pool.submit(_run_chunk, arg) for arg in chunk_args]
            for completed, fut in enumerate(as_completed(futures), start=1):
                partials.append(fut.result())
                if on_chunk is not None:
                    on_chunk(completed, total_chunks)

    merge_t0 = time.perf_counter()
    index = merge(partials)
    merge_sec = time.perf_counter() - merge_t0

    wall_sec = time.perf_counter() - wall_t0
    parent_cpu_sec = time.process_time() - parent_cpu_t0
    total_cpu_sec = (
        parent_cpu_sec + sum(p.cpu_sec for p in partials)
        if executor == "processes"
        else parent_cpu_sec
    )

    metrics = BuildMetrics(
        executor=executor,
        workers=resolved_workers,
        chunks=total_chunks,
        wall_sec=wall_sec,
        cpu_sec=total_cpu_sec,
        merge_sec=merge_sec,
    )
    return index, metrics
