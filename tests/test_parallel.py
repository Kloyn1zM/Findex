"""Tests for Lab 5: parallel map-reduce indexing, byte-identity, and worker error handling."""

import json
from array import array
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.index import DocMeta, Posting, build_index_from_docs
from findex.parallel import (
    ExecutorKind,
    PartialIndex,
    build_index_parallel,
    build_partial,
    merge,
)
from findex.store import save

runner = CliRunner()


def _failing_worker(value: int) -> int:
    """Module-level picklable worker that intentionally fails (for spawn-safe error tests)."""
    raise RuntimeError(f"worker failure on {value}")


@pytest.mark.parametrize("include_positions", [False, True])
def test_serial_threads_processes_produce_byte_identical_indexes(
    tmp_path: Path,
    include_positions: bool,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    # Mix plain .txt files and sharded .jsonl (including a malformed line)
    for i in range(6):
        (corpus / f"{i:02d}_note.txt").write_text(
            f"Python concurrency lab {i} GIL threads processes map reduce.",
            encoding="utf-8",
        )
    for shard in range(3):
        lines = [
            json.dumps(
                {"title": f"Shard {shard} Doc {j}", "text": f"parallel indexing {shard} {j}"}
            )
            for j in range(4)
        ]
        if shard == 1:
            lines.insert(2, "{corrupted json line")
        (corpus / f"z_shard_{shard:02d}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    # Lab 4 baseline (single-pass build_index_from_docs)
    baseline = build_index_from_docs(iter_documents(corpus), include_positions=include_positions)
    baseline_file = tmp_path / "baseline.bin"
    save(baseline, baseline_file, fmt="binary")
    baseline_bytes = baseline_file.read_bytes()

    executors: list[ExecutorKind] = ["serial", "threads", "processes"]
    for exec_kind in executors:
        built, metrics = build_index_parallel(
            corpus,
            executor=exec_kind,
            workers=3,
            include_positions=include_positions,
        )
        assert metrics.executor == exec_kind
        assert dict(built) == dict(baseline)
        assert built.doc_lengths == baseline.doc_lengths
        assert built.doc_meta == baseline.doc_meta
        assert built.doc_texts == baseline.doc_texts

        out_file = tmp_path / f"{exec_kind}.bin"
        save(built, out_file, fmt="binary")
        assert out_file.read_bytes() == baseline_bytes


def test_merge_handles_out_of_order_and_interleaved_partials(tmp_path: Path) -> None:
    f0 = tmp_path / "01.txt"
    f1 = tmp_path / "02.txt"
    f0.write_text("alpha beta", encoding="utf-8")
    f1.write_text("beta gamma", encoding="utf-8")

    p0 = build_partial([f0], chunk_index=0, start_doc_id=0)
    p1 = build_partial([f1], chunk_index=1, start_doc_id=1)

    # Supply partials out of order (as as_completed often does)
    merged = merge([p1, p0])
    assert merged["beta"] == [Posting(doc_id=0, tf=1), Posting(doc_id=1, tf=1)]
    assert merged.num_docs == 2

    # Interleaved doc_ids (non-contiguous chunk indices) trigger two-pointer merge
    manual_a = PartialIndex(
        chunk_index=10,
        start_doc_id=0,
        num_docs=2,
        terms=["x"],
        counts=array("I", [2]),
        doc_ids=array("I", [0, 5]),
        tfs=array("I", [1, 2]),
        positions=array("I"),
        doc_lengths={0: 1, 5: 2},
        doc_meta={0: DocMeta(0, "A", "a"), 5: DocMeta(5, "C", "c")},
        doc_texts={},
        has_positions=False,
    )
    manual_b = PartialIndex(
        chunk_index=20,
        start_doc_id=2,
        num_docs=1,
        terms=["x"],
        counts=array("I", [1]),
        doc_ids=array("I", [2]),
        tfs=array("I", [3]),
        positions=array("I"),
        doc_lengths={2: 3},
        doc_meta={2: DocMeta(2, "B", "b")},
        doc_texts={},
        has_positions=False,
    )
    interleaved = merge([manual_a, manual_b])
    assert [p.doc_id for p in interleaved["x"]] == [0, 2, 5]


def test_worker_exceptions_propagate_cleanly_in_threads_and_processes() -> None:
    with ThreadPoolExecutor(max_workers=2) as tpool:
        fut = tpool.submit(_failing_worker, 1)
        with pytest.raises(RuntimeError, match="worker failure on 1"):
            fut.result()

    with ProcessPoolExecutor(max_workers=2) as ppool:
        fut = ppool.submit(_failing_worker, 2)
        with pytest.raises(RuntimeError, match="worker failure on 2"):
            fut.result()


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_cli_index_supports_all_three_executors(
    mini_corpus_dir: Path,
    tmp_path: Path,
    executor: str,
) -> None:
    out_file = tmp_path / f"cli_{executor}.bin"
    res = runner.invoke(
        app,
        [
            "index",
            str(mini_corpus_dir),
            "--out",
            str(out_file),
            "--executor",
            executor,
            "--workers",
            "2",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_file.exists()
    assert f"executor={executor}" in res.output
