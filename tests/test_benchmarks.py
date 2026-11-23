"""pytest-benchmark regression tests for build_index and search (Lab 8 M1)."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from findex.index import build_index
from findex.scoring import BM25
from findex.search import search
from findex.store import open_index

_T = TypeVar("_T")


class BenchmarkFixture(Protocol):
    """Minimal strict-typing protocol for pytest-benchmark's fixture."""

    def __call__(self, func: Callable[..., _T], *args: object, **kwargs: object) -> _T: ...


def test_benchmark_build_index(
    benchmark: BenchmarkFixture,
    mini_corpus_dir: Path,
) -> None:
    ix = benchmark(build_index, mini_corpus_dir, executor="serial", include_positions=False)
    assert ix.num_docs == 5


def test_benchmark_search(
    benchmark: BenchmarkFixture,
    saved_index_path: Path,
) -> None:
    with open_index(saved_index_path) as ix:
        hits = benchmark(search, ix, "python OR concurrency", scorer=BM25(), k=5)
        assert len(hits) >= 1
