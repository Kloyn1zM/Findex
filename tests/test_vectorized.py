"""Tests for Lab 8 M2: NumPy-vectorized BM25/TF-IDF ranking identity vs Lab 3 Python scorer."""

from pathlib import Path

import pytest

from findex.scoring import BM25, Scorer, TfIdf
from findex.search import clear_search_cache, search
from findex.store import open_index


@pytest.mark.parametrize("scorer", [BM25(), BM25(k1=1.2, b=0.5), TfIdf()])
@pytest.mark.parametrize(
    "query",
    [
        "python",
        "concurrency",
        "python OR rust",
        "python AND concurrency",
        "python NOT rust",
        '"event loop"',
        "search engine",
        "nonexistentterm",
    ],
)
def test_numpy_scorer_matches_python_scorer_identically(
    saved_index_path: Path,
    scorer: Scorer,
    query: str,
) -> None:
    clear_search_cache()
    with open_index(saved_index_path) as ix:
        py_hits = search(ix, query, scorer=scorer, k=10, backend="python")
        np_hits = search(ix, query, scorer=scorer, k=10, backend="numpy")

    assert [h.doc_id for h in np_hits] == [h.doc_id for h in py_hits]
    for np_h, py_h in zip(np_hits, py_hits, strict=True):
        assert np_h.score == pytest.approx(py_h.score, rel=1e-12, abs=1e-12)
        assert np_h.title == py_h.title
        assert np_h.snippet == py_h.snippet
