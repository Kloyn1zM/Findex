"""Tests for Lab 3: Index object model, query AST, ranking sanity checks, decorators, snippets."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse
from findex.scoring import BM25, SearchResult, TfIdf, make_snippet
from findex.search import clear_search_cache, search, search_cache_info
from findex.store import open_index, save
from findex.timing import timed


def _make_index(tmp_path: Path, docs: dict[str, str], *, positions: bool = True) -> Index:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, body in docs.items():
        (corpus / name).write_text(body, encoding="utf-8")
    return build_index(corpus, include_positions=positions)


def test_index_mapping_protocol_and_properties(tmp_path: Path) -> None:
    ix = _make_index(
        tmp_path,
        {"01.txt": "python search engine", "02.txt": "python web"},
    )
    assert isinstance(ix, Mapping)
    assert len(ix) == 4
    assert "python" in ix
    assert "missing" not in ix
    assert ix.num_docs == 2
    assert ix.avg_doc_length == 2.5
    assert ix.doc_length(0) == 3
    assert ix.df("python") == 2
    assert ix.df("missing") == 0
    assert len(ix["python"]) == 2
    assert set(ix) == {"python", "search", "engine", "web"}
    assert repr(ix) == "Index(terms=4, docs=2)"

    with pytest.raises(KeyError):
        _ = ix["missing"]


def test_open_index_context_manager_closes_on_normal_and_error(tmp_path: Path) -> None:
    ix = _make_index(tmp_path, {"01.txt": "hello world"})
    out = tmp_path / "ix.bin"
    save(ix, out)

    with open_index(out) as loaded:
        assert not loaded.closed
        assert "hello" in loaded
    assert loaded.closed

    with pytest.raises(RuntimeError, match="boom"), open_index(out) as loaded_err:
        assert not loaded_err.closed
        raise RuntimeError("boom")
    assert loaded_err.closed


def test_query_parser_tree_equality_and_operators() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))
    assert parse("python AND (async OR await) NOT java") == And(
        And(Term("python"), Or(Term("async"), Term("await"))),
        Not(Term("java")),
    )
    assert parse('"exact phrase" AND python') == And(
        Phrase(("exact", "phrase")),
        Term("python"),
    )

    # Programmatic operator composition
    manual = Term("python") & (Term("async") | Term("await")) & ~Term("java")
    assert manual == parse("python AND (async OR await) NOT java")

    for bad in ("", "   ", "(python", 'unclosed "phrase', "OR python", '""'):
        with pytest.raises(QuerySyntaxError):
            parse(bad)


def test_phrase_search_requires_adjacent_positions(tmp_path: Path) -> None:
    ix = _make_index(
        tmp_path,
        {
            "01.txt": "python event loop is fast",
            "02.txt": "event handling in python loop",
            "03.txt": "loop event order reversed",
        },
        positions=True,
    )
    assert parse('"event loop"').evaluate(ix) == [0]
    assert parse("event loop").evaluate(ix) == [0, 1, 2]


def test_ranking_sanity_check_rare_term_outranks_common(tmp_path: Path) -> None:
    docs = {
        "01.txt": "common filler text here",
        "02.txt": "common more text here",
        "03.txt": "common another line here",
        "04.txt": "common rareword here",
    }
    ix = _make_index(tmp_path, docs)
    for scorer in (TfIdf(), BM25()):
        res = search(ix, "common OR rareword", scorer=scorer, k=4)
        assert res[0].doc_id == 3
        assert res[0].score > res[1].score


def test_ranking_sanity_check_bm25_tf_saturation(tmp_path: Path) -> None:
    docs = {
        "01.txt": "quantum " + "filler " * 25,
        "02.txt": "quantum " * 19 + "filler " * 7,
        "03.txt": "quantum " * 20 + "filler " * 6,
    }
    ix = _make_index(tmp_path, docs)
    bm25 = BM25(k1=1.5, b=0.75)
    s1 = bm25("quantum", ix["quantum"][0], ix)
    s19 = bm25("quantum", ix["quantum"][1], ix)
    s20 = bm25("quantum", ix["quantum"][2], ix)

    # 20th occurrence adds less than 1% of the first occurrence's score
    assert (s20 - s19) < 0.01 * s1


def test_ranking_sanity_check_bm25_prefers_shorter_doc(tmp_path: Path) -> None:
    docs = {
        "01.txt": "nebula star",
        "02.txt": "nebula " + "padding " * 100,
    }
    ix = _make_index(tmp_path, docs)
    res = search(ix, "nebula", scorer=BM25(), k=2)
    assert res[0].doc_id == 0
    assert res[0].score > res[1].score


def test_search_result_ordering_timed_wraps_and_cache(tmp_path: Path) -> None:
    r_low = SearchResult(doc_id=1, score=0.5, title="Low")
    r_high = SearchResult(doc_id=2, score=2.5, title="High")
    assert r_low < r_high
    assert sorted([r_high, r_low]) == [r_low, r_high]

    @timed
    def sample_fn(x: int) -> int:
        """Sample docstring."""
        return x + 1

    assert sample_fn.__name__ == "sample_fn"
    assert sample_fn.__doc__ == "Sample docstring."
    assert sample_fn(41) == 42

    ix = _make_index(tmp_path, {"01.txt": "alpha beta gamma"})
    clear_search_cache()
    _ = search(ix, "alpha", scorer=BM25(), k=5)
    first_info = search_cache_info()
    _ = search(ix, "alpha", scorer=BM25(), k=5)
    second_info = search_cache_info()
    assert second_info.hits == first_info.hits + 1


def test_snippet_highlighting() -> None:
    text = "The Python programming language emphasizes code readability and clean syntax."
    snippet = make_snippet(text, ["python", "syntax"], window=80)
    assert "**Python**" in snippet
    assert "**syntax**" in snippet
