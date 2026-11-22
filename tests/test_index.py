"""Tests for Lab 2: dataclasses/hashing, inverted index, Boolean merge, and persistence."""

from pathlib import Path

import pytest

from findex.index import DocMeta, Posting, build_index
from findex.search import boolean_search, merge_and, merge_and_not, merge_or
from findex.store import StorageFormat, load, save


def test_posting_and_docmeta_are_frozen_slotted_and_hashable() -> None:
    p1 = Posting(doc_id=1, tf=3, positions=(0, 4, 9))
    p2 = Posting(doc_id=1, tf=3, positions=(0, 4, 9))
    m1 = DocMeta(doc_id=1, title="Python", path="doc1.txt")

    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert len({p1, p2}) == 1
    assert len({m1}) == 1
    assert not hasattr(p1, "__dict__")
    assert not hasattr(m1, "__dict__")

    with pytest.raises(AttributeError):
        p1.tf = 5  # type: ignore[misc] # intentionally mutating frozen dataclass to assert AttributeError


def test_build_index_streaming_sorted_with_and_without_positions(tmp_path: Path) -> None:
    (tmp_path / "01_alpha.txt").write_text("Python search engine search", encoding="utf-8")
    (tmp_path / "02_beta.txt").write_text("Rust engine fast", encoding="utf-8")
    (tmp_path / "03_gamma.txt").write_text("Python fast search", encoding="utf-8")

    ix = build_index(tmp_path, include_positions=False)
    assert not ix.has_positions
    assert ix.doc_lengths == {0: 4, 1: 3, 2: 3}
    assert ix.postings["search"] == [Posting(doc_id=0, tf=2), Posting(doc_id=2, tf=1)]

    ix_pos = build_index(tmp_path, include_positions=True)
    assert ix_pos.has_positions
    assert ix_pos.postings["search"] == [
        Posting(doc_id=0, tf=2, positions=(1, 3)),
        Posting(doc_id=2, tf=1, positions=(2,)),
    ]


def test_two_pointer_merge_operations() -> None:
    a = [1, 3, 5, 7, 10]
    b = [2, 3, 6, 7, 12]

    assert merge_and(a, b) == [3, 7]
    assert merge_or(a, b) == [1, 2, 3, 5, 6, 7, 10, 12]
    assert merge_and_not(a, b) == [1, 5, 10]
    assert merge_and([], a) == []
    assert merge_or([], a) == a


def test_boolean_search_engines_agree(tmp_path: Path) -> None:
    (tmp_path / "01.txt").write_text("python web fastapi", encoding="utf-8")
    (tmp_path / "02.txt").write_text("python search bm25", encoding="utf-8")
    (tmp_path / "03.txt").write_text("java search lucene", encoding="utf-8")
    (tmp_path / "04.txt").write_text("rust systems fast", encoding="utf-8")

    ix = build_index(tmp_path)

    queries_expected: list[tuple[str, list[int]]] = [
        ("python search", [1]),
        ("python AND search", [1]),
        ("fastapi OR lucene", [0, 2]),
        ("search NOT java", [1]),
        ("NOT python", [2, 3]),
        ("nonexistent", []),
    ]

    for query, expected in queries_expected:
        assert boolean_search(ix, query, engine="merge") == expected
        assert boolean_search(ix, query, engine="set") == expected


@pytest.mark.parametrize("fmt", ["pickle", "binary"])
@pytest.mark.parametrize("include_positions", [False, True])
def test_save_and_load_round_trip(
    tmp_path: Path,
    fmt: StorageFormat,
    include_positions: bool,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "01.txt").write_text("Café python python", encoding="utf-8")
    (corpus_dir / "02.txt").write_text("Київ search café", encoding="utf-8")

    original = build_index(corpus_dir, include_positions=include_positions)
    out_file = tmp_path / f"index.{fmt}"
    save(original, out_file, fmt=fmt)

    loaded = load(out_file)
    assert loaded.has_positions == original.has_positions
    assert loaded.doc_lengths == original.doc_lengths
    assert loaded.doc_meta == original.doc_meta
    assert loaded.postings == original.postings
