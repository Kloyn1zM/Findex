"""Property-based tests using Hypothesis (Lab 4, M2)."""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index_from_docs
from findex.query import merge_and, merge_and_not, merge_or
from findex.store import load, save
from findex.tokenize import tokenize


@given(text=st.text(max_size=250))
def test_tokenize_no_empty_tokens_and_idempotent(text: str) -> None:
    """Property: tokenize never yields empty strings and is idempotent under re-tokenization."""
    tokens = list(tokenize(text))
    assert all(tok != "" for tok in tokens)
    assert list(tokenize(" ".join(tokens))) == tokens


@given(
    raw_a=st.lists(st.integers(min_value=0, max_value=500), max_size=80),
    raw_b=st.lists(st.integers(min_value=0, max_value=500), max_size=80),
)
def test_merge_operations_match_set_oracle(raw_a: list[int], raw_b: list[int]) -> None:
    """Oracle property: two-pointer merge_and/or/and_not always agree with Python sets."""
    a = sorted(set(raw_a))
    b = sorted(set(raw_b))

    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_and_not(a, b) == sorted(set(a) - set(b))


@given(
    texts=st.lists(st.text(max_size=120), min_size=1, max_size=15),
)
def test_postings_always_strictly_sorted_invariant(texts: list[str]) -> None:
    """Invariant: every term's postings list has strictly increasing doc_ids and positive tf."""
    docs = [
        Document(doc_id=i, path=Path(f"doc_{i}.txt"), text=text, title=f"Doc {i}")
        for i, text in enumerate(texts)
    ]
    ix = build_index_from_docs(docs, include_positions=True)

    for plist in ix.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(set(doc_ids))
        for p in plist:
            assert p.tf == len(p.positions)
            assert p.tf > 0


@settings(max_examples=30, deadline=None)
@given(
    texts=st.lists(st.text(max_size=120), min_size=1, max_size=8),
    include_positions=st.booleans(),
)
def test_binary_save_load_round_trip_property(
    texts: list[str],
    include_positions: bool,
) -> None:
    """Round-trip property: load(save(index)) reproduces the exact same Index."""
    docs = [
        Document(doc_id=i, path=Path(f"doc_{i}.txt"), text=text, title=f"Doc {i}")
        for i, text in enumerate(texts)
    ]
    original = build_index_from_docs(docs, include_positions=include_positions)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "prop_index.bin"
        save(original, out_path, fmt="binary")
        reloaded = load(out_path, fmt="binary")

    assert reloaded.has_positions == original.has_positions
    assert reloaded.doc_lengths == original.doc_lengths
    assert reloaded.doc_meta == original.doc_meta
    assert reloaded.doc_texts == original.doc_texts
    assert dict(reloaded) == dict(original)
