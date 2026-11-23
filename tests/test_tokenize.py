"""Tests for Lab 1: generators, Unicode tokenization, corpus streaming, and stats."""

import inspect
import json
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.stats import compute_stats_eager, compute_stats_lazy
from findex.tokenize import tokenize, tokenize_spans


def test_functions_are_generators() -> None:
    assert inspect.isgeneratorfunction(iter_documents)
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(tokenize_spans)


def test_empty_and_whitespace_input() -> None:
    assert list(tokenize("")) == []
    assert list(tokenize("   \t\n  ")) == []


def test_mixed_case_and_casefold() -> None:
    text = "Python PYTHON PyThOn Straße"
    assert list(tokenize(text)) == ["python", "python", "python", "strasse"]


def test_unicode_nfc_combining_marks() -> None:
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    assert list(tokenize(composed)) == list(tokenize(decomposed)) == ["café"]


def test_cyrillic_and_ukrainian_apostrophes() -> None:
    text = "П'ять комп’ютерів і девʼять зв'язків у Києві!"
    assert list(tokenize(text)) == [
        "п'ять",
        "комп’ютерів",
        "і",
        "девʼять",
        "зв'язків",
        "у",
        "києві",
    ]


def test_punctuation_and_hyphens_policy() -> None:
    text = "State-of-the-art, don't stop... 'quoted' — word-pair!"
    assert list(tokenize(text)) == [
        "state",
        "of",
        "the",
        "art",
        "don't",
        "stop",
        "quoted",
        "word",
        "pair",
    ]


def test_digits_and_alphanumeric_tokens() -> None:
    text = "Lab 01 in 2026: Python3.13 and UTF-8."
    assert list(tokenize(text)) == [
        "lab",
        "01",
        "in",
        "2026",
        "python3",
        "13",
        "and",
        "utf",
        "8",
    ]


def test_tokenize_spans_positions() -> None:
    spans = list(tokenize_spans("Hello, світ!"))
    assert len(spans) == 2
    assert spans[0].term == "hello"
    assert spans[0].position == 0
    assert spans[0].start_char == 0
    assert spans[1].term == "світ"
    assert spans[1].position == 1
    assert spans[1].start_char == 7


def test_iter_documents_txt_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "a_doc.txt").write_text("First plain text doc.", encoding="utf-8")
    jsonl_file = tmp_path / "b_docs.jsonl"
    jsonl_content = "\n".join(
        [
            json.dumps({"title": "JSON 1", "text": "Second doc from JSONL."}),
            "{broken json line",
            json.dumps({"title": "JSON 2", "text": "Third doc."}),
        ]
    )
    jsonl_file.write_text(jsonl_content, encoding="utf-8")

    stream = iter_documents(tmp_path)
    first_doc = next(stream)
    assert first_doc.doc_id == 0
    assert "First" in first_doc.text

    remaining = list(stream)
    assert len(remaining) == 2
    assert remaining[0].doc_id == 1
    assert remaining[1].doc_id == 2


def test_eager_and_lazy_stats_equivalence(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"file_{i}.txt").write_text(f"Token test {i} token.", encoding="utf-8")

    lazy = compute_stats_lazy(tmp_path, limit=3)
    eager = compute_stats_eager(tmp_path, limit=3)

    assert lazy.num_docs == eager.num_docs == 3
    assert lazy.total_tokens == eager.total_tokens == 12
    assert lazy.vocab_size == eager.vocab_size
    assert lazy.top_terms == eager.top_terms


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("", []),
        ("Hello, WORLD!", ["hello", "world"]),
        ("caf\u00e9 cafe\u0301", ["café", "café"]),
        ("Straße STRASSE", ["strasse", "strasse"]),
        ("don't can't п'ять м’ясо", ["don't", "can't", "п'ять", "м’ясо"]),
        ("state-of-the-art", ["state", "of", "the", "art"]),
        ("Python3.13 — 2026!", ["python3", "13", "2026"]),
    ],
)
def test_tokenize_parametrized_cases(raw_text: str, expected: list[str]) -> None:
    assert list(tokenize(raw_text)) == expected
