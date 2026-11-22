"""Shared pytest fixtures for findex test suite (Lab 4, M2)."""

from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.store import save


@pytest.fixture(scope="session")
def mini_corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small, deterministic multi-document corpus once per test session."""
    corpus = tmp_path_factory.mktemp("mini_corpus")
    docs = {
        "01_python.txt": (
            "Python programming language emphasizes readability. "
            "Python event loop powers async web services."
        ),
        "02_search.txt": (
            "Full text search engine builds an inverted index and ranks documents with BM25."
        ),
        "03_unicode.txt": ("Café in Київ serves п'ять cups of coffee along Straße avenues."),
        "04_rust.txt": (
            "Rust systems programming language provides memory safety without garbage collection."
        ),
        "05_long_python.txt": ("Python " + "filler " * 40 + "documentation and tooling overview."),
    }
    for filename, content in docs.items():
        (corpus / filename).write_text(content, encoding="utf-8")
    return corpus


@pytest.fixture(scope="session")
def mini_index(mini_corpus_dir: Path) -> Index:
    """Session-scoped positional index built from mini_corpus_dir (treat as read-only)."""
    return build_index(mini_corpus_dir, include_positions=True)


@pytest.fixture(scope="session")
def saved_index_path(
    tmp_path_factory: pytest.TempPathFactory,
    mini_index: Index,
) -> Path:
    """Persist the session-scoped mini_index to a binary file for CLI/load tests."""
    out = tmp_path_factory.mktemp("saved_index") / "mini.bin"
    save(mini_index, out, fmt="binary")
    return out
