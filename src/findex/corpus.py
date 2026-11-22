"""Lazy streaming corpus reader for plain-text files and JSON-lines."""

import gzip
import json
import logging
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    """Single document yielded by the corpus stream."""

    doc_id: int
    path: Path
    text: str
    title: str = ""


@contextmanager
def _open_text_stream(path: Path) -> Generator[TextIO]:
    """Open a plain-text or gzip-compressed file in UTF-8 mode."""
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with path.open(mode="r", encoding="utf-8", errors="replace") as fh:
            yield fh


def _iter_jsonl_file(path: Path, start_id: int = 0) -> Iterator[Document]:
    """Lazily yield documents line-by-line from a .jsonl or .jsonl.gz file."""
    doc_id = start_id
    try:
        with _open_text_stream(path) as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw_obj: object = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("Skipped malformed JSON in %s:%d (%s)", path, line_no, exc)
                    continue

                if not isinstance(raw_obj, dict):
                    log.warning("Expected JSON object in %s:%d", path, line_no)
                    continue

                record = cast("dict[str, object]", raw_obj)
                text_val = record.get("text") or record.get("abstract") or ""
                title_val = record.get("title") or record.get("id") or f"{path.name}:{line_no}"
                url_val = record.get("url") or f"{path}:{line_no}"

                yield Document(
                    doc_id=doc_id,
                    path=Path(str(url_val)),
                    text=str(text_val),
                    title=str(title_val),
                )
                doc_id += 1
    except OSError as exc:
        log.warning("Failed to read file %s: %s", path, exc)


def collect_corpus_files(root: Path) -> list[Path]:
    """Return a deterministically sorted list of supported corpus file paths."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    if root.is_file():
        return [root]

    files: list[Path] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        suffixes = file_path.suffixes
        if ".jsonl" in suffixes or (suffixes and suffixes[0] in {".txt", ".md"}):
            files.append(file_path)
    return files


def count_documents_in_file(path: Path) -> int:
    """Fast estimate of document count in a corpus file for upfront doc_id range planning."""
    if ".jsonl" not in path.suffixes:
        return 1
    try:
        with _open_text_stream(path) as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def iter_documents_from_paths(
    paths: Iterable[Path],
    *,
    start_id: int = 0,
) -> Iterator[Document]:
    """Lazily yield Document objects from an explicit sequence of file paths."""
    doc_id = start_id
    for file_path in paths:
        suffixes = file_path.suffixes
        if ".jsonl" in suffixes:
            for doc in _iter_jsonl_file(file_path, start_id=doc_id):
                yield doc
                doc_id = doc.doc_id + 1
        else:
            try:
                with _open_text_stream(file_path) as fh:
                    text = fh.read()
                yield Document(doc_id=doc_id, path=file_path, text=text, title=file_path.stem)
                doc_id += 1
            except OSError as exc:
                log.warning("Failed to read file %s: %s", file_path, exc)


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily stream Document objects from a file or directory tree.

    Supports .txt, .md, and .jsonl files (including .gz-compressed variants).
    Holds only one document in memory at a time.
    """
    files = collect_corpus_files(root)
    yield from iter_documents_from_paths(files, start_id=0)
