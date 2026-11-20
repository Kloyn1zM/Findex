"""Swappable ranking scorers (TF-IDF, BM25), SearchResult model, and snippet highlighter."""

import functools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from findex.index import Index, Posting
from findex.tokenize import normalize_text, tokenize_spans


class Scorer(Protocol):
    """Structural interface for term-document relevance scorers."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """Log-scaled TF-IDF scorer."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0 or index.num_docs == 0:
            return 0.0
        tf_weight = 1.0 + math.log(posting.tf)
        idf = math.log(1.0 + index.num_docs / df)
        return tf_weight * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25 scorer with term-frequency saturation (k1) and length normalization (b)."""

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0 or index.num_docs == 0:
            return 0.0
        avg_len = index.avg_doc_length
        if avg_len <= 0.0:
            return 0.0

        n = index.num_docs
        doc_len = index.doc_length(posting.doc_id)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
        return idf * (posting.tf * (self.k1 + 1.0)) / (posting.tf + norm)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@functools.total_ordering
@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single ranked hit ordered by relevance score."""

    doc_id: int
    score: float
    title: str
    snippet: str = ""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return (self.score, self.doc_id) < (other.score, other.doc_id)

    def __str__(self) -> str:
        return f"[{self.doc_id}] {self.title} (score={self.score:.4f})"


def make_snippet(
    text: str,
    terms: Sequence[str],
    *,
    window: int = 80,
    highlight_start: str = "**",
    highlight_end: str = "**",
) -> str:
    """Extract a +-window character excerpt around the first matched term and highlight hits."""
    if not text:
        return ""

    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return ""

    term_set = {normalize_text(t) for t in terms if t}
    anchor_start = 0

    if term_set:
        for span in tokenize_spans(clean_text):
            if span.term in term_set:
                anchor_start = span.start_char
                break

    start = max(0, anchor_start - window)
    end = min(len(clean_text), anchor_start + window)

    # Nudge boundaries to nearby spaces for readability
    if start > 0:
        space_pos = clean_text.find(" ", start)
        if space_pos != -1 and space_pos < anchor_start:
            start = space_pos + 1
    if end < len(clean_text):
        space_pos = clean_text.rfind(" ", anchor_start, end)
        if space_pos != -1 and space_pos > start:
            end = space_pos

    excerpt = clean_text[start:end]

    if term_set:
        pattern = re.compile(
            r"\b("
            + "|".join(re.escape(t) for t in sorted(term_set, key=len, reverse=True))
            + r")\b",
            flags=re.IGNORECASE | re.UNICODE,
        )
        excerpt = pattern.sub(rf"{highlight_start}\1{highlight_end}", excerpt)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    return f"{prefix}{excerpt}{suffix}"
