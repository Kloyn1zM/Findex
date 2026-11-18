"""findex — full-text search engine."""

from findex.corpus import Document, iter_documents
from findex.tokenize import TokenSpan, tokenize, tokenize_spans

__all__ = [
    "Document",
    "TokenSpan",
    "iter_documents",
    "tokenize",
    "tokenize_spans",
]
__version__ = "0.1.0"
