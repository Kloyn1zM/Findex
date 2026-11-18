"""Unicode-aware streaming tokenizer.

Tokenization policy:
1. Unicode normalization: NFC so composed ('é', U+00E9) and decomposed
   ('e' + U+0301) sequences produce identical tokens.
2. Case folding: str.casefold() for aggressive locale-independent matching
   (e.g., German 'ß' -> 'ss').
3. Apostrophes: internal apostrophes (ASCII ', typographic ’, modifier ʼ)
   are preserved inside words (e.g., "don't", "п'ять", "комп’ютер").
4. Hyphens: treated as token separators ("state-of-the-art" -> four tokens)
   to maximize recall; exact multi-word matches are handled by phrase search.
5. Digits: kept as standalone or alphanumeric tokens ("2026", "python3").
"""

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\w+(?:['’ʼ]\w+)*", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Normalized token with token index and character offsets."""

    term: str
    position: int
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization and case folding."""
    return unicodedata.normalize("NFC", text).casefold()


def tokenize_spans(text: str) -> Iterator[TokenSpan]:
    """Lazily yield TokenSpan objects using re.finditer."""
    if not text:
        return
    normalized = normalize_text(text)
    for pos, match in enumerate(TOKEN_PATTERN.finditer(normalized)):
        yield TokenSpan(
            term=match.group(0),
            position=pos,
            start_char=match.start(),
            end_char=match.end(),
        )


def tokenize(text: str) -> Iterator[str]:
    """Lazily yield normalized tokens from text in constant memory."""
    if not text:
        return
    normalized = normalize_text(text)
    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group(0)
