"""Recursive-descent query parser and operator-overloaded query AST (Lab 3, M3)."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from findex.index import Index, Posting
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    """Raised when a search query string cannot be parsed."""


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Two-pointer intersection of two sorted doc_id sequences in O(len(a) + len(b))."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        da = a[i]
        db = b[j]
        if da == db:
            result.append(da)
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Two-pointer union of two sorted doc_id sequences in O(len(a) + len(b))."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        da = a[i]
        db = b[j]
        if da == db:
            result.append(da)
            i += 1
            j += 1
        elif da < db:
            result.append(da)
            i += 1
        else:
            result.append(db)
            j += 1
    if i < len_a:
        result.extend(a[i:])
    if j < len_b:
        result.extend(b[j:])
    return result


def merge_and_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Two-pointer difference (sorted elements in a that are absent from sorted b)."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        da = a[i]
        db = b[j]
        if da == db:
            i += 1
            j += 1
        elif da < db:
            result.append(da)
            i += 1
        else:
            j += 1
    if i < len_a:
        result.extend(a[i:])
    return result


def _find_posting(postings: list[Posting], doc_id: int) -> Posting | None:
    """Binary search a doc_id inside a sorted postings list."""
    idx = bisect_left(postings, doc_id, key=lambda p: p.doc_id)
    if idx < len(postings) and postings[idx].doc_id == doc_id:
        return postings[idx]
    return None


def _positions_adjacent(prev_positions: Sequence[int], next_positions: Sequence[int]) -> list[int]:
    """Two-pointer merge finding p2 in next_positions where (p2 - 1) is in prev_positions."""
    matched: list[int] = []
    i = 0
    j = 0
    len_a = len(prev_positions)
    len_b = len(next_positions)
    while i < len_a and j < len_b:
        expected = prev_positions[i] + 1
        actual = next_positions[j]
        if actual == expected:
            matched.append(actual)
            i += 1
            j += 1
        elif expected < actual:
            i += 1
        else:
            j += 1
    return matched


class QueryNode:
    """Base node supporting Python bitwise operators (&, |, ~) for query composition."""

    __slots__ = ()

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> list[int]:
        raise NotImplementedError

    def positive_terms(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> list[int]:
        if self.value not in index:
            return []
        return [p.doc_id for p in index[self.value]]

    def positive_terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> list[int]:
        if not self.terms:
            return []
        if not index.has_positions:
            raise ValueError("Phrase queries require an index built with --positions")

        for term in self.terms:
            if term not in index:
                return []

        candidates = [p.doc_id for p in index[self.terms[0]]]
        for term in self.terms[1:]:
            candidates = merge_and(candidates, [p.doc_id for p in index[term]])
            if not candidates:
                return []

        if len(self.terms) == 1:
            return candidates

        postings_lists = [index[term] for term in self.terms]
        matched_docs: list[int] = []

        for doc_id in candidates:
            first_posting = _find_posting(postings_lists[0], doc_id)
            if first_posting is None:
                continue
            current_positions: Sequence[int] = first_posting.positions
            for plist in postings_lists[1:]:
                nxt = _find_posting(plist, doc_id)
                if nxt is None:
                    current_positions = ()
                    break
                current_positions = _positions_adjacent(current_positions, nxt.positions)
                if not current_positions:
                    break
            if current_positions:
                matched_docs.append(doc_id)

        return matched_docs

    def positive_terms(self) -> tuple[str, ...]:
        return self.terms


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> list[int]:
        if isinstance(self.right, Not):
            return merge_and_not(self.left.evaluate(index), self.right.operand.evaluate(index))
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> list[int]:
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> list[int]:
        all_docs = sorted(index.doc_meta)
        return merge_and_not(all_docs, self.operand.evaluate(index))

    def positive_terms(self) -> tuple[str, ...]:
        return ()


TokenKind = Literal["WORD", "PHRASE", "AND", "OR", "NOT", "LPAREN", "RPAREN"]


@dataclass(frozen=True, slots=True)
class _LexToken:
    kind: TokenKind
    text: str


def _lex_query(query: str) -> list[_LexToken]:
    tokens: list[_LexToken] = []
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(_LexToken("LPAREN", "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(_LexToken("RPAREN", ")"))
            i += 1
            continue
        if ch == '"':
            end = query.find('"', i + 1)
            if end == -1:
                raise QuerySyntaxError("Unclosed quote in query")
            tokens.append(_LexToken("PHRASE", query[i + 1 : end]))
            i = end + 1
            continue

        start = i
        while i < n and not query[i].isspace() and query[i] not in '()"':
            i += 1
        word = query[start:i]
        if word == "AND":
            tokens.append(_LexToken("AND", word))
        elif word == "OR":
            tokens.append(_LexToken("OR", word))
        elif word == "NOT":
            tokens.append(_LexToken("NOT", word))
        else:
            tokens.append(_LexToken("WORD", word))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_LexToken]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> _LexToken | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> _LexToken:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self._tokens:
            raise QuerySyntaxError("Empty query")
        node = self._parse_or()
        if self._peek() is not None:
            raise QuerySyntaxError(f"Unexpected token: {self._peek()!r}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self._peek()) is not None and tok.kind == "OR":
            self._consume()
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self._peek()) is not None and tok.kind not in {"OR", "RPAREN"}:
            if tok.kind == "AND":
                self._consume()
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> QueryNode:
        tok = self._peek()
        if tok is not None and tok.kind == "NOT":
            self._consume()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        tok = self._peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")

        if tok.kind == "LPAREN":
            self._consume()
            node = self._parse_or()
            closing = self._peek()
            if closing is None or closing.kind != "RPAREN":
                raise QuerySyntaxError("Missing closing ')' in query")
            self._consume()
            return node

        if tok.kind == "PHRASE":
            self._consume()
            terms = tuple(tokenize(tok.text))
            if not terms:
                raise QuerySyntaxError("Empty phrase in quotes")
            return Phrase(terms)

        if tok.kind == "WORD":
            self._consume()
            terms = list(tokenize(tok.text))
            if not terms:
                raise QuerySyntaxError(f"Invalid empty term: {tok.text!r}")
            node: QueryNode = Term(terms[0])
            for extra in terms[1:]:
                node = And(node, Term(extra))
            return node

        raise QuerySyntaxError(f"Unexpected operator {tok.text!r}")


def parse_query(query: str) -> QueryNode:
    """Parse a query string into an AST with precedence: NOT > AND (explicit/implicit) > OR."""
    return _Parser(_lex_query(query)).parse()


parse = parse_query
