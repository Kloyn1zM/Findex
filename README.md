# findex — Full-Text & Semantic Search Engine

A full-text search engine built incrementally in Python (3.13+, `uv`, `ruff`, `pyright`, `pytest`) from a constant-memory streaming corpus pipeline to a parallel inverted index, ranked query engine, installable CLI, async web crawler, and FastAPI web service.

## Corpus

The corpus consists of **5,000 articles** from **Simple English Wikipedia** (`wikimedia/wikipedia`, snapshot `20231101.simple` on Hugging Face), filtered to articles with at least 600 characters and stored as 40 sharded JSON-lines files (`data/wiki_00.jsonl` – `data/wiki_39.jsonl`, **23.47 MiB** total).

`data/` is excluded via `.gitignore`. To reproduce the corpus locally:

~~~bash
uv run --with datasets python -c "
import json
from pathlib import Path
from datasets import load_dataset
out = Path('data'); out.mkdir(exist_ok=True)
ds = load_dataset('wikimedia/wikipedia', '20231101.simple', split='train', streaming=True)
count, shard = 0, 0
fh = (out / f'wiki_{shard:02d}.jsonl').open('w', encoding='utf-8')
try:
    for row in ds:
        text = row['text'].strip()
        if len(text) < 600: continue
        fh.write(json.dumps({'id': row['id'], 'url': row['url'], 'title': row['title'], 'text': text}, ensure_ascii=False) + '\n')
        count += 1
        if count >= 5000: break
        if count % 125 == 0:
            fh.close(); shard += 1
            fh = (out / f'wiki_{shard:02d}.jsonl').open('w', encoding='utf-8')
finally:
    fh.close()
"
~~~

---

## Lab 01 — Streams, Not Lists: Corpus Intake & Tokenization

### What was built
- **`src/findex/corpus.py`**: lazy generator `iter_documents(root) -> Iterator[Document]` supporting `.txt`, `.md`, `.jsonl`, and `.gz` streams with explicit `encoding="utf-8", errors="replace"` and resilient logging for malformed lines. Calling `next(iter_documents(Path("data")))` returns the first document (`id=0, title='April'`) in **9.89 ms** without reading the rest of the corpus.
- **`src/findex/tokenize.py`**: streaming `tokenize(text) -> Iterator[str]` (plus `tokenize_spans` retaining token/character offsets for later phrase queries and snippets) built on `unicodedata.normalize("NFC", ...)`, `str.casefold()`, and `re.finditer`.
- **`src/findex/stats.py`**: single-pass streaming pipeline computing document count, total tokens, vocabulary size, top-50 terms, elapsed time, and peak traced memory (`tracemalloc`), with `--limit N` via `itertools.islice` and `--compare-eager`.
- **`tests/test_tokenize.py`**: 10 pytest checks covering generator verification, empty input, mixed case + `casefold` (`Straße` → `strasse`), Unicode NFC combining marks (`café` vs `cafe\u0301`), Cyrillic + apostrophes, punctuation/hyphens, digits, spans, JSONL/TXT streaming, and eager/lazy equivalence.

### Tokenization policy
- **Unicode & Case**: NFC normalization followed by `str.casefold()` so visually identical composed/decomposed characters and locale-specific case pairs map to the same token.
- **Apostrophes**: internal apostrophes (`'`, `’`, `ʼ`) are kept inside words (`don't`, `п'ять`, `комп’ютер`).
- **Hyphens**: treated as separators (`state-of-the-art` → `state`, `of`, `the`, `art`) to preserve single-term recall; multi-word matching is deferred to phrase queries in Lab 3.
- **Digits**: retained as standalone or alphanumeric tokens (`2026`, `python3`).

### Corpus statistics (full corpus)
- **Documents**: 5,000
- **Total tokens**: 3,780,617
- **Vocabulary size**: 129,327 unique terms
- **Top-5 terms**: `the` (201,674), `of` (112,347), `and` (91,843), `in` (77,744), `a` (63,743)

### Eager vs. Lazy benchmark (`tracemalloc`)

| Version | Documents | Peak memory | Elapsed |
|---|---|---|---|
| eager (lists) | 5,000 | 266.02 MiB | 16.813 s |
| lazy (generators) | 5,000 | 10.27 MiB | 16.486 s |

The lazy pipeline cuts peak Python heap usage by **~25.9×** (from 266.02 MiB down to 10.27 MiB) while running slightly faster. In the eager version, all 5,000 `Document` texts are loaded into RAM at once, followed by a list-of-lists and a flat list holding 3,780,617 Python `str` objects (where each string object and 8-byte list pointer adds ~50–60 bytes of overhead on top of the 23.47 MiB raw text). In the lazy version, upstream stages hold only the single currently processed document and one regex match at a time; peak memory is bounded by the legitimate sink — the `Counter` hash table storing 129,327 unique terms and their counts — which is why it stays at 10.27 MiB rather than zero.

### Running Lab 01

~~~bash
uv run pytest
uv run python -m findex.stats data/ --limit 100
uv run python -m findex.stats data/ --compare-eager
~~~

---

## Lab 02 — The Inverted Index: Dictionaries, Hashing, and Memory

### What was built
- **`src/findex/index.py`**: immutable, hashable `@dataclass(frozen=True, slots=True)` records (`Posting`, `DocMeta`) and `build_index()` that consumes the Lab 1 lazy stream in a single pass using `defaultdict(list)` and per-document `Counter` flushing. Because `doc_id`s arrive in strictly increasing order, every postings list is naturally sorted by `doc_id`. Supports optional `--positions` for Lab 3 phrase search.
- **`src/findex/search.py`**: Boolean query evaluation (`AND`, `OR`, `NOT`, implicit `AND`) with two interchangeable backends: `--engine merge` (two-pointer linear merge on sorted postings) and `--engine set` (Python `set` operations).
- **`src/findex/store.py`**: persistence in two formats — `pickle` (for trusted local files only) and a safe, compact `binary` format (JSON metadata header + contiguous little-endian `array('I')` buffers).
- **`tests/test_index.py`**: 8 pytest tests verifying dataclass hashing/immutability/slots, sorted postings with and without positions, merge correctness, merge-vs-set equivalence, and save/load round trips for both formats.

### M2: Two-pointer merge vs. `set` benchmark

| Pair | Terms (df) | Op | Merge (µs) | Set (µs) | Winner |
|---|---|---|---|---|---|
| Most common | `the` (4,981), `of` (4,931) | AND | 1248.07 | 1193.59 | set |
| Most common | `the` (4,981), `of` (4,931) | OR | 1274.58 | 1080.87 | set |
| Rare | `gallium` (2), `maigmo` (2) | AND | 0.80 | 0.77 | set |
| Rare | `gallium` (2), `maigmo` (2) | OR | 1.05 | 0.89 | set |

Although two-pointer merge has $O(n + m)$ asymptotic complexity without hashing or sorting, Python's built-in `set` intersection/union and `sorted()` (Timsort) execute entirely in compiled C, outperforming a pure-Python `while` loop on lists up to ~5,000 IDs. Two-pointer merge remains essential for streaming/compressed postings and extends directly to positional phrase queries in Lab 3.

### M3: Persistence formats & `--positions` cost

| Format | Positions | File size (MiB) | Save (s) | Load (s) |
|---|---|---|---|---|
| pickle | no | 27.60 | 10.076 | 12.346 |
| binary (`array('I')`) | no | 14.68 | 0.741 | 4.606 |
| pickle | yes | 39.11 | 13.637 | 11.228 |
| binary (`array('I')`) | yes | 29.10 | 1.684 | 6.322 |

- The custom `binary` format is **1.88× smaller** on disk (without positions), **13.6× faster to save**, and **2.7× faster to load** than `pickle`.
- Storing token positions roughly doubles the binary index on disk (14.68 MiB → 29.10 MiB) and increases build peak RAM to ~335 MiB due to per-posting position tuples, unlocking phrase queries in Lab 3.
- **Security note on `pickle`**: `pickle.load()` is a remote-code-execution (RCE) vulnerability on untrusted input because pickled payloads can invoke arbitrary callables via `__reduce__`. In `findex`, `pickle` is restricted to locally generated files only; the `binary` format parses only JSON metadata and raw `uint32` arrays, eliminating code-execution risk.

### M4: Memory study (`tracemalloc`, full 5,000-document corpus, no positions)

| Postings representation | Peak memory (build) | Index file size | Load time |
|---|---|---|---|
| `list[Posting]` with plain `@dataclass` | 173.34 MiB | 31.98 MiB | 1.998 s |
| `list[Posting]` with `slots=True` | 122.24 MiB | 27.19 MiB | 8.601 s |
| `array('I')` pairs `(doc_ids, tfs)` — no objects | 48.96 MiB | 18.64 MiB | 0.382 s |

Across the 5,000-document corpus, the index contains over a million `(doc_id, tf)` postings. With a plain `@dataclass`, every `Posting` allocates a per-instance `__dict__` hash table (~100+ bytes of overhead on top of the object header and two integer references), pushing peak build RAM to **173.34 MiB**. Adding `slots=True` eliminates per-instance dictionaries in favor of fixed C-level struct slots, dropping peak RAM by **51.1 MiB (~1.42× smaller)** to **122.24 MiB**. Replacing `Posting` objects with parallel `array('I')` buffers removes both Python object headers and boxed 28-byte `PyLongObject` integers, packing each `(doc_id, tf)` pair into just 8 contiguous bytes — cutting peak RAM to **48.96 MiB (3.54× smaller than plain dataclasses)** and loading in **0.382 s** (avoiding reconstructing over a million Python objects).

### Running Lab 02

~~~bash
uv run python -m findex.index data/ --out data/index.bin --positions
uv run python -m findex.search data/index.bin "python programming"
uv run python -m findex.search data/index.bin "ukraine NOT russia" --engine set --limit 5
~~~