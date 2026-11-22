# findex — Full-Text & Semantic Search Engine

[![CI](https://github.com/Kloyn1zM/Findex/actions/workflows/ci.yml/badge.svg)](https://github.com/Kloyn1zM/Findex/actions/workflows/ci.yml)

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

---

## Lab 03 — Ranking and the Object Model: Dunders, Protocols, Query Language

### What was built
- **`src/findex/index.py`**: refactored the inverted index into `Index(collections.abc.Mapping[str, list[Posting]])` implementing `__len__`, `__contains__`, `__getitem__`, `__iter__`, `__repr__`, `num_docs`, `df()`, `doc_length()`, and `functools.cached_property` for `avg_doc_length` (756.1 tokens on the full corpus).
- **`src/findex/store.py`**: added `@contextmanager def open_index(path)` guaranteeing `index.close()` on normal exit and exceptions, and upgraded the binary format (`FIDX0002`) to store raw document texts alongside postings for snippet generation.
- **`src/findex/scoring.py`**: defined a structural `Scorer` `Protocol` with two swappable implementations (`TfIdf` and `BM25(k1=1.5, b=0.75)`), `@total_ordering` `SearchResult`, and `make_snippet()` extracting a $\pm 80$-character highlighted window.
- **`src/findex/query.py`**: recursive-descent query parser supporting implicit/explicit `AND`, `OR`, `NOT`, parentheses, and `"quoted phrases"` (via two-pointer adjacency merge over positional postings), producing operator-overloaded AST nodes (`Term`, `Phrase`, `And`, `Or`, `Not` with `&`, `|`, `~`).
- **`src/findex/timing.py` & `src/findex/search.py`**: `@timed` decorator (with `functools.wraps`) on `build_index`, `load`, and `search`; `@functools.lru_cache(maxsize=256)` on the ranked ID lookup path; `heapq.nlargest` for $O(n \log k)$ top-$k$ selection.
- **`tests/test_ranking_and_query.py`**: 9 pytest tests covering the `Mapping` protocol, `open_index` exception safety, parser AST tree equality, phrase queries, all three ranking sanity checks, `@timed`/`lru_cache`, and snippet highlighting.

### M2: Ranking sanity checks (5,000-document corpus, `BM25(k1=1.5, b=0.75)`)

1. **Rare term outranks common term** (`the OR gallium`, where `df("gallium")=2` vs `df("the")=4,981`):
   - `#1 [4996] Gallium` (`score=17.6699`, `has_gallium=True`)
   - `#2 [4994] Arsenic` (`score=6.4172`, `has_gallium=True`)
   - `#3 [4145] Solstice` (`score=0.0096`, `has_gallium=False`)
2. **BM25 term-frequency saturation** (`term="computer"`, `doc_id=6`, `len=855`, `df=437`):
   - `tf=1` → `2.3009` | `tf=5` → `4.5815` | `tf=19` → `5.6049` | `tf=20` → `5.6273`
   - The 20th occurrence adds only **+0.02245** (**0.98%** of the first occurrence).
3. **Document-length normalization** (`term="astronomy"`, `tf=1`):
   - Short doc `[974] List of satellites of the Milky Way` (`len=121`) → `BM25=6.9699`
   - Long doc `[463] Albert Einstein` (`len=4,537`) → `BM25=1.3339`

### M4: `@timed` and `lru_cache` log evidence

~~~text
DEBUG findex.timing: search took 4.39 ms
  After 1st call: SearchCacheInfo(hits=0, misses=1, maxsize=256, currsize=1)
DEBUG findex.search: search cache HIT for query='solar system planets' (CacheInfo(hits=1, misses=1, maxsize=256, currsize=1))
DEBUG findex.timing: search took 4.23 ms
  After 2nd call: SearchCacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
~~~

Notice that the cache hit still reports ~4.23 ms end-to-end because `_cached_ranked_ids` caches the `(doc_id, score)` ranking step, while `search()` still generates highlighted snippets over the top-5 raw article texts on every call.

### M4: Precision@5 evaluation (10 hand-labeled queries, TF-IDF vs. BM25)

| # | Query | TF-IDF P@5 | BM25 P@5 | Notes on top-5 differences |
|---|---|---|---|---|
| 1 | `python programming` | 0.60 (3/5) | 0.60 (3/5) | Identical top-5 (`Programming language`, `C`, `Variable` relevant) |
| 2 | `solar system planets` | 1.00 (5/5) | 0.80 (4/5) | TF-IDF returns 5 core articles; BM25 pulls `Natural satellite` into #4 |
| 3 | `world war europe` | 0.20 (1/5) | 0.60 (3/5) | BM25 surfaces `World War`, `Europe`, `Kingdom of Yugoslavia`; TF-IDF favors long country pages |
| 4 | `dna genetics cell` | 0.80 (4/5) | 0.80 (4/5) | Same 5 documents reordered (`Genetics`, `Gene`, `Protein`, `Evolution` relevant) |
| 5 | `roman empire caesar` | 0.60 (3/5) | 0.40 (2/5) | BM25 length norm over-boosts tiny year stubs (`2`, `5`, `7`) over `Rome` |
| 6 | `linux operating system` | 1.00 (5/5) | 0.80 (4/5) | TF-IDF returns 5/5 Linux/OS pages; BM25 puts broad `Personal computer` at #5 |
| 7 | `albert einstein relativity` | 0.80 (4/5) | 0.80 (4/5) | Both rank `Einstein`, `Special/General relativity`, `Gravity` in top-5 |
| 8 | `periodic table chemical elements` | 0.60 (3/5) | 0.80 (4/5) | BM25 elevates element groups (`Halogen`, `Transition metal`) over `Atom`/`Hydrogen` |
| 9 | `ancient egypt pharaoh` | 0.60 (3/5) | 0.60 (3/5) | Same 5 documents (`Pharaoh`, `Ancient Egypt`, `Pyramid` relevant; `Moses`/`Aaron` noisy) |
| 10 | `jazz music instruments` | 0.40 (2/5) | 0.80 (4/5) | BM25 ranks `Jazz` #1 and surfaces `Saxophone` & `Tuba`; TF-IDF ranks `Music`/`Classical`/`Violin` |
| **Mean** | **10 queries** | **0.66** | **0.70** | **BM25 +0.04 macro P@5** |

BM25 outperforms log-TF-IDF overall (**0.70 vs. 0.66**) by preventing very long encyclopedic articles (like generic country pages in Q3 or `Music`/`Orchestra` in Q10) from dominating solely through raw term repetition. At the same time, Q5 (`roman empire caesar`) exposes a real-world BM25 failure mode: with $b=0.75$, very short Wikipedia year-stub articles (`2`, `5`, `7`) receive a strong length bonus and outrank the much longer `Rome` article.

### Running Lab 03

~~~bash
uv run python -m findex.index data/ --out data/index.bin --positions
uv run python -m findex.search data/index.bin "\"programming language\" OR python" -k 5 -v
uv run python -m findex.search data/index.bin "jazz music instruments" --scorer tfidf -k 5
~~~

---

## Lab 04 — Make It a Real Tool: Typing, Testing, Packaging, and CLI

### What was built
- **Strict static typing (`pyright` strict, 0 errors)**: annotated all public APIs across `src/findex/` using Python 3.12+ generics (`Iterator`, `Iterable`, `X | None`, PEP 695 `type DocId = int`), structural `Scorer(Protocol)`, and `Literal`/`StrEnum` CLI parameters. Every `# type: ignore` is justified inline.
- **Test suite (`pytest` + `Hypothesis`)**: expanded to **40 tests** including a session-scoped fixture corpus in `tests/conftest.py`, parametrized Unicode tokenizer cases, **4 Hypothesis property-based tests** (`tests/test_properties.py` checking tokenizer idempotence/non-emptiness, merge-vs-set oracle equivalence, sorted postings invariant, and binary save/load round-trip), `@pytest.mark.slow`, and CLI integration tests via `typer.testing.CliRunner`.
- **Installable CLI (`src/findex/cli.py`)**: packaged with PEP 621 `pyproject.toml`, `uv.lock`, and `[project.scripts] findex = "findex.cli:app"`. Provides `findex index`, `findex search`, and `findex stats` with `rich` tables/progress bars, `--json` JSON-lines output to `stdout`, `-v`/`-vv` diagnostic logging to `stderr`, and clean exit code `1` (no tracebacks) on expected errors.
- **CI & Release**: added `.github/workflows/ci.yml` running `uv sync --locked`, `ruff check`, `ruff format --check`, `pyright`, and `pytest --cov=findex` on every push/PR; published wheel `findex-0.4.0-py3-none-any.whl` to GitHub Release `v0.4.0`.

### Test coverage (`pytest --cov=findex`)

| Module | Stmts | Miss | Coverage |
|---|---|---|---|
| `src/findex/__init__.py` | 1 | 0 | 100% |
| `src/findex/timing.py` | 14 | 0 | 100% |
| `src/findex/tokenize.py` | 25 | 1 | 96% |
| `src/findex/cli.py` | 154 | 8 | 95% |
| `src/findex/query.py` | 286 | 19 | 93% |
| `src/findex/store.py` | 133 | 10 | 92% |
| `src/findex/scoring.py` | 81 | 11 | 86% |
| `src/findex/corpus.py` | 75 | 18 | 76% |
| `src/findex/index.py` | 145 | 36 | 75% |
| `src/findex/search.py` | 189 | 59 | 69% |
| `src/findex/stats.py` | 90 | 35 | 61% |
| **TOTAL** | **1,193** | **197** | **83%** |

**What is not covered and why:** core library and CLI modules sit at **86–100%** coverage (`cli.py` 95%, `tokenize.py` 96%, `query.py` 93%, `store.py` 92%). Most uncovered lines are the legacy `python -m` `argparse` entry points retained from Labs 1–3 (`index.py`, `search.py`, `stats.py`), big-endian `byteswap()` guards on x86-64 hardware (`store.py`), and defensive `OSError` handlers (`corpus.py`).

### Installing and running `findex` (`v0.4.0`)

~~~bash
# Install directly from the v0.4.0 GitHub Release wheel
uv tool install https://github.com/Kloyn1zM/Findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl

# Or build and install locally
uv build
uv tool install --force .

# CLI usage
findex index data/ --out data/index.bin --positions
findex stats data/index.bin --top 10
findex search data/index.bin "solar system planets" --k 5
findex search data/index.bin "solar system planets" --k 5 --json
findex -vv search data/index.bin "solar system planets" --k 3
~~~