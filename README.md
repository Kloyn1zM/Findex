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