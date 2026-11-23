# Lab 08 — Baseline Profiling Report (Before Optimization)

Profiling was captured on the full 5,000-document / 129,327-term Simple English Wikipedia index (`data/index.bin`) **before** any Lab 8 code changes.

- **Flame graphs (`py-spy`, 250 Hz)**:
  - Search (250 uncached 3-term queries): `docs/lab08_search.svg` (2,139 samples, 0 errors)
  - Serial index build (5,000 docs): `docs/lab08_index.svg` (2,671 samples, 0 errors)
- **Deterministic call profiles (`cProfile`, sorted by `cumtime`)**:
  - Search: 50 distinct 3-term queries (no LRU cache hits) — `0.789 s` cumulative (`798,736` function calls)
  - Indexing: 5,000 docs (`executor="serial"`, `store_text=False`) — `16.879 s` cumulative (`12,018,753` function calls)

## Top-3 Hotspots & Hypotheses

| # | Target | Function / Path | `cProfile` cumtime | Dominant Layer | Hypothesis (written before patching) |
|---|---|---|---:|---|---|
| 1 | `search` | `_compute_ranked_ids` → `BM25.score()` | `0.431 s` (54.6% of search) | **Python** (56,382 `score()` calls, 56k `df()`, 112k `num_docs()`, 56k `math.log()`) | Packing postings into contiguous NumPy `int32` arrays (`doc_ids`, `tfs`, `doc_lengths`) and computing BM25/TF-IDF via vectorized ufuncs + `np.argpartition` will eliminate per-document Python dispatch/boxing and yield a **5×–30× speedup** on common and multi-term queries (with minimal or <1× gain on very rare terms due to C-call overhead). |
| 2 | `search` | `make_snippet()` (`re.Pattern.sub` + `tokenize_spans`) | `0.345 s` (43.7% of search) | **Mixed** (`0.157 s` native C regex, `0.142 s` Python span scan) | Runs only on the final `k` returned hits (constant wrt corpus match count). Native `re.sub` is already in C, so micro-optimizing Python loops around it is low-leverage compared to vectorizing scoring over tens of thousands of candidate postings. |
| 3 | `index` | `build_partial` (`tokenize` + `Counter`) & `merge()` | `11.828 s` + `4.910 s` (99.2% of serial build) | **Python** (3.78M generator yields + per-posting object creation in `merge`) | Already parallelized across CPU cores in Lab 5 (`1.48×`). Replacing per-posting Python objects with flat NumPy `int32` arrays (and optional `memmap` / compact storage) targets the remaining object-allocation bottleneck. |

## Outcomes After Optimization (M2 + M3)

| Target | Hypothesis (Before) | Measured Outcome (After) | Verdict |
|---|---|---|---|
| `BM25` / `TfIdf` scoring (`src/findex/vectorized.py`) | 5×–30× speedup on common/multi-term queries; minimal or <1× on tiny rare-term lists | **53.6×** on `the` (`df=4,981`: `13.405 → 0.250 ms`), **23.6×** on `world` (`df=1,938`: `5.218 → 0.221 ms`), **7.8×** on 3-term OR, **2.9×** on 3-term AND, and **0.85×** on rare `aachen` (`df=10`: `0.045 → 0.053 ms`) with **100% identical top-10 ranking** | **Confirmed** — Python per-posting loop eliminated; small postings confirm C-dispatch overhead floor. |
| `make_snippet()` | Native `re.sub` dominates; constant cost on final `k` hits, low leverage | Left untouched; after scoring speedup, snippet formatting is the main remaining per-query cost on top-`k` results | **Confirmed** — correctly avoided premature micro-optimization. |
| Semantic + Hybrid retrieval (`src/findex/semantic.py`) | Brute-force `E @ q` on ~20k chunks stays in single-digit ms and rescues paraphrase queries where BM25 fails | `19,789` chunks (`384-d`, `28.99 MiB`) embedded with `all-MiniLM-L6-v2`; on 5 paraphrase queries P@5 jumped from **0.120 (keyword)** to **0.840 (semantic/hybrid)**; overall 15-query P@5: **0.880 (hybrid)** vs **0.867 (semantic)** vs **0.587 (keyword)** | **Confirmed** — RRF combines exact-term precision with semantic recall without raw score normalization. |