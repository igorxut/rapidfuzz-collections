# Benchmark Design Notes

Rationale, prototype tracking, and historical investigation findings behind the commands and numbers referenced from [README.md](README.md). This file is a record of *why* certain benchmarks, prototypes, and index strategies exist and what past runs found — not a source of current performance guarantees. Generated reports under [`reports/`](reports/README.md) are local, untracked artifacts. This document therefore records portable relationships and architecture decisions rather than treating absolute timings or allocations as canonical evidence. Re-run the cited runner on one environment and compare ratios and winner consistency within that report.

## Production and prototype policy

Published performance conclusions about the package must come from production cases that construct classes imported from `rapidfuzz-collections`:

- `list` and `tuple`;
- `fuzzy-dict-sequence` and `fuzzy-dict-keyed`;
- `fuzzy-set-sequence` and `fuzzy-set-keyed`;
- `frozen-dict` and `frozen-keyed-dict`;
- `frozen-set` and `frozen-keyed-set`.

Prototype cases exist to test an architecture before it is added to `src`. They must be named and selected explicitly, and their results must not be presented as current package performance. Running `index_strategy_benchmark.py` without `--cases` selects production cases only.

The active strategy prototype is:

| Case                           | Implementation                 | Purpose                                                          | Status                               |
|--------------------------------|--------------------------------|------------------------------------------------------------------|--------------------------------------|
| `ordered-dict` / `ordered-set` | `OrderedUniqueFuzzy*Prototype` | Dense RapidFuzz choices with hash-based exact and mutation state | Retained for architecture comparison |

The active prototype follows the production ranking contract: scorer quality first, exact equality for equal scores, then insertion order. This keeps its measurements focused on storage architecture rather than different public results.

The ordered prototype is included only by runner presets that explicitly add `ORDERED_PROTOTYPE_CASES`. Its historical results remain in the `index_strategy`, `index_strategy_large`, and `index_strategy_weighted` report directories.

The old benchmark-only immutable keyed index and frozen facades were removed after their design was implemented by `ImmutableFuzzyKeyedIndex`, `FrozenFuzzyDict`, and `FrozenFuzzySet` in `src`. Their conclusions remain reproducible through the production frozen cases and the frozen reports.

`baseline_benchmark.py` intentionally retains the following earlier architecture experiments:

| Prototype                          | Question preserved by the experiment                        |
|------------------------------------|-------------------------------------------------------------|
| `ExplicitFuzzyIndex`               | Cost of a separate immutable sequence index                 |
| `MutableExplicitFuzzyIndex`        | Cost of incrementally maintained mutable sequence state     |
| `TombstoneFuzzyIndex`              | Whether soft deletion avoids expensive rebuilds             |
| `CompactDeleteFuzzyIndex`          | Whether compact deletion is preferable without stable slots |
| `ExplicitFuzzyMappingIndex`        | Separate immutable key-index design                         |
| `MutableExplicitFuzzyMappingIndex` | Separate mutable key-index design                           |

These prototypes are not candidates for deletion merely because production code now exists. Keep them until a documented successor covers the same experiment and its comparison no longer provides useful evidence.

## Data profile rationale

The synthetic profiles isolate different costs:

- `unique`: baseline lookup/build behavior without duplicates or normalized collisions;
- `mixed`: normalizer rejection of non-string and short-string inputs;
- `duplicates`: repeated values with casing/spacing variation;
- `collision-*`: normalized-key ambiguity and its cost on lookup and mutation paths.

`build_queries()` also records a `normalized_collision_exact` query when a profile contains two distinct original values with the same normalized form. The query is the later original value, so it measures the case where scorer quality ties but exact equality must outrank an earlier normalized-equivalent candidate.

## Why `SEQUENCE` is the default strategy

The primary purpose of the library is fast repeated fuzzy lookup over collection data. For that workload, sequence storage is the safest default because RapidFuzz scans list/tuple-like choices efficiently and because the strategy keeps the lookup path direct.

The final narrow strategy matrices consistently favor `SEQUENCE` across the ordinary read-only, lookup-heavy, and batch-heavy pairs that motivate the default. The extended matrices challenge that result with additional scorers and normalizers. These comparisons show why keyed storage cannot be the silent default: it does not consistently win the package's primary read workload.

Evidence: `run_index_strategy_final_narrow_benchmark.py` and `run_index_strategy_final_extended_benchmark.py`, comparing paired production cases at the same item count, profile, normalizer, scorer, and workload.

## Why `KEYED` still exists

Keyed storage is useful for domains where the fuzzy search candidates are unique and hashable, which is naturally true for dict keys and set members. It avoids some position-oriented bookkeeping and can improve selected mutation and construction workloads. The production KEYED index also keeps canonical exact values. For mutable collections, its keyed choice mapping and reverse collision state usually make KEYED larger than SEQUENCE even when KEYED is faster.

Paired final-matrix rows show `KEYED` winning selected build and bulk-mutation workloads, especially with mixed or collision-heavy data. The same mutable matrices generally show a memory premium for its exact-value, choice-mapping, and reverse-collision state. Compare the strategy ratio and memory ratio in a single generated report rather than carrying machine-specific measurements into this design record.

The strongest keyed cases are not ordinary point lookup. They are construction, bulk fuzzy deletion/retention, collision-heavy data, and selected batch or normalizer/scorer combinations. That is why `IndexStrategy.KEYED` is exposed as an explicit strategy instead of being hidden behind a separate collection class.

## Why there is no automatic strategy selection

Automatic strategy selection would have to infer the future workload from the initial data. The benchmarks show that the best choice depends on information the constructor usually does not have:

- whether the collection will be read-heavy or mutation-heavy;
- whether future queries will be close matches, misses, or matches that differ only after normalization;
- whether the scorer is a similarity scorer, a distance scorer, or a composite scorer such as `WRatio`;
- whether the normalizer is cheap or expensive;
- whether normalized collisions are rare or common;
- whether the caller values build time, steady lookup time, or mutation time.

A hidden automatic choice would make performance harder to predict and could produce large regressions for users whose data looks similar at construction time but is queried differently later. The public `strategy` parameter keeps the choice explicit and reproducible.

## Why frozen dict/set do not have one guaranteed best index

Frozen collections remove mutation cost from the equation, so keyed storage can drop mutation-only state. The keyed form still keeps canonical exact values, but current measurements show that it often remains a strong construction and memory candidate. It is not a universal replacement for sequence storage.

The paired frozen matrices commonly favor `KEYED` for build cost or memory while favoring `SEQUENCE` for point lookup. Neither relationship holds for every scorer, normalizer, and profile, so the narrow and extended frozen reports must be evaluated together.

The practical rule is:

- choose `IndexStrategy.SEQUENCE` when frozen point lookup is the priority;
- try `IndexStrategy.KEYED` when build cost, memory, or scan-heavy batch work matters more;
- benchmark with your own scorer, normalizer, and query distribution before treating either strategy as final for a large frozen reference table.

## Why list and tuple stay sequence-only

Lists and tuples can contain duplicates, unhashable values, and meaningful positions. A keyed value index cannot represent that model without adding an artificial slot identity and a separate slot-to-value mapping. Once that state exists, the design has effectively returned to sequence storage with extra bookkeeping.

This is why only dict/set facades accept `strategy`. For `FuzzyList` and `FuzzyTuple`, sequence storage is not merely a default; it is the storage shape that matches the collection contract.

## Why `cdist` is opt-in

RapidFuzz `process.cdist` is valuable when the caller needs a score matrix or has measured a workload where full matrix scoring is faster. Collection fuzzy lookup usually needs a top-one or top-many result, not the full matrix.

The ordinary batch methods preserve exact shortcuts and allow RapidFuzz `extractOne` to prune candidate work as strong matches are found. A `cdist` path computes query-by-choice matrix chunks and then reduces them back to the same top-one result shape. Before matrix computation, it still normalizes queries and resolves provably optimal exact matches, so exact queries do not need to enter the matrix. That can be useful, but it is not the default collection lookup model.

Use the explicit `*_batch_cdist` methods only after measuring your workload. Use RapidFuzz `process.cdist` directly when you need the full matrix.

## Practical strategy rules

Use these rules before running your own benchmarks:

| Data and workload                          | Start with | Try next                                            |
|--------------------------------------------|------------|-----------------------------------------------------|
| Dict/set, ordinary point lookup            | `SEQUENCE` | `KEYED` only if measured faster                     |
| Dict/set, lookup-heavy workload            | `SEQUENCE` | `KEYED` only for a demonstrated local win           |
| Dict/set, bulk fuzzy deletion or retention | `KEYED`    | compare against `SEQUENCE` for unique data          |
| Mutable dict/set, memory sensitive         | `SEQUENCE` | use `KEYED` only if its speed win is more important |
| Frozen dict/set, build or memory sensitive | `KEYED`    | compare read-only point lookup against `SEQUENCE`   |
| Frozen dict/set, read-heavy point lookup   | `SEQUENCE` | `KEYED` for scan-heavy batch work                   |

The final choice should be made on your data. The public API keeps both strategies because neither one dominates all realistic collection workloads.

## Historical findings

These sections summarize specific past investigations that are still reproducible with the flags shown. They describe the measured synthetic profiles from one local run, not a universal speed guarantee — regenerate the underlying report to check current behavior; see [reports/README.md](reports/README.md).

### Full-result ranking cost

`run_index_strategy_full_result_benchmark.py` is the 100,000-item matrix for detecting the additional Python-side ranking cost of `find_many(limit=None)` (reproducible with `index_strategy_benchmark.py --full-result-only`). It covers both scorer directions and three normalized collision densities without running unrelated operations.

The first 100,000-item run showed why result cardinality matters. With `ratio`, 49,163 to 97,909 values passed the cutoff and a second Python sort became visible; with `Levenshtein.distance`, only 226 to 294 values passed and the ranking overhead was small. Profiling the high-cardinality paths attributed about 13-15% of the profiled call to the redundant sort. The production indexes now preserve the order already returned by RapidFuzz for unbounded searches and promote exact candidates only within equal-score groups.

An alternating in-process comparison on prebuilt 100,000-value indexes avoided the process-level noise seen between complete benchmark runs. In that check, the optimized non-exact `ratio` path was about 6-7% faster for SEQUENCE and 11-13% faster for KEYED. Exact collision queries improved by up to about 3% for SEQUENCE and 14-23% for KEYED.

### Exact shortcuts after incremental deletion

A mutable sequence index stores the source value, the normalized value, the sequence of RapidFuzz choices, the exact-value registry, and a choice-to-source-position mapping when some values are not searchable. Incremental sequence deletion avoids rebuilding normalized fuzzy choices, but it invalidates exact-value source positions. The legacy implementation scanned all current values on every exact fuzzy lookup. Production now rebuilds only the exact shortcut dictionaries on the first lookup and reuses them afterward; insertions, sorting, replacements, and large or complex deletions mark the whole index dirty for a full rebuild on the next fuzzy query instead. For `KEYED`, mutable dicts and sets use direct `add`/`remove`, a batched `batch_remove`, and switch to a full rebuild as the cheaper path when a deletion removes a large enough fraction of the collection.

The focused 1,000/10,000/100,000-item matrix (`hot_path_benchmark.py`) shows the tradeoff explicitly:

- one exact lookup is 35-92% slower because it pays the one-time rebuild;
- two exact lookups are already 9-32% faster;
- three exact lookups are 42-54% faster;
- ten exact lookups are about 85% faster;
- one hundred exact lookups are 97-98% faster.

The largest measured repeated-lookup case retained the same strong relative improvement while paying a temporary allocation to replace the exact dictionaries. This policy is intended for repeated lookup workloads; it deliberately keeps incremental deletion itself inexpensive. Across the measured sizes, one close fuzzy lookup was 4-18% slower, while ten lookups were 11-14% faster. The one-time exact-registry rebuild is therefore visible but does not change the O(n) fuzzy-scoring cost.

### Exact tie-breaking under normalized collisions

Top-one lookup paths use exact original values or keys to resolve equal-score ambiguity: scorer quality remains primary, exact equality only breaks equal-score ties, and source position or insertion order resolves any remaining tie. An exact shortcut is enabled only when native RapidFuzz metadata confirms the configured scorer direction and proves that the exact candidate has the optimal possible score; the score cutoff still applies, and custom scorers without compatible metadata scan every searchable candidate. KEYED additionally maps every exact searchable value to the canonical object stored by the collection, which matters because Python can consider distinct objects equal (such as `True` and `1`).

The dedicated `collision-50` matrix shows that exact tie resolution keeps top-one lookup in the same performance class as ordinary exact lookup across both strategies. `find_many(limit=1)` is intentionally more expensive because it preserves the ranked multi-result contract before trimming the result.

### Custom scorer score materialization

Ranked extraction is useful for native RapidFuzz scorers, but a custom scorer fallback previously sorted all scored values before placing them back into source order. `fuzzy_score_all()`/`fuzzy_iter_scores()` solve a different problem than `find_many`: scoring the source positionally, yielding `None` for non-searchable, rejected, and below-cutoff positions, so they never apply the top-one exact shortcut. Production now writes custom-scorer results directly to their output positions and retains ranked extraction for compatible RapidFuzz scorers.

Across constant and wrapped-`ratio` custom scorers, direct materialization was 5-27% faster in the focused matrix (`hot_path_benchmark.py`). At the largest measured size, peak traced allocation fell by roughly 36-57%. Use `iter_scores` when streaming consumption can avoid materializing the output list altogether.
