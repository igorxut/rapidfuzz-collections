# Benchmarks

Scripts for measuring build, lookup, and mutation performance of `rapidfuzz-collections`. For rationale, prototype tracking, and historical investigation findings behind the numbers below, see [DESIGN.md](DESIGN.md). Generated report files are described in [reports/README.md](reports/README.md).

> Benchmark results depend on workload, scorer, normalizer, data profile, and machine. Numbers here and in generated reports are reference data from one environment, not universal performance guarantees. Re-run locally before relying on any specific figure.

## Setup

From the repository root:

```bash
pip install -e ".[dev]"
```

The `benchmark` extra installs NumPy for benchmark paths that exercise RapidFuzz cdist-style APIs. The `dev` extra keeps the checkout aligned with the normal test/lint environment.

## Files

| File                          | Purpose                                                                                                              |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `baseline_benchmark.py`       | Historical architecture experiments and production baselines                                                         |
| `index_strategy_benchmark.py` | Production strategy matrix plus explicitly selected active prototypes                                                |
| `hot_path_benchmark.py`       | Legacy-vs-production checks for deletion shortcuts and custom `score_all` paths                                      |
| `real_data_benchmark.py`      | Build/lookup benchmarks against committed real entity matching datasets                                              |
| `utils.py`                    | Low-level timing (`measure_timings`), memory (`measure_peak_kib`), and report (`write_benchmark_reports`) primitives |
| `datasets.py`                 | Deterministic dataset factories (`build_values`, `build_queries`) and `DataProfile`                                  |
| `runner_args.py`              | Argument presets and `build_runner_args()` used by the pytest wrappers                                               |
| `run_index_strategy_*.py`     | Thin pytest wrappers: call `main(build_runner_args(...))`                                                            |
| `run_hot_path_benchmark.py`   | Thin pytest wrapper for the focused hot-path matrix                                                                  |

## Production vs. prototype cases

Production cases construct classes imported from `rapidfuzz-collections` (`list`, `tuple`, `fuzzy-dict-sequence`/`fuzzy-dict-keyed`, `fuzzy-set-sequence`/`fuzzy-set-keyed`, `frozen-dict`/`frozen-keyed-dict`, `frozen-set`/`frozen-keyed-set`). Running `index_strategy_benchmark.py` without `--cases` selects production cases only.

`baseline_benchmark.py` and the `ordered-dict`/`ordered-set` cases in `index_strategy_benchmark.py` also carry internal architecture prototypes (named explicitly, e.g. `ExplicitFuzzyIndex`, `ordered-dict`). **Prototype results are not production package performance claims** — see [DESIGN.md](DESIGN.md#production-and-prototype-policy) for what each prototype exists to answer.

## How to run

Every script can be run either directly (`python benchmarks/<script>.py`) or as a module (`python -m benchmarks.<script>`); the direct form does not require the current directory to be the project root. Pass `--help` to any script for the full flag reference.

```bash
# Full baseline (all sections, default settings)
python -m benchmarks.baseline_benchmark

# Smoke run
python -m benchmarks.baseline_benchmark --items 200 --repeats 2 --groups build sequence

# Select scorer and data profile
python -m benchmarks.baseline_benchmark --items 1000 --scorer wratio --profile duplicates

# Strategy matrix
python -m benchmarks.index_strategy_benchmark \
  --items 1000 --repeats 3 \
  --cases fuzzy-dict-sequence fuzzy-dict-keyed \
  --profiles unique collision-20

# Explicit prototype comparison
python -m benchmarks.index_strategy_benchmark \
  --items 1000 --repeats 3 \
  --cases fuzzy-dict-sequence fuzzy-dict-keyed ordered-dict \
  --profiles unique collision-20

# Focused deletion-shortcut and custom-score materialization A/B matrix
python -m benchmarks.hot_path_benchmark \
  --items 1000 10000 100000 \
  --repeats 5 \
  --lookup-counts 1 2 3 10 100

# Real-data build/lookup benchmark
python benchmarks/real_data_benchmark.py

# Via pytest (thin wrappers, saves JSON + CSV to benchmarks/reports/)
python -m pytest benchmarks/run_index_strategy_facade_smoke_benchmark.py -v

# Final narrow strategy matrices used for stable architecture comparisons
python -m pytest benchmarks/run_index_strategy_final_narrow_benchmark.py -v

# Broader scorer/normalizer matrix used to challenge narrow conclusions
python -m pytest benchmarks/run_index_strategy_final_extended_benchmark.py -v
```

`--exact-tie-only` and `--full-result-only` on `index_strategy_benchmark.py` select focused sub-matrices used in past investigations; see [DESIGN.md](DESIGN.md#historical-findings) for what they were used to measure.

## Report output format

Every benchmark script (`baseline_benchmark.py`, `index_strategy_benchmark.py`, `hot_path_benchmark.py`, `real_data_benchmark.py`) writes its results the same way, via `write_benchmark_reports()` in `utils.py`:

- a fixed `--output-dir` subfolder under `benchmarks/reports/`, created if missing (each script has its own default; override with `--output-dir`);
- exactly two files per run: `<stem>.json` and `<stem>.csv`;
- the JSON file wraps the result rows together with `environment_metadata()` (Python version, platform, `rapidfuzz`/`rapidfuzz-collections` versions);
- the CSV file contains only the raw result rows, with no environment metadata.

Markdown and plain-table output are not produced by any benchmark script.

## `benchmarks/reports/`

Each subfolder under `benchmarks/reports/` holds the `<stem>.json`/`<stem>.csv` pair from one local run (e.g. `baseline/`, `index_strategy/`, `index_hot_paths/`, `real_data/`, plus a few named investigation runs such as `index_strategy_full_result_100000/`). This data is **generated locally and not tracked in git** — see [reports/README.md](reports/README.md). Regenerate it with the commands above to compare against your own hardware and workload.

## Which reports support architectural conclusions

| Conclusion                                                                                           | Primary runner and generated report folders                                                                                                                                             |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SEQUENCE` is the general read-heavy default                                                         | `run_index_strategy_final_narrow_benchmark.py`; `index_strategy_final_narrow_mutable_1000`, `index_strategy_final_narrow_mutable_10000`, and `index_strategy_final_narrow_frozen_10000` |
| `KEYED` can win build- and mutation-heavy workloads                                                  | Narrow mutable reports above, challenged by `run_index_strategy_final_extended_benchmark.py` and its mutable report                                                                     |
| Frozen strategy depends on build/memory versus lookup priority                                       | Narrow and extended frozen reports                                                                                                                                                      |
| Full-result ranking should avoid redundant Python sorting                                            | `run_index_strategy_full_result_benchmark.py`; `index_strategy_full_result_100000`                                                                                                      |
| Lazy exact-shortcut rebuild and direct custom-score materialization are beneficial for repeated work | `run_hot_path_benchmark.py`; `index_hot_paths`                                                                                                                                          |
| Exact tie-breaking remains correct under normalized collisions                                       | `run_index_strategy_exact_tie_benchmark.py`; `index_strategy_exact_tie_10000`                                                                                                           |

Report folders are generated locally and are not canonical artifacts. Compare strategy ratios and winner consistency within one report environment; do not compare absolute timings across folders produced on different machines or dependency versions.

## What each benchmark measures

### `baseline_benchmark.py`

Measures raw fuzzy-lookup performance across **all collection types** (FuzzyList, FuzzyDict, FuzzySet, FrozenFuzzySet) and compares them against internal architecture prototypes.

Sections (pass with `--groups`):

| Section             | What it tests                                               |
|---------------------|-------------------------------------------------------------|
| `build`             | Construction and normalization cost                         |
| `sequence`          | Single and batch lookups on FuzzyList                       |
| `mapping`           | Key lookups on FuzzyDict                                    |
| `set`               | Membership on FuzzySet / FrozenFuzzySet                     |
| `keyed-choices`     | SEQUENCE vs KEYED strategy for dict/set                     |
| `batch-api`         | Native RapidFuzz `cdist` / `cpdist` paths                   |
| `mutation`          | Append, set, and discard costs                              |
| `index-comparison`  | FuzzySequenceIndex vs MutableFuzzySequenceIndex             |
| `deletion-heavy`    | Repeated fuzzy deletes; tombstone/compact-delete prototypes |
| `replacement-heavy` | Positional replace + query cycles                           |
| `interleaved`       | Mixed insert/delete/query workloads                         |
| `score-hint`        | Effect of `score_hint` on `extractOne` speed                |
| `collision-cost`    | Exact-delete cost vs normalized-collision density           |
| `advanced-top-one`  | Bounded cdist top-one batch API                             |

### `index_strategy_benchmark.py`

Measures how **index strategy** (SEQUENCE vs KEYED) and **collection type** affect all standard operations: build, lookup, contains, find-many, batch, mutation, and composite workloads. Production cases use the package classes from `src`; `ordered-dict`/`ordered-set` are explicit architecture prototypes.

The normalized-collision operations (`lookup:normalized-collision-exact`, `find-one:normalized-collision-exact`, `find-many:1-normalized-collision-exact`, `batch-cdist:normalized-collision-exact`, `mutation:fuzzy-discard-normalized-collision-exact`) verify and measure the public tie policy: the harness fails before timing if `find_one` and `find_many(limit=1)` do not select the exact original value in that scenario.

### `hot_path_benchmark.py`

Retains two minimal legacy policies solely as reproducible A/B baselines: repeated exact-value scans after an incremental sequence deletion, and ranked extraction before position-aligned custom-scorer materialization. The production comparisons use the current public index classes. See [DESIGN.md](DESIGN.md#historical-findings) for the tradeoffs these baselines were used to quantify.

## Real-data benchmark

`real_data_benchmark.py` measures build time, single-query lookup time, and peak memory for six collection types against committed entity matching datasets under `examples/data/`. It is opt-in and is not run by pytest automatically. Results are written under `benchmarks/reports/real_data/`. The benchmark measures timing and memory behavior only; it does not assert entity-resolution quality. Each dataset case currently repeats one positive query, so these rows are smoke-level real-data timings rather than evidence for mixed positive/typo/miss workloads.

## Data profiles

All profiles are deterministic; shuffled profiles use `random.Random(42)`.

| Profile        | Generated values                                                                                                                                     |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `unique`       | `Alpha Phone {i:06d} Model {i % 97:02d}` — one unique string per slot                                                                                |
| `mixed`        | Every 7th → `int`, 7th+1 → `None`, 7th+2 → `"XS"` (too short for normalizer), rest → `Beta Tablet {i:06d} Series {i % 97:02d}`                       |
| `duplicates`   | Repeated normalized groups cycling through canonical, lowercase-padded, and uppercase variants of `Coffee Grinder {group:06d} Type {group % 97:02d}` |
| `collision-0`  | 0 % normalized collisions; all unique                                                                                                                |
| `collision-5`  | 5 % of values share a normalized form with a padded-lowercase variant                                                                                |
| `collision-20` | 20 % collision density                                                                                                                               |
| `collision-50` | 50 % collision density                                                                                                                               |

See [DESIGN.md](DESIGN.md#data-profile-rationale) for why each profile exists.

## Interpreting metrics

| Metric      | Meaning                                                  |
|-------------|----------------------------------------------------------|
| `best_ms`   | Minimum elapsed time over all repeats (least OS noise)   |
| `median_ms` | Median elapsed time (more robust for noisy environments) |
| `peak_kib`  | Peak traced memory from a single `tracemalloc` run       |

`best_ms` is the primary comparison metric. `median_ms` helps detect high variance. `peak_kib` reflects per-call allocation, not resident set size.

Custom scorers are covered by deterministic scorer-call tests in `tests/test_sequence_index.py` and `tests/test_keyed_index.py`. Call counts are algorithmic diagnostics rather than timing measurements, so assertions provide a clearer regression signal than wall-clock benchmark rows.

## Reproducibility

All dataset generators use fixed seeds and deterministic iteration order: `build_values`/`build_queries` are pure functions of `(items, profile)`, and `build_values_with_collision_rate` shuffles with `random.Random(42)`. No global state is modified between benchmark calls.

Run with `--repeats 1` for a quick smoke check; use `--repeats 5` or higher for stable timing comparisons.
