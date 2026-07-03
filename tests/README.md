# Tests

Behavior tests for the public `rapidfuzz-collections` API, run with pytest. Tests are deterministic and primarily check observable behavior. Focused white-box tests cover selected index invariants that cannot be isolated as precisely through the public facades. Performance is not asserted; for timing and memory measurements, see [benchmarks/README.md](../benchmarks/README.md).

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The `dev` extra already installs `numpy`, which `test_batch_cdist.py` requires. The separate `cdist` extra is the runtime option for users who need matrix matching without installing the complete development toolchain; combining `dev` and `cdist` is supported but redundant for dependency installation.

A healthy run should collect about 1,000 tests and report 100% branch coverage. The exact test count may increase as the suite grows.

Optional coverage report:

```bash
python -m pytest -q --cov=rapidfuzz_collections --cov-report=term-missing
```

Run a focused file:

```bash
python -m pytest tests/test_fuzzy_dict.py -q
```

Run only public API contract tests:

```bash
python -m pytest tests/test_public_api.py -q
```

## Test groups

| Group                     | Files                                                                                                                                                                                                                               |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fuzzy collection behavior | `test_fuzzy_dict.py`, `test_frozen_fuzzy_dict.py`, `test_fuzzy_set.py`, `test_frozen_fuzzy_set.py`, `test_fuzzy_list.py`, `test_fuzzy_tuple.py`                                                                                     |
| Builtin compatibility     | `test_fuzzy_dict_compatibility.py`, `test_frozen_fuzzy_dict_compatibility.py`, `test_fuzzy_set_compatibility.py`, `test_frozen_fuzzy_set_compatibility.py`, `test_fuzzy_list_compatibility.py`, `test_fuzzy_tuple_compatibility.py` |
| Keyed strategy            | `test_fuzzy_dict_keyed_strategy.py`, `test_fuzzy_set_keyed_strategy.py`, `test_strategy_fuzzy_dict_set.py`                                                                                                                          |
| Sequence/keyed index      | `test_sequence_index.py`, `test_keyed_index.py`                                                                                                                                                                                     |
| Batch cdist               | `test_batch_cdist.py`                                                                                                                                                                                                               |
| Benchmark infrastructure  | `test_benchmark_infrastructure.py`                                                                                                                                                                                                  |
| Examples                  | `test_examples.py`                                                                                                                                                                                                                  |
| Normalization             | `test_normalization.py`                                                                                                                                                                                                             |
| Per-query overrides       | `test_per_query_overrides.py`                                                                                                                                                                                                       |
| Public API                | `test_public_api.py`                                                                                                                                                                                                                |
| Randomized compatibility  | `test_randomized_compatibility.py`                                                                                                                                                                                                  |
| Realistic data scenarios  | `test_data_fixtures.py`                                                                                                                                                                                                             |

Notes on each group:

- **Fuzzy collection behavior** tests each collection's own API — construction, fuzzy lookup, mutation, and standard dunder methods.
- **Builtin compatibility** tests compare a Fuzzy collection against the matching builtin (`dict`, `set`, `list`, `tuple`) on the same inputs, to keep the collections usable as close-to-drop-in replacements.
- **Keyed strategy** tests exercise `IndexStrategy.KEYED` specifically, plus the `SEQUENCE`/`KEYED` result-shape equivalence check in `test_strategy_fuzzy_dict_set.py`.
- **Sequence/keyed index** tests exercise `FuzzySequenceIndex` and the keyed index classes directly, below the collection facades.
- **Batch cdist** tests exercise the `numpy`-backed batch matching methods and require the `cdist` extra.
- **Benchmark infrastructure** tests run the fast strategy-matrix pre-flight check and verify benchmark defaults without performing performance assertions.
- **Examples** tests execute every documented runnable script in an isolated subprocess so public API changes cannot silently break the examples.
- **Per-query overrides** tests distinguish omitted configuration from explicit overrides such as `score_cutoff=None` and verify scorer/scorer-type rules.
- **Public API** tests check `rapidfuzz_collections.__all__`, exact visible public method sets, and public signatures via `tests/helpers.py`.
- **Randomized compatibility** tests run fixed-seed randomized operation sequences in lockstep against a Fuzzy collection and its builtin counterpart, checking both builtin-compatible state and synchronized exact fuzzy lookup after every mutation.

## Shared fixtures and helpers

- `tests/data.py` — reusable sample data (`PRODUCT_CATALOG`, `PRODUCT_NAMES`, `PRODUCT_PRICES`, `PRODUCT_QUERIES`, `UNICODE_PRODUCT_NAMES`, `NORMALIZED_COLLISION_VALUES`, `DUPLICATE_PRODUCT_NAMES`, `MIXED_SEARCH_VALUES`, `MAPPING_VALUES`) used by `test_data_fixtures.py` and other tests that need realistic values.
- `tests/helpers.py` — shared test utilities, including `HashableCycleNode` and `SearchableEqualityKey` fixtures, normalization helpers (`casefold_string`, `normalize_equality_key`, `normalize_boolean_one`, `require_not_none`), and public API introspection helpers (`public_methods`, `signature_text`, `mapping_match_signature`, `keyed_value_match_signature`, `positioned_value_match_signature`) used by `test_public_api.py`.

## Guidelines for writing tests

- Prefer behavior tests over implementation tests. Assert on inputs and outputs of the public API rather than internal state. White-box assertions are permitted when a private invariant is itself the subject of a focused test and no equally precise public observation exists. Current examples include lazy/dirty rebuild state, strategy preservation, source-index translation, exact-match shortcut tables, retained configuration, and scorer call counts. Keep such assertions local, name the invariant clearly, and do not treat private attributes as supported user API.
- Keep randomized tests deterministic. Seed `random.Random` explicitly, as `test_randomized_compatibility.py` does, instead of relying on unseeded global randomness.
- Do not put benchmark assertions in tests. Timing and memory measurements belong in [benchmarks/README.md](../benchmarks/README.md), not in the pytest suite.
- Update `test_public_api.py` only when the exported public contract intentionally changes (new/removed public names, signature changes).
- Add strategy-specific tests when a behavior differs internally but must keep the same public result shape across `SEQUENCE` and `KEYED`.
