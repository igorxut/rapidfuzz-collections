# Examples

Each script demonstrates one public collection or index API and one usage pattern against real or inline data.

These examples are intended for repository checkouts. They are not installed as package data and rely on files under `examples/data/`.

## Setup

Install the package in editable mode from the repository root before running any script:

```bash
pip install -e .
```

Then run any script directly from the repository root:

```bash
python examples/<script>.py
```

## Scripts

| Script                                            | Collection                 | Demonstrates                                                                                      |
|---------------------------------------------------|----------------------------|---------------------------------------------------------------------------------------------------|
| `bulk_publication_lookup_with_cdist.py`           | `FuzzyTuple`               | Advanced bounded cdist top-one lookup for bulk publication resolution                             |
| `catalog_lookup_with_fuzzy_dict.py`               | `FrozenFuzzyDict`          | Compact key → payload lookup with a decision policy                                               |
| `database_ids_with_keyed_index.py`                | `ImmutableFuzzyKeyedIndex` | Returning stable database IDs from fuzzy queries                                                  |
| `mutable_reference_data.py`                       | `FuzzyDict`                | Runtime mutation and its effect on fuzzy lookup results                                           |
| `ordered_list_cleanup_with_fuzzy_list.py`         | `FuzzyList`                | Ordered duplicate values, positional matches, and fuzzy cleanup                                   |
| `record_lookup_with_sequence_index.py`            | `FuzzySequenceIndex`       | Record-as-value lookup, custom normalization, and a per-query cutoff override                     |
| `reference_data_normalization_with_fuzzy_dict.py` | `FrozenFuzzyDict`          | Alias table with `None` sentinels, decision policy, and explicit cutoff disabling for maintenance |
| `vocabulary_lookup_with_fuzzy_set.py`             | `FrozenFuzzySet`           | Fuzzy lookup over a fixed vocabulary of category labels                                           |

```bash
python examples/bulk_publication_lookup_with_cdist.py
python examples/catalog_lookup_with_fuzzy_dict.py
python examples/database_ids_with_keyed_index.py
python examples/mutable_reference_data.py
python examples/ordered_list_cleanup_with_fuzzy_list.py
python examples/record_lookup_with_sequence_index.py
python examples/reference_data_normalization_with_fuzzy_dict.py
python examples/vocabulary_lookup_with_fuzzy_set.py
```

`bulk_publication_lookup_with_cdist.py` requires NumPy from the optional cdist extra:

```bash
pip install -e ".[cdist]"
```

The cdist example demonstrates bounded matrix chunks, not a universal performance recommendation. Try the ordinary batch methods first and select cdist only after measuring the application's scorer, cutoff, query distribution, batch size, and memory constraints.

## How to read the output

The printed decisions are illustrative business policies, not guarantees that a fuzzy score alone identifies the correct entity. Examples deliberately show both candidates that can be accepted under a chosen policy and candidates that should remain subject to manual review. Production thresholds and review rules must be validated against the application's own data and error costs.

## Datasets

The datasets under `examples/data/` are included for local experimentation only. They are not part of the Python package distribution and are not covered by this project's source code license. See [examples/data/NOTICE.md](data/NOTICE.md) for dataset sources, license, attribution, and modification notes.
