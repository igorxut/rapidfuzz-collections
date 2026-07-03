# rapidfuzz-collections

[![PyPI version](https://img.shields.io/pypi/v/rapidfuzz-collections)](https://pypi.org/project/rapidfuzz-collections/)
[![Python versions](https://img.shields.io/pypi/pyversions/rapidfuzz-collections)](https://www.python.org/downloads/)
[![Wheel](https://img.shields.io/pypi/wheel/rapidfuzz-collections)](https://pypi.org/project/rapidfuzz-collections/)
[![CI](https://github.com/igorxut/rapidfuzz-collections/actions/workflows/ci.yml/badge.svg)](https://github.com/igorxut/rapidfuzz-collections/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/igorxut/rapidfuzz-collections/branch/main/graph/badge.svg)](https://codecov.io/gh/igorxut/rapidfuzz-collections)
[![Typed](https://img.shields.io/badge/typed-py.typed-blue)](https://peps.python.org/pep-0561/)
[![License](https://img.shields.io/github/license/igorxut/rapidfuzz-collections)](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/LICENSE)

`rapidfuzz-collections` provides collection facades that keep Python's builtin collection behavior while adding fuzzy lookup powered by [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/).

Use it when your data naturally belongs in a list, tuple, set, or dict, but you also need typo-tolerant lookup over the stored values or mapping keys.

---

## Contents

1. [What This Library Adds](#what-this-library-adds)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Choosing a Collection](#choosing-a-collection)
5. [Practical Examples](#practical-examples)
6. [Runnable Examples](#runnable-examples)
7. [Lookup Model](#lookup-model)
8. [Index Strategies](#index-strategies)
9. [Configuration](#configuration)
10. [Batch Lookup and cdist](#batch-lookup-and-cdist)
11. [Mutation and Rebuilds](#mutation-and-rebuilds)
12. [Result Objects](#result-objects)
13. [Public Method Reference](#public-method-reference)
14. [Performance Guidance](#performance-guidance)
15. [Advanced Index APIs](#advanced-index-apis)
16. [Design Boundaries](#design-boundaries)
17. [Development Checks](#development-checks)
18. [Third-party Example Datasets](#third-party-example-datasets)
19. [License](#license)

For index-strategy rationale, historical benchmark investigations, and reproducible numbers behind the guidance in this README, see [`benchmarks/DESIGN.md`](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/benchmarks/DESIGN.md).

---

## What This Library Adds

RapidFuzz already provides the fuzzy matching algorithms. This library does not replace RapidFuzz and does not reimplement its scorers.

RapidFuzz provides:

- string similarity and distance scorers, such as `WRatio`, `ratio`, and Levenshtein distance;
- high-performance extraction utilities such as `process.extractOne`;
- matrix scoring utilities such as `process.cdist`;
- scorer-specific behavior, score cutoffs, score hints, and scorer kwargs.

`rapidfuzz-collections` adds:

- builtin-like collections that store your original values unchanged;
- cached normalized lookup choices for repeated fuzzy searches;
- exact-value registries and deterministic equal-score tie-breaking;
- mutation-aware index maintenance for mutable collections;
- consistent result objects for value and mapping lookups.

The boundary is deliberate:

- Choose RapidFuzz directly when you need raw string metrics, custom matrix scoring, or one-off matching between plain sequences.
- Choose `rapidfuzz-collections` when your data has collection semantics and fuzzy lookup is a repeated operation over that collection.

Official RapidFuzz resources:

- Documentation: <https://rapidfuzz.github.io/RapidFuzz/>
- GitHub: <https://github.com/rapidfuzz/RapidFuzz>

---

## Installation

```bash
pip install rapidfuzz-collections
```

Python 3.14 or later is required. RapidFuzz is installed as the runtime fuzzy matching dependency.

Install the optional `cdist` extra only if you plan to use the explicit bounded matrix batch methods:

```bash
pip install "rapidfuzz-collections[cdist]"
```

The `cdist` extra installs NumPy for the opt-in `*_batch_cdist` methods. The ordinary fuzzy lookup methods do not require NumPy.

---

## Quick Start

```python
from rapidfuzz_collections import FuzzyDict, FuzzyList

products = FuzzyList(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

match = products.fuzzy_find_one("Alpa Phone")
print(match.value)  # "Alpha Phone"

catalog = FuzzyDict({"Alpha Phone": 499, "Beta Tablet": 799})

price = catalog.fuzzy_get("beta tablt")
print(price)  # 799
```

Misses return `None`, or a default for direct `fuzzy_get` helpers:

```python
print(products.fuzzy_find_one("Coffee Grinder"))  # None
print(catalog.fuzzy_get("Coffee Grinder", default=0))  # 0
```

---

## Choosing a Collection

| Need                                          | Collection        |
|-----------------------------------------------|-------------------|
| Ordered values, duplicates allowed, mutable   | `FuzzyList`       |
| Ordered values, duplicates allowed, immutable | `FuzzyTuple`      |
| Unique values, mutable                        | `FuzzySet`        |
| Unique values, immutable and hashable         | `FrozenFuzzySet`  |
| Key to value mapping, mutable                 | `FuzzyDict`       |
| Key to value mapping, immutable               | `FrozenFuzzyDict` |

Choose by data model first:

- Use a sequence when order, duplicates, or positions matter.
- Use a set when values are unique and the value itself is the result.
- Use a dict when a fuzzy-matched key should retrieve a payload.
- Use frozen collections for read-many reference data.
- Use mutable collections when values are changed after construction.

Then choose an index strategy only for dict/set facades. See [Index Strategies](#index-strategies).

---

## Practical Examples

### Command palette

A command palette is an ordered list of unique or repeated labels. The result value is the command label, and the position can still be useful for UI state.

```python
from rapidfuzz_collections import FuzzyList

commands = FuzzyList([
    "Open Settings",
    "Open Recent File",
    "Toggle Sidebar",
    "Format Document",
])

match = commands.fuzzy_find_one("format doc")
if match is not None:
    print(match.value)
```

### Product catalog lookup

A catalog often needs fuzzy lookup by product name while returning a price, identifier, or metadata record.

```python
from rapidfuzz_collections import FuzzyDict

catalog = FuzzyDict({
    "Alpha Phone 128GB": {"sku": "AP-128", "price": 499},
    "Beta Tablet 11 inch": {"sku": "BT-11", "price": 799},
})

item = catalog.fuzzy_find_item("beta tab 11")
if item is not None:
    print(item.key, item.value["sku"])
```

### Tag validation

A set is useful when the matched value itself is enough. Duplicate tags are ignored, and fuzzy containment is explicit.

```python
from rapidfuzz_collections import FuzzySet

allowed_tags = FuzzySet(["python", "machine learning", "data science"])

if allowed_tags.fuzzy_contains("machne learnig"):
    print(allowed_tags.fuzzy_get("machne learnig"))
```

### Immutable reference table

Frozen collections are appropriate for data that is loaded once and queried many times, such as a country lookup table or a command alias map.

```python
from rapidfuzz_collections import FrozenFuzzyDict

countries = FrozenFuzzyDict({
    "United States": "US",
    "United Kingdom": "GB",
    "Georgia": "GE",
})

print(countries.fuzzy_get("Unted Stats"))  # "US"
```

### Custom normalizer for structured values

The collection stores original objects unchanged. The normalizer controls only what text is searched.

```python
from rapidfuzz_collections import FuzzyList


def person_normalizer(person):
    if isinstance(person, dict):
        return f"{person['first']} {person['last']}".casefold()
    return None


people = FuzzyList(
    [
        {"first": "Ada", "last": "Lovelace", "id": 1},
        {"first": "Grace", "last": "Hopper", "id": 2},
    ],
    normalizer=person_normalizer,
)

match = people.fuzzy_find_one("grace hoppr")
print(match.value["id"])  # 2
```

---

## Runnable Examples

The [`examples/`](https://github.com/igorxut/rapidfuzz-collections/tree/v1.0.0/examples) directory contains runnable scripts that demonstrate the library against real or inline data. Each script covers one collection class and one usage pattern.

Install the package in editable mode before running any script:

```bash
pip install -e .
python examples/<script>.py
```

See [`examples/README.md`](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/examples/README.md) for the full list of scripts, setup instructions, and dataset licensing notes.

---

## Lookup Model

Every top-one fuzzy lookup follows the same broad sequence:

1. Normalize the query. If the normalizer returns `None`, return no fuzzy match.
2. Score searchable candidates and apply the configured score cutoff.
3. Rank accepted candidates by scorer quality: highest similarity or lowest distance.
4. Among candidates with the same score, prefer a hashable stored value or key equal to the original query.
5. Resolve any remaining tie by source position or insertion order.

Consequently, `fuzzy_find_one(query)` selects the same candidate as the first result of `fuzzy_find_many(query, limit=1)`. The same rule is used by top-one retrieval and mutation methods such as `fuzzy_get`, `fuzzy_discard`, and `fuzzy_remove`.

For a compatible native RapidFuzz scorer, an exact candidate may return immediately when scorer metadata proves that its score is optimal. This is an optimization only; it does not change the ranking contract. Custom scorers without compatible metadata evaluate every searchable candidate needed to determine the best score.

```python
from rapidfuzz_collections import FuzzyList

names = FuzzyList(["ALPHA", "alpha"], score_cutoff=0)

print(names.fuzzy_find_one("alpha").value)  # "alpha"
print(names.fuzzy_find_many("alpha", limit=1)[0].value)  # "alpha"
```

Both stored strings normalize to the same text and receive the same default score. The exact original value wins that score tie even though it appears later. For unhashable sequence values, no exact-value registry is available, so equal-score ties are resolved by source position.

The original collection data is not replaced by normalized data. Normalized choices are cached beside the collection to avoid repeating normalization work on every query.

The lookup domain depends on collection type:

| Collection type                | Fuzzy search domain |
|--------------------------------|---------------------|
| `FuzzyList`, `FuzzyTuple`      | stored values       |
| `FuzzySet`, `FrozenFuzzySet`   | stored values       |
| `FuzzyDict`, `FrozenFuzzyDict` | mapping keys        |

Mapping value lookup is intentionally key-based. To fuzzy-search mapping values, store those values in a value collection or create a separate mapping whose keys are the searchable values.

---

## Index Strategies

`FuzzyDict`, `FuzzySet`, `FrozenFuzzyDict`, and `FrozenFuzzySet` accept a `strategy` parameter:

```python
from rapidfuzz_collections import FuzzyDict, IndexStrategy

catalog = FuzzyDict(
    {"Alpha Phone": 499, "Beta Tablet": 799},
    strategy=IndexStrategy.SEQUENCE,
)
```

### `IndexStrategy.SEQUENCE`

`SEQUENCE` stores normalized choices in sequence order. It is the default because it is the safest general read-heavy baseline in current benchmarks.

Use it when:

- you do not know the workload shape yet;
- point lookups and ordinary batch lookups dominate;
- you need the explicit `*_batch_cdist` methods;
- you prefer the most predictable default.

### `IndexStrategy.KEYED`

`KEYED` stores normalized choices keyed by each unique hashable value or key. It can reduce build cost or selected bulk mutation costs for dict/set domains, especially when normalized collisions are common. It also stores a canonical exact-value mapping so an equal-but-not-identical query returns the object actually held by the collection.

Try it when:

- the collection is a dict or set facade;
- keys or values are unique and hashable;
- build cost or selected mutation paths matter;
- fuzzy discard/retain operations are common;
- local benchmarks show a keyed win for your data.

`KEYED` is not a universal faster mode. Keep `SEQUENCE` as the baseline for large read-heavy collections unless your own measurements say otherwise. For mutable collections, KEYED generally uses more memory than SEQUENCE. Frozen KEYED collections can still reduce both build cost and memory.

Both strategies return the same public result classes. Dict and set facades are position-free: `Match.index` and `MappingMatch.index` are always `None` for those facades.

Neither strategy is a universal winner across all workloads. For the benchmark rows and reasoning behind the guidance above, see [Why `SEQUENCE` is the default strategy](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/benchmarks/DESIGN.md#why-sequence-is-the-default-strategy) and [Why `KEYED` still exists](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/benchmarks/DESIGN.md#why-keyed-still-exists) in `benchmarks/DESIGN.md`.

---

## Configuration

All collection facades accept these keyword-only options:

```python
normalizer = None
scorer = WRatio
scorer_kwargs = None
scorer_type = ScorerType.SIMILARITY
score_cutoff = 80
score_hint = None
strategy = IndexStrategy.SEQUENCE  # dict/set facades only
```

### Normalization

The normalizer converts a stored object or query into searchable text. Return `None` to exclude a value or query from every fuzzy method, including contains, find-one, find-many, count, and fuzzy mutation methods. The original value remains available through ordinary exact collection operations such as `in`, indexing, or mapping lookup.

The default normalizer:

- accepts strings only;
- strips leading and trailing whitespace;
- applies `casefold()`;
- excludes strings shorter than three characters.

When `normalizer=None`, indexes use an optimized built-in callable with behavior equivalent to `Normalizer.default()`. A `Normalizer` instance can be supplied directly when a custom pipeline is needed. Treat any supplied normalizer as immutable after constructing a collection or index: stored choices are normalized and cached during index maintenance, so later mutation of the callable could make query normalization inconsistent with those cached values.

`Normalizer` builder methods mutate the instance by appending operations. Do not keep configuring an instance after passing it to a collection:

```python
from rapidfuzz_collections import FuzzyList, Normalizer

normalizer = Normalizer().isinstance_str().strip()
products = FuzzyList(["  Keyboard  ", "  Mouse  "], normalizer=normalizer)

# Unsafe: cached choices used the old pipeline, while later queries use the
# mutated pipeline.
normalizer.casefold()
```

Instead, complete the pipeline first and then treat the callable as immutable:

```python
normalizer = Normalizer().isinstance_str().strip().casefold()
products = FuzzyList(["  Keyboard  ", "  Mouse  "], normalizer=normalizer)
# Do not mutate normalizer after this point.
```

The same rule applies to any custom mutable or stateful callable. The caller is responsible for keeping its behavior stable for the lifetime of the collection or index. To change normalization rules, construct a new configured collection, for example with `with_config(normalizer=...)`, instead of mutating the callable already in use.

### Scorers

Native RapidFuzz scorers use its optimized process path. Custom scorers are also supported and are called directly, so they do not need to accept RapidFuzz's internal keyword arguments. `ScorerType` determines their ordering and cutoff semantics. Use `ScorerType.SIMILARITY` when higher scores are better, and `ScorerType.DISTANCE` when lower scores are better. Pass the enum member itself; raw values such as `0`, `1`, or `"distance"` are rejected instead of being interpreted implicitly.

An exact candidate returns immediately only when native RapidFuzz metadata confirms both the configured scorer direction and that candidate's optimal score. Otherwise, the lookup evaluates the candidates required to determine the best score. Exact equality then breaks equal-score ties; it never replaces a better scorer result. A custom scorer without compatible metadata evaluates every searchable candidate, so a non-exact value with a better custom score still wins.

```python
from rapidfuzz.distance import Levenshtein
from rapidfuzz_collections import FuzzyList, ScorerType

words = FuzzyList(
    ["kitten", "sitting", "mitten"],
    scorer=Levenshtein.distance,
    scorer_type=ScorerType.DISTANCE,
    score_cutoff=2,
)
```

### Score cutoffs

`score_cutoff` controls which candidates are accepted:

- for similarity scorers, candidates below the cutoff are rejected;
- for distance scorers, candidates above the cutoff are rejected;
- `None` disables cutoff filtering.

### Score hints

`score_hint` is forwarded to RapidFuzz as an expected score. It can help RapidFuzz choose an internal implementation path, but it does not change the semantic result. Leave it as `None` unless you have measured your workload.

### Scorer kwargs

Use `scorer_kwargs` for scorer-specific options:

```python
from rapidfuzz.distance import Levenshtein
from rapidfuzz_collections import FuzzyList, ScorerType

values = FuzzyList(
    ["kitten", "sitting"],
    scorer=Levenshtein.distance,
    scorer_type=ScorerType.DISTANCE,
    scorer_kwargs={"weights": (1, 1, 2)},
    score_cutoff=None,
)
```

### `with_config`

`with_config(...)` returns a new collection over the same logical data with selected fuzzy options changed. The source collection is not mutated.

```python
strict = FuzzyList(["Alpha Phone", "Beta Tablet"], score_cutoff=95)
permissive = strict.with_config(score_cutoff=60)
```

### Per-query overrides

Every fuzzy lookup method also accepts `scorer`, `scorer_kwargs`, `scorer_type`, `score_cutoff`, and `score_hint` as keyword-only arguments. Passing one of them overrides the collection's default for that single call only; the collection itself is not changed, and the collection's own defaults still apply to every other call:

```python
products = FuzzyList(["Alpha Phone", "Beta Tablet"], score_cutoff=90)

products.fuzzy_find_one("Alpa Phone")  # uses score_cutoff=90
products.fuzzy_find_one("Alpa Phone", score_cutoff=60)  # uses score_cutoff=60, just for this call
```

This is a lighter-weight alternative to `with_config(...)` when only a single query needs different matching behavior, since it avoids building a second collection. Omit an argument to keep using the collection's default; passing `None` for `scorer_kwargs` or `score_cutoff` is a meaningful value (no extra scorer kwargs / no cutoff filtering), not the same as omitting it.

When `scorer` is overridden without `scorer_type`, the score direction is inferred from compatible RapidFuzz metadata. Custom scorers without that metadata must provide `scorer_type` explicitly; otherwise the query raises `ValueError` instead of risking reversed ranking or cutoff semantics.

`normalizer` and `strategy` cannot be overridden per query because they affect how the index itself is built and searched. They remain fixed for the lifetime of a collection. Change them through `with_config(...)`, which builds a new index, or by constructing a new collection.

---

## Batch Lookup and cdist

Use ordinary batch methods first:

```python
products = FuzzyList(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

matches = products.fuzzy_find_one_batch([
    "Alpa Phone",
    "Bta Tablet",
    "Missing",
])
```

Batch methods preserve query order. Top-one and direct retrieval methods return one result per query; multi-match methods return one result list per query. For each query, the same ranking order is applied: scorer quality, exact equality, then collection order.

### Ordinary batch methods

| Collection family | Top-one batch method                            | Many-match batch method                           |
|-------------------|-------------------------------------------------|---------------------------------------------------|
| Sequence          | `fuzzy_find_one_batch`                          | `fuzzy_find_many_batch`                           |
| Set               | `fuzzy_find_one_batch`                          | `fuzzy_find_many_batch`                           |
| Mapping           | `fuzzy_find_key_batch`, `fuzzy_find_item_batch` | `fuzzy_find_keys_batch`, `fuzzy_find_items_batch` |

### Explicit cdist methods

The `*_batch_cdist` methods are advanced opt-in methods. They compute bounded query-by-choice matrix chunks using RapidFuzz `process.cdist`, immediately reduce each query to its top-one result, and return the same semantic result as the ordinary top-one batch methods.

They are useful only after measurement on your workload. The ordinary batch methods are the default because RapidFuzz `extractOne` can prune candidate scoring as it finds strong matches, while `cdist` computes all pairs in each matrix chunk.

| Collection family           | cdist method                  | Result meaning                     |
|-----------------------------|-------------------------------|------------------------------------|
| Sequence                    | `fuzzy_find_one_batch_cdist`  | best value match per query         |
| Set                         | `fuzzy_find_one_batch_cdist`  | best value match per query         |
| Mapping                     | `fuzzy_find_key_batch_cdist`  | best key match per query           |
| Mapping                     | `fuzzy_find_item_batch_cdist` | best key/value match per query     |
| Standalone sequence indexes | `find_one_batch_cdist`        | best indexed value match per query |

Requirements and limits:

- install `rapidfuzz-collections[cdist]`;
- use `IndexStrategy.SEQUENCE` for dict/set facades;
- `IndexStrategy.KEYED` raises `NotImplementedError`;
- custom scorers are adapted for RapidFuzz matrix calls while receiving only the explicitly configured `scorer_kwargs`;
- use RapidFuzz directly if you need the full score matrix.

---

## Mutation and Rebuilds

Mutable collections keep an internal fuzzy index synchronized with exact collection storage.

Top-one fuzzy mutations use the same deterministic resolver as top-one reads: scorer quality is primary, exact equality breaks equal-score ties, and source or insertion order breaks remaining ties. Thus `fuzzy_discard(query)` removes the value that `fuzzy_find_one(query)` would return, while `fuzzy_discard_all` and `fuzzy_retain_all` operate on every candidate that passes the score cutoff.

Some mutations can update the index incrementally. Other mutations mark the index dirty, and the next fuzzy query rebuilds derived lookup state once.

Practical rules:

- Appending to `FuzzyList` is cheap.
- Adding to `FuzzySet` is cheap when the value is new.
- Updating an existing `FuzzyDict` value does not change the fuzzy key index.
- Positional insertions, replacements, and large complex deletions are more likely to require a rebuild.

For measured rebuild cost after incremental deletion, see [Exact shortcuts after incremental deletion](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/benchmarks/DESIGN.md#exact-shortcuts-after-incremental-deletion) in `benchmarks/DESIGN.md`.

---

## Result Objects

### `Match[T]`

Returned by value collections, set collections, mapping key methods, and standalone sequence indexes.

| Field              | Meaning                                         |
|--------------------|-------------------------------------------------|
| `value`            | original matched value or key                   |
| `score`            | RapidFuzz scorer output                         |
| `index`            | source position, or `None` for dict/set facades |
| `query`            | original query object                           |
| `normalized_query` | normalized query text                           |
| `normalized_value` | normalized matched value/key text               |

### `MappingMatch[K, V]`

Returned by mapping item methods.

| Field              | Meaning                           |
|--------------------|-----------------------------------|
| `key`              | original matched mapping key      |
| `value`            | payload stored under that key     |
| `score`            | RapidFuzz scorer output           |
| `index`            | always `None` for mapping facades |
| `query`            | original query object             |
| `normalized_query` | normalized query text             |
| `normalized_key`   | normalized matched key text       |

Scores are scorer-dependent. For `ScorerType.SIMILARITY`, higher is better. For `ScorerType.DISTANCE`, lower is better.

### `ValueMatch[T]` and `KeyValueMatch[K, V]`

Returned by the standalone `ImmutableFuzzyKeyedIndex`/`MutableFuzzyKeyedIndex` classes described in [Advanced Index APIs](#advanced-index-apis). Keyed indexes do not track sequence positions, so these result types have no `index` field at all, rather than `index=None`. Collection facades built on a keyed index adapt these results to `Match`/`MappingMatch` with `index=None`.

`ValueMatch[T]`:

| Field              | Meaning                          |
|--------------------|----------------------------------|
| `value`            | original collection value        |
| `score`            | RapidFuzz scorer output          |
| `query`            | original query object            |
| `normalized_query` | normalized query text            |
| `normalized_value` | normalized form of matched value |

`KeyValueMatch[K, V]`:

| Field              | Meaning                        |
|--------------------|--------------------------------|
| `key`              | original matched mapping key   |
| `value`            | payload stored under that key  |
| `score`            | RapidFuzz scorer output        |
| `query`            | original query object          |
| `normalized_query` | normalized query text          |
| `normalized_key`   | normalized form of matched key |

---

## Public Method Reference

| Method                                     | `FuzzyList` | `FuzzyTuple` | `FuzzySet` | `FrozenFuzzySet` | `FuzzyDict` | `FrozenFuzzyDict` | Notes                                          |
|--------------------------------------------|:-----------:|:------------:|:----------:|:----------------:|:-----------:|:-----------------:|------------------------------------------------|
| `fuzzy_find_one(query)`                    |     yes     |     yes      |    yes     |       yes        |     no      |        no         | best value match                               |
| `fuzzy_find_many(query, limit=5)`          |     yes     |     yes      |    yes     |       yes        |     no      |        no         | best value matches                             |
| `fuzzy_find_index(query)`                  |     yes     |     yes      |     no     |        no        |     no      |        no         | source index of the best value match           |
| `fuzzy_count(query)`                       |     yes     |     yes      |     no     |        no        |     no      |        no         | number of matching sequence values             |
| `fuzzy_get(query, default=None)`           |     yes     |     yes      |    yes     |       yes        |     yes     |        yes        | direct value/payload retrieval                 |
| `fuzzy_get_batch(queries, default=None)`   |     yes     |     yes      |    yes     |       yes        |     yes     |        yes        | direct batch value/payload retrieval           |
| `fuzzy_contains(query)`                    |     yes     |     yes      |    yes     |       yes        |     no      |        no         | fuzzy value containment                        |
| `fuzzy_contains_key(query)`                |     no      |      no      |     no     |        no        |     yes     |        yes        | fuzzy key containment                          |
| `fuzzy_find_key(query)`                    |     no      |      no      |     no     |        no        |     yes     |        yes        | best key match                                 |
| `fuzzy_find_item(query)`                   |     no      |      no      |     no     |        no        |     yes     |        yes        | best key/value match                           |
| `fuzzy_find_keys(query, limit=5)`          |     no      |      no      |     no     |        no        |     yes     |        yes        | best key matches                               |
| `fuzzy_find_items(query, limit=5)`         |     no      |      no      |     no     |        no        |     yes     |        yes        | best key/value matches                         |
| `fuzzy_find_one_batch(queries)`            |     yes     |     yes      |    yes     |       yes        |     no      |        no         | ordinary top-one batch                         |
| `fuzzy_find_many_batch(queries, limit=5)`  |     yes     |     yes      |    yes     |       yes        |     no      |        no         | ordinary many-match batch                      |
| `fuzzy_find_key_batch(queries)`            |     no      |      no      |     no     |        no        |     yes     |        yes        | ordinary key batch                             |
| `fuzzy_find_item_batch(queries)`           |     no      |      no      |     no     |        no        |     yes     |        yes        | ordinary item batch                            |
| `fuzzy_find_keys_batch(queries, limit=5)`  |     no      |      no      |     no     |        no        |     yes     |        yes        | ordinary many-key batch                        |
| `fuzzy_find_items_batch(queries, limit=5)` |     no      |      no      |     no     |        no        |     yes     |        yes        | ordinary many-item batch                       |
| `fuzzy_find_one_batch_cdist(queries)`      |     yes     |     yes      |    yes     |       yes        |     no      |        no         | advanced top-one batch, sequence strategy only |
| `fuzzy_find_key_batch_cdist(queries)`      |     no      |      no      |     no     |        no        |     yes     |        yes        | advanced key batch, sequence strategy only     |
| `fuzzy_find_item_batch_cdist(queries)`     |     no      |      no      |     no     |        no        |     yes     |        yes        | advanced item batch, sequence strategy only    |
| `fuzzy_score_all(query)`                   |     yes     |     yes      |    yes     |       yes        |     yes     |        yes        | one score slot per stored item/key             |
| `fuzzy_iter_scores(query)`                 |     yes     |     yes      |    yes     |       yes        |     yes     |        yes        | streaming score slots                          |
| `fuzzy_discard(query)`                     |     yes     |      no      |    yes     |        no        |     yes     |        no         | remove best fuzzy match                        |
| `fuzzy_remove(query)`                      |     yes     |      no      |     no     |        no        |     no      |        no         | list-only remove with error on miss            |
| `fuzzy_discard_all(query)`                 |     yes     |      no      |    yes     |        no        |     yes     |        no         | remove all fuzzy matches                       |
| `fuzzy_retain_all(query)`                  |     yes     |      no      |    yes     |        no        |     yes     |        no         | keep only fuzzy matches                        |
| `with_config(**overrides)`                 |     yes     |     yes      |    yes     |       yes        |     yes     |        yes        | return reconfigured collection                 |
| `fromkeys(keys, value=None, **config)`     |     no      |      no      |     no     |        no        |     yes     |        yes        | mapping factory                                |

---

## Performance Guidance

Start with the defaults:

- default normalizer;
- `WRatio` scorer;
- `score_cutoff=80`;
- `IndexStrategy.SEQUENCE`;
- ordinary batch methods instead of `cdist`.

Measure before switching:

- Try `IndexStrategy.KEYED` for dict/set workloads dominated by construction cost, normalized collisions, or selected bulk fuzzy mutations. Treat lower memory as a possible frozen-collection benefit, not a general KEYED property.
- Try `*_batch_cdist` only for large batch workloads where scorer choice and query distribution make full matrix scoring worthwhile.
- Try `score_hint` only when a specific scorer/data distribution benefits from it.

Do not assume that lower-level RapidFuzz APIs are always faster through the facades. Collection lookup includes exact-value tie resolution, normalization caches, result adaptation, and mutation state.

For the benchmark data and reasoning behind this guidance, see [Practical strategy rules](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/benchmarks/DESIGN.md#practical-strategy-rules) in `benchmarks/DESIGN.md`.

---

## Advanced Index APIs

Most users should use collection facades. Standalone indexes are available for advanced users who already manage storage separately:

- `FuzzySequenceIndex`
- `MutableFuzzySequenceIndex`
- `ImmutableFuzzyKeyedIndex`
- `MutableFuzzyKeyedIndex`

Use standalone indexes when:

- you do not need a collection facade;
- you can keep exact storage and index storage synchronized yourself;
- you need lower-level access to index lookup behavior.

Do not expose a collection's internal index and mutate it separately. That would desynchronize the collection's exact storage from its fuzzy lookup state.

The keyed index classes return `ValueMatch`/`KeyValueMatch` results; see [`ValueMatch[T]` and `KeyValueMatch[K, V]`](#valuematcht-and-keyvaluematchk-v).

---

## Design Boundaries

This library is intentionally not:

- a full-text search engine;
- a database index;
- a sublinear approximate nearest-neighbor index;
- a replacement for RapidFuzz scorers and distance functions;
- a general matrix-scoring wrapper around `process.cdist`;
- a compatibility layer for historical APIs.

It is a collection-oriented layer over RapidFuzz:

- store original data in familiar Python collection shapes;
- cache normalized lookup data;
- keep fuzzy lookup state synchronized with mutations;
- expose predictable fuzzy result objects.

When in doubt, first choose the collection that matches your data model. Then choose configuration and strategy based on measured workload behavior.

---

## Development Checks

```bash
pip install -e ".[dev]"
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

See [`tests/README.md`](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/tests/README.md) for the test-suite structure and local coverage commands.

---

## Third-party Example Datasets

The repository includes third-party example datasets under `examples/data/` for documentation, examples, and local experimentation.

These datasets are not part of the Python package distribution and are not covered by this project's source code license. See [`examples/data/NOTICE.md`](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/examples/data/NOTICE.md) for dataset sources, licenses, attribution, and modification notes.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/LICENSE).

---

For the design rationale, benchmark methodology, and historical investigation findings behind the index-strategy and performance guidance in this README, see [`benchmarks/DESIGN.md`](https://github.com/igorxut/rapidfuzz-collections/blob/v1.0.0/benchmarks/DESIGN.md).
