"""Tests for opt-in bounded ``cdist`` fuzzy lookup operations."""

import sys

import pytest
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

import rapidfuzz_collections.indexes.sequence_index as sequence_index_module
from rapidfuzz_collections import (
    FrozenFuzzyDict,
    FrozenFuzzySet,
    FuzzyDict,
    FuzzyList,
    FuzzySequenceIndex,
    FuzzySet,
    FuzzyTuple,
    IndexStrategy,
    MutableFuzzySequenceIndex,
    ScorerType,
)
from tests.helpers import (
    SearchableEqualityKey,
    casefold_string,
    normalize_equality_key,
    require_not_none,
)


@pytest.mark.parametrize(
    "collection",
    [
        FuzzyList(["Alpha Phone", "Beta Tablet", "Gamma Camera"], scorer=ratio, score_cutoff=50),
        FuzzyTuple(["Alpha Phone", "Beta Tablet", "Gamma Camera"], scorer=ratio, score_cutoff=50),
        FuzzySet(["Alpha Phone", "Beta Tablet", "Gamma Camera"], scorer=ratio, score_cutoff=50),
        FrozenFuzzySet(["Alpha Phone", "Beta Tablet", "Gamma Camera"], scorer=ratio, score_cutoff=50),
    ],
)
def test_value_facade_cdist_matches_standard_batch_lookup(collection):
    queries = ["Alpha Phone", "  BETA TABLET  ", "Gama Camera", "Missing"]

    assert collection.fuzzy_find_one_batch_cdist(queries, query_chunk_size=2, choice_chunk_size=1) == (
        collection.fuzzy_find_one_batch(queries)
    )


@pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
def test_public_index_cdist_matches_standard_batch_lookup(index_class):
    index = index_class(["Alpha Phone", "Beta Tablet", "Gamma Camera"], scorer=ratio, score_cutoff=50)
    queries = ["Alpha Phone", "Bta Tablet", "Missing"]

    assert index.find_one_batch_cdist(queries, query_chunk_size=2, choice_chunk_size=1) == index.find_one_batch(queries)


@pytest.mark.parametrize(
    "collection",
    [
        FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2, "Gamma Camera": 3}, scorer=ratio, score_cutoff=50),
        FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2, "Gamma Camera": 3}, scorer=ratio, score_cutoff=50),
    ],
)
def test_mapping_facade_cdist_returns_standard_mapping_matches(collection):
    queries = ["Alpha Phone", "  BETA TABLET  ", "Gama Camera", "Missing"]

    assert collection.fuzzy_find_item_batch_cdist(queries, query_chunk_size=2, choice_chunk_size=1) == (
        collection.fuzzy_find_item_batch(queries)
    )


@pytest.mark.parametrize(
    "collection, method_name",
    [
        (FuzzyDict({"Alpha Phone": 1}, strategy=IndexStrategy.KEYED), "fuzzy_find_key_batch_cdist"),
        (FuzzyDict({"Alpha Phone": 1}, strategy=IndexStrategy.KEYED), "fuzzy_find_item_batch_cdist"),
        (FrozenFuzzyDict({"Alpha Phone": 1}, strategy=IndexStrategy.KEYED), "fuzzy_find_key_batch_cdist"),
        (FrozenFuzzyDict({"Alpha Phone": 1}, strategy=IndexStrategy.KEYED), "fuzzy_find_item_batch_cdist"),
        (FuzzySet(["Alpha Phone"], strategy=IndexStrategy.KEYED), "fuzzy_find_one_batch_cdist"),
        (FrozenFuzzySet(["Alpha Phone"], strategy=IndexStrategy.KEYED), "fuzzy_find_one_batch_cdist"),
    ],
)
def test_cdist_lookup_rejects_keyed_strategy(collection, method_name):
    method = getattr(collection, method_name)

    with pytest.raises(NotImplementedError, match="cdist is not available for IndexStrategy.KEYED"):
        method(["Alpa Phone"])


def test_cdist_lookup_rebuilds_dirty_mutable_index_before_scoring():
    values = FuzzyList(["Alpha Phone", "Gamma Camera"], scorer=ratio, score_cutoff=50)
    values.insert(1, "Beta Tablet")

    assert values.fuzzy_find_one_batch_cdist(["Bta Tablet"], choice_chunk_size=1) == values.fuzzy_find_one_batch(
        ["Bta Tablet"]
    )


def test_cdist_lookup_uses_current_shortcuts_after_dense_fast_delete():
    values = FuzzyList(["Alpha Phone", "  alpha phone  ", "Beta Tablet"], scorer=ratio, score_cutoff=50)
    del values[0]

    assert not values._index.is_dirty
    assert values.fuzzy_find_one_batch_cdist(["ALPHA PHONE"]) == values.fuzzy_find_one_batch(["ALPHA PHONE"])


def test_cdist_lookup_translates_sparse_source_slots_after_fast_delete():
    values = FuzzyList(["xy", "Alpha Phone", "Beta Tablet"], scorer=ratio, score_cutoff=50)
    del values[1]

    accelerated = values.fuzzy_find_one_batch_cdist(["Bta Tablet"], choice_chunk_size=1)

    assert not values._index.is_dirty
    assert accelerated == values.fuzzy_find_one_batch(["Bta Tablet"])
    first_match = require_not_none(accelerated[0])
    assert first_match.index == 1


def test_cdist_lookup_preserves_distance_cutoff_and_source_order_ties():
    values = FuzzyList(
        ["cat", "cut", "cot"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=1,
    )

    assert values.fuzzy_find_one_batch_cdist(["cit", "unrelated"], choice_chunk_size=1) == (
        values.fuzzy_find_one_batch(["cit", "unrelated"])
    )
    match = require_not_none(values.fuzzy_find_one_batch_cdist(["cit"], choice_chunk_size=1)[0])
    assert match.index == 0


def test_cdist_lookup_uses_exact_equality_to_break_score_tie():
    values = FuzzyList(["ALPHA", "alpha"], normalizer=casefold_string, scorer=ratio, score_cutoff=0)

    standard = values.fuzzy_find_one_batch(["alpha"])
    accelerated = values.fuzzy_find_one_batch_cdist(["alpha"], choice_chunk_size=1)

    assert accelerated == standard
    assert require_not_none(accelerated[0]).value == "alpha"


def test_cdist_custom_scorer_uses_exact_equality_to_break_score_tie():
    def equal_score(_query: str, _choice: str) -> int:
        return 50

    values = FuzzyList(
        ["ALPHA", "alpha"],
        normalizer=casefold_string,
        scorer=equal_score,
        score_cutoff=0,
    )

    match = require_not_none(values.fuzzy_find_one_batch_cdist(["alpha"], choice_chunk_size=1)[0])

    assert match.value == "alpha"


def test_cdist_custom_scorer_evaluates_all_choices_then_rescores_selected_pair():
    calls = 0

    def counting_similarity(_query: str, _choice: str) -> int:
        nonlocal calls
        calls += 1
        return 50

    values = FuzzyList(
        ["Alpha Phone", "Beta Tablet", "Gamma Camera"],
        scorer=counting_similarity,
        score_cutoff=0,
    )

    values.fuzzy_find_one_batch_cdist(["unindexed query"])

    assert calls == 4


def test_cdist_ignores_equal_source_value_rejected_during_index_build():
    indexed = SearchableEqualityKey("shared", None)
    searchable = SearchableEqualityKey("other", "other")
    query = SearchableEqualityKey("shared", "shared")
    values = FuzzyList(
        [indexed, searchable],
        normalizer=normalize_equality_key,
        scorer=ratio,
        score_cutoff=0,
    )

    match = require_not_none(values.fuzzy_find_one_batch_cdist([query], choice_chunk_size=1)[0])

    assert match.value is searchable


def test_cdist_lookup_forwards_configured_score_hint(monkeypatch):
    values = FuzzyList(["Alpha Phone"], scorer=ratio, score_cutoff=50, score_hint=100)
    original_cdist = sequence_index_module.process.cdist
    observed_hints: list[int | float | None] = []

    def recording_cdist(*args, **kwargs):
        observed_hints.append(kwargs["score_hint"])
        return original_cdist(*args, **kwargs)

    monkeypatch.setattr(sequence_index_module.process, "cdist", recording_cdist)

    require_not_none(values.fuzzy_find_one_batch_cdist(["Alpa Phone"])[0])
    assert observed_hints == [100]


@pytest.mark.parametrize("workers", [1, -1, 4])
def test_cdist_lookup_forwards_configured_workers(monkeypatch, workers):
    # -1 is RapidFuzz's documented sentinel for "use all available CPU cores";
    # this must pass through untouched rather than being rejected or clamped.
    values = FuzzyList(["Alpha Phone"], scorer=ratio, score_cutoff=50)
    original_cdist = sequence_index_module.process.cdist
    observed_workers: list[int] = []

    def recording_cdist(*args, **kwargs):
        observed_workers.append(kwargs["workers"])
        return original_cdist(*args, **kwargs)

    monkeypatch.setattr(sequence_index_module.process, "cdist", recording_cdist)

    require_not_none(values.fuzzy_find_one_batch_cdist(["Alpa Phone"], workers=workers)[0])
    assert observed_workers == [workers]


def test_cdist_lookup_preserves_close_float_score_ordering():
    # noinspection PyUnusedLocal
    def scorer(_query, choice, *, score_cutoff=None):
        del score_cutoff
        return {"first value": 90.00000001, "second value": 90.00000002}[choice]

    values = FuzzyList(["First Value", "Second Value"], scorer=scorer, score_cutoff=80)

    assert values.fuzzy_find_one_batch_cdist(["different query"], choice_chunk_size=1) == (
        values.fuzzy_find_one_batch(["different query"])
    )
    match = require_not_none(values.fuzzy_find_one_batch_cdist(["different query"], choice_chunk_size=1)[0])
    assert match.value == "Second Value"


@pytest.mark.parametrize("parameter", ["query_chunk_size", "choice_chunk_size"])
def test_cdist_lookup_rejects_non_positive_chunk_sizes(parameter):
    kwargs = {parameter: 0}

    with pytest.raises(ValueError, match="must be greater than 0"):
        FuzzyList(["Alpha Phone"]).fuzzy_find_one_batch_cdist(["Alpa Phone"], **kwargs)


@pytest.mark.parametrize("parameter", ["query_chunk_size", "choice_chunk_size"])
def test_cdist_lookup_rejects_non_integer_chunk_sizes(parameter):
    kwargs = {parameter: True}

    with pytest.raises(TypeError, match="must be an integer"):
        # noinspection PyTypeChecker
        FuzzyList(["Alpha Phone"]).fuzzy_find_one_batch_cdist(["Alpa Phone"], **kwargs)


@pytest.mark.parametrize(
    ("collection", "method_name"),
    [
        (FuzzyDict({"Alpha Phone": 1}), "fuzzy_find_key_batch_cdist"),
        (FuzzyDict({"Alpha Phone": 1}), "fuzzy_find_item_batch_cdist"),
        (FrozenFuzzyDict({"Alpha Phone": 1}), "fuzzy_find_key_batch_cdist"),
        (FrozenFuzzyDict({"Alpha Phone": 1}), "fuzzy_find_item_batch_cdist"),
        (FuzzySet(["Alpha Phone"]), "fuzzy_find_one_batch_cdist"),
        (FrozenFuzzySet(["Alpha Phone"]), "fuzzy_find_one_batch_cdist"),
    ],
)
@pytest.mark.parametrize("parameter", ["query_chunk_size", "choice_chunk_size"])
def test_facade_cdist_rejects_non_positive_chunk_sizes(collection, method_name, parameter):
    kwargs = {parameter: 0}
    method = getattr(collection, method_name)

    with pytest.raises(ValueError, match="must be greater than 0"):
        method(["Alpa Phone"], **kwargs)


@pytest.mark.parametrize(
    ("collection", "method_name"),
    [
        (FuzzyDict({"Alpha Phone": 1}), "fuzzy_find_key_batch_cdist"),
        (FuzzyDict({"Alpha Phone": 1}), "fuzzy_find_item_batch_cdist"),
        (FrozenFuzzyDict({"Alpha Phone": 1}), "fuzzy_find_key_batch_cdist"),
        (FrozenFuzzyDict({"Alpha Phone": 1}), "fuzzy_find_item_batch_cdist"),
        (FuzzySet(["Alpha Phone"]), "fuzzy_find_one_batch_cdist"),
        (FrozenFuzzySet(["Alpha Phone"]), "fuzzy_find_one_batch_cdist"),
    ],
)
@pytest.mark.parametrize("parameter", ["query_chunk_size", "choice_chunk_size"])
def test_facade_cdist_rejects_non_integer_chunk_sizes(collection, method_name, parameter):
    kwargs = {parameter: True}
    method = getattr(collection, method_name)

    with pytest.raises(TypeError, match="must be an integer"):
        # noinspection PyTypeChecker
        method(["Alpa Phone"], **kwargs)


def test_cdist_lookup_reports_optional_numpy_dependency(monkeypatch):
    values = FuzzyList(["Alpha Phone"], scorer=ratio, score_cutoff=50)
    monkeypatch.setitem(sys.modules, "numpy", None)

    with pytest.raises(ModuleNotFoundError, match=r"rapidfuzz-collections\[cdist\]"):
        values.fuzzy_find_one_batch_cdist(["Alpa Phone"])


def test_cdist_lookup_non_normalizable_query_returns_none_without_numpy(monkeypatch):
    values = FuzzyList(["Alpha Phone", "Beta Tablet"], scorer=ratio, score_cutoff=50)
    monkeypatch.setitem(sys.modules, "numpy", None)

    assert values.fuzzy_find_one_batch_cdist([42]) == [None]


def test_cdist_lookup_unhashable_query_returns_none():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"], scorer=ratio, score_cutoff=50)

    assert values.fuzzy_find_one_batch_cdist([[1, 2, 3]]) == [None]


def test_cdist_forwards_scorer_kwargs_as_dict_not_unpacked():
    # RapidFuzz expects scorer options under `scorer_kwargs`, not expanded as
    # top-level keyword arguments.
    values = FuzzyList(
        ["kitten", "sitting"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=10,
        scorer_kwargs={"weights": (1, 1, 1)},
    )

    result = values.fuzzy_find_one_batch_cdist(["kitten"])
    match = require_not_none(result[0])
    assert match.value == "kitten"
    assert match.score == 0
