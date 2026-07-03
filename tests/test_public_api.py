import inspect
from typing import cast

import pytest

import rapidfuzz_collections as rfc
from rapidfuzz_collections import (
    FrozenFuzzyDict,
    FrozenFuzzySet,
    FuzzyDict,
    FuzzyList,
    FuzzySequenceIndex,
    FuzzySet,
    FuzzyTuple,
    ImmutableFuzzyKeyedIndex,
    IndexStrategy,
    MutableFuzzyKeyedIndex,
    MutableFuzzySequenceIndex,
    Normalizer,
    ScorerType,
)
from tests.helpers import public_methods


def test_package_exports_expected_public_names():
    assert rfc.__all__ == [
        "FrozenFuzzyDict",
        "FrozenFuzzySet",
        "FuzzyDict",
        "FuzzyList",
        "FuzzySequenceIndex",
        "FuzzySet",
        "FuzzyTuple",
        "ImmutableFuzzyKeyedIndex",
        "IndexStrategy",
        "KeyValueMatch",
        "MappingMatch",
        "Match",
        "MutableFuzzyKeyedIndex",
        "MutableFuzzySequenceIndex",
        "Normalizer",
        "ScorerType",
        "ValueMatch",
    ]

    for name in rfc.__all__:
        assert hasattr(rfc, name)

    assert hasattr(rfc, "FuzzySequenceIndex")
    assert hasattr(rfc, "MutableFuzzySequenceIndex")
    assert isinstance(IndexStrategy.SEQUENCE, IndexStrategy)
    assert not hasattr(rfc, "FuzzyKeyedDict")
    assert not hasattr(rfc, "FuzzyKeyedSet")
    assert not hasattr(rfc, "FuzzySequenceDict")
    assert not hasattr(rfc, "FuzzySequenceSet")
    assert not hasattr(rfc, "FrozenFuzzySequenceIndex")
    assert not hasattr(rfc, "fuzzy_find_one_batch")


def test_collection_facades_do_not_expose_internal_indexes():
    assert not hasattr(FuzzyList(), "fuzzy_index")
    assert not hasattr(FuzzyTuple(), "fuzzy_index")
    assert not hasattr(FuzzySet(), "fuzzy_index")
    assert not hasattr(FrozenFuzzySet(), "fuzzy_index")
    assert not hasattr(FuzzyDict(), "fuzzy_key_index")
    assert not hasattr(FrozenFuzzyDict(), "fuzzy_key_index")
    assert not hasattr(FuzzyList(), "search_index")
    assert not hasattr(FuzzyTuple(), "search_index")
    assert not hasattr(FuzzySet(), "search_index")
    assert not hasattr(FrozenFuzzySet(), "search_index")
    assert not hasattr(FuzzyDict(), "key_search_index")
    assert not hasattr(FrozenFuzzyDict(), "key_search_index")
    assert not hasattr(FuzzySet(strategy=IndexStrategy.KEYED), "search_index")
    assert not hasattr(FuzzyDict(strategy=IndexStrategy.KEYED), "key_search_index")


def test_runtime_public_objects_do_not_have_instance_dicts():
    assert not hasattr(FuzzyList(), "__dict__")
    assert not hasattr(FuzzyTuple(), "__dict__")
    assert not hasattr(FuzzySet(), "__dict__")
    assert not hasattr(FrozenFuzzySet(), "__dict__")
    assert not hasattr(FuzzyDict(), "__dict__")
    assert not hasattr(FrozenFuzzyDict(), "__dict__")
    assert not hasattr(FuzzySet(strategy=IndexStrategy.KEYED), "__dict__")
    assert not hasattr(FuzzyDict(strategy=IndexStrategy.KEYED), "__dict__")
    assert not hasattr(Normalizer(), "__dict__")
    assert not hasattr(FuzzySequenceIndex(), "__dict__")
    assert not hasattr(ImmutableFuzzyKeyedIndex(), "__dict__")
    assert not hasattr(MutableFuzzyKeyedIndex(), "__dict__")
    assert not hasattr(MutableFuzzySequenceIndex(), "__dict__")


def test_sequence_index_public_methods():
    expected_methods = {
        "config_kwargs",
        "contains",
        "find_many",
        "find_many_batch",
        "find_one",
        "find_one_batch",
        "find_one_batch_cdist",
        "iter_scores",
        "normalize",
        "score_all",
    }

    assert public_methods(FuzzySequenceIndex) == expected_methods
    assert public_methods(MutableFuzzySequenceIndex) == expected_methods | {
        "append",
        "delete_at",
        "delete_at_positions",
        "delete_value",
        "insert_at",
        "keep_at_positions",
        "replace_at",
        "sort",
    }


def test_fuzzy_dict_public_methods():
    assert public_methods(FuzzyDict) == {
        "clear",
        "copy",
        "fromkeys",
        "fuzzy_contains_key",
        "fuzzy_discard",
        "fuzzy_discard_all",
        "fuzzy_find_item",
        "fuzzy_find_item_batch",
        "fuzzy_find_item_batch_cdist",
        "fuzzy_find_items",
        "fuzzy_find_items_batch",
        "fuzzy_find_key",
        "fuzzy_find_key_batch",
        "fuzzy_find_key_batch_cdist",
        "fuzzy_find_keys",
        "fuzzy_find_keys_batch",
        "fuzzy_get",
        "fuzzy_get_batch",
        "fuzzy_iter_scores",
        "fuzzy_retain_all",
        "fuzzy_score_all",
        "get",
        "items",
        "keys",
        "pop",
        "popitem",
        "setdefault",
        "update",
        "values",
        "with_config",
    }


def test_fuzzy_list_public_methods():
    assert public_methods(FuzzyList) == {
        "append",
        "clear",
        "copy",
        "count",
        "extend",
        "fuzzy_contains",
        "fuzzy_count",
        "fuzzy_discard",
        "fuzzy_discard_all",
        "fuzzy_find_index",
        "fuzzy_find_many",
        "fuzzy_find_many_batch",
        "fuzzy_find_one",
        "fuzzy_find_one_batch",
        "fuzzy_find_one_batch_cdist",
        "fuzzy_get",
        "fuzzy_get_batch",
        "fuzzy_iter_scores",
        "fuzzy_remove",
        "fuzzy_retain_all",
        "fuzzy_score_all",
        "index",
        "insert",
        "pop",
        "remove",
        "reverse",
        "sort",
        "with_config",
    }


def test_fuzzy_set_public_methods():
    assert public_methods(FuzzySet) == {
        "add",
        "clear",
        "copy",
        "difference",
        "difference_update",
        "discard",
        "fuzzy_contains",
        "fuzzy_discard",
        "fuzzy_discard_all",
        "fuzzy_find_many",
        "fuzzy_find_many_batch",
        "fuzzy_find_one",
        "fuzzy_find_one_batch",
        "fuzzy_find_one_batch_cdist",
        "fuzzy_get",
        "fuzzy_get_batch",
        "fuzzy_iter_scores",
        "fuzzy_retain_all",
        "fuzzy_score_all",
        "intersection",
        "intersection_update",
        "isdisjoint",
        "issubset",
        "issuperset",
        "pop",
        "remove",
        "symmetric_difference",
        "symmetric_difference_update",
        "union",
        "update",
        "with_config",
    }


def test_fuzzy_tuple_public_methods():
    assert public_methods(FuzzyTuple) == {
        "copy",
        "count",
        "fuzzy_contains",
        "fuzzy_count",
        "fuzzy_find_index",
        "fuzzy_find_many",
        "fuzzy_find_many_batch",
        "fuzzy_find_one",
        "fuzzy_find_one_batch",
        "fuzzy_find_one_batch_cdist",
        "fuzzy_get",
        "fuzzy_get_batch",
        "fuzzy_iter_scores",
        "fuzzy_score_all",
        "index",
        "with_config",
    }


def test_frozen_fuzzy_set_public_methods():
    assert public_methods(FrozenFuzzySet) == {
        "copy",
        "difference",
        "fuzzy_contains",
        "fuzzy_find_many",
        "fuzzy_find_many_batch",
        "fuzzy_find_one",
        "fuzzy_find_one_batch",
        "fuzzy_find_one_batch_cdist",
        "fuzzy_get",
        "fuzzy_get_batch",
        "fuzzy_iter_scores",
        "fuzzy_score_all",
        "intersection",
        "isdisjoint",
        "issubset",
        "issuperset",
        "symmetric_difference",
        "union",
        "with_config",
    }


def test_normalizer_public_methods():
    assert public_methods(Normalizer) == {
        "capitalize",
        "casefold",
        "custom",
        "default",
        "endswith",
        "exact_length",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "isinstance_str",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        "lower",
        "lstrip",
        "max_length",
        "min_length",
        "not_empty_str",
        "re_sub",
        "removeprefix",
        "removesuffix",
        "replace",
        "rstrip",
        "startswith",
        "strip",
        "upper",
    }


def test_frozen_fuzzy_dict_public_methods():
    assert public_methods(FrozenFuzzyDict) == {
        "copy",
        "fromkeys",
        "fuzzy_contains_key",
        "fuzzy_find_item",
        "fuzzy_find_item_batch",
        "fuzzy_find_item_batch_cdist",
        "fuzzy_find_items",
        "fuzzy_find_items_batch",
        "fuzzy_find_key",
        "fuzzy_find_key_batch",
        "fuzzy_find_key_batch_cdist",
        "fuzzy_find_keys",
        "fuzzy_find_keys_batch",
        "fuzzy_get",
        "fuzzy_get_batch",
        "fuzzy_iter_scores",
        "fuzzy_score_all",
        "get",
        "items",
        "keys",
        "values",
        "with_config",
    }


def test_batch_method_signatures():
    overridable = ("scorer", "scorer_kwargs", "scorer_type", "score_cutoff", "score_hint")

    for method, default_name, default_type in (
        (FuzzyList.fuzzy_get_batch, "default", "T"),
        (FuzzyDict.fuzzy_get_batch, "default", "V"),
    ):
        signature = inspect.signature(method)
        parameters = signature.parameters

        assert list(parameters) == ["self", "queries", default_name, *overridable]
        assert str(parameters["queries"].annotation) == "collections.abc.Iterable[object]"
        assert str(parameters[default_name].annotation) == f"{default_type} | None"
        assert parameters[default_name].default is None
        assert str(signature.return_annotation) == f"list[{default_type} | None]"

        for name in overridable:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is not inspect.Parameter.empty
            assert parameters[name].default is not None
        override_defaults = [parameters[name].default for name in overridable]
        assert all(default is override_defaults[0] for default in override_defaults)


def test_score_hint_is_part_of_public_configuration_api():
    facades = (
        FuzzyDict,
        FuzzyList,
        FuzzySet,
        FuzzyTuple,
        FrozenFuzzySet,
        FrozenFuzzyDict,
    )

    for facade in facades:
        assert "score_hint" in inspect.signature(facade).parameters
        assert "score_hint" in inspect.signature(facade.with_config).parameters

    assert "score_hint" in inspect.signature(FrozenFuzzyDict.fromkeys).parameters
    assert "score_hint" in inspect.signature(FuzzyDict.fromkeys).parameters


def test_per_query_overrides_are_keyword_only_on_representative_facade_methods():
    overridable = ("scorer", "scorer_kwargs", "scorer_type", "score_cutoff", "score_hint")

    representative_methods = (
        FuzzyList.fuzzy_find_one,
        FuzzyTuple.fuzzy_find_one,
        FuzzySet.fuzzy_find_one,
        FrozenFuzzySet.fuzzy_find_one,
        FuzzyDict.fuzzy_find_key,
        FrozenFuzzyDict.fuzzy_find_key,
    )

    for method in representative_methods:
        parameters = inspect.signature(method).parameters

        for name in overridable:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is not inspect.Parameter.empty
            assert parameters[name].default is not None
        override_defaults = [parameters[name].default for name in overridable]
        assert all(default is override_defaults[0] for default in override_defaults)


def test_index_strategy_is_part_of_dict_set_configuration_api():
    strategy_facades = (
        FuzzyDict,
        FuzzySet,
        FrozenFuzzyDict,
        FrozenFuzzySet,
    )

    for facade in strategy_facades:
        assert "strategy" in inspect.signature(facade).parameters
        assert "strategy" in inspect.signature(facade.with_config).parameters

    assert "strategy" in inspect.signature(FuzzyDict.fromkeys).parameters
    assert "strategy" in inspect.signature(FrozenFuzzyDict.fromkeys).parameters


@pytest.mark.parametrize(
    "facade, values",
    [
        (FuzzyDict, {"Alpha Phone": 1}),
        (FuzzySet, ["Alpha Phone"]),
        (FrozenFuzzyDict, {"Alpha Phone": 1}),
        (FrozenFuzzySet, ["Alpha Phone"]),
    ],
)
def test_invalid_index_strategy_is_rejected_by_constructors(facade, values):
    with pytest.raises(ValueError, match="strategy must be IndexStrategy.SEQUENCE or IndexStrategy.KEYED"):
        facade(values, strategy="invalid")


@pytest.mark.parametrize(
    "collection",
    [
        FuzzyDict({"Alpha Phone": 1}),
        FuzzySet(["Alpha Phone"]),
        FrozenFuzzyDict({"Alpha Phone": 1}),
        FrozenFuzzySet(["Alpha Phone"]),
    ],
)
def test_invalid_index_strategy_is_rejected_by_with_config(collection):
    with pytest.raises(ValueError, match="strategy must be IndexStrategy.SEQUENCE or IndexStrategy.KEYED"):
        collection.with_config(strategy="invalid")


@pytest.mark.parametrize(
    "constructor, values",
    [
        (FuzzyList, ["Alpha Phone"]),
        (FuzzyTuple, ["Alpha Phone"]),
        (FuzzyDict, {"Alpha Phone": 1}),
        (FuzzySet, ["Alpha Phone"]),
        (FrozenFuzzyDict, {"Alpha Phone": 1}),
        (FrozenFuzzySet, ["Alpha Phone"]),
        (FuzzySequenceIndex, ["Alpha Phone"]),
        (MutableFuzzySequenceIndex, ["Alpha Phone"]),
        (ImmutableFuzzyKeyedIndex, ["Alpha Phone"]),
        (MutableFuzzyKeyedIndex, ["Alpha Phone"]),
    ],
)
@pytest.mark.parametrize("invalid_scorer_type", [0, 1, "distance", None])
def test_invalid_scorer_type_is_rejected_by_constructors(constructor, values, invalid_scorer_type):
    scorer_type = cast(ScorerType, cast(object, invalid_scorer_type))

    with pytest.raises(
        TypeError,
        match="scorer_type must be ScorerType.DISTANCE or ScorerType.SIMILARITY",
    ):
        constructor(values, scorer_type=scorer_type)


@pytest.mark.parametrize(
    "collection",
    [
        FuzzyList(["Alpha Phone"]),
        FuzzyTuple(["Alpha Phone"]),
        FuzzyDict({"Alpha Phone": 1}),
        FuzzySet(["Alpha Phone"]),
        FrozenFuzzyDict({"Alpha Phone": 1}),
        FrozenFuzzySet(["Alpha Phone"]),
    ],
)
def test_invalid_scorer_type_is_rejected_by_with_config(collection):
    scorer_type = cast(ScorerType, cast(object, ScorerType.DISTANCE.value))

    with pytest.raises(
        TypeError,
        match="scorer_type must be ScorerType.DISTANCE or ScorerType.SIMILARITY",
    ):
        collection.with_config(scorer_type=scorer_type)
