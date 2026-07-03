"""Compatibility checks comparing ``FrozenFuzzyDict`` against the builtin ``dict``."""

from collections import UserDict
from collections.abc import Hashable
from copy import copy, deepcopy
from pickle import dumps, loads

import pytest

from rapidfuzz_collections import FrozenFuzzyDict, FuzzyDict


def test_frozen_fuzzy_dict_union_accepts_arbitrary_mapping():
    values = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    result = values | UserDict({"b": "two"})

    assert isinstance(result, FrozenFuzzyDict)
    assert dict(result) == {"Alpha Phone": 1, "b": "two"}
    assert result.fuzzy_get("Alpha Phone") == 1
    assert result.fuzzy_get("Alpa Phone") is None


def test_frozen_fuzzy_dict_keys_view_is_reversible_and_has_mapping():
    values = FrozenFuzzyDict({"a": 1, "b": 2})

    # noinspection PyTypeChecker
    assert list(reversed(values.keys())) == ["b", "a"]
    # noinspection PyUnresolvedReferences
    assert values.keys().mapping == values._data


def test_frozen_fuzzy_dict_items_view_is_reversible_and_has_mapping():
    values = FrozenFuzzyDict({"a": 1, "b": 2})

    # noinspection PyTypeChecker
    assert list(reversed(values.items())) == [("b", 2), ("a", 1)]
    # noinspection PyUnresolvedReferences
    assert values.items().mapping == values._data


def test_frozen_fuzzy_dict_values_view_is_reversible():
    values = FrozenFuzzyDict({"a": 1, "b": 2})

    # noinspection PyTypeChecker
    assert list(reversed(values.values())) == [2, 1]


def test_frozen_fuzzy_dict_or_rejects_non_dict_iterable_of_pairs():
    values = FrozenFuzzyDict({"a": 1})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values | [("b", 2)]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = [("b", 2)] | values
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = {"a": 1} | [("b", 2)]


def test_frozen_fuzzy_dict_and_fuzzy_dict_support_union():
    values = FrozenFuzzyDict({"a": 1})
    mutable = FuzzyDict({"b": 2})

    frozen_result = values | mutable
    mutable_result = mutable | values

    assert isinstance(frozen_result, FrozenFuzzyDict)
    assert isinstance(mutable_result, FuzzyDict)
    assert dict(frozen_result) == {"a": 1, "b": 2}
    assert dict(mutable_result) == {"b": 2, "a": 1}


def test_frozen_fuzzy_dict_constructor_stays_permissive_for_iterable_of_pairs():
    values = FrozenFuzzyDict([("a", 1), ("b", 2)])
    builtin = dict([("a", 1), ("b", 2)])

    assert dict(values) == builtin


def test_frozen_fuzzy_dict_or_with_dict_matches_builtin_semantics():
    values = FrozenFuzzyDict({"a": 1})
    builtin = {"a": 1}

    result = values | {"b": 2}
    builtin_result = builtin | {"b": 2}

    # noinspection PyTypeChecker
    assert dict(result) == builtin_result


def test_frozen_fuzzy_dict_construction_matches_builtin_for_equivalent_mapping():
    source = {"a": 1, "b": 2}

    assert dict(FrozenFuzzyDict(source)) == source


def test_frozen_fuzzy_dict_len_and_membership_match_builtin():
    source = {"a": 1, "b": 2}
    values = FrozenFuzzyDict(source)

    assert len(values) == len(source)
    assert ("a" in values) == ("a" in source)
    assert ("z" in values) == ("z" in source)


def test_frozen_fuzzy_dict_iteration_order_matches_builtin_insertion_order():
    source = {"a": 1, "b": 2, "c": 3}

    assert list(FrozenFuzzyDict(source)) == list(source)


def test_frozen_fuzzy_dict_equality_against_builtin_dict():
    source = {"a": 1, "b": 2}

    assert FrozenFuzzyDict(source) == source
    assert source == FrozenFuzzyDict(source)


def test_frozen_fuzzy_dict_is_not_hashable_like_builtin_dict():
    values = FrozenFuzzyDict({"a": 1})

    assert not isinstance(values, Hashable)
    assert not isinstance({"a": 1}, Hashable)


def test_frozen_fuzzy_dict_copy_and_deepcopy_round_trip():
    values = FrozenFuzzyDict({"a": 1, "b": 2}, score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FrozenFuzzyDict)
        assert dict(result) == dict(values)


def test_frozen_fuzzy_dict_pickle_round_trip_preserves_content_and_fuzzy_lookup():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    restored = loads(dumps(values))

    assert isinstance(restored, FrozenFuzzyDict)
    assert dict(restored) == dict(values)
    assert restored.fuzzy_get("Alpa Phone") == 1


def test_frozen_fuzzy_dict_repr_does_not_crash():
    assert repr(FrozenFuzzyDict({"a": 1}))
