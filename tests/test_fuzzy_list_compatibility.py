"""Compatibility checks comparing ``FuzzyList`` against the builtin ``list``."""

from collections.abc import Hashable
from copy import copy, deepcopy
from pickle import dumps, loads

import pytest

from rapidfuzz_collections import FuzzyList


class RepeatCount:
    """Provide a noninteger repetition count through ``__index__``."""

    def __index__(self) -> int:
        return 2


def test_fuzzy_list_reflected_addition_preserves_fuzzy_type_and_config():
    values = FuzzyList(["Alpha Phone"], score_cutoff=100)

    result = ["Beta Tablet"] + values

    assert isinstance(result, FuzzyList)
    assert list(result) == ["Beta Tablet", "Alpha Phone"]
    assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
    assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_reflected_addition_rejects_non_list():
    values = FuzzyList([2])

    # noinspection PyTypeChecker
    assert values.__radd__((1,)) is NotImplemented


def test_fuzzy_list_repetition_accepts_supports_index():
    values = FuzzyList([1, 2])

    assert list(values * RepeatCount()) == [1, 2, 1, 2]
    assert list(RepeatCount() * values) == [1, 2, 1, 2]

    values *= RepeatCount()
    assert list(values) == [1, 2, 1, 2]


def test_fuzzy_list_lt_matches_builtin_lexicographic_ordering():
    assert (FuzzyList([1, 2]) < FuzzyList([1, 3])) == ([1, 2] < [1, 3]) is True
    assert (FuzzyList([1, 2]) < [1, 3]) is True
    assert (FuzzyList([1, 3]) < FuzzyList([1, 2])) is False


def test_fuzzy_list_le_matches_builtin_lexicographic_ordering():
    assert (FuzzyList([1, 2]) <= FuzzyList([1, 2])) == ([1, 2] <= [1, 2]) is True
    assert (FuzzyList([1, 3]) <= [1, 2]) is False


def test_fuzzy_list_gt_matches_builtin_lexicographic_ordering():
    assert (FuzzyList([1, 3]) > FuzzyList([1, 2])) == ([1, 3] > [1, 2]) is True
    assert (FuzzyList([1, 2]) > [1, 3]) is False


def test_fuzzy_list_ge_matches_builtin_lexicographic_ordering():
    assert (FuzzyList([1, 2]) >= FuzzyList([1, 2])) == ([1, 2] >= [1, 2]) is True
    assert (FuzzyList([1, 2]) >= [1, 3]) is False


def test_fuzzy_list_comparisons_against_unrelated_type_return_not_implemented():
    values = FuzzyList([1, 2])

    assert values.__lt__(("x",)) is NotImplemented
    assert values.__le__(("x",)) is NotImplemented
    assert values.__gt__(("x",)) is NotImplemented
    assert values.__ge__(("x",)) is NotImplemented

    with pytest.raises(TypeError):
        _ = values < ("x",)


def test_fuzzy_list_construction_matches_builtin_for_equivalent_iterable():
    source = ["a", "b", "a"]

    assert list(FuzzyList(source)) == source


def test_fuzzy_list_len_and_membership_match_builtin():
    source = ["a", "b"]
    values = FuzzyList(source)

    assert len(values) == len(source)
    assert ("a" in values) == ("a" in source)
    assert ("z" in values) == ("z" in source)


def test_fuzzy_list_equality_against_builtin_list():
    source = ["a", "b"]

    assert FuzzyList(source) == source
    assert source == FuzzyList(source)


def test_fuzzy_list_is_not_hashable_like_builtin_list():
    values = FuzzyList(["a"])

    assert not isinstance(values, Hashable)
    assert not isinstance(["a"], Hashable)


def test_fuzzy_list_copy_and_deepcopy_round_trip():
    values = FuzzyList(["a", "b"], score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FuzzyList)
        assert list(result) == list(values)


def test_fuzzy_list_pickle_round_trip_preserves_content_and_fuzzy_lookup():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    restored = loads(dumps(values))

    assert isinstance(restored, FuzzyList)
    assert list(restored) == list(values)
    assert restored.fuzzy_find_one("Alpa Phone") is not None


def test_fuzzy_list_repr_does_not_crash():
    assert repr(FuzzyList(["a"]))
