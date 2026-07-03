"""Compatibility checks comparing ``FuzzyTuple`` against the builtin ``tuple``."""

from copy import copy, deepcopy
from pickle import dumps, loads

import pytest

from rapidfuzz_collections import FuzzyTuple


class RepeatCount:
    """Provide a noninteger repetition count through ``__index__``."""

    def __index__(self) -> int:
        return 2


def test_fuzzy_tuple_reflected_addition_preserves_fuzzy_type_and_config():
    values = FuzzyTuple(["Alpha Phone"], score_cutoff=100)

    result = ("Beta Tablet",) + values

    assert isinstance(result, FuzzyTuple)
    assert tuple(result) == ("Beta Tablet", "Alpha Phone")
    assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
    assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_tuple_reflected_addition_rejects_non_tuple():
    values = FuzzyTuple([2])

    # noinspection PyTypeChecker
    assert values.__radd__([1]) is NotImplemented


def test_fuzzy_tuple_repetition_accepts_supports_index():
    values = FuzzyTuple([1, 2])

    assert tuple(values * RepeatCount()) == (1, 2, 1, 2)
    assert tuple(RepeatCount() * values) == (1, 2, 1, 2)


def test_fuzzy_tuple_lt_matches_builtin_lexicographic_ordering():
    # noinspection PyTypeChecker
    assert (FuzzyTuple([1, 2]) < FuzzyTuple([1, 3])) == ((1, 2) < (1, 3)) is True
    assert (FuzzyTuple([1, 2]) < (1, 3)) is True
    assert (FuzzyTuple([1, 3]) < FuzzyTuple([1, 2])) is False


def test_fuzzy_tuple_le_matches_builtin_lexicographic_ordering():
    assert (FuzzyTuple([1, 2]) <= FuzzyTuple([1, 2])) == ((1, 2) <= (1, 2)) is True
    assert (FuzzyTuple([1, 3]) <= (1, 2)) is False


def test_fuzzy_tuple_gt_matches_builtin_lexicographic_ordering():
    # noinspection PyTypeChecker
    assert (FuzzyTuple([1, 3]) > FuzzyTuple([1, 2])) == ((1, 3) > (1, 2)) is True
    assert (FuzzyTuple([1, 2]) > (1, 3)) is False


def test_fuzzy_tuple_ge_matches_builtin_lexicographic_ordering():
    assert (FuzzyTuple([1, 2]) >= FuzzyTuple([1, 2])) == ((1, 2) >= (1, 2)) is True
    assert (FuzzyTuple([1, 2]) >= (1, 3)) is False


def test_fuzzy_tuple_comparisons_against_unrelated_type_return_not_implemented():
    values = FuzzyTuple([1, 2])

    assert values.__lt__(["x"]) is NotImplemented
    assert values.__le__(["x"]) is NotImplemented
    assert values.__gt__(["x"]) is NotImplemented
    assert values.__ge__(["x"]) is NotImplemented

    with pytest.raises(TypeError):
        _ = values < ["x"]


def test_fuzzy_tuple_construction_matches_builtin_for_equivalent_iterable():
    source = ("a", "b", "a")

    assert tuple(FuzzyTuple(source)) == source


def test_fuzzy_tuple_len_and_membership_match_builtin():
    source = ("a", "b")
    values = FuzzyTuple(source)

    assert len(values) == len(source)
    assert ("a" in values) == ("a" in source)
    assert ("z" in values) == ("z" in source)


def test_fuzzy_tuple_equality_against_builtin_tuple():
    source = ("a", "b")

    assert FuzzyTuple(source) == source
    assert source == FuzzyTuple(source)


def test_fuzzy_tuple_is_hashable_like_builtin_tuple():
    values = FuzzyTuple(("a", "b"))

    assert isinstance(hash(values), int)
    assert hash(values) == hash(FuzzyTuple(("a", "b")))


def test_fuzzy_tuple_copy_and_deepcopy_round_trip():
    values = FuzzyTuple(("a", "b"), score_cutoff=100)

    for result in (copy(values), deepcopy(values)):
        assert isinstance(result, FuzzyTuple)
        assert tuple(result) == tuple(values)


def test_fuzzy_tuple_pickle_round_trip_preserves_content_and_fuzzy_lookup():
    values = FuzzyTuple(("Alpha Phone", "Beta Tablet"))

    restored = loads(dumps(values))

    assert isinstance(restored, FuzzyTuple)
    assert tuple(restored) == tuple(values)
    assert restored.fuzzy_find_one("Alpa Phone") is not None


def test_fuzzy_tuple_repr_does_not_crash():
    assert repr(FuzzyTuple(("a",)))
