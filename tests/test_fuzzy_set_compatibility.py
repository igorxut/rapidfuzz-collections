"""Compatibility checks comparing ``FuzzySet`` against the builtin ``set``."""

from collections.abc import Hashable, Iterator, Set
from copy import copy, deepcopy
from pickle import dumps, loads

import pytest

from rapidfuzz_collections import FuzzySet, IndexStrategy


class UnhashableSet(Set[int]):
    """Provide an unhashable non-builtin set-like value."""

    __hash__ = None

    def __contains__(self, value: object) -> bool:
        return value == 1

    def __iter__(self) -> Iterator[int]:
        return iter((1,))

    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize("strategy", [IndexStrategy.KEYED, IndexStrategy.SEQUENCE])
def test_fuzzy_set_membership_and_removal_accept_mutable_set_equivalent(strategy):
    values = FuzzySet([frozenset({1})], strategy=strategy)

    assert {1} in values

    values.remove({1})

    assert not values


@pytest.mark.parametrize("strategy", [IndexStrategy.KEYED, IndexStrategy.SEQUENCE])
def test_fuzzy_set_discard_accepts_mutable_set_equivalent(strategy):
    values = FuzzySet([frozenset({1})], strategy=strategy)

    values.discard({1})

    assert not values


def test_fuzzy_set_membership_rejects_unhashable_non_set():
    values = FuzzySet(["value"])

    with pytest.raises(TypeError, match="unhashable type"):
        _ = [] in values


@pytest.mark.parametrize("candidate", [UnhashableSet(), {1: None}.keys()])
@pytest.mark.parametrize("strategy", [IndexStrategy.KEYED, IndexStrategy.SEQUENCE])
def test_fuzzy_set_membership_and_removal_reject_non_builtin_unhashable_sets(candidate, strategy):
    values = FuzzySet([frozenset({1})], strategy=strategy)

    with pytest.raises(TypeError, match="unhashable type"):
        _ = candidate in values
    with pytest.raises(TypeError, match="unhashable type"):
        values.discard(candidate)
    with pytest.raises(TypeError, match="unhashable type"):
        # noinspection PyTypeChecker
        values.remove(candidate)


def test_fuzzy_set_issubset_accepts_arbitrary_iterable():
    values = FuzzySet({"a", "b"})
    builtin = {"a", "b"}

    assert values.issubset(["a", "b", "c"]) == builtin.issubset(["a", "b", "c"]) is True
    assert values.issubset(["a"]) == builtin.issubset(["a"]) is False


def test_fuzzy_set_issuperset_accepts_arbitrary_iterable():
    values = FuzzySet({"a", "b", "c"})
    builtin = {"a", "b", "c"}

    assert values.issuperset(["a", "b"]) == builtin.issuperset(["a", "b"]) is True
    assert values.issuperset(["a", "z"]) == builtin.issuperset(["a", "z"]) is False


def test_fuzzy_set_isdisjoint_matches_builtin():
    values = FuzzySet({"a", "b"})
    builtin = {"a", "b"}

    assert values.isdisjoint(["c", "d"]) == builtin.isdisjoint(["c", "d"]) is True
    assert values.isdisjoint(["a", "z"]) == builtin.isdisjoint(["a", "z"]) is False


@pytest.mark.parametrize("method_name", ["isdisjoint", "issuperset"])
@pytest.mark.parametrize("strategy", [IndexStrategy.KEYED, IndexStrategy.SEQUENCE])
def test_fuzzy_set_named_relations_reject_unhashable_elements(method_name, strategy):
    values = FuzzySet([frozenset({1})], strategy=strategy)

    with pytest.raises(TypeError, match="unhashable type"):
        getattr(values, method_name)([{1}])


def test_fuzzy_set_named_operations_support_mixed_hashable_types():
    values = FuzzySet([1])

    union = values.union(["two"])
    symmetric_difference = values.symmetric_difference(["two"])

    assert list(union) == [1, "two"]
    assert list(symmetric_difference) == [1, "two"]


def test_fuzzy_set_and_rejects_plain_list():
    values = FuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values & ["a", "c"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = {"a", "b"} & ["a", "c"]


def test_fuzzy_set_or_rejects_plain_list():
    values = FuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values | ["c"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = {"a", "b"} | ["c"]


def test_fuzzy_set_sub_rejects_plain_list():
    values = FuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values - ["a"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = {"a", "b"} - ["a"]


def test_fuzzy_set_xor_rejects_plain_list():
    values = FuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values ^ ["a", "c"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = {"a", "b"} ^ ["a", "c"]


def test_fuzzy_set_reverse_operators_reject_plain_list():
    values = FuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ["a", "c"] & values
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ["c"] | values
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ["a"] - values
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = ["a", "c"] ^ values


def test_fuzzy_set_in_place_operators_reject_plain_list():
    values = FuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        values &= ["a"]
    with pytest.raises(TypeError):
        values |= ["c"]
    with pytest.raises(TypeError):
        values -= ["a"]
    with pytest.raises(TypeError):
        values ^= ["a"]


def test_fuzzy_set_named_methods_stay_permissive_for_plain_list():
    values = FuzzySet({"a", "b"})
    builtin = {"a", "b"}

    assert set(values.union(["c"])) == builtin.union(["c"])
    assert set(values.intersection(["a", "z"])) == builtin.intersection(["a", "z"])
    assert set(values.difference(["a"])) == builtin.difference(["a"])
    assert set(values.symmetric_difference(["a", "c"])) == builtin.symmetric_difference(["a", "c"])


def test_fuzzy_set_update_family_stays_permissive_for_plain_list():
    values = FuzzySet({"a", "b"})
    builtin = {"a", "b"}

    values.update(["c"])
    builtin.update(["c"])
    assert set(values) == builtin

    values.intersection_update(["a", "c"])
    builtin.intersection_update(["a", "c"])
    assert set(values) == builtin

    values.difference_update(["a"])
    builtin.difference_update(["a"])
    assert set(values) == builtin

    values.symmetric_difference_update(["z"])
    builtin.symmetric_difference_update(["z"])
    assert set(values) == builtin


def test_fuzzy_set_operators_accept_real_set_operand():
    values = FuzzySet({"a", "b"})
    builtin = {"a", "b"}

    # noinspection PyTypeChecker
    assert set(values & {"a", "c"}) == builtin & {"a", "c"}
    # noinspection PyTypeChecker
    assert set(values | {"c"}) == builtin | {"c"}
    # noinspection PyTypeChecker
    assert set(values - {"a"}) == builtin - {"a"}
    # noinspection PyTypeChecker
    assert set(values ^ {"a", "c"}) == builtin ^ {"a", "c"}


def test_fuzzy_set_construction_matches_builtin_for_equivalent_iterable():
    source = ["a", "b", "a"]

    assert set(FuzzySet(source)) == set(source)


def test_fuzzy_set_len_and_membership_match_builtin():
    source = {"a", "b"}
    values = FuzzySet(source)

    assert len(values) == len(source)
    assert ("a" in values) == ("a" in source)
    assert ("z" in values) == ("z" in source)


def test_fuzzy_set_equality_against_builtin_set():
    source = {"a", "b"}

    assert FuzzySet(source) == source
    assert source == FuzzySet(source)


def test_fuzzy_set_is_not_hashable_like_builtin_set():
    values = FuzzySet({"a"})

    assert not isinstance(values, Hashable)
    assert not isinstance({"a"}, Hashable)


def test_fuzzy_set_copy_and_deepcopy_round_trip():
    values = FuzzySet({"a", "b"}, score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FuzzySet)
        assert set(result) == set(values)


def test_fuzzy_set_pickle_round_trip_preserves_content_and_fuzzy_lookup():
    values = FuzzySet({"Alpha Phone", "Beta Tablet"})

    restored = loads(dumps(values))

    assert isinstance(restored, FuzzySet)
    assert set(restored) == set(values)
    assert restored.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_fuzzy_set_repr_does_not_crash():
    assert repr(FuzzySet({"a"}))
