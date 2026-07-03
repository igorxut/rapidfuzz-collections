"""Compatibility checks comparing ``FrozenFuzzySet`` against the builtin ``frozenset``."""

from copy import copy, deepcopy
from pickle import dumps, loads

import pytest

from rapidfuzz_collections import FrozenFuzzySet


def test_frozen_fuzzy_set_issubset_accepts_arbitrary_iterable():
    values = FrozenFuzzySet({"a", "b"})
    builtin = frozenset({"a", "b"})

    assert values.issubset(["a", "b", "c"]) == builtin.issubset(["a", "b", "c"]) is True
    assert values.issubset(["a"]) == builtin.issubset(["a"]) is False


def test_frozen_fuzzy_set_issuperset_accepts_arbitrary_iterable():
    values = FrozenFuzzySet({"a", "b", "c"})
    builtin = frozenset({"a", "b", "c"})

    assert values.issuperset(["a", "b"]) == builtin.issuperset(["a", "b"]) is True
    assert values.issuperset(["a", "z"]) == builtin.issuperset(["a", "z"]) is False


def test_frozen_fuzzy_set_isdisjoint_matches_builtin():
    values = FrozenFuzzySet({"a", "b"})
    builtin = frozenset({"a", "b"})

    assert values.isdisjoint(["c", "d"]) == builtin.isdisjoint(["c", "d"]) is True
    assert values.isdisjoint(["a", "z"]) == builtin.isdisjoint(["a", "z"]) is False


@pytest.mark.parametrize("method_name", ["isdisjoint", "issuperset"])
def test_frozen_fuzzy_set_named_relations_reject_unhashable_elements(method_name):
    values = FrozenFuzzySet([frozenset({1})])

    with pytest.raises(TypeError, match="unhashable type"):
        getattr(values, method_name)([{1}])


def test_frozen_fuzzy_set_named_operations_support_mixed_hashable_types():
    values = FrozenFuzzySet([1])

    union = values.union(["two"])
    symmetric_difference = values.symmetric_difference(["two"])

    assert list(union) == [1, "two"]
    assert list(symmetric_difference) == [1, "two"]


def test_frozen_fuzzy_set_and_rejects_plain_list():
    values = FrozenFuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values & ["a", "c"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = frozenset({"a", "b"}) & ["a", "c"]


def test_frozen_fuzzy_set_or_rejects_plain_list():
    values = FrozenFuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values | ["c"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = frozenset({"a", "b"}) | ["c"]


def test_frozen_fuzzy_set_sub_rejects_plain_list():
    values = FrozenFuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values - ["a"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = frozenset({"a", "b"}) - ["a"]


def test_frozen_fuzzy_set_xor_rejects_plain_list():
    values = FrozenFuzzySet({"a", "b"})

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values ^ ["a", "c"]
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = frozenset({"a", "b"}) ^ ["a", "c"]


def test_frozen_fuzzy_set_reverse_operators_reject_plain_list():
    values = FrozenFuzzySet({"a", "b"})

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


def test_frozen_fuzzy_set_named_methods_stay_permissive_for_plain_list():
    values = FrozenFuzzySet({"a", "b"})
    builtin = frozenset({"a", "b"})

    assert set(values.union(["c"])) == builtin.union(["c"])
    assert set(values.intersection(["a", "z"])) == builtin.intersection(["a", "z"])
    assert set(values.difference(["a"])) == builtin.difference(["a"])
    assert set(values.symmetric_difference(["a", "c"])) == builtin.symmetric_difference(["a", "c"])


def test_frozen_fuzzy_set_operators_accept_real_set_operand():
    values = FrozenFuzzySet({"a", "b"})
    builtin = frozenset({"a", "b"})

    # noinspection PyTypeChecker
    assert set(values & {"a", "c"}) == builtin & {"a", "c"}
    # noinspection PyTypeChecker
    assert set(values | {"c"}) == builtin | {"c"}
    # noinspection PyTypeChecker
    assert set(values - {"a"}) == builtin - {"a"}
    # noinspection PyTypeChecker
    assert set(values ^ {"a", "c"}) == builtin ^ {"a", "c"}


def test_frozen_fuzzy_set_construction_matches_builtin_for_equivalent_iterable():
    source = ["a", "b", "a"]

    assert set(FrozenFuzzySet(source)) == set(source)


def test_frozen_fuzzy_set_len_and_membership_match_builtin():
    source = frozenset({"a", "b"})
    values = FrozenFuzzySet(source)

    assert len(values) == len(source)
    assert ("a" in values) == ("a" in source)
    assert ("z" in values) == ("z" in source)


def test_frozen_fuzzy_set_equality_against_builtin_frozenset():
    source = frozenset({"a", "b"})

    assert FrozenFuzzySet(source) == source
    assert source == FrozenFuzzySet(source)


def test_frozen_fuzzy_set_is_hashable_like_builtin_frozenset():
    values = FrozenFuzzySet({"a", "b"})

    assert isinstance(hash(values), int)
    assert hash(values) == hash(FrozenFuzzySet({"a", "b"}))


def test_frozen_fuzzy_set_copy_and_deepcopy_round_trip():
    values = FrozenFuzzySet({"a", "b"}, score_cutoff=100)

    for result in (copy(values), deepcopy(values)):
        assert isinstance(result, FrozenFuzzySet)
        assert set(result) == set(values)


def test_frozen_fuzzy_set_pickle_round_trip_preserves_content_and_fuzzy_lookup():
    values = FrozenFuzzySet({"Alpha Phone", "Beta Tablet"})

    restored = loads(dumps(values))

    assert isinstance(restored, FrozenFuzzySet)
    assert set(restored) == set(values)
    assert restored.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_frozen_fuzzy_set_repr_does_not_crash():
    assert repr(FrozenFuzzySet({"a"}))
