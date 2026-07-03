from copy import copy, deepcopy

import pytest
from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FrozenFuzzySet, FuzzySet, IndexStrategy, Match, ScorerType
from tests.helpers import HashableCycleNode, require_not_none


@pytest.mark.parametrize("strategy", list(IndexStrategy))
def test_fuzzy_set_add_is_atomic_when_normalization_fails(strategy):
    def rejecting_normalizer(value: object) -> str:
        if value == "bad":
            raise RuntimeError("normalization failed")
        if not isinstance(value, str):
            raise TypeError("test normalizer accepts strings only")
        return value

    values = FuzzySet(["alpha"], normalizer=rejecting_normalizer, strategy=strategy)

    with pytest.raises(RuntimeError, match="normalization failed"):
        values.add("bad")

    assert list(values) == ["alpha"]
    assert "bad" not in values


def test_fuzzy_set_keyed_query_accepts_value_with_misleading_hashable_abc():
    query = (["x"],)
    values = FuzzySet(["x"], normalizer=str, score_cutoff=None, strategy=IndexStrategy.KEYED)

    assert values.fuzzy_find_one(query) is not None


def test_fuzzy_set_and_frozen_fuzzy_set_support_binary_operators():
    mutable = FuzzySet(["alpha", "beta"])
    frozen = FrozenFuzzySet(["beta", "gamma"])

    mutable_difference = mutable - frozen
    mutable_intersection = mutable & frozen
    mutable_symmetric_difference = mutable ^ frozen
    mutable_union = mutable | frozen
    frozen_difference = frozen - mutable
    frozen_intersection = frozen & mutable
    frozen_symmetric_difference = frozen ^ mutable
    frozen_union = frozen | mutable

    assert isinstance(mutable_difference, FuzzySet)
    assert isinstance(mutable_intersection, FuzzySet)
    assert isinstance(mutable_symmetric_difference, FuzzySet)
    assert isinstance(mutable_union, FuzzySet)
    assert isinstance(frozen_difference, FrozenFuzzySet)
    assert isinstance(frozen_intersection, FrozenFuzzySet)
    assert isinstance(frozen_symmetric_difference, FrozenFuzzySet)
    assert isinstance(frozen_union, FrozenFuzzySet)
    assert set(mutable_difference) == {"alpha"}
    assert set(mutable_intersection) == {"beta"}
    assert set(mutable_symmetric_difference) == {"alpha", "gamma"}
    assert set(frozen_difference) == {"gamma"}
    assert set(frozen_intersection) == {"beta"}
    assert set(frozen_symmetric_difference) == {"alpha", "gamma"}


def test_fuzzy_set_behaves_like_mutable_set_and_rebuilds_index():
    values = FuzzySet(["Alpha Phone"])

    values.add("Beta Tablet")
    assert set(values) == {"Alpha Phone", "Beta Tablet"}
    assert list(values) == ["Alpha Phone", "Beta Tablet"]
    assert values.fuzzy_get("Bta Tablet") == "Beta Tablet"

    values.discard("Beta Tablet")
    assert values.fuzzy_get("Bta Tablet") is None


def test_fuzzy_set_ignores_duplicate_values():
    values = FuzzySet(["Alpha Phone", "Alpha Phone"])

    assert len(values) == 1
    assert list(values) == ["Alpha Phone"]


def test_fuzzy_set_deepcopy_preserves_recursive_graph():
    node = HashableCycleNode()
    values = FuzzySet([node])
    node.owner = values

    result = deepcopy(values)
    copied_node = next(iter(result))

    assert copied_node.owner is result


def test_fuzzy_set_returns_match_for_exact_value():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    match = values.fuzzy_find_one("Alpha Phone")

    assert match == Match(
        value="Alpha Phone",
        score=100,
        index=None,
        query="Alpha Phone",
        normalized_query="alpha phone",
        normalized_value="alpha phone",
    )


def test_fuzzy_set_uses_insertion_order_for_normalized_exact_value():
    values = FuzzySet(["Alpha Phone", "  alpha phone  "])

    match = values.fuzzy_find_one("ALPHA PHONE")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index is None


def test_fuzzy_set_excludes_unsearchable_value_from_fuzzy_lookup():
    values = FuzzySet(["xy", "Alpha Phone"])

    assert values.fuzzy_find_one("xy") is None
    assert "xy" in values


def test_fuzzy_set_find_many_returns_match_objects():
    values = FuzzySet(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    matches = values.fuzzy_find_many("Alpha", limit=2)

    assert len(matches) == 2
    assert all(isinstance(match, Match) for match in matches)
    assert {match.value for match in matches} <= {"Alpha Phone", "Alpha Case"}


def test_fuzzy_set_batch_methods_preserve_query_order():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    one_matches = values.fuzzy_find_one_batch(["Bta Tablet", "Coffee Grinder"])
    many_matches = values.fuzzy_find_many_batch(["Alpha", "Coffee Grinder"], limit=1)
    matched_values = values.fuzzy_get_batch(["Alpa Phone", "Coffee Grinder"], default="missing")

    first_match = require_not_none(one_matches[0])
    assert first_match.value == "Beta Tablet"
    assert one_matches[1] is None
    assert [[match.value for match in matches] for matches in many_matches] == [
        ["Alpha Phone"],
        [],
    ]
    assert matched_values == ["Alpha Phone", "missing"]


def test_fuzzy_set_supports_distance_scorer():
    values = FuzzySet(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    match = values.fuzzy_find_one("Alph Phone")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.score <= 2


def test_fuzzy_set_get_returns_default_on_miss():
    values = FuzzySet(["Alpha Phone"])

    assert values.fuzzy_get("Coffee Grinder") is None
    assert values.fuzzy_get("Coffee Grinder", default="missing") == "missing"
    assert not values.fuzzy_contains("Coffee Grinder")


def test_fuzzy_set_discards_best_single_match():
    values = FuzzySet(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    values.fuzzy_discard("Alpa Phone")

    assert list(values) == ["Alpha Case", "Beta Tablet"]
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_set_discard_is_noop_on_miss():
    values = FuzzySet(["Alpha Phone"])

    values.fuzzy_discard("Coffee Grinder")

    assert list(values) == ["Alpha Phone"]


def test_fuzzy_set_discards_all_matches_and_returns_count():
    values = FuzzySet(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    removed = values.fuzzy_discard_all("Alpha")

    assert removed == 2
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_set_retain_all_keeps_only_matches_and_returns_count():
    values = FuzzySet(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    removed = values.fuzzy_retain_all("Alpha")

    assert removed == 1
    assert list(values) == ["Alpha Phone", "Alpha Case"]


def test_fuzzy_set_discards_all_duplicate_normalized_values():
    values = FuzzySet(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

    removed = values.fuzzy_discard_all("ALPHA PHONE")

    assert removed == 2
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_set_retain_all_keeps_duplicate_normalized_values():
    values = FuzzySet(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

    removed = values.fuzzy_retain_all("ALPHA PHONE")

    assert removed == 1
    assert list(values) == ["Alpha Phone", "  alpha phone  "]


def test_fuzzy_set_does_not_fuzzy_discard_unsearchable_value():
    values = FuzzySet(["xy", "Alpha Phone"])

    values.fuzzy_discard("xy")

    assert list(values) == ["xy", "Alpha Phone"]


def test_fuzzy_set_discard_all_supports_distance_scorer():
    values = FuzzySet(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    removed = values.fuzzy_discard_all("Alph Phone")

    assert removed == 1
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_set_repr():
    values = FuzzySet(["Alpha Phone"])

    assert repr(values) == "FuzzySet(['Alpha Phone'])"


def test_fuzzy_set_contains_returns_true_for_exact_value():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    assert "Alpha Phone" in values
    assert "Coffee Grinder" not in values


def test_fuzzy_set_add_ignores_duplicate():
    values = FuzzySet(["Alpha Phone"])

    values.add("Alpha Phone")

    assert len(values) == 1


def test_fuzzy_set_discard_noop_when_value_absent():
    values = FuzzySet(["Alpha Phone"])

    values.discard("Coffee Grinder")

    assert list(values) == ["Alpha Phone"]


def test_fuzzy_set_discard_all_noop_on_miss_returns_zero():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    removed = values.fuzzy_discard_all("Coffee Grinder")

    assert removed == 0
    assert list(values) == ["Alpha Phone", "Beta Tablet"]


def test_fuzzy_set_retain_all_noop_when_all_match_returns_zero():
    values = FuzzySet(["Alpha Phone", "Alpha Case"])

    removed = values.fuzzy_retain_all("Alpha")

    assert removed == 0
    assert list(values) == ["Alpha Phone", "Alpha Case"]


def test_fuzzy_set_discard_all_empties_set():
    values = FuzzySet(["Alpha Phone", "Alpha Case"])

    removed = values.fuzzy_discard_all("Alpha")

    assert removed == 2
    assert list(values) == []
    assert len(values) == 0
    assert values.fuzzy_get("Alpha Phone") is None


def test_fuzzy_set_retain_all_removes_all_when_no_match():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    removed = values.fuzzy_retain_all("Coffee Grinder")

    assert removed == 2
    assert list(values) == []


def test_fuzzy_set_inherited_clear_updates_fuzzy_index():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    values.clear()

    assert list(values) == []
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_set_inherited_pop_updates_fuzzy_index():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    popped = values.pop()

    assert popped == "Alpha Phone"
    assert list(values) == ["Beta Tablet"]
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_set_inherited_remove_updates_fuzzy_index():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    values.remove("Alpha Phone")

    assert list(values) == ["Beta Tablet"]
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_set_inherited_update_updates_fuzzy_index():
    values = FuzzySet(["Alpha Phone"])

    values.update(["Beta Tablet", "Gamma Watch"])

    assert list(values) == ["Alpha Phone", "Beta Tablet", "Gamma Watch"]
    assert values.fuzzy_get("Gama Watch") == "Gamma Watch"


def test_fuzzy_set_union_returns_fuzzy_set():
    values = FuzzySet(["Alpha Phone"])

    result = values | {"Beta Tablet"}

    assert isinstance(result, FuzzySet)
    assert list(result) == ["Alpha Phone", "Beta Tablet"]
    assert result.fuzzy_get("Bta Tablet") == "Beta Tablet"


def test_fuzzy_set_algebra_preserves_score_cutoff():
    values = FuzzySet(["Alpha Phone"], score_cutoff=100)

    result = values | {"Beta Tablet"}

    assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
    assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_set_copy_preserves_config():
    values = FuzzySet(["Alpha Phone"], score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FuzzySet)
        assert list(result) == ["Alpha Phone"]
        assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
        assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_set_binary_algebra_preserves_order_and_type():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"])

    # noinspection PyTypeChecker
    assert list(values & {"Beta Tablet", "Gamma Watch"}) == ["Beta Tablet"]
    # noinspection PyTypeChecker
    assert list(values - {"Beta Tablet"}) == ["Alpha Phone"]
    # noinspection PyTypeChecker
    assert list(values ^ {"Beta Tablet", "Gamma Watch"}) == ["Alpha Phone", "Gamma Watch"]
    assert isinstance(values & {"Beta Tablet"}, FuzzySet)
    assert isinstance(values - {"Beta Tablet"}, FuzzySet)
    assert isinstance(values ^ {"Beta Tablet", "Gamma Watch"}, FuzzySet)


def test_fuzzy_set_reverse_algebra_uses_left_operand_order_and_preserves_config():
    values = FuzzySet(["Alpha Phone"], score_cutoff=100)

    union = {"Beta Tablet"} | values
    difference = {"Beta Tablet", "Alpha Phone"} - values
    symmetric_difference = {"Beta Tablet", "Alpha Phone"} ^ values

    # noinspection PyTypeChecker
    assert set(union) == {"Beta Tablet", "Alpha Phone"}
    # noinspection PyUnresolvedReferences
    assert union.fuzzy_get("Alpa Phone") is None
    # noinspection PyTypeChecker
    assert list(difference) == ["Beta Tablet"]
    # noinspection PyTypeChecker
    assert set(symmetric_difference) == {"Beta Tablet"}


def test_fuzzy_set_in_place_algebra_updates_fuzzy_index():
    values = FuzzySet(["Alpha Phone", "Beta Tablet"], score_cutoff=100)

    values |= {"Gamma Watch"}
    assert list(values) == ["Alpha Phone", "Beta Tablet", "Gamma Watch"]
    assert values.fuzzy_get("Gama Watch") is None
    assert values.fuzzy_get("Gamma Watch") == "Gamma Watch"

    values &= {"Alpha Phone", "Gamma Watch"}
    assert list(values) == ["Alpha Phone", "Gamma Watch"]

    values -= {"Alpha Phone"}
    assert list(values) == ["Gamma Watch"]
    assert values.fuzzy_get("Alpha Phone") is None

    values ^= {"Gamma Watch", "Delta Cable"}
    assert list(values) == ["Delta Cable"]
    assert values.fuzzy_get("Delta Cable") == "Delta Cable"


def test_fuzzy_set_named_algebra_supports_multiple_inputs_and_preserves_config():
    values = FuzzySet(["Alpha Phone", "Beta Tablet", "Gamma Watch"], score_cutoff=100)

    union = values.union(["Delta Cable"], ["Epsilon Case"])
    intersection = values.intersection(["Alpha Phone", "Beta Tablet"], ["Beta Tablet", "Delta Cable"])
    difference = values.difference(["Alpha Phone"], ["Gamma Watch"])
    symmetric_difference = values.symmetric_difference(["Gamma Watch", "Delta Cable"])

    assert list(union) == ["Alpha Phone", "Beta Tablet", "Gamma Watch", "Delta Cable", "Epsilon Case"]
    assert union.fuzzy_get("Alpa Phone") is None
    assert list(intersection) == ["Beta Tablet"]
    assert list(difference) == ["Beta Tablet"]
    assert list(symmetric_difference) == ["Alpha Phone", "Beta Tablet", "Delta Cable"]


def test_fuzzy_set_named_update_operations_keep_index_synchronized():
    values = FuzzySet(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

    values.intersection_update(["Alpha Phone", "Beta Tablet"], ["Beta Tablet", "Delta Cable"])
    assert list(values) == ["Beta Tablet"]

    values.update(["Gamma Watch", "Delta Cable"])
    values.difference_update(["Gamma Watch"], ["Unknown"])
    assert list(values) == ["Beta Tablet", "Delta Cable"]

    values.symmetric_difference_update(["Delta Cable", "Epsilon Case"])
    assert list(values) == ["Beta Tablet", "Epsilon Case"]
    assert values.fuzzy_get("Epslon Case") == "Epsilon Case"


def test_fuzzy_set_iter_scores_matches_materialized_scores():
    values = FuzzySet(["Alpha Phone", "xy", "Beta Tablet"])

    assert list(values.fuzzy_iter_scores("Alpa Phone")) == values.fuzzy_score_all("Alpa Phone")


def test_fuzzy_set_with_config_updates_policy_without_mutating_source():
    values = FuzzySet(["Alpha Phone"], score_cutoff=100)

    permissive = values.with_config(score_cutoff=None)

    assert list(permissive) == ["Alpha Phone"]
    assert values.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_fuzzy_set_with_config_preserves_score_hint():
    values = FuzzySet(["Alpha Phone"], score_hint=100)

    assert values.with_config(score_cutoff=None)._index._score_hint == 100


def test_fuzzy_set_rand_returns_other_values_present_in_set():
    values = FuzzySet(["alpha", "beta", "gamma"])

    result = frozenset(["alpha", "delta"]) & values

    assert isinstance(result, FuzzySet)
    assert set(result) == {"alpha"}


def test_fuzzy_set_difference_update_no_args_is_noop():
    values = FuzzySet(["alpha", "beta"])

    values.difference_update()

    assert set(values) == {"alpha", "beta"}


def test_fuzzy_set_intersection_no_args_returns_copy():
    values = FuzzySet(["alpha", "beta"])

    result = values.intersection()

    assert set(result) == {"alpha", "beta"}


def test_fuzzy_set_intersection_update_no_args_is_noop():
    values = FuzzySet(["alpha", "beta"])

    values.intersection_update()

    assert set(values) == {"alpha", "beta"}
