from copy import copy, deepcopy

from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FrozenFuzzySet, IndexStrategy, Match, ScorerType
from tests.helpers import HashableCycleNode, require_not_none


def test_frozen_fuzzy_set_behaves_like_immutable_set_with_construction_order():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet", "Alpha Phone"])

    assert len(values) == 2
    assert "Alpha Phone" in values
    assert "Gamma Watch" not in values
    assert list(values) == ["Alpha Phone", "Beta Tablet"]
    assert hash(values) == hash(frozenset({"Alpha Phone", "Beta Tablet"}))


def test_frozen_fuzzy_set_deepcopy_preserves_recursive_graph():
    node = HashableCycleNode()
    values = FrozenFuzzySet([node])
    node.owner = values

    result = deepcopy(values)
    copied_node = next(iter(result))

    assert copied_node.owner is result


def test_frozen_fuzzy_set_returns_match_for_exact_value():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"])

    match = values.fuzzy_find_one("Alpha Phone")

    assert match == Match(
        value="Alpha Phone",
        score=100,
        index=None,
        query="Alpha Phone",
        normalized_query="alpha phone",
        normalized_value="alpha phone",
    )


def test_frozen_fuzzy_set_uses_construction_order_for_normalized_exact_value():
    values = FrozenFuzzySet(["Alpha Phone", "  alpha phone  "])

    match = values.fuzzy_find_one("ALPHA PHONE")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index is None


def test_frozen_fuzzy_set_excludes_unsearchable_value_from_fuzzy_lookup():
    values = FrozenFuzzySet(["xy", "Alpha Phone"])

    assert values.fuzzy_find_one("xy") is None
    assert "xy" in values


def test_frozen_fuzzy_set_find_many_returns_match_objects():
    values = FrozenFuzzySet(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    matches = values.fuzzy_find_many("Alpha", limit=2)

    assert len(matches) == 2
    assert all(isinstance(match, Match) for match in matches)
    assert {match.value for match in matches} <= {"Alpha Phone", "Alpha Case"}


def test_frozen_fuzzy_set_batch_methods_preserve_query_order():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"])

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


def test_frozen_fuzzy_set_supports_distance_scorer():
    values = FrozenFuzzySet(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    match = values.fuzzy_find_one("Alph Phone")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.score <= 2


def test_frozen_fuzzy_set_repr():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"])

    assert repr(values) == "FrozenFuzzySet(('Alpha Phone', 'Beta Tablet'))"


def test_frozen_fuzzy_set_fuzzy_get_returns_matched_value():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"])

    assert values.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_frozen_fuzzy_set_get_returns_default_on_miss():
    values = FrozenFuzzySet(["Alpha Phone"])

    assert values.fuzzy_get("Coffee Grinder") is None
    assert values.fuzzy_get("Coffee Grinder", default="missing") == "missing"
    assert not values.fuzzy_contains("Coffee Grinder")


def test_frozen_fuzzy_set_union_returns_frozen_fuzzy_set():
    values = FrozenFuzzySet(["Alpha Phone"])

    result = values | {"Beta Tablet"}

    assert isinstance(result, FrozenFuzzySet)
    assert list(result) == ["Alpha Phone", "Beta Tablet"]
    assert result.fuzzy_get("Bta Tablet") == "Beta Tablet"


def test_frozen_fuzzy_set_algebra_preserves_score_cutoff():
    values = FrozenFuzzySet(["Alpha Phone"], score_cutoff=100)

    result = values | {"Beta Tablet"}

    assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
    assert result.fuzzy_get("Alpa Phone") is None


def test_frozen_fuzzy_set_copy_preserves_config():
    values = FrozenFuzzySet(["Alpha Phone"], score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FrozenFuzzySet)
        assert list(result) == ["Alpha Phone"]
        assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
        assert result.fuzzy_get("Alpa Phone") is None


def test_frozen_fuzzy_set_binary_algebra_preserves_order_and_type():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"])

    # noinspection PyTypeChecker
    assert list(values & {"Beta Tablet", "Gamma Watch"}) == ["Beta Tablet"]
    # noinspection PyTypeChecker
    assert list(values - {"Beta Tablet"}) == ["Alpha Phone"]
    # noinspection PyTypeChecker
    assert list(values ^ {"Beta Tablet", "Gamma Watch"}) == ["Alpha Phone", "Gamma Watch"]
    assert isinstance(values & {"Beta Tablet"}, FrozenFuzzySet)
    assert isinstance(values - {"Beta Tablet"}, FrozenFuzzySet)
    assert isinstance(values ^ {"Beta Tablet", "Gamma Watch"}, FrozenFuzzySet)


def test_frozen_fuzzy_set_reverse_algebra_uses_left_operand_order_and_preserves_config():
    values = FrozenFuzzySet(["Alpha Phone"], score_cutoff=100)

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


def test_frozen_fuzzy_set_named_algebra_supports_multiple_inputs_and_preserves_config():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet", "Gamma Watch"], score_cutoff=100)

    union = values.union(["Delta Cable"], ["Epsilon Case"])
    intersection = values.intersection(["Alpha Phone", "Beta Tablet"], ["Beta Tablet", "Delta Cable"])
    difference = values.difference(["Alpha Phone"], ["Gamma Watch"])
    symmetric_difference = values.symmetric_difference(["Gamma Watch", "Delta Cable"])

    assert list(union) == ["Alpha Phone", "Beta Tablet", "Gamma Watch", "Delta Cable", "Epsilon Case"]
    assert union.fuzzy_get("Alpa Phone") is None
    assert list(intersection) == ["Beta Tablet"]
    assert list(difference) == ["Beta Tablet"]
    assert list(symmetric_difference) == ["Alpha Phone", "Beta Tablet", "Delta Cable"]


def test_frozen_fuzzy_set_iter_scores_matches_materialized_scores():
    values = FrozenFuzzySet(["Alpha Phone", "xy", "Beta Tablet"])

    assert list(values.fuzzy_iter_scores("Alpa Phone")) == values.fuzzy_score_all("Alpa Phone")


def test_frozen_fuzzy_set_with_config_updates_policy_without_mutating_source():
    values = FrozenFuzzySet(["Alpha Phone"], score_cutoff=100)

    permissive = values.with_config(score_cutoff=None)

    assert list(permissive) == ["Alpha Phone"]
    assert values.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_frozen_fuzzy_set_with_config_preserves_score_hint():
    values = FrozenFuzzySet(["Alpha Phone"], score_hint=100)

    assert values.with_config(score_cutoff=None)._index._score_hint == 100


def test_frozen_fuzzy_set_rand_returns_other_values_present_in_set():
    values = FrozenFuzzySet(["alpha", "beta", "gamma"])

    result = frozenset(["alpha", "delta"]) & values

    assert isinstance(result, FrozenFuzzySet)
    assert set(result) == {"alpha"}


def test_frozen_fuzzy_set_intersection_no_args_returns_copy():
    values = FrozenFuzzySet(["alpha", "beta"])

    result = values.intersection()

    assert set(result) == {"alpha", "beta"}


def test_frozen_fuzzy_set_keyed_strategy_returns_same_public_result_class():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

    match = values.fuzzy_find_one("Alpa Phone")

    assert isinstance(match, Match)
    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index is None


def test_frozen_fuzzy_set_keyed_strategy_excludes_unsearchable_value():
    values = FrozenFuzzySet(["xy", "Alpha Phone"], strategy=IndexStrategy.KEYED)

    assert values.fuzzy_find_one("xy") is None
    assert "xy" in values


def test_frozen_fuzzy_set_keyed_strategy_preserves_order_and_aligned_scores():
    values = FrozenFuzzySet(["Alpha Phone", "xy", "Beta Tablet"], strategy="keyed")

    results = values.fuzzy_score_all("Alpa Phone")

    assert list(values) == ["Alpha Phone", "xy", "Beta Tablet"]
    assert repr(values) == "FrozenFuzzySet(('Alpha Phone', 'xy', 'Beta Tablet'))"
    assert len(results) == 3
    first_result = require_not_none(results[0])
    assert isinstance(first_result, Match)
    assert first_result.value == "Alpha Phone"
    assert first_result.index is None
    assert results[1] is None
    assert results == list(values.fuzzy_iter_scores("Alpa Phone"))


def test_frozen_fuzzy_set_keyed_strategy_batch_lookup_uses_keyed_path():
    values = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

    results = values.fuzzy_find_one_batch(["Alpa Phone", 42, "Beta Tablet"])

    first_result = require_not_none(results[0])
    third_result = require_not_none(results[2])
    assert first_result.value == "Alpha Phone"
    assert results[1] is None
    assert third_result.value == "Beta Tablet"


def test_frozen_fuzzy_set_keyed_strategy_is_preserved_by_copy_and_config():
    values = FrozenFuzzySet(["Alpha Phone"], strategy=IndexStrategy.KEYED, score_cutoff=100)

    copied = values.copy()
    permissive = values.with_config(score_cutoff=None)

    assert copied._strategy == IndexStrategy.KEYED
    assert permissive._strategy == IndexStrategy.KEYED
    assert copied.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == "Alpha Phone"
