from copy import copy, deepcopy

import pytest
from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FuzzyTuple, Match, ScorerType
from tests.helpers import casefold_string, require_not_none


def test_fuzzy_tuple_accepts_value_with_misleading_hashable_abc():
    value = (["x"],)

    values = FuzzyTuple([value], normalizer=str, score_cutoff=None)

    assert values.fuzzy_get(value) == value


def test_fuzzy_tuple_behaves_like_sequence():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    assert len(values) == 2
    assert values[0] == "Alpha Phone"
    assert values[1:] == ("Beta Tablet",)
    assert list(values) == ["Alpha Phone", "Beta Tablet"]


def test_fuzzy_tuple_returns_match_for_exact_value():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    match = values.fuzzy_find_one("Alpha Phone")

    assert match == Match(
        value="Alpha Phone",
        score=100,
        index=0,
        query="Alpha Phone",
        normalized_query="alpha phone",
        normalized_value="alpha phone",
    )


def test_fuzzy_tuple_uses_source_order_for_normalized_exact_value():
    values = FuzzyTuple(["Alpha Phone", "  alpha phone  "])

    match = values.fuzzy_find_one("ALPHA PHONE")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index == 0


def test_fuzzy_tuple_prefers_exact_value_on_equal_score():
    values = FuzzyTuple(["ALPHA", "alpha"], normalizer=casefold_string, score_cutoff=0)

    match = require_not_none(values.fuzzy_find_one("alpha"))
    matches = values.fuzzy_find_many("alpha", limit=2)

    assert match == matches[0]
    assert [result.value for result in matches] == ["alpha", "ALPHA"]


def test_fuzzy_tuple_excludes_unsearchable_value_from_fuzzy_lookup():
    values = FuzzyTuple(["xy", "Alpha Phone"])

    assert values.fuzzy_find_one("xy") is None
    assert values[0] == "xy"


def test_fuzzy_tuple_find_many_returns_match_objects():
    values = FuzzyTuple(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    matches = values.fuzzy_find_many("Alpha", limit=2)

    assert len(matches) == 2
    assert all(isinstance(match, Match) for match in matches)
    assert {match.value for match in matches} <= {"Alpha Phone", "Alpha Case"}


def test_fuzzy_tuple_batch_methods_preserve_query_order():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

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


def test_fuzzy_tuple_add_returns_fuzzy_tuple_and_preserves_config():
    values = FuzzyTuple(["Alpha Phone"], score_cutoff=100)

    result = values + ("Beta Tablet",)

    assert isinstance(result, FuzzyTuple)
    assert tuple(result) == ("Alpha Phone", "Beta Tablet")
    assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
    assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_tuple_add_rejects_non_tuple_iterable():
    values = FuzzyTuple(["Alpha Phone"])

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values + ["Beta Tablet"]


def test_fuzzy_tuple_copy_preserves_config():
    values = FuzzyTuple(["Alpha Phone"], score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FuzzyTuple)
        assert tuple(result) == ("Alpha Phone",)
        assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
        assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_tuple_deepcopy_preserves_recursive_graph():
    holder: list[object] = []
    values = FuzzyTuple([holder])
    holder.append(values)

    result = deepcopy(values)

    assert result[0][0] is result


def test_fuzzy_tuple_mul_returns_fuzzy_tuple_and_preserves_config():
    values = FuzzyTuple(["Alpha Phone"], score_cutoff=100)

    result = values * 2
    reverse_result = 2 * values

    assert isinstance(result, FuzzyTuple)
    assert isinstance(reverse_result, FuzzyTuple)
    assert tuple(result) == ("Alpha Phone", "Alpha Phone")
    assert tuple(reverse_result) == ("Alpha Phone", "Alpha Phone")
    assert result.fuzzy_get("Alpa Phone") is None
    assert reverse_result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_tuple_reversed_iterates_in_reverse_order():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    assert list(reversed(values)) == ["Beta Tablet", "Alpha Phone"]


def test_fuzzy_tuple_supports_distance_scorer():
    values = FuzzyTuple(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    match = values.fuzzy_find_one("Alph Phone")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.score <= 2


def test_fuzzy_tuple_repr():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    assert repr(values) == "FuzzyTuple(('Alpha Phone', 'Beta Tablet'))"


def test_fuzzy_tuple_value_equality_and_hash_match_builtin_tuple_semantics():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    assert values == FuzzyTuple(["Alpha Phone", "Beta Tablet"], score_cutoff=100)
    assert values == ("Alpha Phone", "Beta Tablet")
    assert ("Alpha Phone", "Beta Tablet") == values
    assert values != ("Beta Tablet", "Alpha Phone")
    assert values != ["Alpha Phone", "Beta Tablet"]
    assert hash(values) == hash(("Alpha Phone", "Beta Tablet"))

    with pytest.raises(TypeError):
        hash(FuzzyTuple([["unhashable"]]))


def test_fuzzy_tuple_fuzzy_contains_returns_true_for_match():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    assert values.fuzzy_contains("Alpa Phone") is True
    assert values.fuzzy_contains("Coffee Grinder") is False


def test_fuzzy_tuple_fuzzy_get_returns_matched_value():
    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"])

    assert values.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_fuzzy_tuple_get_returns_default_on_miss():
    values = FuzzyTuple(["Alpha Phone"])

    assert values.fuzzy_get("Coffee Grinder") is None
    assert values.fuzzy_get("Coffee Grinder", default="missing") == "missing"
    assert not values.fuzzy_contains("Coffee Grinder")


def test_fuzzy_tuple_counts_all_matches():
    values = FuzzyTuple(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    assert values.fuzzy_count("Alpha") == 2
    assert values.fuzzy_count("Coffee Grinder") == 0


def test_fuzzy_tuple_finds_index_or_raises_value_error():
    values = FuzzyTuple(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    assert values.fuzzy_find_index("Alpa Phone") == 0

    with pytest.raises(ValueError, match="Coffee Grinder"):
        values.fuzzy_find_index("Coffee Grinder")


def test_fuzzy_tuple_counts_duplicate_normalized_values():
    values = FuzzyTuple(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

    assert values.fuzzy_count("ALPHA PHONE") == 2


def test_fuzzy_tuple_counts_duplicate_exact_values():
    values = FuzzyTuple(["Alpha Phone", "Alpha Phone", "Beta Tablet"])

    assert values.fuzzy_count("Alpha Phone") == 2


def test_fuzzy_tuple_does_not_count_unsearchable_value():
    values = FuzzyTuple(["xy", "Alpha Phone"])

    assert values.fuzzy_count("xy") == 0


def test_fuzzy_tuple_count_supports_distance_scorer():
    values = FuzzyTuple(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    assert values.fuzzy_count("Alph Phone") == 1


def test_fuzzy_tuple_iter_scores_matches_materialized_scores():
    values = FuzzyTuple(["Alpha Phone", "xy", "Beta Tablet"])

    assert list(values.fuzzy_iter_scores("Alpa Phone")) == values.fuzzy_score_all("Alpa Phone")


def test_fuzzy_tuple_iter_scores_reuses_cached_normalized_values():
    normalized_inputs: list[object] = []

    def normalize(value: object) -> str | None:
        normalized_inputs.append(value)
        if not isinstance(value, str):
            return None
        return value.casefold()

    values = FuzzyTuple(["Alpha Phone", "Beta Tablet"], normalizer=normalize, score_cutoff=0)

    list(values.fuzzy_iter_scores("Alpa Phone"))

    assert normalized_inputs == ["Alpha Phone", "Beta Tablet", "Alpa Phone"]


def test_fuzzy_tuple_with_config_updates_policy_without_mutating_source():
    values = FuzzyTuple(["Alpha Phone"], score_cutoff=100)

    permissive = values.with_config(score_cutoff=None)

    assert tuple(permissive) == ("Alpha Phone",)
    assert values.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_fuzzy_tuple_with_config_preserves_score_hint():
    values = FuzzyTuple(["Alpha Phone"], score_hint=100)

    assert values.with_config(score_cutoff=None)._index._score_hint == 100
