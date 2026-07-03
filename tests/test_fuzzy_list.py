from collections.abc import Hashable
from copy import copy, deepcopy

import pytest
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

from rapidfuzz_collections import FuzzyList, Match, ScorerType
from rapidfuzz_collections.indexes import FuzzySequenceIndex
from tests.helpers import casefold_string, require_not_none


def test_fuzzy_list_accepts_value_with_misleading_hashable_abc():
    value = (["x"],)

    values = FuzzyList([value], normalizer=str, score_cutoff=None)

    assert values.fuzzy_get(value) == value


def test_sequence_index_returns_match_for_exact_value():
    index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

    match = index.find_one("Alpha Phone")

    assert match == Match(
        value="Alpha Phone",
        score=100,
        index=0,
        query="Alpha Phone",
        normalized_query="alpha phone",
        normalized_value="alpha phone",
    )


def test_sequence_index_returns_first_normalized_exact_match():
    index = FuzzySequenceIndex(["Alpha Phone", "  alpha phone  "])

    match = index.find_one("ALPHA PHONE")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index == 0
    assert match.normalized_query == "alpha phone"


def test_sequence_index_returns_fuzzy_close_match():
    index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

    match = index.find_one("Alpa Phone")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index == 0


def test_sequence_index_returns_none_for_unsearchable_query():
    index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

    assert index.find_one(None) is None
    assert index.find_one("x") is None


def test_sequence_index_find_many_excludes_unsearchable_value():
    index = FuzzySequenceIndex(["xy", "Alpha Phone"])

    matches = index.find_many("xy")

    assert matches == []


def test_sequence_index_find_many_returns_duplicate_exact_values():
    index = FuzzySequenceIndex(["Alpha Phone", "Alpha Phone", "Beta Tablet"])

    matches = index.find_many("Alpha Phone", limit=None)

    assert [match.index for match in matches] == [0, 1]
    assert [match.value for match in matches] == ["Alpha Phone", "Alpha Phone"]


def test_sequence_index_find_many_exact_values_respects_limit():
    index = FuzzySequenceIndex(["Alpha Phone", "Alpha Phone", "Alpha Phone"])

    matches = index.find_many("Alpha Phone", limit=2)

    assert [match.index for match in matches] == [0, 1]


def test_sequence_index_batch_methods_preserve_query_order():
    index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

    one_matches = index.find_one_batch(["Bta Tablet", "Coffee Grinder", "Alpha Phone"])
    many_matches = index.find_many_batch(["Alpha", "Coffee Grinder"], limit=1)

    first_match = require_not_none(one_matches[0])
    assert one_matches[1] is None
    third_match = require_not_none(one_matches[2])
    assert first_match.value == "Beta Tablet"
    assert third_match.value == "Alpha Phone"
    assert [[match.value for match in matches] for matches in many_matches] == [
        ["Alpha Phone"],
        [],
    ]


def test_sequence_index_ignores_unsearchable_values():
    index = FuzzySequenceIndex([None, 1, "xy", "Alpha Phone"])

    match = index.find_one("alpha")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index == 3


def test_sequence_index_supports_distance_scorer():
    index = FuzzySequenceIndex(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    match = index.find_one("Alph Phone")

    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.score <= 2


def test_sequence_index_exact_shortcut_uses_distance_score():
    index = FuzzySequenceIndex(
        ["Alpha Phone"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    match = index.find_one("Alpha Phone")

    assert match is not None
    assert match.score == 0


def test_fuzzy_list_behaves_like_mutable_sequence_and_rebuilds_index():
    values = FuzzyList(["Alpha Phone"])

    values.append("Beta Tablet")
    assert list(values) == ["Alpha Phone", "Beta Tablet"]
    assert values.fuzzy_get("Bta Tablet") == "Beta Tablet"

    values[1] = "Gamma Watch"
    assert values.fuzzy_get("Bta Tablet") is None
    assert values.fuzzy_get("Gama Watch") == "Gamma Watch"


def test_fuzzy_list_find_many_returns_match_objects():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    matches = values.fuzzy_find_many("Alpha", limit=2)

    assert len(matches) == 2
    assert all(isinstance(match, Match) for match in matches)
    assert {match.value for match in matches} <= {"Alpha Phone", "Alpha Case"}


def test_fuzzy_list_batch_methods_preserve_query_order():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

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


def test_fuzzy_list_counts_all_matches():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    assert values.fuzzy_count("Alpha") == 2
    assert values.fuzzy_count("Coffee Grinder") == 0


def test_fuzzy_list_finds_index_or_raises_value_error():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    assert values.fuzzy_find_index("Alpa Phone") == 0

    with pytest.raises(ValueError, match="Coffee Grinder"):
        values.fuzzy_find_index("Coffee Grinder")


def test_fuzzy_list_discards_best_single_match():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    values.fuzzy_discard("Alpa Phone")

    assert list(values) == ["Alpha Case", "Beta Tablet"]
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_discard_prefers_exact_value_on_equal_score():
    values = FuzzyList(["ALPHA", "alpha"], normalizer=casefold_string, score_cutoff=0)

    values.fuzzy_discard("alpha")

    assert list(values) == ["ALPHA"]


def test_fuzzy_list_discard_is_noop_on_miss():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    values.fuzzy_discard("Coffee Grinder")

    assert list(values) == ["Alpha Phone", "Beta Tablet"]


def test_fuzzy_list_remove_deletes_best_match():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    values.fuzzy_remove("Alpa Phone")

    assert list(values) == ["Beta Tablet"]


def test_fuzzy_list_remove_raises_value_error_on_miss():
    values = FuzzyList(["Alpha Phone"])

    with pytest.raises(ValueError, match="Coffee Grinder"):
        values.fuzzy_remove("Coffee Grinder")


def test_fuzzy_list_discards_all_matches_and_returns_count():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    removed = values.fuzzy_discard_all("Alpha")

    assert removed == 2
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_list_retain_all_keeps_only_matches_and_returns_count():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    removed = values.fuzzy_retain_all("Alpha")

    assert removed == 1
    assert list(values) == ["Alpha Phone", "Alpha Case"]


def test_fuzzy_list_counts_duplicate_normalized_values():
    values = FuzzyList(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

    assert values.fuzzy_count("ALPHA PHONE") == 2


def test_fuzzy_list_counts_duplicate_exact_values():
    values = FuzzyList(["Alpha Phone", "Alpha Phone", "Beta Tablet"])

    assert values.fuzzy_count("Alpha Phone") == 2


def test_fuzzy_list_discards_all_duplicate_exact_values():
    values = FuzzyList(["Alpha Phone", "Alpha Phone", "Beta Tablet"])

    removed = values.fuzzy_discard_all("Alpha Phone")

    assert removed == 2
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_list_discards_all_duplicate_normalized_values():
    values = FuzzyList(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

    removed = values.fuzzy_discard_all("ALPHA PHONE")

    assert removed == 2
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_list_retain_all_keeps_duplicate_normalized_values():
    values = FuzzyList(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

    removed = values.fuzzy_retain_all("ALPHA PHONE")

    assert removed == 1
    assert list(values) == ["Alpha Phone", "  alpha phone  "]


def test_fuzzy_list_does_not_count_unsearchable_value():
    values = FuzzyList(["xy", "Alpha Phone"])

    assert values.fuzzy_count("xy") == 0


def test_fuzzy_list_does_not_fuzzy_discard_unsearchable_value():
    values = FuzzyList(["xy", "Alpha Phone"])

    values.fuzzy_discard("xy")

    assert list(values) == ["xy", "Alpha Phone"]
    assert "xy" in values


def test_fuzzy_list_negative_slice_before_unsearchable_tail_keeps_index_aligned():
    def normalizer(value: object) -> str | None:
        return value if isinstance(value, str) else None

    values = FuzzyList(
        ["Alpha Phone", "Beta Tablet", 2],
        normalizer=normalizer,
        scorer=ratio,
        score_cutoff=50,
    )

    del values[:-1]

    assert list(values) == [2]
    assert values.fuzzy_find_one("Beta") is None


def test_fuzzy_list_count_supports_distance_scorer():
    values = FuzzyList(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    assert values.fuzzy_count("Alph Phone") == 1


def test_fuzzy_list_discard_all_supports_distance_scorer():
    values = FuzzyList(
        ["Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    removed = values.fuzzy_discard_all("Alph Phone")

    assert removed == 1
    assert list(values) == ["Beta Tablet"]


def test_fuzzy_list_repr():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    assert repr(values) == "FuzzyList(['Alpha Phone', 'Beta Tablet'])"


def test_fuzzy_list_value_equality_matches_builtin_list_semantics():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    assert values == FuzzyList(["Alpha Phone", "Beta Tablet"], score_cutoff=100)
    assert values == ["Alpha Phone", "Beta Tablet"]
    assert ["Alpha Phone", "Beta Tablet"] == values
    assert values != ["Beta Tablet", "Alpha Phone"]
    assert values != ("Alpha Phone", "Beta Tablet")

    assert not isinstance(values, Hashable)


def test_fuzzy_list_recursive_repr_and_deepcopy_preserve_self_reference():
    values: FuzzyList[object] = FuzzyList([])
    values.append(values)

    result = deepcopy(values)

    assert repr(values) == "FuzzyList([...])"
    assert result is not values
    assert result[0] is result


def test_fuzzy_list_getitem_int():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    assert values[0] == "Alpha Phone"
    assert values[-1] == "Beta Tablet"


def test_fuzzy_list_getitem_slice():
    values = FuzzyList(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

    assert values[0:2] == ["Alpha Phone", "Beta Tablet"]
    assert values[1:] == ["Beta Tablet", "Gamma Watch"]


def test_fuzzy_list_delitem_int():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    del values[0]

    assert list(values) == ["Beta Tablet"]
    assert values.fuzzy_get("Alpha Phone") is None


def test_fuzzy_list_delitem_slice():
    values = FuzzyList(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

    del values[0:2]

    assert list(values) == ["Gamma Watch"]


def test_fuzzy_list_fuzzy_contains_returns_true_for_match():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    assert values.fuzzy_contains("Alpa Phone") is True
    assert values.fuzzy_contains("Coffee Grinder") is False


def test_fuzzy_list_fuzzy_find_one_returns_match():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    match = values.fuzzy_find_one("Alpa Phone")

    assert match is not None
    assert match.value == "Alpha Phone"


def test_fuzzy_list_insert_marks_index_dirty_and_query_works():
    values = FuzzyList(["Beta Tablet"])

    values.insert(0, "Alpha Phone")

    assert values._index.is_dirty
    assert values.fuzzy_get("Alpa Phone") == "Alpha Phone"
    assert list(values) == ["Alpha Phone", "Beta Tablet"]


def test_fuzzy_list_discard_all_noop_on_miss_returns_zero():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    removed = values.fuzzy_discard_all("Coffee Grinder")

    assert removed == 0
    assert list(values) == ["Alpha Phone", "Beta Tablet"]


def test_fuzzy_list_retain_all_noop_when_all_match_returns_zero():
    values = FuzzyList(["Alpha Phone", "Alpha Case"])

    removed = values.fuzzy_retain_all("Alpha")

    assert removed == 0
    assert list(values) == ["Alpha Phone", "Alpha Case"]


def test_fuzzy_list_discard_all_removes_all_returns_empty():
    values = FuzzyList(["Alpha Phone", "Alpha Case"])

    removed = values.fuzzy_discard_all("Alpha")

    assert removed == 2
    assert list(values) == []
    assert values.fuzzy_get("Alpha Phone") is None


def test_fuzzy_list_retain_all_removes_all_when_no_match():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    removed = values.fuzzy_retain_all("Coffee Grinder")

    assert removed == 2
    assert list(values) == []


def test_fuzzy_list_inherited_clear_updates_fuzzy_index():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    values.clear()

    assert list(values) == []
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_inherited_extend_updates_fuzzy_index_incrementally():
    values = FuzzyList(["Alpha Phone"])

    values.extend(["Beta Tablet", "Gamma Watch"])

    assert list(values) == ["Alpha Phone", "Beta Tablet", "Gamma Watch"]
    assert values.fuzzy_get("Gama Watch") == "Gamma Watch"


def test_fuzzy_list_add_returns_fuzzy_list_and_preserves_config():
    values = FuzzyList(["Alpha Phone"], score_cutoff=100)

    result = values + ["Beta Tablet"]

    assert isinstance(result, FuzzyList)
    assert list(result) == ["Alpha Phone", "Beta Tablet"]
    assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
    assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_add_rejects_non_list_iterable():
    values = FuzzyList(["Alpha Phone"])

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = values + ("Beta Tablet",)


def test_fuzzy_list_copy_preserves_config():
    values = FuzzyList(["Alpha Phone"], score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FuzzyList)
        assert list(result) == ["Alpha Phone"]
        assert result.fuzzy_get("Alpha Phone") == "Alpha Phone"
        assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_inherited_iadd_updates_fuzzy_index():
    values = FuzzyList(["Alpha Phone"])

    values += ["Beta Tablet"]

    assert list(values) == ["Alpha Phone", "Beta Tablet"]
    assert values.fuzzy_get("Bta Tablet") == "Beta Tablet"


def test_fuzzy_list_inherited_imul_updates_fuzzy_index():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    values *= 2

    assert list(values) == ["Alpha Phone", "Beta Tablet", "Alpha Phone", "Beta Tablet"]
    assert values.fuzzy_count("Alpha Phone") == 2


def test_fuzzy_list_mul_returns_fuzzy_list_and_preserves_config():
    values = FuzzyList(["Alpha Phone"], score_cutoff=100)

    result = values * 2
    reverse_result = 2 * values

    assert isinstance(result, FuzzyList)
    assert isinstance(reverse_result, FuzzyList)
    assert list(result) == ["Alpha Phone", "Alpha Phone"]
    assert list(reverse_result) == ["Alpha Phone", "Alpha Phone"]
    assert result.fuzzy_get("Alpa Phone") is None
    assert reverse_result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_inherited_pop_updates_fuzzy_index():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    popped = values.pop(0)

    assert popped == "Alpha Phone"
    assert list(values) == ["Beta Tablet"]
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_inherited_remove_updates_fuzzy_index():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    values.remove("Alpha Phone")

    assert list(values) == ["Beta Tablet"]
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_list_inherited_reverse_updates_fuzzy_index_order():
    values = FuzzyList(["Alpha Phone", "Alpha Case", "Beta Tablet"])

    values.reverse()

    match = values.fuzzy_find_one("Alpa Phone")
    assert list(values) == ["Beta Tablet", "Alpha Case", "Alpha Phone"]
    assert match is not None
    assert match.value == "Alpha Phone"
    assert match.index == 2


def test_fuzzy_list_reversed_iterates_in_reverse_order():
    values = FuzzyList(["Alpha Phone", "Beta Tablet"])

    assert list(reversed(values)) == ["Beta Tablet", "Alpha Phone"]


def test_fuzzy_list_sort_updates_fuzzy_index_order():
    values = FuzzyList(["Beta Tablet", "Alpha Phone"])

    values.sort()

    assert list(values) == ["Alpha Phone", "Beta Tablet"]
    assert values.fuzzy_find_index("Bta Tablet") == 1


def test_fuzzy_list_iter_scores_matches_materialized_scores_without_list_allocation():
    values = FuzzyList(["Alpha Phone", "xy", "Beta Tablet"])

    results = values.fuzzy_iter_scores("Alpa Phone")

    assert iter(results) is results
    assert list(results) == values.fuzzy_score_all("Alpa Phone")


def test_fuzzy_list_with_config_builds_new_policy_without_mutating_source():
    values = FuzzyList(["Alpha Phone"], score_cutoff=100)

    permissive = values.with_config(score_cutoff=None)

    assert permissive is not values
    assert list(permissive) == ["Alpha Phone"]
    assert values.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_fuzzy_list_with_config_none_normalizer_restores_default_normalization():
    values = FuzzyList(["Alpha Phone"], normalizer=lambda value: None)

    defaulted = values.with_config(normalizer=None)

    assert values.fuzzy_get("Alpa Phone") is None
    assert defaulted.fuzzy_get("Alpa Phone") == "Alpha Phone"


def test_fuzzy_list_with_config_preserves_and_clears_score_hint():
    values = FuzzyList(["Alpha Phone"], score_hint=100)

    preserved = values.copy()
    cleared = values.with_config(score_hint=None)

    assert preserved._index._score_hint == 100
    assert values.fuzzy_get("Alpa Phone") == "Alpha Phone"
    assert cleared._index._score_hint is None


def test_fuzzy_list_unhashable_query_does_not_use_exact_tiebreak():
    def normalizer(value: object) -> str | None:
        if isinstance(value, list):
            return "item"
        return None

    # ["a"] at position 1 equals the query by value, but ["a"] is unhashable so
    # _exact_source_indexes returns () — exact tie-breaking is skipped.
    # Source position 0 (["b"]) wins instead.
    lst = FuzzyList(
        [["b"], ["a"]],
        normalizer=normalizer,
        score_cutoff=0,
    )
    result = lst.fuzzy_find_one(["a"])

    assert result is not None
    assert result.value == ["b"]


def test_fuzzy_list_rejects_non_callable_normalizer():
    with pytest.raises(TypeError, match="normalizer"):
        FuzzyList(["Alpha Phone"], normalizer=42)  # type: ignore[arg-type]


def test_fuzzy_list_rejects_non_callable_scorer():
    with pytest.raises(TypeError, match="scorer"):
        FuzzyList(["Alpha Phone"], scorer="ratio")  # type: ignore[arg-type]
