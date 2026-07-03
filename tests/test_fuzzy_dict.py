from copy import copy, deepcopy

import pytest
from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FuzzyDict, IndexStrategy, MappingMatch, Match, ScorerType
from tests.helpers import require_not_none


@pytest.mark.parametrize("strategy", list(IndexStrategy))
def test_fuzzy_dict_new_key_assignment_is_atomic_when_normalization_fails(strategy):
    def rejecting_normalizer(value: object) -> str:
        if value == "bad":
            raise RuntimeError("normalization failed")
        if not isinstance(value, str):
            raise TypeError("test normalizer accepts strings only")
        return value

    values = FuzzyDict({"alpha": 1}, normalizer=rejecting_normalizer, strategy=strategy)

    with pytest.raises(RuntimeError, match="normalization failed"):
        values["bad"] = 2

    assert dict(values) == {"alpha": 1}


def test_fuzzy_dict_keyed_query_accepts_value_with_misleading_hashable_abc():
    query = (["x"],)
    values = FuzzyDict({"x": 1}, normalizer=str, score_cutoff=None, strategy=IndexStrategy.KEYED)

    assert values.fuzzy_find_key(query) is not None


def test_fuzzy_dict_behaves_like_mutable_mapping_and_rebuilds_key_index():
    values = FuzzyDict({"Alpha Phone": 1})

    values["Beta Tablet"] = 2
    assert dict(values) == {"Alpha Phone": 1, "Beta Tablet": 2}
    assert values.fuzzy_get("Bta Tablet") == 2

    del values["Beta Tablet"]
    assert values.fuzzy_get("Bta Tablet") is None


def test_fuzzy_dict_value_update_does_not_touch_key_index():
    values = FuzzyDict({"Alpha Phone": 1})

    values["Alpha Phone"] = 2

    assert values.fuzzy_get("Alpa Phone") == 2


def test_fuzzy_dict_returns_key_match_for_exact_key():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    match = values.fuzzy_find_key("Alpha Phone")

    assert match == Match(
        value="Alpha Phone",
        score=100,
        index=None,
        query="Alpha Phone",
        normalized_query="alpha phone",
        normalized_value="alpha phone",
    )


def test_fuzzy_dict_returns_item_match_for_fuzzy_key():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    match = values.fuzzy_find_item("Alpa Phone")

    assert match is not None
    assert match.key == "Alpha Phone"
    assert match.value == 1
    assert match.score >= 80
    assert match.index is None
    assert match.query == "Alpa Phone"
    assert match.normalized_query == "alpa phone"
    assert match.normalized_key == "alpha phone"


def test_fuzzy_dict_uses_insertion_order_for_normalized_exact_key():
    values = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2})

    match = values.fuzzy_find_item("ALPHA PHONE")

    assert match is not None
    assert match.key == "Alpha Phone"
    assert match.value == 1
    assert match.index is None


def test_fuzzy_dict_excludes_unsearchable_key_from_fuzzy_lookup():
    values = FuzzyDict({"xy": 1, "Alpha Phone": 2})

    assert values.fuzzy_find_item("xy") is None
    assert values["xy"] == 1


def test_fuzzy_dict_find_items_returns_mapping_matches():
    values = FuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    matches = values.fuzzy_find_items("Alpha", limit=2)

    assert len(matches) == 2
    assert all(isinstance(match, MappingMatch) for match in matches)
    assert {match.key for match in matches} <= {"Alpha Phone", "Alpha Case"}
    assert {match.value for match in matches} <= {1, 2}


def test_fuzzy_dict_batch_methods_preserve_query_order():
    values = FuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    key_matches = values.fuzzy_find_key_batch(["Bta Tablet", "Coffee Grinder"])
    item_matches = values.fuzzy_find_item_batch(["Alpa Phone", "Coffee Grinder"])
    key_groups = values.fuzzy_find_keys_batch(["Alpha", "Coffee Grinder"], limit=1)
    item_groups = values.fuzzy_find_items_batch(["Alpha", "Coffee Grinder"], limit=1)
    matched_values = values.fuzzy_get_batch(["Alpa Phone", "Coffee Grinder"], default=0)

    first_key_match = require_not_none(key_matches[0])
    assert key_matches[1] is None
    first_item_match = require_not_none(item_matches[0])
    assert item_matches[1] is None
    assert first_key_match.value == "Beta Tablet"
    assert first_item_match.key == "Alpha Phone"
    assert [[match.value for match in matches] for matches in key_groups] == [
        ["Alpha Phone"],
        [],
    ]
    assert [[match.key for match in matches] for matches in item_groups] == [
        ["Alpha Phone"],
        [],
    ]
    assert matched_values == [1, 0]


def test_fuzzy_dict_supports_distance_scorer():
    values = FuzzyDict(
        {"Alpha Phone": 1, "Beta Tablet": 2},
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    match = values.fuzzy_find_item("Alph Phone")

    assert match is not None
    assert match.key == "Alpha Phone"
    assert match.value == 1
    assert match.score <= 2


def test_fuzzy_dict_get_returns_default_on_miss():
    values = FuzzyDict({"Alpha Phone": 1})

    assert values.fuzzy_get("Coffee Grinder") is None
    assert values.fuzzy_get("Coffee Grinder", default=0) == 0
    assert not values.fuzzy_contains_key("Coffee Grinder")


def test_fuzzy_dict_discards_best_single_key_match():
    values = FuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    values.fuzzy_discard("Alpa Phone")

    assert dict(values) == {"Alpha Case": 2, "Beta Tablet": 3}
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_dict_discard_is_noop_on_miss():
    values = FuzzyDict({"Alpha Phone": 1})

    values.fuzzy_discard("Coffee Grinder")

    assert dict(values) == {"Alpha Phone": 1}


def test_fuzzy_dict_discards_all_key_matches_and_returns_count():
    values = FuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    removed = values.fuzzy_discard_all("Alpha")

    assert removed == 2
    assert dict(values) == {"Beta Tablet": 3}


def test_fuzzy_dict_retain_all_keeps_only_key_matches_and_returns_count():
    values = FuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    removed = values.fuzzy_retain_all("Alpha")

    assert removed == 1
    assert dict(values) == {"Alpha Phone": 1, "Alpha Case": 2}


def test_fuzzy_dict_discards_all_duplicate_normalized_keys():
    values = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2, "Beta Tablet": 3})

    removed = values.fuzzy_discard_all("ALPHA PHONE")

    assert removed == 2
    assert dict(values) == {"Beta Tablet": 3}


def test_fuzzy_dict_retain_all_keeps_duplicate_normalized_keys():
    values = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2, "Beta Tablet": 3})

    removed = values.fuzzy_retain_all("ALPHA PHONE")

    assert removed == 1
    assert dict(values) == {"Alpha Phone": 1, "  alpha phone  ": 2}


def test_fuzzy_dict_repr():
    values = FuzzyDict({"Alpha Phone": 1})

    assert repr(values) == "FuzzyDict({'Alpha Phone': 1})"


def test_fuzzy_dict_deepcopy_preserves_self_reference():
    values: FuzzyDict[str, object] = FuzzyDict()
    values["self"] = values

    result = deepcopy(values)

    assert result is not values
    assert result["self"] is result


def test_fuzzy_dict_len():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    assert len(values) == 2


def test_fuzzy_dict_discard_all_noop_on_miss_returns_zero():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    removed = values.fuzzy_discard_all("Coffee Grinder")

    assert removed == 0
    assert dict(values) == {"Alpha Phone": 1, "Beta Tablet": 2}


def test_fuzzy_dict_retain_all_noop_when_all_match_returns_zero():
    values = FuzzyDict({"Alpha Phone": 1, "Alpha Case": 2})

    removed = values.fuzzy_retain_all("Alpha")

    assert removed == 0
    assert dict(values) == {"Alpha Phone": 1, "Alpha Case": 2}


def test_fuzzy_dict_does_not_fuzzy_discard_unsearchable_key():
    values = FuzzyDict({"xy": 1, "Alpha Phone": 2})

    values.fuzzy_discard("xy")

    assert dict(values) == {"xy": 1, "Alpha Phone": 2}


def test_fuzzy_dict_discard_all_supports_distance_scorer():
    values = FuzzyDict(
        {"Alpha Phone": 1, "Beta Tablet": 2},
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
    )

    removed = values.fuzzy_discard_all("Alph Phone")

    assert removed == 1
    assert dict(values) == {"Beta Tablet": 2}


def test_fuzzy_dict_inherited_clear_updates_key_index():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    values.clear()

    assert dict(values) == {}
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_dict_copy_preserves_config():
    values = FuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FuzzyDict)
        assert dict(result) == {"Alpha Phone": 1}
        assert result.fuzzy_get("Alpha Phone") == 1
        assert result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_dict_inherited_pop_updates_key_index():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    popped = values.pop("Alpha Phone")

    assert popped == 1
    assert dict(values) == {"Beta Tablet": 2}
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_dict_inherited_popitem_updates_key_index():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    key, value = values.popitem()

    assert (key, value) == ("Beta Tablet", 2)
    assert dict(values) == {"Alpha Phone": 1}
    assert values.fuzzy_get("Bta Tablet") is None


def test_fuzzy_dict_inherited_setdefault_updates_key_index_for_new_key():
    values = FuzzyDict({"Alpha Phone": 1})

    result = values.setdefault("Beta Tablet", 2)

    assert result == 2
    assert dict(values) == {"Alpha Phone": 1, "Beta Tablet": 2}
    assert values.fuzzy_get("Bta Tablet") == 2


def test_fuzzy_dict_inherited_update_updates_key_index():
    values = FuzzyDict({"Alpha Phone": 1})

    values.update({"Beta Tablet": 2})

    assert dict(values) == {"Alpha Phone": 1, "Beta Tablet": 2}
    assert values.fuzzy_get("Bta Tablet") == 2


def test_fuzzy_dict_inherited_ior_updates_key_index():
    values = FuzzyDict({"Alpha Phone": 1})

    values |= {"Beta Tablet": 2}

    assert dict(values) == {"Alpha Phone": 1, "Beta Tablet": 2}
    assert values.fuzzy_get("Bta Tablet") == 2


def test_fuzzy_dict_union_preserves_config():
    values = FuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    result = values | {"Beta Tablet": 2}
    reverse_result = {"Beta Tablet": 2} | values

    assert isinstance(result, FuzzyDict)
    assert isinstance(reverse_result, FuzzyDict)
    assert dict(result) == {"Alpha Phone": 1, "Beta Tablet": 2}
    assert dict(reverse_result) == {"Beta Tablet": 2, "Alpha Phone": 1}
    assert result.fuzzy_get("Alpa Phone") is None
    assert reverse_result.fuzzy_get("Alpa Phone") is None


def test_fuzzy_dict_reversed_iterates_keys_in_reverse_order():
    values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    assert list(reversed(values)) == ["Beta Tablet", "Alpha Phone"]


def test_fuzzy_dict_fromkeys_builds_configured_fuzzy_mapping():
    values = FuzzyDict.fromkeys(["Alpha Phone", "Beta Tablet"], 0, score_cutoff=100)

    assert dict(values) == {"Alpha Phone": 0, "Beta Tablet": 0}
    assert values.fuzzy_get("Alpha Phone") == 0
    assert values.fuzzy_get("Alpa Phone") is None


def test_fuzzy_dict_iter_scores_streams_mapping_matches_in_key_order():
    values = FuzzyDict({"Alpha Phone": 1, "xy": 2, "Beta Tablet": 3})

    results = values.fuzzy_iter_scores("Alpa Phone")

    assert iter(results) is results
    matches = list(results)
    first_match = require_not_none(matches[0])
    assert isinstance(first_match, MappingMatch)
    assert first_match.key == "Alpha Phone"
    assert first_match.value == 1
    assert matches[1] is None


def test_fuzzy_dict_with_config_updates_key_policy_without_mutating_source():
    values = FuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    permissive = values.with_config(score_cutoff=None)

    assert dict(permissive) == {"Alpha Phone": 1}
    assert values.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == 1


def test_fuzzy_dict_score_all_returns_one_result_per_key():
    values = FuzzyDict({"Alpha Phone": 1, "xy": 2, "Beta Tablet": 3})

    results = values.fuzzy_score_all("Alpa Phone")

    assert len(results) == 3
    first_result = require_not_none(results[0])
    assert isinstance(first_result, MappingMatch)
    assert first_result.key == "Alpha Phone"
    assert first_result.value == 1
    assert results[1] is None  # "xy" excluded by default normalizer min_length(3)
    assert results == list(values.fuzzy_iter_scores("Alpa Phone"))


def test_fuzzy_dict_fromkeys_and_with_config_preserve_score_hint():
    values = FuzzyDict.fromkeys(["Alpha Phone"], 1, score_hint=100)

    derived = values.with_config(score_cutoff=None)

    assert values._key_index._score_hint == 100
    assert derived._key_index._score_hint == 100


def test_fuzzy_dict_rejects_non_callable_normalizer():
    with pytest.raises(TypeError, match="normalizer"):
        FuzzyDict({"Alpha Phone": 1}, normalizer=42)  # type: ignore[arg-type]


def test_fuzzy_dict_rejects_non_callable_scorer():
    with pytest.raises(TypeError, match="scorer"):
        FuzzyDict({"Alpha Phone": 1}, scorer="ratio")  # type: ignore[arg-type]
