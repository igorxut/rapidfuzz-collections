from copy import copy, deepcopy

import pytest
from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FrozenFuzzyDict, IndexStrategy, MappingMatch, Match, ScorerType
from tests.helpers import require_not_none


def test_frozen_fuzzy_dict_behaves_like_immutable_mapping():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    assert len(values) == 2
    assert "Alpha Phone" in values
    assert "Coffee Grinder" not in values
    assert values["Alpha Phone"] == 1
    assert dict(values) == {"Alpha Phone": 1, "Beta Tablet": 2}


def test_frozen_fuzzy_dict_preserves_insertion_order():
    values = FrozenFuzzyDict({"Beta Tablet": 2, "Alpha Phone": 1})

    assert list(values) == ["Beta Tablet", "Alpha Phone"]


def test_frozen_fuzzy_dict_duplicate_keys_last_wins():
    values = FrozenFuzzyDict([("Alpha Phone", 1), ("Alpha Phone", 2)])

    assert len(values) == 1
    assert values["Alpha Phone"] == 2


def test_frozen_fuzzy_dict_deepcopy_preserves_recursive_graph():
    holder: list[object] = []
    values = FrozenFuzzyDict({"holder": holder})
    holder.append(values)

    result = deepcopy(values)

    assert result["holder"][0] is result


def test_frozen_fuzzy_dict_returns_key_match_for_exact_key():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    match = values.fuzzy_find_key("Alpha Phone")

    assert match == Match(
        value="Alpha Phone",
        score=100,
        index=None,
        query="Alpha Phone",
        normalized_query="alpha phone",
        normalized_value="alpha phone",
    )


def test_frozen_fuzzy_dict_returns_item_match_for_fuzzy_key():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    match = values.fuzzy_find_item("Alpa Phone")

    assert match is not None
    assert match.key == "Alpha Phone"
    assert match.value == 1
    assert match.score >= 80
    assert match.index is None
    assert match.query == "Alpa Phone"
    assert match.normalized_query == "alpa phone"
    assert match.normalized_key == "alpha phone"


def test_frozen_fuzzy_dict_uses_insertion_order_for_normalized_exact_key():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2})

    match = values.fuzzy_find_item("ALPHA PHONE")

    assert match is not None
    assert match.key == "Alpha Phone"
    assert match.value == 1
    assert match.index is None


def test_frozen_fuzzy_dict_excludes_unsearchable_key_from_fuzzy_lookup():
    values = FrozenFuzzyDict({"xy": 1, "Alpha Phone": 2})

    assert values.fuzzy_find_item("xy") is None
    assert values["xy"] == 1


def test_frozen_fuzzy_dict_find_items_returns_mapping_matches():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    matches = values.fuzzy_find_items("Alpha", limit=2)

    assert len(matches) == 2
    assert all(isinstance(match, MappingMatch) for match in matches)
    assert {match.key for match in matches} <= {"Alpha Phone", "Alpha Case"}
    assert {match.value for match in matches} <= {1, 2}


def test_frozen_fuzzy_dict_batch_methods_preserve_query_order():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

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


def test_frozen_fuzzy_dict_supports_distance_scorer():
    values = FrozenFuzzyDict(
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


def test_frozen_fuzzy_dict_get_returns_default_on_miss():
    values = FrozenFuzzyDict({"Alpha Phone": 1})

    assert values.fuzzy_get("Coffee Grinder") is None
    assert values.fuzzy_get("Coffee Grinder", default=0) == 0
    assert not values.fuzzy_contains_key("Coffee Grinder")


def test_frozen_fuzzy_dict_find_many_keys_returns_match_objects():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Alpha Case": 2, "Beta Tablet": 3})

    matches = values.fuzzy_find_keys("Alpha", limit=None)

    assert len(matches) == 2
    assert all(isinstance(m, Match) for m in matches)
    assert {m.value for m in matches} == {"Alpha Phone", "Alpha Case"}


def test_frozen_fuzzy_dict_has_no_mutation_methods():
    values = FrozenFuzzyDict({"Alpha Phone": 1})

    assert not hasattr(values, "fuzzy_discard")
    assert not hasattr(values, "fuzzy_discard_all")
    assert not hasattr(values, "fuzzy_retain_all")

    with pytest.raises(TypeError):
        values["New Key"] = 99  # type: ignore[index]

    with pytest.raises(TypeError):
        del values["Alpha Phone"]  # type: ignore[attr-defined]


def test_frozen_fuzzy_dict_repr():
    values = FrozenFuzzyDict({"Alpha Phone": 1})

    assert repr(values) == "FrozenFuzzyDict({'Alpha Phone': 1})"


def test_frozen_fuzzy_dict_copy_preserves_config():
    values = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    for result in (values.copy(), copy(values), deepcopy(values)):
        assert isinstance(result, FrozenFuzzyDict)
        assert dict(result) == {"Alpha Phone": 1}
        assert result.fuzzy_get("Alpha Phone") == 1
        assert result.fuzzy_get("Alpa Phone") is None


def test_frozen_fuzzy_dict_union_preserves_config():
    values = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    result = values | {"Beta Tablet": 2}
    reverse_result = {"Beta Tablet": 2} | values

    assert isinstance(result, FrozenFuzzyDict)
    assert isinstance(reverse_result, FrozenFuzzyDict)
    assert dict(result) == {"Alpha Phone": 1, "Beta Tablet": 2}
    assert dict(reverse_result) == {"Beta Tablet": 2, "Alpha Phone": 1}
    assert result.fuzzy_get("Alpa Phone") is None
    assert reverse_result.fuzzy_get("Alpa Phone") is None


def test_frozen_fuzzy_dict_reversed_iterates_keys_in_reverse_order():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    assert list(reversed(values)) == ["Beta Tablet", "Alpha Phone"]


def test_frozen_fuzzy_dict_accepts_mapping():
    source = {"Alpha Phone": 1, "Beta Tablet": 2}
    values = FrozenFuzzyDict(source)

    assert dict(values) == source


def test_frozen_fuzzy_dict_fuzzy_get_returns_matched_value():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})

    assert values.fuzzy_get("Alpa Phone") == 1


def test_frozen_fuzzy_dict_iter_scores_streams_mapping_matches_in_key_order():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "xy": 2, "Beta Tablet": 3})

    matches = list(values.fuzzy_iter_scores("Alpa Phone"))

    first_match = require_not_none(matches[0])
    assert isinstance(first_match, MappingMatch)
    assert first_match.key == "Alpha Phone"
    assert first_match.value == 1
    assert matches[1] is None


def test_frozen_fuzzy_dict_with_config_updates_key_policy_without_mutating_source():
    values = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

    permissive = values.with_config(score_cutoff=None)

    assert dict(permissive) == {"Alpha Phone": 1}
    assert values.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == 1


def test_frozen_fuzzy_dict_score_all_returns_one_result_per_key():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "xy": 2, "Beta Tablet": 3})

    results = values.fuzzy_score_all("Alpa Phone")

    assert len(results) == 3
    first_result = require_not_none(results[0])
    assert isinstance(first_result, MappingMatch)
    assert first_result.key == "Alpha Phone"
    assert first_result.value == 1
    assert results[1] is None  # "xy" excluded by default normalizer min_length(3)
    assert results == list(values.fuzzy_iter_scores("Alpa Phone"))


def test_frozen_fuzzy_dict_fromkeys_builds_mapping_from_iterable():
    values = FrozenFuzzyDict.fromkeys(["Alpha Phone", "Beta Tablet"], 42)

    assert dict(values) == {"Alpha Phone": 42, "Beta Tablet": 42}
    assert values.fuzzy_get("Alpa Phone") == 42


def test_frozen_fuzzy_dict_fromkeys_accepts_keyed_strategy():
    values = FrozenFuzzyDict.fromkeys(
        ["Alpha Phone", "Beta Tablet"],
        42,
        strategy=IndexStrategy.KEYED,
    )

    match = values.fuzzy_find_item("Alpa Phone")

    assert values._strategy == IndexStrategy.KEYED
    assert match is not None
    assert match.key == "Alpha Phone"
    assert match.value == 42
    assert match.index is None


def test_frozen_fuzzy_dict_with_config_preserves_score_hint():
    values = FrozenFuzzyDict({"Alpha Phone": 1}, score_hint=100)

    assert values.with_config(score_cutoff=None)._key_index._score_hint == 100


def test_frozen_fuzzy_dict_keyed_strategy_returns_same_public_result_classes():
    values = FrozenFuzzyDict(
        {"Alpha Phone": 1, "Beta Tablet": 2},
        strategy=IndexStrategy.KEYED,
    )

    key_match = values.fuzzy_find_key("Alpa Phone")
    item_match = values.fuzzy_find_item("Alpa Phone")

    assert isinstance(key_match, Match)
    assert isinstance(item_match, MappingMatch)
    assert key_match is not None
    assert item_match is not None
    assert key_match.value == "Alpha Phone"
    assert item_match.key == "Alpha Phone"
    assert key_match.index is None
    assert item_match.index is None


def test_frozen_fuzzy_dict_keyed_strategy_excludes_unsearchable_keys():
    values = FrozenFuzzyDict({"xy": 1, "Alpha Phone": 2}, strategy=IndexStrategy.KEYED)

    assert values.fuzzy_find_item("xy") is None
    assert values["xy"] == 1


def test_frozen_fuzzy_dict_keyed_strategy_preserves_aligned_score_output():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "xy": 2, "Beta Tablet": 3}, strategy="keyed")

    results = values.fuzzy_score_all("Alpa Phone")

    assert len(results) == 3
    first_result = require_not_none(results[0])
    assert isinstance(first_result, MappingMatch)
    assert first_result.key == "Alpha Phone"
    assert first_result.index is None
    assert results[1] is None
    assert results == list(values.fuzzy_iter_scores("Alpa Phone"))


def test_frozen_fuzzy_dict_keyed_strategy_batch_lookup_uses_keyed_path():
    values = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2}, strategy=IndexStrategy.KEYED)

    results = values.fuzzy_find_key_batch(["Alpa Phone", 42, "Beta Tablet"])

    first_result = require_not_none(results[0])
    third_result = require_not_none(results[2])
    assert first_result.value == "Alpha Phone"
    assert results[1] is None
    assert third_result.value == "Beta Tablet"


def test_frozen_fuzzy_dict_keyed_strategy_is_preserved_by_copy_and_config():
    values = FrozenFuzzyDict({"Alpha Phone": 1}, strategy=IndexStrategy.KEYED, score_cutoff=100)

    copied = values.copy()
    permissive = values.with_config(score_cutoff=None)

    assert copied._strategy == IndexStrategy.KEYED
    assert permissive._strategy == IndexStrategy.KEYED
    assert copied.fuzzy_get("Alpa Phone") is None
    assert permissive.fuzzy_get("Alpa Phone") == 1
