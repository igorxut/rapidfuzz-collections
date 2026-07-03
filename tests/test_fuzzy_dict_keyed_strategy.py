from copy import copy, deepcopy

from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FuzzyDict, IndexStrategy, MappingMatch, Match, ScorerType
from tests.helpers import require_not_none


def test_fuzzy_lookup_returns_position_free_key_and_item_matches():
    collection = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2}, strategy=IndexStrategy.KEYED)

    key_match = collection.fuzzy_find_key("alpha phne")
    item_match = collection.fuzzy_find_item("alpha phne")

    assert isinstance(key_match, Match)
    assert key_match.value == "Alpha Phone"
    assert key_match.index is None
    assert isinstance(item_match, MappingMatch)
    assert item_match.key == "Alpha Phone"
    assert item_match.value == 1
    assert item_match.index is None


def test_unsearchable_key_is_excluded_from_fuzzy_lookup():
    collection = FuzzyDict({1: "numeric", "Alpha Phone": "phone"}, strategy=IndexStrategy.KEYED)

    assert collection.fuzzy_find_item(1) is None
    assert collection[1] == "numeric"
    assert collection.fuzzy_find_item("1") is None


def test_removing_first_normalized_collision_promotes_next_key():
    collection = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2}, strategy=IndexStrategy.KEYED)

    first_match = collection.fuzzy_find_key("ALPHA PHONE")

    assert first_match is not None
    assert first_match.value == "Alpha Phone"

    del collection["Alpha Phone"]

    next_match = collection.fuzzy_find_key("ALPHA PHONE")

    assert next_match is not None
    assert next_match.value == "  alpha phone  "


def test_fuzzy_discard_all_removes_normalized_collisions():
    collection = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2, "Beta Tablet": 3}, strategy=IndexStrategy.KEYED)

    removed = collection.fuzzy_discard_all("alpha phone")

    assert removed == 2
    assert dict(collection) == {"Beta Tablet": 3}


def test_removing_later_normalized_collision_keeps_first_key():
    collection = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2}, strategy=IndexStrategy.KEYED)

    del collection["  alpha phone  "]

    match = collection.fuzzy_find_key("ALPHA PHONE")
    assert match is not None
    assert match.value == "Alpha Phone"


def test_three_normalized_collisions_retain_order_while_removed():
    collection = FuzzyDict({"Alpha Phone": 1, " alpha phone ": 2, "ALPHA PHONE": 3}, strategy=IndexStrategy.KEYED)

    del collection["Alpha Phone"]
    first_remaining = collection.fuzzy_find_key("alpha phone")
    del collection[" alpha phone "]
    last_remaining = collection.fuzzy_find_key("alpha phone")

    assert first_remaining is not None
    assert first_remaining.value == " alpha phone "
    assert last_remaining is not None
    assert last_remaining.value == "ALPHA PHONE"


def test_mutation_score_stream_and_configuration_preserve_mapping_behavior():
    collection = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2}, score_cutoff=80, strategy=IndexStrategy.KEYED)
    collection["Alpha Phone"] = 10
    collection["Gamma Watch"] = 3

    assert collection.fuzzy_get("alpha phne") == 10
    assert list(collection.fuzzy_iter_scores("alpha phone")) == collection.fuzzy_score_all("alpha phone")

    relaxed = collection.with_config(score_cutoff=0)
    shallow = copy(collection)
    deep = deepcopy(collection)

    assert relaxed.fuzzy_get("missing query") is not None
    assert shallow["Alpha Phone"] == 10
    assert deep["Gamma Watch"] == 3
    assert dict(collection) == {"Alpha Phone": 10, "Beta Tablet": 2, "Gamma Watch": 3}


def test_batch_lookup_mutation_and_derived_constructor_api():
    collection = FuzzyDict.fromkeys(["Alpha Phone", "Beta Tablet"], value=1, strategy=IndexStrategy.KEYED)

    collection["Gamma Watch"] = 3

    assert len(collection) == 3
    assert repr(collection).startswith("FuzzyDict(")
    assert collection.fuzzy_contains_key("alpha phne")
    assert not collection.fuzzy_contains_key(object())
    assert [match.value if match else None for match in collection.fuzzy_find_key_batch(["alpha phne", object()])] == [
        "Alpha Phone",
        None,
    ]
    assert [match.value for match in collection.fuzzy_find_keys("Alpha Phone")] == ["Alpha Phone"]
    first_key_group = collection.fuzzy_find_keys_batch(["alpha phne"])[0]
    assert first_key_group[0].value == "Alpha Phone"
    batch_item = collection.fuzzy_find_item_batch(["alpha phne"])[0]
    batch_item = require_not_none(batch_item)
    assert batch_item.value == 1
    assert collection.fuzzy_find_items("alpha phne")[0].key == "Alpha Phone"
    first_item_group = collection.fuzzy_find_items_batch(["alpha phne"])[0]
    assert first_item_group[0].key == "Alpha Phone"
    assert collection.fuzzy_get_batch(["alpha phne", object()], default=-1) == [1, -1]

    collection.fuzzy_discard("alpha phne")
    collection.fuzzy_discard(object())
    removed = collection.fuzzy_retain_all("gamma watch")

    assert removed == 1
    assert dict(collection) == {"Gamma Watch": 3}


def test_mapping_union_and_reverse_iteration_preserve_configuration():
    collection = FuzzyDict({"Alpha Phone": 1}, score_cutoff=100, strategy=IndexStrategy.KEYED)

    collection |= {"Beta Tablet": 2}
    merged = collection | {"Gamma Watch": 3}
    reverse_merged = {"Gamma Watch": 3} | collection

    assert list(reversed(collection)) == ["Beta Tablet", "Alpha Phone"]
    # noinspection PyTypeChecker
    assert list(merged) == ["Alpha Phone", "Beta Tablet", "Gamma Watch"]
    # noinspection PyTypeChecker
    assert list(reverse_merged) == ["Gamma Watch", "Alpha Phone", "Beta Tablet"]
    # noinspection PyUnresolvedReferences
    assert merged.fuzzy_get("Gama Watch") is None


def test_distance_scorer_and_unsearchable_score_paths():
    collection = FuzzyDict(
        {1: "integer", "Alpha Phone": "phone", "Beta Tablet": "tablet"},
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
        strategy=IndexStrategy.KEYED,
    )

    exact = collection.fuzzy_find_key("Alpha Phone")

    assert exact is not None
    assert exact.score == 0
    assert collection.fuzzy_find_keys(object()) == []
    assert collection.fuzzy_score_all(object()) == [None, None, None]
    scored = collection.fuzzy_score_all("alpha phne")
    assert scored[0] is None
    second_score = require_not_none(scored[1])
    assert second_score.key == "Alpha Phone"
    assert scored[2] is None
    del collection[1]


def test_four_way_collision_sequential_removal():
    collection = FuzzyDict(
        {"Alpha Phone": 1, " alpha phone ": 2, "ALPHA PHONE": 3, "  alpha phone  ": 4},
        strategy=IndexStrategy.KEYED,
    )

    assert require_not_none(collection.fuzzy_find_key("alpha phone")).value == "Alpha Phone"
    del collection["Alpha Phone"]
    assert require_not_none(collection.fuzzy_find_key("alpha phone")).value == " alpha phone "
    del collection[" alpha phone "]
    assert require_not_none(collection.fuzzy_find_key("alpha phone")).value == "ALPHA PHONE"
    del collection["ALPHA PHONE"]
    assert require_not_none(collection.fuzzy_find_key("alpha phone")).value == "  alpha phone  "
    del collection["  alpha phone  "]
    assert collection.fuzzy_find_key("alpha phone") is None


def test_remove_all_collision_members_then_readd():
    collection = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2}, strategy=IndexStrategy.KEYED)

    del collection["Alpha Phone"]
    del collection["  alpha phone  "]

    collection["ALPHA PHONE"] = 3

    match = collection.fuzzy_find_key("alpha phone")
    assert match is not None
    assert match.value == "ALPHA PHONE"
    matches = collection.fuzzy_find_keys("alpha phone")
    assert len(matches) == 1
    assert matches[0].value == "ALPHA PHONE"


def test_alternating_add_remove_collision_cycles():
    collection = FuzzyDict({"Alpha Phone": 1, "  alpha phone  ": 2}, strategy=IndexStrategy.KEYED)

    del collection["Alpha Phone"]
    collection["ALPHA PHONE"] = 3

    match = collection.fuzzy_find_key("alpha phone")
    assert match is not None
    assert match.value == "  alpha phone  "


def test_find_many_excludes_removed_collision_values():
    collection = FuzzyDict({"Alpha Phone": 1, " alpha phone ": 2, "  alpha phone  ": 3}, strategy=IndexStrategy.KEYED)

    del collection[" alpha phone "]

    matches = collection.fuzzy_find_keys("alpha phone")
    assert len(matches) == 2
    match_values = {m.value for m in matches}
    assert match_values == {"Alpha Phone", "  alpha phone  "}


def test_fuzzy_retain_all_no_op_when_everything_is_retained():
    collection = FuzzyDict({"Alpha Phone": 1, "Alpha Phone Case": 2}, strategy=IndexStrategy.KEYED)

    removed = collection.fuzzy_retain_all("alpha phone")

    assert removed == 0
    assert dict(collection) == {"Alpha Phone": 1, "Alpha Phone Case": 2}


def test_fuzzy_retain_all_batch_delete_path():
    items = {
        "Alpha Phone": 1,
        "Alpha Phone Case": 2,
        "Alpha Phone Charger": 3,
        "Alpha Phone Stand": 4,
        "Beta Tablet": 5,
    }
    collection = FuzzyDict(items, strategy=IndexStrategy.KEYED)

    removed = collection.fuzzy_retain_all("alpha phone")

    assert removed == 1
    assert set(collection.keys()) == {
        "Alpha Phone",
        "Alpha Phone Case",
        "Alpha Phone Charger",
        "Alpha Phone Stand",
    }
    assert collection.fuzzy_find_key("alpha phone") is not None


def test_fuzzy_retain_all_rebuild_path():
    items = {f"Discontinued Accessory {i}": i for i in range(20)}
    items["Alpha Phone"] = 99
    collection = FuzzyDict(items, strategy=IndexStrategy.KEYED)
    removed = collection.fuzzy_retain_all("alpha phone")
    assert removed == 20
    assert list(collection.keys()) == ["Alpha Phone"]
    assert collection.fuzzy_find_key("alpha phne") is not None


def test_fuzzy_retain_all_index_correct_after_rebuild():
    a, b = "Alpha Phone", "  alpha phone  "
    items = {**{f"X{i}": i for i in range(10)}, a: 1, b: 2}
    collection = FuzzyDict(items, strategy=IndexStrategy.KEYED)
    collection.fuzzy_retain_all("alpha phone")
    assert set(collection.keys()) == {a, b}
    match = collection.fuzzy_find_key("ALPHA PHONE")
    assert match is not None
    assert match.value == a
    del collection[a]
    assert require_not_none(collection.fuzzy_find_key("ALPHA PHONE")).value == b
