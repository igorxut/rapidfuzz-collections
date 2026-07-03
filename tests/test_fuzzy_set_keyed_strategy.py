from copy import copy, deepcopy

from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import FrozenFuzzySet, FuzzySet, IndexStrategy, Match, ScorerType
from tests.helpers import require_not_none


def test_fuzzy_lookup_returns_position_free_value_match():
    collection = FuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

    match = collection.fuzzy_find_one("alpha phne")

    assert isinstance(match, Match)
    assert match.value == "Alpha Phone"
    assert match.index is None


def test_unsearchable_value_is_excluded_from_fuzzy_lookup():
    collection = FuzzySet([1, "Alpha Phone"], strategy=IndexStrategy.KEYED)

    assert collection.fuzzy_get(1) is None
    assert 1 in collection
    assert collection.fuzzy_find_one("1") is None


def test_removing_first_normalized_collision_promotes_next_value():
    collection = FuzzySet(["Alpha Phone", "  alpha phone  "], strategy=IndexStrategy.KEYED)

    assert collection.fuzzy_get("ALPHA PHONE") == "Alpha Phone"

    collection.discard("Alpha Phone")

    assert collection.fuzzy_get("ALPHA PHONE") == "  alpha phone  "


def test_fuzzy_discard_all_removes_normalized_collisions():
    collection = FuzzySet(["Alpha Phone", "  alpha phone  ", "Beta Tablet"], strategy=IndexStrategy.KEYED)

    removed = collection.fuzzy_discard_all("alpha phone")

    assert removed == 2
    assert list(collection) == ["Beta Tablet"]


def test_removing_later_normalized_collision_keeps_first_value():
    collection = FuzzySet(["Alpha Phone", "  alpha phone  "], strategy=IndexStrategy.KEYED)

    collection.discard("  alpha phone  ")

    assert collection.fuzzy_get("ALPHA PHONE") == "Alpha Phone"


def test_three_normalized_collisions_retain_order_while_removed():
    collection = FuzzySet(["Alpha Phone", " alpha phone ", "ALPHA PHONE"], strategy=IndexStrategy.KEYED)

    collection.discard("Alpha Phone")
    assert collection.fuzzy_get("alpha phone") == " alpha phone "
    collection.discard(" alpha phone ")
    assert collection.fuzzy_get("alpha phone") == "ALPHA PHONE"


def test_mutation_score_stream_and_configuration_preserve_set_behavior():
    collection = FuzzySet(["Alpha Phone", "Beta Tablet"], score_cutoff=80, strategy=IndexStrategy.KEYED)
    collection.add("Gamma Watch")

    assert list(collection) == ["Alpha Phone", "Beta Tablet", "Gamma Watch"]
    assert list(collection.fuzzy_iter_scores("alpha phone")) == collection.fuzzy_score_all("alpha phone")

    relaxed = collection.with_config(score_cutoff=0)
    shallow = copy(collection)
    deep = deepcopy(collection)

    assert relaxed.fuzzy_get("missing query") is not None
    assert list(shallow) == list(collection)
    assert list(deep) == list(collection)


def test_named_exact_set_operations_preserve_keyed_collection_contract():
    collection = FuzzySet(["Alpha", "Beta", "Gamma"], strategy=IndexStrategy.KEYED)

    assert list(collection.difference(["Beta"])) == ["Alpha", "Gamma"]
    assert list(collection.intersection(["Beta", "Gamma"])) == ["Beta", "Gamma"]
    assert list(collection.symmetric_difference(["Gamma", "Delta"])) == ["Alpha", "Beta", "Delta"]
    assert list(collection.union(["Delta"])) == ["Alpha", "Beta", "Gamma", "Delta"]

    collection.difference_update(["Alpha"])
    collection.intersection_update(["Beta", "Gamma", "Delta"])
    collection.symmetric_difference_update(["Gamma", "Delta"])

    assert list(collection) == ["Beta", "Delta"]


def test_set_algebra_operators_preserve_keyed_collection_contract():
    collection = FuzzySet(["Alpha", "Beta"], score_cutoff=100, strategy=IndexStrategy.KEYED)
    other = FrozenFuzzySet(["Gamma", "Beta"])

    # noinspection PyTypeChecker
    assert list(collection & other) == ["Beta"]
    # noinspection PyTypeChecker
    assert list(collection | FrozenFuzzySet(["Gamma"])) == ["Alpha", "Beta", "Gamma"]
    # noinspection PyTypeChecker
    assert list(other | collection) == ["Gamma", "Beta", "Alpha"]
    # noinspection PyTypeChecker
    assert list(other & collection) == ["Beta"]
    # noinspection PyTypeChecker
    assert list(collection - FrozenFuzzySet(["Beta"])) == ["Alpha"]
    # noinspection PyTypeChecker
    assert list(other - collection) == ["Gamma"]
    # noinspection PyTypeChecker
    assert list(collection ^ other) == ["Alpha", "Gamma"]
    # noinspection PyTypeChecker
    assert list(other ^ collection) == ["Gamma", "Alpha"]

    collection |= {"Gamma"}
    collection &= {"Beta", "Gamma"}
    collection -= {"Beta"}
    collection ^= {"Delta"}

    assert list(collection) == ["Gamma", "Delta"]
    assert collection.fuzzy_get("Gama") is None


def test_batch_lookup_mutation_and_retain_api():
    collection = FuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

    collection.add("Gamma Watch")
    collection.add("Gamma Watch")

    assert "Gamma Watch" in collection
    assert repr(collection).startswith("FuzzySet(")
    assert collection.fuzzy_contains("alpha phne")
    assert not collection.fuzzy_contains(object())
    assert [match.value for match in collection.fuzzy_find_many("Alpha Phone")] == ["Alpha Phone"]
    assert collection.fuzzy_find_many(object()) == []
    first_group = collection.fuzzy_find_many_batch(["alpha phne"])[0]
    assert first_group[0].value == "Alpha Phone"
    assert collection.fuzzy_find_one_batch(["alpha phne", object()])[1] is None
    assert collection.fuzzy_get_batch(["alpha phne", object()], default="missing") == ["Alpha Phone", "missing"]

    collection.fuzzy_discard("alpha phne")
    collection.fuzzy_discard(object())
    removed = collection.fuzzy_retain_all("gamma watch")
    collection.discard("absent")

    assert removed == 1
    assert list(collection) == ["Gamma Watch"]
    assert list(collection.intersection()) == ["Gamma Watch"]
    collection.intersection_update()


def test_fuzzy_retain_all_no_op_when_everything_is_retained():
    collection = FuzzySet(["Alpha Phone", "Alpha Phone Case"], strategy=IndexStrategy.KEYED)

    removed = collection.fuzzy_retain_all("alpha phone")

    assert removed == 0
    assert set(collection) == {"Alpha Phone", "Alpha Phone Case"}


def test_fuzzy_retain_all_batch_delete_path():
    collection = FuzzySet(
        [
            "Alpha Phone",
            "Alpha Phone Case",
            "Alpha Phone Charger",
            "Alpha Phone Stand",
            "Beta Tablet",
        ],
        strategy=IndexStrategy.KEYED,
    )

    removed = collection.fuzzy_retain_all("alpha phone")

    assert removed == 1
    assert set(collection) == {
        "Alpha Phone",
        "Alpha Phone Case",
        "Alpha Phone Charger",
        "Alpha Phone Stand",
    }
    assert collection.fuzzy_get("alpha phone") == "Alpha Phone"


def test_distance_scorer_and_unsearchable_score_paths():
    collection = FuzzySet(
        [1, "Alpha Phone", "Beta Tablet"],
        scorer=Levenshtein.distance,
        scorer_type=ScorerType.DISTANCE,
        score_cutoff=2,
        strategy=IndexStrategy.KEYED,
    )

    exact = collection.fuzzy_find_one("Alpha Phone")

    assert exact is not None
    assert exact.score == 0
    assert collection.fuzzy_score_all(object()) == [None, None, None]
    scored = collection.fuzzy_score_all("alpha phne")
    assert scored[0] is None
    second_score = require_not_none(scored[1])
    assert second_score.value == "Alpha Phone"
    assert scored[2] is None
    collection.discard(1)


def test_four_way_collision_sequential_removal():
    collection = FuzzySet(
        ["Alpha Phone", " alpha phone ", "ALPHA PHONE", "  alpha phone  "], strategy=IndexStrategy.KEYED
    )

    assert require_not_none(collection.fuzzy_find_one("alpha phone")).value == "Alpha Phone"
    collection.discard("Alpha Phone")
    assert require_not_none(collection.fuzzy_find_one("alpha phone")).value == " alpha phone "
    collection.discard(" alpha phone ")
    assert require_not_none(collection.fuzzy_find_one("alpha phone")).value == "ALPHA PHONE"
    collection.discard("ALPHA PHONE")
    assert require_not_none(collection.fuzzy_find_one("alpha phone")).value == "  alpha phone  "
    collection.discard("  alpha phone  ")
    assert collection.fuzzy_find_one("alpha phone") is None


def test_remove_all_collision_members_then_readd():
    collection = FuzzySet(["Alpha Phone", "  alpha phone  "], strategy=IndexStrategy.KEYED)

    collection.discard("Alpha Phone")
    collection.discard("  alpha phone  ")

    collection.add("ALPHA PHONE")

    match = collection.fuzzy_find_one("alpha phone")
    assert match is not None
    assert match.value == "ALPHA PHONE"
    matches = collection.fuzzy_find_many("alpha phone")
    assert len(matches) == 1
    assert matches[0].value == "ALPHA PHONE"


def test_alternating_add_remove_collision_cycles():
    collection = FuzzySet(["Alpha Phone", "  alpha phone  "], strategy=IndexStrategy.KEYED)

    collection.discard("Alpha Phone")
    collection.add("ALPHA PHONE")

    match = collection.fuzzy_find_one("alpha phone")
    assert match is not None
    assert match.value == "  alpha phone  "


def test_find_many_excludes_removed_collision_values():
    collection = FuzzySet(["Alpha Phone", " alpha phone ", "  alpha phone  "], strategy=IndexStrategy.KEYED)

    collection.discard(" alpha phone ")

    matches = collection.fuzzy_find_many("alpha phone")
    assert len(matches) == 2
    match_values = {m.value for m in matches}
    assert match_values == {"Alpha Phone", "  alpha phone  "}
