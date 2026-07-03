from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import (
    FrozenFuzzyDict,
    FrozenFuzzySet,
    FuzzyDict,
    FuzzySet,
    IndexStrategy,
    MappingMatch,
    Match,
    ScorerType,
)
from tests.helpers import casefold_string, normalize_boolean_one, require_not_none
from tests.helpers import mapping_match_signature as mapping_signature
from tests.helpers import positioned_value_match_signature as value_signature


class TestFuzzyDictStrategy:
    def test_default_strategy_is_sequence(self):
        values = FuzzyDict({"Alpha Phone": 1})

        assert values._strategy == IndexStrategy.SEQUENCE

    def test_strategies_return_same_public_result_shape(self):
        sequence = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})
        keyed = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2}, strategy=IndexStrategy.KEYED)

        sequence_item = sequence.fuzzy_find_item("Alpa Phone")
        keyed_item = keyed.fuzzy_find_item("Alpa Phone")
        sequence_key = sequence.fuzzy_find_key("Alpa Phone")
        keyed_key = keyed.fuzzy_find_key("Alpa Phone")

        assert isinstance(sequence_item, MappingMatch)
        assert isinstance(keyed_item, MappingMatch)
        assert isinstance(sequence_key, Match)
        assert isinstance(keyed_key, Match)
        assert mapping_signature(sequence_item) == mapping_signature(keyed_item)
        assert value_signature(sequence_key) == value_signature(keyed_key)
        assert sequence_item.index is None
        assert keyed_item.index is None

    def test_unsearchable_key_is_excluded_by_both_strategies(self):
        sequence = FuzzyDict({"xy": 1, "Alpha Phone": 2})
        keyed = FuzzyDict({"xy": 1, "Alpha Phone": 2}, strategy=IndexStrategy.KEYED)

        assert sequence.fuzzy_find_item("xy") is None
        assert keyed.fuzzy_find_item("xy") is None
        assert sequence["xy"] == keyed["xy"] == 1

    def test_exact_equal_query_returns_stored_key_for_both_strategies(self):
        sequence = FuzzyDict({True: "value"}, normalizer=normalize_boolean_one)
        keyed = FuzzyDict(
            {True: "value"},
            strategy=IndexStrategy.KEYED,
            normalizer=normalize_boolean_one,
        )

        assert require_not_none(sequence.fuzzy_find_item(1)).key is True
        assert require_not_none(keyed.fuzzy_find_item(1)).key is True
        assert sequence.fuzzy_contains_key(1)
        assert keyed.fuzzy_contains_key(1)

    def test_score_all_and_iter_scores_are_aligned_for_keyed_strategy(self):
        values = FuzzyDict({"Alpha Phone": 1, "xy": 2, "Beta Tablet": 3}, strategy="keyed")

        results = values.fuzzy_score_all("Alpa Phone")

        assert len(results) == 3
        first_result = require_not_none(results[0])
        assert isinstance(first_result, MappingMatch)
        assert first_result.key == "Alpha Phone"
        assert first_result.index is None
        assert results[1] is None
        assert results == list(values.fuzzy_iter_scores("Alpa Phone"))

    def test_mutation_methods_preserve_strategy_and_results(self):
        values = FuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2}, strategy=IndexStrategy.KEYED)

        values["Gamma Watch"] = 3
        del values["Beta Tablet"]
        values.fuzzy_discard("Gama Watch")

        assert values._strategy == IndexStrategy.KEYED
        assert dict(values) == {"Alpha Phone": 1}

    def test_with_config_can_switch_strategy(self):
        values = FuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

        switched = values.with_config(score_cutoff=None, strategy=IndexStrategy.KEYED)

        assert switched._strategy == IndexStrategy.KEYED
        assert values.fuzzy_get("Alpa Phone") is None
        assert switched.fuzzy_get("Alpa Phone") == 1

    def test_distance_scorer_exact_score_is_strategy_independent(self):
        sequence = FuzzyDict(
            {"Alpha Phone": 1},
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
        )
        keyed = FuzzyDict(
            {"Alpha Phone": 1},
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
            strategy=IndexStrategy.KEYED,
        )

        assert require_not_none(sequence.fuzzy_find_key("Alpha Phone")).score == 0
        assert require_not_none(keyed.fuzzy_find_key("Alpha Phone")).score == 0

    def test_exact_key_wins_equal_score_tie_and_is_removed_by_both_strategies(self):
        for strategy in IndexStrategy:
            values = FuzzyDict(
                {"ALPHA": 1, "alpha": 2},
                normalizer=casefold_string,
                score_cutoff=0,
                strategy=strategy,
            )

            assert values.fuzzy_get("alpha") == 2
            assert [match.key for match in values.fuzzy_find_items("alpha", limit=2)] == ["alpha", "ALPHA"]
            values.fuzzy_discard("alpha")
            assert dict(values) == {"ALPHA": 1}


class TestFuzzySetStrategy:
    def test_default_strategy_is_sequence(self):
        values = FuzzySet(["Alpha Phone"])

        assert values._strategy == IndexStrategy.SEQUENCE

    def test_strategies_return_same_public_result_shape(self):
        sequence = FuzzySet(["Alpha Phone", "Beta Tablet"])
        keyed = FuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

        sequence_match = sequence.fuzzy_find_one("Alpa Phone")
        keyed_match = keyed.fuzzy_find_one("Alpa Phone")

        assert isinstance(sequence_match, Match)
        assert isinstance(keyed_match, Match)
        assert value_signature(sequence_match) == value_signature(keyed_match)
        assert sequence_match.index is None
        assert keyed_match.index is None

    def test_unsearchable_value_is_excluded_by_both_strategies(self):
        sequence = FuzzySet(["xy", "Alpha Phone"])
        keyed = FuzzySet(["xy", "Alpha Phone"], strategy=IndexStrategy.KEYED)

        assert sequence.fuzzy_find_one("xy") is None
        assert keyed.fuzzy_find_one("xy") is None
        assert "xy" in sequence
        assert "xy" in keyed

    def test_exact_equal_query_returns_stored_value_for_both_strategies(self):
        sequence = FuzzySet([True], normalizer=normalize_boolean_one)
        keyed = FuzzySet(
            [True],
            strategy=IndexStrategy.KEYED,
            normalizer=normalize_boolean_one,
        )

        assert require_not_none(sequence.fuzzy_find_one(1)).value is True
        assert require_not_none(keyed.fuzzy_find_one(1)).value is True
        assert sequence.fuzzy_contains(1)
        assert keyed.fuzzy_contains(1)

    def test_score_all_and_iter_scores_are_aligned_for_keyed_strategy(self):
        values = FuzzySet(["Alpha Phone", "xy", "Beta Tablet"], strategy="keyed")

        results = values.fuzzy_score_all("Alpa Phone")

        assert list(values) == ["Alpha Phone", "xy", "Beta Tablet"]
        assert len(results) == 3
        first_result = require_not_none(results[0])
        assert isinstance(first_result, Match)
        assert first_result.value == "Alpha Phone"
        assert first_result.index is None
        assert results[1] is None
        assert results == list(values.fuzzy_iter_scores("Alpa Phone"))

    def test_set_operations_preserve_strategy(self):
        values = FuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

        union = values | {"Gamma Watch"}
        intersection = union & {"Alpha Phone", "Gamma Watch"}

        assert isinstance(union, FuzzySet)
        assert isinstance(intersection, FuzzySet)
        assert union._strategy == IndexStrategy.KEYED
        assert intersection._strategy == IndexStrategy.KEYED
        assert list(intersection) == ["Alpha Phone", "Gamma Watch"]

    def test_mutation_methods_preserve_strategy_and_results(self):
        values = FuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

        values.add("Gamma Watch")
        values.discard("Beta Tablet")
        values.fuzzy_discard("Gama Watch")

        assert values._strategy == IndexStrategy.KEYED
        assert list(values) == ["Alpha Phone"]

    def test_with_config_can_switch_strategy(self):
        values = FuzzySet(["Alpha Phone"], score_cutoff=100)

        switched = values.with_config(score_cutoff=None, strategy=IndexStrategy.KEYED)

        assert switched._strategy == IndexStrategy.KEYED
        assert values.fuzzy_get("Alpa Phone") is None
        assert switched.fuzzy_get("Alpa Phone") == "Alpha Phone"

    def test_exact_value_wins_equal_score_tie_and_is_removed_by_both_strategies(self):
        for strategy in IndexStrategy:
            values = FuzzySet(
                ["ALPHA", "alpha"],
                normalizer=casefold_string,
                score_cutoff=0,
                strategy=strategy,
            )

            assert values.fuzzy_get("alpha") == "alpha"
            assert [match.value for match in values.fuzzy_find_many("alpha", limit=2)] == ["alpha", "ALPHA"]
            values.fuzzy_discard("alpha")
            assert list(values) == ["ALPHA"]


class TestFrozenFuzzyDictStrategy:
    def test_strategies_return_same_public_result_shape(self):
        sequence = FrozenFuzzyDict({"Alpha Phone": 1, "Beta Tablet": 2})
        keyed = FrozenFuzzyDict(
            {"Alpha Phone": 1, "Beta Tablet": 2},
            strategy=IndexStrategy.KEYED,
        )

        sequence_item = require_not_none(sequence.fuzzy_find_item("Alpa Phone"))
        keyed_item = require_not_none(keyed.fuzzy_find_item("Alpa Phone"))
        sequence_key = require_not_none(sequence.fuzzy_find_key("Alpa Phone"))
        keyed_key = require_not_none(keyed.fuzzy_find_key("Alpa Phone"))

        assert mapping_signature(sequence_item) == mapping_signature(keyed_item)
        assert value_signature(sequence_key) == value_signature(keyed_key)
        assert sequence_item.index is None
        assert sequence_key.index is None

    def test_unsearchable_key_is_excluded_by_both_strategies(self):
        sequence = FrozenFuzzyDict({"xy": 1, "Alpha Phone": 2})
        keyed = FrozenFuzzyDict(
            {"xy": 1, "Alpha Phone": 2},
            strategy=IndexStrategy.KEYED,
        )

        assert sequence.fuzzy_find_item("xy") is None
        assert keyed.fuzzy_find_item("xy") is None
        assert sequence["xy"] == keyed["xy"] == 1

    def test_exact_equal_query_returns_stored_key_for_both_strategies(self):
        sequence = FrozenFuzzyDict({True: "value"}, normalizer=normalize_boolean_one)
        keyed = FrozenFuzzyDict(
            {True: "value"},
            strategy=IndexStrategy.KEYED,
            normalizer=normalize_boolean_one,
        )

        assert require_not_none(sequence.fuzzy_find_item(1)).key is True
        assert require_not_none(keyed.fuzzy_find_item(1)).key is True
        assert sequence.fuzzy_contains_key(1)
        assert keyed.fuzzy_contains_key(1)

    def test_with_config_can_switch_strategy(self):
        values = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=100)

        switched = values.with_config(score_cutoff=None, strategy=IndexStrategy.KEYED)

        assert switched._strategy == IndexStrategy.KEYED
        assert values.fuzzy_get("Alpa Phone") is None
        assert switched.fuzzy_get("Alpa Phone") == 1


class TestFrozenFuzzySetStrategy:
    def test_strategies_return_same_public_result_shape(self):
        sequence = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"])
        keyed = FrozenFuzzySet(["Alpha Phone", "Beta Tablet"], strategy=IndexStrategy.KEYED)

        sequence_match = require_not_none(sequence.fuzzy_find_one("Alpa Phone"))
        keyed_match = require_not_none(keyed.fuzzy_find_one("Alpa Phone"))

        assert value_signature(sequence_match) == value_signature(keyed_match)
        assert sequence_match.index is None

    def test_unsearchable_value_is_excluded_by_both_strategies(self):
        sequence = FrozenFuzzySet(["xy", "Alpha Phone"])
        keyed = FrozenFuzzySet(["xy", "Alpha Phone"], strategy=IndexStrategy.KEYED)

        assert sequence.fuzzy_find_one("xy") is None
        assert keyed.fuzzy_find_one("xy") is None
        assert "xy" in sequence
        assert "xy" in keyed

    def test_exact_equal_query_returns_stored_value_for_both_strategies(self):
        sequence = FrozenFuzzySet([True], normalizer=normalize_boolean_one)
        keyed = FrozenFuzzySet(
            [True],
            strategy=IndexStrategy.KEYED,
            normalizer=normalize_boolean_one,
        )

        assert require_not_none(sequence.fuzzy_find_one(1)).value is True
        assert require_not_none(keyed.fuzzy_find_one(1)).value is True
        assert sequence.fuzzy_contains(1)
        assert keyed.fuzzy_contains(1)

    def test_with_config_can_switch_strategy(self):
        values = FrozenFuzzySet(["Alpha Phone"], score_cutoff=100)

        switched = values.with_config(score_cutoff=None, strategy=IndexStrategy.KEYED)

        assert switched._strategy == IndexStrategy.KEYED
        assert values.fuzzy_get("Alpa Phone") is None
        assert switched.fuzzy_get("Alpa Phone") == "Alpha Phone"
