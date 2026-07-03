"""Direct tests for keyed fuzzy indexes."""

import pytest
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

from rapidfuzz_collections import Normalizer, ScorerType
from rapidfuzz_collections.indexes import ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex
from tests.helpers import (
    SearchableEqualityKey,
    casefold_string,
    normalize_equality_key,
    require_not_none,
)
from tests.helpers import keyed_value_match_signature as match_signature


def test_mutable_keyed_add_is_atomic_when_normalization_fails():
    def rejecting_normalizer(value: object) -> str:
        if value == "bad":
            raise RuntimeError("normalization failed")
        if not isinstance(value, str):
            raise TypeError("test normalizer accepts strings only")
        return value

    index = MutableFuzzyKeyedIndex(["alpha"], normalizer=rejecting_normalizer)

    with pytest.raises(RuntimeError, match="normalization failed"):
        index.add("bad")

    assert tuple(index._exact_values) == ("alpha",)
    assert tuple(index._choices) == ("alpha",)


@pytest.mark.parametrize("index_class", [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex])
def test_custom_scorer_fallback_evaluates_each_searchable_choice(index_class):
    calls = 0

    def counting_similarity(_query: str, _value: str) -> int:
        nonlocal calls
        calls += 1
        return 50

    index = index_class(
        ["Alpha Phone", "Beta Tablet", "Gamma Camera"],
        scorer=counting_similarity,
        score_cutoff=0,
    )

    index.find_one("unindexed query")
    assert calls == 3

    calls = 0
    index.find_many("unindexed query", limit=None)
    assert calls == 3


@pytest.mark.parametrize("index_class", [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex])
def test_equal_score_non_exact_matches_preserve_insertion_order(index_class):
    def constant_similarity(_query: str, _value: str) -> int:
        return 50

    values = ["Gamma Camera", "Alpha Phone", "Beta Tablet"]
    index = index_class(values, scorer=constant_similarity, score_cutoff=0)

    matches = index.find_many("unindexed query", limit=None)

    assert [match.value for match in matches] == values


@pytest.mark.parametrize("index_class", [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex])
def test_custom_score_all_aligns_unsearchable_accepted_and_rejected_values(index_class):
    def normalizer(value: object) -> str | None:
        return value.casefold() if isinstance(value, str) and value != "skip" else None

    def scorer(_query: str, value: str) -> int:
        return 100 if value == "alpha phone" else 0

    values = ["skip", "Alpha Phone", "Beta Tablet"]
    index = index_class(values, normalizer=normalizer, scorer=scorer, score_cutoff=50)

    results = index.score_all(values, "query")

    assert results[0] is None
    assert require_not_none(results[1]).value == "Alpha Phone"
    assert results[2] is None


class TestImmutableFuzzyKeyedIndex:
    def test_find_one_returns_normalized_exact_first_value(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone", "  alpha phone  "])

        match = index.find_one("ALPHA PHONE")

        assert match is not None
        assert match.value == "Alpha Phone"
        assert match.score == 100
        assert match.normalized_query == "alpha phone"
        assert match.normalized_value == "alpha phone"

    def test_find_many_returns_hashable_values_without_positions(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone", "Beta Tablet"])

        matches = index.find_many("Alpa Phone", limit=None)

        assert matches
        assert matches[0].value == "Alpha Phone"
        assert not hasattr(matches[0], "index")

    def test_exact_match_excludes_unsearchable_value(self):
        index = ImmutableFuzzyKeyedIndex([1, "Alpha Phone"])

        assert index.exact_match(1, query=1) is None

    def test_exact_match_returns_stored_equal_value(self):
        def normalizer(value: object) -> str:
            if value in (True, 1):
                return "boolean-one"
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings, True, and 1 only")
            return value

        index = ImmutableFuzzyKeyedIndex([True], normalizer=normalizer)

        match = index.exact_match(1, query=1)

        assert match is not None
        assert match.value is True

    def test_exact_shortcut_uses_configured_score_and_cutoff(self):
        # noinspection PyUnusedLocal
        def score_exact_pair(left: str, right: str, **kwargs: object) -> int:
            return 40

        index = ImmutableFuzzyKeyedIndex(
            ["Alpha Phone"],
            scorer=score_exact_pair,
            score_cutoff=50,
        )

        assert index.find_one("Alpha Phone") is None

        accepted = ImmutableFuzzyKeyedIndex(
            ["Alpha Phone"],
            scorer=score_exact_pair,
            score_cutoff=40,
        ).find_one("Alpha Phone")
        assert accepted is not None
        assert accepted.score == 40

    def test_unsearchable_query_returns_no_fuzzy_matches(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone"])

        assert index.find_one(1) is None
        assert index.find_many(1) == []
        assert index.score_all(["Alpha Phone"], 1) == [None]
        assert list(index.iter_scores(["Alpha Phone"], 1)) == [None]

    def test_simple_custom_similarity_scorer_uses_direct_path(self):
        def normalizer(value: object) -> str | None:
            return value.casefold() if isinstance(value, str) else None

        def same_initial(left: str, right: str) -> int:
            return 100 if left[:1] == right[:1] else 0

        index = ImmutableFuzzyKeyedIndex(
            ["abc"],
            normalizer=normalizer,
            scorer=same_initial,
            score_cutoff=None,
        )

        match = index.find_one("ax")

        assert match is not None
        assert match.value == "abc"
        assert match.score == 100

    def test_custom_scorer_without_metadata_skips_exact_shortcut(self):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        # noinspection PyUnusedLocal
        def prefer_non_exact(left: str, right: str) -> int:
            return 100 if right == "abd" else 80

        immutable = ImmutableFuzzyKeyedIndex(
            ["abc", "abd"],
            normalizer=normalizer,
            scorer=prefer_non_exact,
            score_cutoff=80,
        )
        mutable = MutableFuzzyKeyedIndex(
            ["abc", "abd"],
            normalizer=normalizer,
            scorer=prefer_non_exact,
            score_cutoff=80,
        )

        immutable_match = require_not_none(immutable.find_one("abc"))
        mutable_match = require_not_none(mutable.find_one("abc"))

        assert immutable_match.value == mutable_match.value == "abd"
        assert immutable_match.score == mutable_match.score == 100

    @pytest.mark.parametrize("index_class", [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex])
    @pytest.mark.parametrize("limit", [2, None])
    def test_exact_equality_breaks_equal_score_tie(self, index_class, limit):
        index = index_class(["ALPHA", "alpha"], normalizer=casefold_string, scorer=ratio, score_cutoff=0)

        match = require_not_none(index.find_one("alpha"))
        matches = index.find_many("alpha", limit=limit)

        assert match == matches[0]
        assert [result.value for result in matches] == ["alpha", "ALPHA"]

    @pytest.mark.parametrize("index_class", [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex])
    @pytest.mark.parametrize("limit", [2, None])
    def test_better_score_remains_ahead_of_exact_equality(self, index_class, limit):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        def prefer_non_exact(_query: str, value: str) -> int:
            return 100 if value == "abd" else 80

        index = index_class(
            ["abc", "abd"],
            normalizer=normalizer,
            scorer=prefer_non_exact,
            score_cutoff=0,
        )

        match = require_not_none(index.find_one("abc"))
        matches = index.find_many("abc", limit=limit)

        assert match == matches[0]
        assert [result.value for result in matches] == ["abd", "abc"]

    def test_compatible_metadata_falls_back_when_shortcut_score_is_rejected(self):
        # noinspection PyUnusedLocal
        def prefer_non_exact(left: str, right: str, **kwargs: object) -> int:
            return 100 if right == "abd" else 80

        prefer_non_exact._RF_ScorerPy = {
            "get_scorer_flags": lambda **kwargs: {
                "optimal_score": 100,
                "worst_score": 0,
                "flags": 0,
            }
        }
        index = ImmutableFuzzyKeyedIndex(
            ["abc", "abd"],
            normalizer=Normalizer().isinstance_str().strip().casefold(),
            scorer=prefer_non_exact,
            score_cutoff=90,
        )

        assert index.exact_match("abc", query="abc") is None
        exact_match = require_not_none(index.find_one("abc"))
        normalized_match = require_not_none(index.find_one(" ABC "))

        assert exact_match.value == normalized_match.value == "abd"

    def test_find_many_with_zero_limit_returns_no_matches(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone"])

        assert index.find_many("Alpha Phone", limit=0) == []

    def test_unsearchable_equal_value_does_not_become_exact_candidate(self):
        indexed = SearchableEqualityKey("shared", None)
        searchable = SearchableEqualityKey("other", "other")
        query = SearchableEqualityKey("shared", "shared")
        index = ImmutableFuzzyKeyedIndex(
            [indexed, searchable],
            normalizer=normalize_equality_key,
            scorer=ratio,
            score_cutoff=0,
        )

        match = require_not_none(index.find_one(query))

        assert match.value is searchable

    def test_scorer_metadata_with_wrong_direction_uses_direct_path(self):
        # noinspection PyUnusedLocal
        def length_distance(left: str, right: str, **kwargs: object) -> int:
            return abs(len(left) - len(right))

        length_distance._RF_ScorerPy = {
            "get_scorer_flags": lambda **kwargs: {
                "optimal_score": 100,
                "worst_score": 0,
                "flags": 0,
            }
        }
        index = ImmutableFuzzyKeyedIndex(
            ["aa", "aaaaaa"],
            normalizer=Normalizer().isinstance_str(),
            scorer=length_distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=2,
        )

        match = require_not_none(index.find_one("xxx"))

        assert match.value == "aa"
        assert match.score == 1

    def test_negative_limit_is_rejected_before_query_normalization(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone"])

        with pytest.raises(ValueError, match="limit"):
            index.find_many(None, limit=-1)

    def test_non_integer_limit_is_rejected_with_type_error(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone"])

        with pytest.raises(TypeError, match="limit"):
            # noinspection PyTypeChecker
            index.find_many(None, limit=True)
        with pytest.raises(TypeError, match="limit"):
            # noinspection PyTypeChecker
            index.find_many(None, limit="2")

    def test_score_all_aligns_to_supplied_value_order(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone", "Beta Tablet"])

        results = index.score_all(["Beta Tablet", 1, "Alpha Phone"], "alpha phone")

        assert results[0] is None
        assert results[1] is None
        third_result = require_not_none(results[2])
        assert third_result.value == "Alpha Phone"

    def test_score_all_preserves_duplicate_source_positions(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone"])

        results = index.score_all(["Alpha Phone", "Alpha Phone"], "Alpha Phone")

        assert [require_not_none(result).value for result in results] == [
            "Alpha Phone",
            "Alpha Phone",
        ]

    def test_normalized_exact_match_supports_none_value(self):
        def normalizer(value: object) -> str | None:
            return "missing" if value is None or value == "MISSING" else None

        # noinspection PyUnusedLocal
        def score_match(left: str, right: str, **kwargs: object) -> int:
            return 73

        index = ImmutableFuzzyKeyedIndex(
            [None],
            normalizer=normalizer,
            scorer=score_match,
            score_cutoff=70,
        )

        match = index.find_one("MISSING")

        assert match is not None
        assert match.value is None
        assert match.score == 73

    def test_falsey_normalizer_is_not_replaced_by_default(self):
        class FalseyNormalizer:
            def __bool__(self) -> bool:
                return False

            def __call__(self, value: object) -> str:
                return "custom"

        normalizer = FalseyNormalizer()
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone"], normalizer=normalizer)

        match = index.find_one("anything")

        assert match is not None
        assert match.normalized_query == "custom"
        assert index.config_kwargs()["normalizer"] is normalizer

    def test_empty_normalized_query_is_preserved_in_match_metadata(self):
        def normalizer(value: object) -> str | None:
            if value == "query":
                return ""
            if value == "stored":
                return "stored-normalized"
            return None

        index = ImmutableFuzzyKeyedIndex(
            ["stored"],
            normalizer=normalizer,
            scorer=ratio,
            score_cutoff=0,
        )

        match = index.find_one("query")

        assert match is not None
        assert match.normalized_query == ""
        assert match.normalized_value == "stored-normalized"

    def test_distance_scorer_exact_returns_zero(self):
        index = ImmutableFuzzyKeyedIndex(
            ["Alpha Phone"],
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
        )

        match = index.find_one("Alpha Phone")

        assert match is not None
        assert match.score == 0

    def test_generic_distance_scorer_uses_configured_direction(self):
        # noinspection PyUnusedLocal
        def length_distance(left: str, right: str, **kwargs: object) -> int:
            return abs(len(left) - len(right))

        index = ImmutableFuzzyKeyedIndex(
            ["a", "aa", "aaaaaa"],
            normalizer=Normalizer().isinstance_str(),
            scorer=length_distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=2,
        )

        one = index.find_one("xxx")
        many = index.find_many("xxx", limit=None)
        scores = index.score_all(["a", "aa", "aaaaaa"], "xxx")
        streamed_scores = list(index.iter_scores(["a", "aa", "aaaaaa"], "xxx"))

        assert one is not None
        assert one.value == "aa"
        assert [(match.value, match.score) for match in many] == [("aa", 1), ("a", 2)]
        assert [None if match is None else match.score for match in scores] == [2, 1, None]
        assert scores == streamed_scores

    def test_custom_scorer_with_incomplete_rapidfuzz_metadata_uses_fallback(self):
        # noinspection PyUnusedLocal
        def similarity(left: str, right: str, **kwargs: object) -> int:
            return 100 if left == right else 0

        similarity._RF_ScorerPy = {}
        index = ImmutableFuzzyKeyedIndex(
            ["stored"],
            normalizer=Normalizer().isinstance_str(),
            scorer=similarity,
            score_cutoff=0,
        )

        match = index.find_one("query")

        assert match is not None
        assert match.value == "stored"

    def test_custom_scorer_with_non_mapping_rapidfuzz_metadata_uses_fallback(self):
        # noinspection PyUnusedLocal
        def similarity(left: str, right: str, **kwargs: object) -> int:
            return 100 if left == right else 0

        similarity._RF_ScorerPy = object()
        index = ImmutableFuzzyKeyedIndex(
            ["stored"],
            normalizer=Normalizer().isinstance_str(),
            scorer=similarity,
            score_cutoff=0,
        )

        match = index.find_one("query")

        assert match is not None
        assert match.value == "stored"

    def test_custom_scorer_with_invalid_rapidfuzz_metadata_uses_fallback(self):
        # noinspection PyUnusedLocal
        def distance(left: str, right: str, **kwargs: object) -> int:
            return abs(len(left) - len(right))

        # noinspection PyUnusedLocal
        def invalid_flags(**kwargs: object) -> dict[str, int]:
            return {}

        distance._RF_ScorerPy = {"get_scorer_flags": invalid_flags}
        index = ImmutableFuzzyKeyedIndex(
            ["aa", "aaaaaa"],
            normalizer=Normalizer().isinstance_str(),
            scorer=distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=2,
        )

        match = index.find_one("xxx")

        assert match is not None
        assert match.value == "aa"

    def test_custom_scorer_with_non_mapping_flags_uses_fallback(self):
        # noinspection PyUnusedLocal
        def similarity(left: str, right: str, **kwargs: object) -> int:
            return 100 if left == right else 0

        # noinspection PyUnusedLocal
        def invalid_flags(**kwargs: object) -> object:
            return object()

        similarity._RF_ScorerPy = {"get_scorer_flags": invalid_flags}
        index = ImmutableFuzzyKeyedIndex(
            ["stored"],
            normalizer=Normalizer().isinstance_str(),
            scorer=similarity,
            score_cutoff=0,
        )

        match = index.find_one("query")

        assert match is not None
        assert match.value == "stored"

    def test_custom_scorer_with_failing_flags_uses_fallback(self):
        # noinspection PyUnusedLocal
        def similarity(left: str, right: str, **kwargs: object) -> int:
            return 100 if left == right else 0

        # noinspection PyUnusedLocal
        def invalid_flags(**kwargs: object) -> dict[str, int]:
            raise TypeError("unsupported scorer metadata")

        similarity._RF_ScorerPy = {"get_scorer_flags": invalid_flags}
        index = ImmutableFuzzyKeyedIndex(
            ["stored"],
            normalizer=Normalizer().isinstance_str(),
            scorer=similarity,
            score_cutoff=0,
        )

        match = index.find_one("query")

        assert match is not None
        assert match.value == "stored"

    def test_config_kwargs_are_independent_from_scorer_kwargs(self):
        index = ImmutableFuzzyKeyedIndex(
            ["Alpha Phone"],
            scorer_kwargs={"weights": [1, 1, 1]},
        )

        config = index.config_kwargs()
        scorer_kwargs = config["scorer_kwargs"]
        assert scorer_kwargs is not None
        scorer_kwargs["weights"].append(2)

        assert index.config_kwargs()["scorer_kwargs"] == {"weights": [1, 1, 1]}


class TestKeyedIndexStrategyParity:
    def test_immutable_and_mutable_read_results_match(self):
        values = ["Alpha Phone", "  alpha phone  ", "Beta Tablet", 1]
        immutable = ImmutableFuzzyKeyedIndex(values)
        mutable = MutableFuzzyKeyedIndex(values)

        assert match_signature(immutable.find_one("ALPHA PHONE")) == match_signature(mutable.find_one("ALPHA PHONE"))
        assert [match_signature(m) for m in immutable.find_many("Alpa Phone", limit=None)] == [
            match_signature(m) for m in mutable.find_many("Alpa Phone", limit=None)
        ]
        assert [match_signature(m) for m in immutable.score_all(values, "alpha phone")] == [
            match_signature(m) for m in mutable.score_all(values, "alpha phone")
        ]
        assert [match_signature(m) for m in immutable.iter_scores(values, "alpha phone")] == [
            match_signature(m) for m in mutable.iter_scores(values, "alpha phone")
        ]

    def test_mutable_remove_preserves_collision_tie_breaking(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

        index.remove("Alpha Phone")
        match = index.find_one("ALPHA PHONE")

        assert match is not None
        assert match.value == "  alpha phone  "

    def test_mutable_remove_ignores_missing_value(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone"])

        index.remove("Coffee Grinder")

        match = index.find_one("Alpha Phone")
        assert match is not None
        assert match.value == "Alpha Phone"

    def test_add_duplicate_value_is_ignored(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone"])

        index.add("Alpha Phone")

        assert len(index.find_many("alpha phone", limit=None)) == 1

    def test_exact_match_tracks_stored_equal_value_after_mutation(self):
        def normalizer(value: object) -> str:
            if value in (True, 1):
                return "boolean-one"
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings, True, and 1 only")
            return value

        index = MutableFuzzyKeyedIndex([True], normalizer=normalizer)

        first_match = index.exact_match(1, query=1)
        assert first_match is not None
        assert first_match.value is True

        index.remove(1)
        index.add(1)

        second_match = index.exact_match(True, query=True)
        assert second_match is not None
        assert type(second_match.value) is int

    def test_remove_updates_unsearchable_exact_value(self):
        index = MutableFuzzyKeyedIndex([True])

        index.remove(1)

        with pytest.raises(KeyError):
            index.exact_match(True, query=True)

    def test_batch_remove_updates_unsearchable_exact_value(self):
        index = MutableFuzzyKeyedIndex([True, 2], normalizer=str)

        index.batch_remove([1])

        with pytest.raises(KeyError):
            index.exact_match(True, query=True)
        match = index.exact_match(2, query=2)
        assert match is not None
        assert match.value == 2

    def test_score_all_ignores_indexed_values_outside_source_values(self):
        index = ImmutableFuzzyKeyedIndex(["Alpha Phone", "Beta Tablet"])

        results = index.score_all(["Alpha Phone"], "Beta Tablet")

        assert results == [None]

    @pytest.mark.parametrize("index_class", [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex])
    def test_score_all_returns_each_supplied_equal_value(self, index_class):
        class EqualKey:
            def __init__(self, value: str, label: str) -> None:
                self.value = value
                self.label = label

            def __eq__(self, other: object) -> bool:
                return isinstance(other, EqualKey) and self.value == other.value

            def __hash__(self) -> int:
                return hash(self.value)

        indexed = EqualKey("alpha", "indexed")
        external = EqualKey("alpha", "external")
        index = index_class(
            [indexed],
            normalizer=Normalizer().custom(lambda value: value.value if isinstance(value, EqualKey) else value),
            score_cutoff=0,
        )

        result = index.score_all([external], "alpha")

        match = require_not_none(result[0])
        assert match.value is external

    def test_batch_remove_skips_missing_and_promotes_remaining_collision(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

        index.batch_remove(["Coffee Grinder", "Alpha Phone"])

        match = index.find_one("alpha phone")
        assert match is not None
        assert match.value == "  alpha phone  "

    def test_batch_remove_keeps_collision_group_when_multiple_values_remain(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone", " alpha phone ", "  alpha phone  "])

        index.batch_remove(["Alpha Phone"])

        matches = index.find_many("alpha phone", limit=None)
        assert [match.value for match in matches] == [" alpha phone ", "  alpha phone  "]

    def test_batch_remove_returns_when_reverse_group_is_missing(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone"])
        index._normalized_to_values.clear()

        index.batch_remove(["Alpha Phone"])

        assert "Alpha Phone" not in index._choices

    def test_remove_returns_when_reverse_group_is_missing(self):
        index = MutableFuzzyKeyedIndex(["Alpha Phone"])
        index._normalized_to_values.clear()

        index.remove("Alpha Phone")

        assert "Alpha Phone" not in index._choices
