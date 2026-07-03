"""Direct tests for FuzzySequenceIndex and MutableFuzzySequenceIndex.

These tests exercise the public index classes directly without going through
collection facades.  They cover behavior that collection tests only reach
indirectly.
"""

import pytest
from rapidfuzz.distance import Levenshtein

from rapidfuzz_collections import Normalizer, ScorerType
from rapidfuzz_collections.indexes import FuzzySequenceIndex, MutableFuzzySequenceIndex
from tests.helpers import casefold_string, require_not_none


def _reject_bad_value(value: object) -> str:
    if value == "bad":
        raise RuntimeError("normalization failed")
    if not isinstance(value, str):
        raise TypeError("test normalizer accepts strings only")
    return value


@pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
def test_sequence_index_accepts_values_with_misleading_hashable_abc(index_class):
    value = (["x"],)

    index = index_class([value], normalizer=str, score_cutoff=None)

    assert index.find_one(value) is not None


@pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
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


@pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
def test_custom_score_all_aligns_unsearchable_accepted_and_rejected_values(index_class):
    def normalizer(value: object) -> str | None:
        return value.casefold() if isinstance(value, str) and value != "skip" else None

    def scorer(_query: str, value: str) -> int:
        return 100 if value == "alpha phone" else 0

    index = index_class(
        ["skip", "Alpha Phone", "Beta Tablet"],
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=50,
    )

    results = index.score_all("query")

    assert results[0] is None
    assert require_not_none(results[1]).value == "Alpha Phone"
    assert results[2] is None


# ---------------------------------------------------------------------------
# Invariant: find_one(q) == find_many(q, limit=1)[0]
# ---------------------------------------------------------------------------


class TestFindOneInvariant:
    """Verify find_one and find_many(limit=1) return the same best match."""

    @pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
    def test_default_scorer_with_tied_exact_match(self, index_class):
        # Default scorer has _RF_ScorerPy metadata, so find_one may use the fast-path
        # for an exact match; find_many(limit=1) always goes through _ranked_matches.
        index = index_class(["ALPHA", "alpha", "beta"], normalizer=casefold_string, score_cutoff=0)
        one = index.find_one("alpha")
        many = index.find_many("alpha", limit=1)
        assert one == (many[0] if many else None)

    @pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
    def test_custom_scorer_without_metadata_bypasses_fast_path(self, index_class):
        # noinspection PyUnusedLocal
        def constant_scorer(query: str, value: str) -> int:
            return 80

        index = index_class(
            ["ALPHA", "alpha", "beta"],
            normalizer=casefold_string,
            scorer=constant_scorer,
            score_cutoff=0,
        )
        one = index.find_one("alpha")
        many = index.find_many("alpha", limit=1)
        assert one == (many[0] if many else None)

    @pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
    def test_no_match_both_return_none(self, index_class):
        index = index_class(["abc", "def"], score_cutoff=90)
        one = index.find_one("xyz")
        many = index.find_many("xyz", limit=1)
        assert one is None
        assert one == (many[0] if many else None)

    @pytest.mark.parametrize("index_class", [FuzzySequenceIndex, MutableFuzzySequenceIndex])
    def test_distance_scorer_exact_match(self, index_class):
        index = index_class(
            ["Alpha Phone", "Beta Tablet"],
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=5,
        )
        one = index.find_one("Alpha Phone")
        many = index.find_many("Alpha Phone", limit=1)
        assert one == (many[0] if many else None)


# ---------------------------------------------------------------------------
# FuzzySequenceIndex
# ---------------------------------------------------------------------------


class TestFuzzySequenceIndexConstruction:
    def test_stores_values_in_source_order(self):
        index = FuzzySequenceIndex(["Beta", "Alpha", "Gamma"])

        assert index.values == ("Beta", "Alpha", "Gamma")
        assert index[0] == "Beta"
        assert index[1:] == ("Alpha", "Gamma")
        assert list(index) == ["Beta", "Alpha", "Gamma"]
        assert list(reversed(index)) == ["Gamma", "Alpha", "Beta"]
        assert len(index) == 3

    def test_empty_index(self):
        index = FuzzySequenceIndex([])

        assert index.values == ()
        assert len(index) == 0
        assert index.find_one("anything") is None
        assert index.contains("anything") is False

    def test_normalized_choices_excludes_unsearchable(self):
        index = FuzzySequenceIndex(["xy", "Alpha Phone"])

        assert "xy" not in index.normalized_choices
        assert "alpha phone" in index.normalized_choices

    def test_normalize_applies_configured_normalizer(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.normalize("Alpha Phone") == "alpha phone"
        assert index.normalize("xy") is None

    def test_falsey_normalizer_is_not_replaced_by_default(self):
        class FalseyNormalizer:
            def __bool__(self) -> bool:
                return False

            def __call__(self, value: object) -> str:
                return "custom"

        normalizer = FalseyNormalizer()
        index = FuzzySequenceIndex(["Alpha Phone"], normalizer=normalizer)

        match = index.find_one("anything")

        assert match is not None
        assert match.normalized_query == "custom"
        assert index.config_kwargs()["normalizer"] is normalizer

    def test_public_configuration_properties_expose_index_settings(self):
        index = FuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=Levenshtein.distance,
            scorer_kwargs={"weights": (1, 1, 1)},
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
            score_hint=1,
        )

        assert index.normalizer(" Alpha Phone ") == "alpha phone"
        assert index.score_cutoff == 3
        assert index.score_hint == 1
        assert index.scorer is Levenshtein.distance
        assert index.scorer_type == ScorerType.DISTANCE

    def test_iter_scores_does_not_raise_when_unsearchable_element_at_tail(self):
        # This used to fail when an unsearchable value sat after all searchable
        # choices.
        index = FuzzySequenceIndex(["Apple", "Banana", 42])

        results = list(index.iter_scores("Apple"))

        assert len(results) == 3
        first_result = require_not_none(results[0])
        assert first_result.value == "Apple"
        assert results[1] is None
        assert results[2] is None

    def test_iter_scores_does_not_raise_when_multiple_unsearchable_at_tail(self):
        index = FuzzySequenceIndex(["Apple", 1, 2])

        results = list(index.iter_scores("Apple"))

        assert len(results) == 3
        require_not_none(results[0])
        assert results[1] is None
        assert results[2] is None


class TestFuzzySequenceIndexFindOne:
    def test_exact_match_returns_score_100(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        match = index.find_one("Alpha Phone")

        assert match is not None
        assert match.value == "Alpha Phone"
        assert match.score == 100
        assert match.index == 0

    def test_normalized_exact_match_returns_first_occurrence(self):
        index = FuzzySequenceIndex(["Alpha Phone", "  alpha phone  "])

        match = index.find_one("ALPHA PHONE")

        assert match is not None
        assert match.value == "Alpha Phone"
        assert match.index == 0

    def test_fuzzy_close_match_returned(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        match = index.find_one("Alpa Phone")

        assert match is not None
        assert match.value == "Alpha Phone"
        assert match.score >= 80

    def test_miss_returns_none(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.find_one("Coffee Grinder") is None

    def test_exact_unsearchable_value_is_excluded_from_fuzzy_lookup(self):
        index = FuzzySequenceIndex(["xy", "Alpha Phone"])

        assert index.find_one("xy") is None
        assert index.values[0] == "xy"

    def test_exact_shortcut_uses_configured_score_and_cutoff(self):
        # noinspection PyUnusedLocal
        def score_exact_pair(left: str, right: str, **kwargs: object) -> int:
            return 40

        index = FuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=score_exact_pair,
            score_cutoff=50,
        )

        assert index.find_one("Alpha Phone") is None

        accepted = FuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=score_exact_pair,
            score_cutoff=40,
        ).find_one("Alpha Phone")
        assert accepted is not None
        assert accepted.score == 40

    def test_simple_custom_similarity_scorer_uses_direct_path(self):
        def normalizer(value: object) -> str | None:
            return value.casefold() if isinstance(value, str) else None

        def same_initial(left: str, right: str) -> int:
            return 100 if left[:1] == right[:1] else 0

        index = FuzzySequenceIndex(
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

        index = FuzzySequenceIndex(
            ["abc", "abd"],
            normalizer=normalizer,
            scorer=prefer_non_exact,
            score_cutoff=80,
        )

        match = require_not_none(index.find_one("abc"))
        batch_match = require_not_none(index.find_one_batch(["abc"])[0])
        cdist_match = require_not_none(index.find_one_batch_cdist(["abc"])[0])

        assert match.value == batch_match.value == cdist_match.value == "abd"
        assert match.score == batch_match.score == cdist_match.score == 100

    @pytest.mark.parametrize("limit", [2, None])
    def test_exact_equality_breaks_equal_score_tie(self, limit):
        index = FuzzySequenceIndex(["ALPHA", "alpha"], normalizer=casefold_string, score_cutoff=0)

        match = require_not_none(index.find_one("alpha"))
        matches = index.find_many("alpha", limit=limit)

        assert match == matches[0]
        assert [result.value for result in matches] == ["alpha", "ALPHA"]

    @pytest.mark.parametrize("limit", [2, None])
    def test_better_score_remains_ahead_of_exact_equality(self, limit):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        def prefer_non_exact(_query: str, value: str) -> int:
            return 100 if value == "abd" else 80

        index = FuzzySequenceIndex(
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
        index = FuzzySequenceIndex(
            ["abc", "abd"],
            normalizer=Normalizer().isinstance_str().strip().casefold(),
            scorer=prefer_non_exact,
            score_cutoff=90,
        )

        exact_match = require_not_none(index.find_one("abc"))
        normalized_match = require_not_none(index.find_one(" ABC "))
        cdist_match = require_not_none(index.find_one_batch_cdist(["abc"])[0])

        assert exact_match.value == normalized_match.value == cdist_match.value == "abd"

    def test_exact_shortcut_reuses_cached_normalized_value(self):
        calls = 0

        def normalizer(value: object) -> str | None:
            nonlocal calls
            calls += 1
            return value.casefold() if isinstance(value, str) and len(value) >= 3 else None

        index = FuzzySequenceIndex(["xy", "Alpha Phone"], normalizer=normalizer)
        calls = 0

        match = index.find_one("Alpha Phone")

        assert match is not None
        assert calls == 1

    def test_unhashable_searchable_query_uses_normalized_shortcut(self):
        def normalizer(value: object) -> str | None:
            if isinstance(value, list):
                return " ".join(value).casefold()
            return value.casefold() if isinstance(value, str) else None

        index = FuzzySequenceIndex(["Alpha Phone"], normalizer=normalizer)

        match = index.find_one(["Alpha", "Phone"])
        batch_match = index.find_one_batch([["Alpha", "Phone"]])[0]

        assert match is not None
        assert batch_match is not None
        assert match.value == batch_match.value == "Alpha Phone"

    def test_exact_shortcut_skips_value_rejected_during_index_build(self):
        calls = 0

        def stateful_normalizer(value: object) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings only")
            return value.casefold()

        index = FuzzySequenceIndex(["Alpha Phone"], normalizer=stateful_normalizer)

        assert index.find_one("Alpha Phone") is None

    def test_cdist_resolver_handles_non_exact_and_rejected_shortcuts(self):
        def list_normalizer(value: object) -> str:
            if isinstance(value, list):
                return "alpha phone"
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings and lists only")
            return value.casefold()

        unhashable_index = FuzzySequenceIndex(["Alpha Phone"], normalizer=list_normalizer)
        unhashable_match, unresolved_query = unhashable_index._resolve_query(["Alpha", "Phone"])
        assert unhashable_match is None
        assert unresolved_query == "alpha phone"

        calls = 0

        def stateful_normalizer(value: object) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings only")
            return value.casefold()

        rejected_index = FuzzySequenceIndex(["Alpha Phone"], normalizer=stateful_normalizer)
        assert rejected_index._resolve_query("Alpha Phone") == (None, "alpha phone")

        # noinspection PyUnusedLocal
        def rejected_score(left: str, right: str, **kwargs: object) -> int:
            return 40

        cutoff_index = FuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=rejected_score,
            score_cutoff=50,
        )
        assert cutoff_index._resolve_query("Alpha Phone") == (None, "alpha phone")

    def test_unsearchable_query_returns_none(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.find_one("xy") is None

    def test_distance_scorer_exact_returns_zero(self):
        index = FuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
        )

        match = index.find_one("Alpha Phone")

        assert match is not None
        assert match.score == 0

    def test_distance_scorer_close_match(self):
        index = FuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
        )

        match = index.find_one("Alph Phone")

        assert match is not None
        assert match.score <= 3

    def test_generic_distance_scorer_uses_lower_scores(self):
        # noinspection PyUnusedLocal
        def length_distance(left: str, right: str, **kwargs: object) -> int:
            return abs(len(left) - len(right))

        index = FuzzySequenceIndex(
            ["aa", "aaaaaa"],
            normalizer=Normalizer().isinstance_str(),
            scorer=length_distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=2,
        )

        match = index.find_one("xxx")

        assert match is not None
        assert match.value == "aa"
        assert match.score == 1


class TestFuzzySequenceIndexFindMany:
    def test_returns_all_above_cutoff_when_limit_none(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Alpha Case", "Beta Tablet"])

        matches = index.find_many("Alpha", limit=None)

        values = {m.value for m in matches}
        assert "Alpha Phone" in values
        assert "Alpha Case" in values
        assert "Beta Tablet" not in values

    def test_respects_limit(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Alpha Case", "Alpha Watch"])

        matches = index.find_many("Alpha", limit=2)

        assert len(matches) <= 2

    def test_zero_limit_returns_no_matches(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.find_many("Alpha Phone", limit=0) == []

    def test_generic_distance_scorer_orders_and_filters_matches(self):
        # noinspection PyUnusedLocal
        def length_distance(left: str, right: str, **kwargs: object) -> int:
            return abs(len(left) - len(right))

        index = FuzzySequenceIndex(
            ["a", "aa", "aaaaaa"],
            normalizer=Normalizer().isinstance_str(),
            scorer=length_distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=2,
        )

        matches = index.find_many("xxx", limit=None)
        scores = index.score_all("xxx")
        streamed_scores = list(index.iter_scores("xxx"))

        assert [(match.value, match.score) for match in matches] == [("aa", 1), ("a", 2)]
        assert [None if match is None else match.score for match in scores] == [2, 1, None]
        assert scores == streamed_scores

    def test_duplicate_exact_values_all_returned(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Alpha Phone"])

        matches = index.find_many("Alpha Phone", limit=None)

        assert len(matches) == 2
        assert all(m.value == "Alpha Phone" for m in matches)
        assert {m.index for m in matches} == {0, 2}

    def test_duplicate_exact_values_limited(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Alpha Phone"])

        matches = index.find_many("Alpha Phone", limit=1)

        assert len(matches) == 1
        assert matches[0].index == 0

    def test_miss_returns_empty_list(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.find_many("Coffee Grinder") == []

    def test_results_ordered_by_score_descending(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Alpha"])

        matches = index.find_many("Alpha", limit=None)

        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_negative_limit_is_rejected_before_query_normalization(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        with pytest.raises(ValueError, match="limit"):
            index.find_many(None, limit=-1)

        with pytest.raises(ValueError, match="limit"):
            index.find_many_batch([], limit=-1)

    def test_non_integer_limit_is_rejected_with_type_error(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        with pytest.raises(TypeError, match="limit"):
            # noinspection PyTypeChecker
            index.find_many(None, limit=True)
        with pytest.raises(TypeError, match="limit"):
            # noinspection PyTypeChecker
            index.find_many(None, limit="2")
        with pytest.raises(TypeError, match="limit"):
            # noinspection PyTypeChecker
            index.find_many_batch([], limit=True)


class TestFuzzySequenceIndexUnhashableQuery:
    def test_find_one_unhashable_query_uses_normalizer(self):
        # A list is unhashable — skips exact path, goes straight to normalizer.
        # The default normalizer rejects non-strings, so result is None.
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.find_one(["Alpha Phone"]) is None

    def test_find_many_unhashable_query_uses_normalizer(self):
        index = FuzzySequenceIndex(["Alpha Phone"])

        assert index.find_many(["Alpha Phone"]) == []


class TestFuzzySequenceIndexUnhashable:
    def test_unhashable_value_excluded_from_exact_index(self):
        # Lists are unhashable — should be stored but not in exact_first_index.
        # They are still included in fuzzy search via normalized form.
        index = FuzzySequenceIndex([["alpha phone"], "Alpha Phone"])

        # The unhashable list is not findable by exact path.
        match = index.find_one("Alpha Phone")
        assert match is not None
        assert match.value == "Alpha Phone"

    def test_unhashable_value_normalizer_rejects_non_str(self):
        # The default normalizer rejects non-strings, so a list value
        # is not searchable at all.
        index = FuzzySequenceIndex([["not", "a", "string"], "Alpha Phone"])

        matches = index.find_many("alpha phone", limit=None)
        assert all(m.value == "Alpha Phone" for m in matches)

    def test_trailing_unsearchable_value_preserves_searchable_indexes(self):
        index = FuzzySequenceIndex(["Alpha Phone", "xy"])

        match = index.find_one("Alpha Phone")

        assert match is not None
        assert match.index == 0

    def test_multiple_searchable_values_after_gap_preserve_source_indexes(self):
        index = FuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        alpha = index.find_one("Alpha Phone")
        beta = index.find_one("Beta Tablet")

        assert alpha is not None
        assert alpha.index == 1
        assert beta is not None
        assert beta.index == 2


class TestFuzzySequenceIndexBatch:
    def test_find_one_batch_preserves_order(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = index.find_one_batch(["Alpa Phone", "Coffee Grinder", "Beta Tablet"])

        first_result = require_not_none(results[0])
        assert first_result.value == "Alpha Phone"
        assert results[1] is None
        third_result = require_not_none(results[2])
        assert third_result.value == "Beta Tablet"

    def test_find_many_batch_preserves_order(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Alpha Case", "Beta Tablet"])

        results = index.find_many_batch(["Alpha", "Coffee Grinder"], limit=1)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[1] == []

    def test_cdist_preserves_exact_and_non_normalizable_query_shortcuts(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = index.find_one_batch_cdist(["Alpha Phone", ["Beta Tablet"], "Bta Tablet"])

        exact_result = require_not_none(results[0])
        fuzzy_result = require_not_none(results[2])
        assert exact_result.value == "Alpha Phone"
        assert results[1] is None
        assert fuzzy_result.value == "Beta Tablet"

    def test_cdist_adapts_simple_custom_distance_scorer(self):
        def normalizer(value: object) -> str | None:
            return value.casefold() if isinstance(value, str) else None

        def length_distance(left: str, right: str) -> int:
            return abs(len(left) - len(right))

        index = FuzzySequenceIndex(
            ["abc", "de"],
            normalizer=normalizer,
            scorer=length_distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=None,
        )

        result = require_not_none(index.find_one_batch_cdist(["xy"])[0])

        assert result.value == "de"
        assert result.score == 0

    def test_cdist_adapter_preserves_configured_scorer_kwargs(self):
        def normalizer(value: object) -> str | None:
            return value.casefold() if isinstance(value, str) else None

        def same_initial(left: str, right: str, *, match_score: int) -> int:
            return match_score if left[:1] == right[:1] else 0

        index = FuzzySequenceIndex(
            ["abc"],
            normalizer=normalizer,
            scorer=same_initial,
            scorer_kwargs={"match_score": 73},
            score_cutoff=None,
        )

        result = require_not_none(index.find_one_batch_cdist(["ax"])[0])

        assert result.value == "abc"
        assert result.score == 73


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — construction and properties
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexConstruction:
    def test_stores_values_in_source_order(self):
        index = MutableFuzzySequenceIndex(["Beta", "Alpha"])

        assert index.values == ("Beta", "Alpha")
        assert list(index) == ["Beta", "Alpha"]
        assert len(index) == 2

    def test_scorer_kwargs_not_stored_by_reference_in_constructor(self):
        weights = [1, 1, 1]
        original_kw: dict[str, object] = {"weights": weights}
        index = MutableFuzzySequenceIndex(["Alpha", "Beta"], scorer_kwargs=original_kw)
        original_kw["injected"] = "surprise"
        weights.append(99)
        assert index.scorer_kwargs == {"weights": [1, 1, 1]}

    def test_public_snapshots_do_not_allow_untracked_state_mutation(self):
        index = MutableFuzzySequenceIndex(
            ["Beta", "Alpha"],
            scorer_kwargs={"weights": [1, 1, 1]},
        )

        exposed_slice = index[:]
        exposed_config = index.config_kwargs()
        exposed_kwargs = index.scorer_kwargs
        assert exposed_kwargs is not None
        config_kwargs = exposed_config["scorer_kwargs"]
        assert config_kwargs is not None
        exposed_slice.append("Gamma")
        config_kwargs["weights"].append(2)
        exposed_kwargs["weights"].append(2)

        assert index[0] == "Beta"
        assert index.values == ("Beta", "Alpha")
        assert index.normalized_choices == ("beta", "alpha")
        assert list(reversed(index)) == ["Alpha", "Beta"]
        assert index.scorer_kwargs == {"weights": [1, 1, 1]}

    def test_normalized_choices_rebuilds_dirty_state(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.replace_at(0, "Beta Tablet")

        assert index.is_dirty
        assert index.normalized_choices == ("beta tablet",)
        assert not index.is_dirty

    def test_sort_reorders_values_and_updates_lookup_positions(self):
        index = MutableFuzzySequenceIndex(["Beta Tablet", "Alpha Phone"])

        index.sort()
        match = index.find_one("Bta Tablet")

        assert index.values == ("Alpha Phone", "Beta Tablet")
        assert match is not None
        assert match.index == 1

    def test_sort_accepts_custom_key_and_reverse_order(self):
        index = MutableFuzzySequenceIndex(["Beta", "Alpha Phone", "Gamma"])

        index.sort(key=len, reverse=True)

        assert index.values == ("Alpha Phone", "Gamma", "Beta")

    def test_not_dirty_after_construction(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        assert not index.is_dirty

    def test_empty_index_not_dirty(self):
        index = MutableFuzzySequenceIndex([])

        assert not index.is_dirty
        assert len(index) == 0

    def test_normalize_applies_configured_normalizer(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        assert index.normalize("Alpha Phone") == "alpha phone"
        assert index.normalize("xy") is None

    def test_trailing_unsearchable_value_creates_source_mapping(self):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        index = MutableFuzzySequenceIndex(["Alpha Phone", 1], normalizer=normalizer)

        assert index._source_indexes == [0]


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — append (O(1), never dirty)
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexAppend:
    def test_append_does_not_dirty_index(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append("Beta Tablet")

        assert not index.is_dirty

    def test_append_value_immediately_queryable(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append("Beta Tablet")
        match = index.find_one("Beta Tablet")

        assert match is not None
        assert match.value == "Beta Tablet"

    def test_append_to_empty_index(self):
        index = MutableFuzzySequenceIndex([])

        index.append("Alpha Phone")

        assert not index.is_dirty
        assert index.find_one("Alpha Phone") is not None

    def test_append_searchable_after_unsearchable_gap_preserves_source_index(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append("xy")
        index.append("Beta Tablet")
        match = index.find_one("Beta Tablet")

        assert match is not None
        assert match.index == 2

    def test_append_unsearchable_value_creates_source_mapping(self):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        index: MutableFuzzySequenceIndex[object] = MutableFuzzySequenceIndex(
            ["Alpha Phone"],
            normalizer=normalizer,
        )

        index.append(1)

        assert index._source_indexes == [0]

    def test_append_searchable_after_existing_gap_extends_source_indexes(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append("xy")
        index.append("Beta Tablet")
        index.append("Gamma Watch")
        match = index.find_one("Gama Watch")

        assert match is not None
        assert match.index == 3

    def test_rebuild_with_multiple_searchable_values_after_gap_preserves_source_indexes(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

        index.insert_at(0, "xy")
        beta = index.find_one("Beta Tablet")
        gamma = index.find_one("Gamma Watch")

        assert beta is not None
        assert beta.index == 2
        assert gamma is not None
        assert gamma.index == 3

    def test_append_after_all_unsearchable_rebuild_preserves_source_index(self):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) and len(value) >= 4 else None

        index = MutableFuzzySequenceIndex(["x"], normalizer=normalizer, score_cutoff=0)
        index.replace_at(0, "x")
        assert index.find_one("none") is None

        index.append("abcd")
        match = index.find_one("abce")

        assert match is not None
        assert match.value == "abcd"
        assert match.index == 1

    def test_append_duplicate_both_found_by_find_many(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append("Alpha Phone")
        matches = index.find_many("Alpha Phone", limit=None)

        assert len(matches) == 2
        assert {m.index for m in matches} == {0, 1}

    def test_append_duplicate_exact_matches_respect_limit(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append("Alpha Phone")
        index.append("Alpha Phone")
        matches = index.find_many("Alpha Phone", limit=2)

        assert [match.index for match in matches] == [0, 1]

    def test_multiple_appends_accumulate(self):
        index = MutableFuzzySequenceIndex([])

        for word in ["Alpha Phone", "Beta Tablet", "Gamma Watch"]:
            index.append(word)

        assert len(index) == 3
        assert not index.is_dirty


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — structural mutation behavior
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexDirtyMutations:
    def test_dense_delete_rebuilds_exact_shortcuts_on_next_lookup(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Beta Tablet"])

        index.delete_at(0)
        assert not index._shortcuts_valid

        matches = index.find_many("Beta Tablet", limit=None)

        assert index._shortcuts_valid
        assert [match.index for match in matches] == [0, 1]
        assert index._exact_first_index["Beta Tablet"] == 0
        assert index._exact_duplicate_indexes["Beta Tablet"] == [1]

    def test_sparse_delete_rebuilds_exact_shortcuts_on_next_lookup(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.delete_at(1)
        assert not index._shortcuts_valid

        match = require_not_none(index.find_one("Beta Tablet"))

        assert index._shortcuts_valid
        assert match.index == 1
        assert index._exact_first_index["Beta Tablet"] == 1

    def test_missing_exact_query_rebuilds_invalid_shortcuts_once(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_at(0)
        assert index._exact_source_indexes("Missing Device") == ()

        assert index._shortcuts_valid
        assert index._exact_source_indexes("Missing Device") == ()

    def test_insert_at_marks_dirty(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.insert_at(0, "Gamma Watch")

        assert index.is_dirty

    def test_insert_at_result_correct_after_query(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.insert_at(0, "Gamma Watch")
        match = index.find_one("Gama Watch")

        assert match is not None
        assert match.value == "Gamma Watch"
        assert not index.is_dirty

    def test_replace_at_int_marks_dirty(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.replace_at(0, "Beta Tablet")

        assert index.is_dirty

    def test_replace_at_int_result_correct_after_query(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.replace_at(0, "Beta Tablet")

        assert index.find_one("Alpha Phone") is None
        assert index.find_one("Beta Tablet") is not None

    def test_replace_at_slice_marks_dirty(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.replace_at(slice(0, 2), ["Gamma Watch", "Delta Clock"])

        assert index.is_dirty

    def test_replace_at_slice_result_correct_after_query(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.replace_at(slice(0, 2), ["Gamma Watch", "Delta Clock"])

        assert index.find_one("Alpha Phone") is None
        assert index.find_one("Gamma Watch") is not None

    def test_delete_at_dense_searchable_value_updates_without_rebuild(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_at(0)

        assert not index.is_dirty

    def test_dense_delete_discards_removed_shortcut_membership(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_at(0)
        index.append("Gamma Watch")

        assert "Alpha Phone" not in index._exact_first_index
        assert "alpha phone" not in index.normalized_choices
        assert "Beta Tablet" in index._exact_first_index
        assert "Gamma Watch" in index._exact_first_index
        gamma_match = require_not_none(index.find_one("Gamma Watch"))
        assert gamma_match.value == "Gamma Watch"

    def test_dense_delete_exact_fallback_skips_unhashable_values(self):
        class UnhashableSearchable:
            __hash__ = None

            def __eq__(self, other):
                raise AssertionError("Unhashable values must not participate in exact lookup.")

        value = UnhashableSearchable()
        index = MutableFuzzySequenceIndex(
            [value, "Alpha Phone", "Beta Tablet"],
            normalizer=lambda candidate: "custom value" if candidate is value else str(candidate).casefold(),
        )

        index.delete_at(1)
        match = index.find_one("Beta Tablet")
        matches = index.find_many("Beta Tablet", limit=None)

        assert match is not None
        assert match.value == "Beta Tablet"
        assert [result.value for result in matches] == ["Beta Tablet"]

    def test_dense_delete_tracks_remaining_normalized_duplicate_membership(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

        index.delete_at(0)
        remaining = index.find_one("ALPHA PHONE")
        index.delete_at(0)

        assert remaining is not None
        assert remaining.value == "  alpha phone  "
        assert index.find_one("ALPHA PHONE") is None

    def test_delete_at_int_value_absent_after_query(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_at(0)

        assert index.find_one("Alpha Phone") is None
        assert index.find_one("Beta Tablet") is not None

    def test_delete_at_dense_searchable_slice_updates_without_rebuild(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

        index.delete_at(slice(0, 2))

        assert not index.is_dirty
        assert len(index) == 1

    def test_delete_value_dense_searchable_updates_without_rebuild_and_returns_true(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        result = index.delete_value("Alpha Phone")

        assert result is True
        assert not index.is_dirty

    def test_delete_value_absent_returns_false_no_dirty(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        result = index.delete_value("Coffee Grinder")

        assert result is False
        assert not index.is_dirty

    def test_delete_at_positions_dense_searchable_updates_without_rebuild(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

        index.delete_at_positions({0, 2})

        assert not index.is_dirty
        assert len(index) == 1

    def test_delete_at_sparse_choices_updates_without_rebuild(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.delete_at(1)
        match = index.find_one("Beta Tabet")

        assert not index.is_dirty
        assert match is not None
        assert match.index == 1

    def test_delete_unsearchable_sparse_value_updates_source_positions(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.delete_at(0)
        match = index.find_one("Beta Tabet")

        assert not index.is_dirty
        assert match is not None
        assert match.index == 1

    def test_sparse_delete_translates_many_fuzzy_match_positions(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Alpha Case", "Beta Tablet"])

        index.delete_at(1)
        matches = index.find_many("Alpa Case", limit=None)

        assert not index.is_dirty
        assert matches
        assert matches[0].value == "Alpha Case"
        assert matches[0].index == 1

    def test_delete_at_positions_correct_values_remain(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

        index.delete_at_positions({0, 2})

        assert index.values == ("Beta Tablet",)

    def test_delete_at_positions_large_dense_batch_keeps_lookup_consistent(self):
        index = MutableFuzzySequenceIndex([f"Product Name {number:03d}" for number in range(140)])

        index.delete_at_positions(set(range(129)))
        match = index.find_one("Product Name 139")

        assert not index.is_dirty
        assert match is not None
        assert match.index == 10

    def test_delete_at_positions_small_sparse_batch_keeps_lookup_consistent(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "zz", "Beta Tablet", "Gamma Watch"])

        index.delete_at_positions({0, 3})
        match = index.find_one("Gama Watch")

        assert not index.is_dirty
        assert match is not None
        assert match.index == 2

    def test_delete_at_positions_large_sparse_batch_uses_measured_incremental_path(self):
        index = MutableFuzzySequenceIndex(["xy", *[f"Product Name {number:04d}" for number in range(1005)]])

        index.delete_at_positions(set(range(1, 1001)))
        assert not index.is_dirty

        match = index.find_one("Product Name 1004")

        assert match is not None
        assert match.index == 5
        assert not index.is_dirty

    def test_delete_at_positions_sparse_batch_above_limit_retains_rebuild_path(self):
        index = MutableFuzzySequenceIndex(["xy", *[f"Product Name {number:04d}" for number in range(1030)]])

        index.delete_at_positions(set(range(1, 1026)))
        assert index.is_dirty

        match = index.find_one("Product Name 1029")

        assert match is not None
        assert match.index == 5
        assert not index.is_dirty

    def test_delete_at_sparse_slice_retains_rebuild_path(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.delete_at(slice(1, 2))

        assert index.is_dirty

    def test_exact_match_survives_sparse_delete_append_sequence(self):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        index = MutableFuzzySequenceIndex([0, "aa", "bb"], normalizer=normalizer)
        index.delete_at(0)
        index.append("aa")
        index.delete_at(0)
        index.append("aa")
        index.delete_at(1)

        match = index.find_one("aa")

        assert match is not None
        assert match.value == "aa"
        assert match.index == 1

    def test_normalized_exact_match_uses_current_choices_after_sparse_delete(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.delete_at(0)

        match = index.find_one("ALPHA PHONE")

        assert match is not None
        assert match.value == "Alpha Phone"

    def test_delete_at_positions_ignores_invalid_positions(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_at_positions({99})

        assert index.values == ("Alpha Phone", "Beta Tablet")
        assert not index.is_dirty

    def test_keep_at_positions_marks_dirty_and_returns_removed_count(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Gamma Watch"])

        removed = index.keep_at_positions({1})

        assert removed == 2
        assert index.is_dirty
        assert index.values == ("Beta Tablet",)

    def test_keep_at_positions_ignores_invalid_positions(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        removed = index.keep_at_positions({0, 99})

        assert removed == 1
        assert index.is_dirty
        assert index.values == ("Alpha Phone",)

    def test_keep_at_positions_no_op_returns_zero(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        removed = index.keep_at_positions({0, 1})

        assert removed == 0
        assert not index.is_dirty


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — lazy rebuild behavior
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexLazyRebuild:
    def test_single_rebuild_for_multiple_mutations(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet", "Gamma Watch"])

        index.delete_at(1)
        index.delete_at(1)
        assert not index.is_dirty

        match = index.find_one("Gama Watch")
        assert match is not None
        assert match.value == "Gamma Watch"

    def test_fuzzy_close_match_uses_identity_source_mapping(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        match = index.find_one("Alpa Phone")

        assert match is not None
        assert match.index == 0
        assert not index.is_dirty

    def test_rebuild_clears_dirty_flag(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone"])

        index.insert_at(1, "Beta Tablet")
        assert index.is_dirty

        index.find_one("anything")
        assert not index.is_dirty

    def test_append_after_dirty_mutation_works_after_query(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.insert_at(1, "Delta Clock")
        index.append("Gamma Watch")

        match = index.find_one("Gama Watch")
        assert match is not None
        assert match.value == "Gamma Watch"

    def test_append_after_sparse_delete_preserves_stable_source_translation(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone", "Beta Tablet"])

        index.delete_at(1)
        index.append("Gamma Watch")
        match = index.find_one("Gama Watch")

        assert not index.is_dirty
        assert match is not None
        assert match.index == 2


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — empty index edge cases
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexEmpty:
    def test_delete_all_then_query_returns_none(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_at_positions({0, 1})

        assert index.find_one("Alpha Phone") is None
        assert index.find_many("Alpha Phone") == []
        assert not index.contains("Alpha Phone")
        assert len(index) == 0

    def test_keep_none_then_query_returns_none(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.keep_at_positions(set())

        assert index.find_one("Alpha Phone") is None
        assert len(index) == 0

    def test_append_to_empty_after_all_deleted(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.delete_at(0)
        index.append("Beta Tablet")

        assert index.find_one("Beta Tablet") is not None
        assert index.find_one("Alpha Phone") is None


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — exact shortcuts bypass rebuild
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexExactShortcuts:
    def test_exact_match_after_dense_delete_and_append_uses_current_state(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])
        index.delete_at(0)
        index.append("Beta Tablet")
        match = index.find_one("Beta Tablet")

        assert match is not None
        assert match.value == "Beta Tablet"

    def test_normalized_exact_match_after_dense_delete_uses_current_first_value(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "  alpha phone  ", "Beta Tablet"])

        index.delete_at(0)
        match = index.find_one("ALPHA PHONE")

        assert match is not None
        assert match.value == "  alpha phone  "
        assert match.index == 0

    def test_unsearchable_exact_value_remains_excluded_after_mutations(self):
        index = MutableFuzzySequenceIndex(["xy", "Alpha Phone"])

        index.delete_at(1)

        assert index.find_one("xy") is None
        assert index.values == ("xy",)


# ---------------------------------------------------------------------------
# MutableFuzzySequenceIndex — batch methods
# ---------------------------------------------------------------------------


class TestMutableFuzzySequenceIndexUnhashable:
    def test_find_one_unhashable_query_returns_none(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        assert index.find_one(["Alpha Phone"]) is None

    def test_find_many_unhashable_query_returns_empty(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone"])

        assert index.find_many(["Alpha Phone"]) == []

    def test_searchable_unhashable_query_uses_normalized_shortcuts(self):
        def normalizer(value: object) -> str | None:
            if isinstance(value, list):
                return " ".join(value).casefold()
            return value.casefold() if isinstance(value, str) else None

        index = MutableFuzzySequenceIndex(["Alpha Phone"], normalizer=normalizer)

        match = index.find_one(["Alpha", "Phone"])
        batch_match = index.find_one_batch([["Alpha", "Phone"]])[0]

        assert match is not None
        assert batch_match is not None
        assert match.value == batch_match.value == "Alpha Phone"


class TestMutableFuzzySequenceIndexShortcutFallbacks:
    def test_cdist_resolver_handles_rejected_shortcuts(self):
        def list_normalizer(value: object) -> str:
            if isinstance(value, list):
                return "alpha phone"
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings and lists only")
            return value.casefold()

        unhashable_index = MutableFuzzySequenceIndex(["Alpha Phone"], normalizer=list_normalizer)
        unhashable_match, unresolved_query = unhashable_index._resolve_query(["Alpha", "Phone"])
        assert unhashable_match is None
        assert unresolved_query == "alpha phone"

        calls = 0

        def stateful_normalizer(value: object) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings only")
            return value.casefold()

        rejected_index = MutableFuzzySequenceIndex(["Alpha Phone"], normalizer=stateful_normalizer)
        assert rejected_index._resolve_query("Alpha Phone") == (None, "alpha phone")

        # noinspection PyUnusedLocal
        def rejected_score(left: str, right: str, **kwargs: object) -> int:
            return 40

        cutoff_index = MutableFuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=rejected_score,
            score_cutoff=50,
        )
        assert cutoff_index._resolve_query("Alpha Phone") == (None, "alpha phone")

    def test_find_one_handles_rejected_shortcuts(self):
        calls = 0

        def stateful_normalizer(value: object) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            if not isinstance(value, str):
                raise TypeError("test normalizer accepts strings only")
            return value.casefold()

        rejected_index = MutableFuzzySequenceIndex(["Alpha Phone"], normalizer=stateful_normalizer)
        assert rejected_index.find_one("Alpha Phone") is None

        # noinspection PyUnusedLocal
        def rejected_score(left: str, right: str, **kwargs: object) -> int:
            return 40

        cutoff_index = MutableFuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=rejected_score,
            score_cutoff=50,
        )
        assert cutoff_index.find_one("Alpha Phone") is None

    def test_append_unhashable_value_stored_but_not_fuzzy_searchable(self):
        index: MutableFuzzySequenceIndex[object] = MutableFuzzySequenceIndex(["Alpha Phone"])

        index.append(["not", "hashable"])

        assert len(index) == 2
        assert not index.is_dirty
        # unhashable is not findable by exact or fuzzy
        assert index.find_one("Alpha Phone") is not None

    def test_rebuild_with_unhashable_values_correct_results(self):
        # Insert → mark dirty → trigger rebuild that processes unhashable values
        index: MutableFuzzySequenceIndex[object] = MutableFuzzySequenceIndex(["Alpha Phone"])
        index.insert_at(0, ["not", "hashable"])
        assert index.is_dirty

        match = index.find_one("Alpha Phone")
        assert match is not None
        assert match.value == "Alpha Phone"


class TestMutableFuzzySequenceIndexDistanceScorer:
    def test_exact_match_returns_score_zero(self):
        index = MutableFuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
        )

        match = index.find_one("Alpha Phone")

        assert match is not None
        assert match.score == 0

    def test_close_match_returns_low_distance(self):
        index = MutableFuzzySequenceIndex(
            ["Alpha Phone"],
            scorer=Levenshtein.distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=3,
        )

        match = index.find_one("Alph Phone")

        assert match is not None
        assert match.score <= 3

    def test_generic_distance_scorer_matches_all_lookup_paths(self):
        # noinspection PyUnusedLocal
        def length_distance(left: str, right: str, **kwargs: object) -> int:
            return abs(len(left) - len(right))

        index = MutableFuzzySequenceIndex(
            ["aa", "aaaaaa"],
            normalizer=Normalizer().isinstance_str(),
            scorer=length_distance,
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=2,
        )

        one = index.find_one("xxx")
        many = index.find_many("xxx", limit=None)
        scores = index.score_all("xxx")
        batch = index.find_one_batch_cdist(["xxx"], query_chunk_size=1, choice_chunk_size=1)

        assert one is not None
        assert one.value == "aa"
        assert [(match.value, match.score) for match in many] == [("aa", 1)]
        assert [None if match is None else match.score for match in scores] == [1, None]
        batch_match = require_not_none(batch[0])
        assert batch_match.value == "aa"

    def test_custom_scorer_without_metadata_skips_exact_shortcut(self):
        def normalizer(value: object) -> str | None:
            return value if isinstance(value, str) else None

        # noinspection PyUnusedLocal
        def prefer_non_exact(left: str, right: str) -> int:
            return 100 if right == "abd" else 80

        index = MutableFuzzySequenceIndex(
            ["abc", "abd"],
            normalizer=normalizer,
            scorer=prefer_non_exact,
            score_cutoff=80,
        )

        match = require_not_none(index.find_one("abc"))
        batch_match = require_not_none(index.find_one_batch(["abc"])[0])
        cdist_match = require_not_none(index.find_one_batch_cdist(["abc"])[0])

        assert match.value == batch_match.value == cdist_match.value == "abd"
        assert match.score == batch_match.score == cdist_match.score == 100

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
        index = MutableFuzzySequenceIndex(
            ["abc", "abd"],
            normalizer=Normalizer().isinstance_str().strip().casefold(),
            scorer=prefer_non_exact,
            score_cutoff=90,
        )

        exact_match = require_not_none(index.find_one("abc"))
        normalized_match = require_not_none(index.find_one(" ABC "))
        cdist_match = require_not_none(index.find_one_batch_cdist(["abc"])[0])

        assert exact_match.value == normalized_match.value == cdist_match.value == "abd"


class TestMutableFuzzySequenceIndexRebuildDuplicates:
    def test_rebuild_after_replacement_handles_duplicates_correctly(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Alpha Phone"])

        index.replace_at(0, "Alpha Phone")
        matches = index.find_many("Alpha Phone", limit=None)

        assert len(matches) == 2
        assert {m.index for m in matches} == {0, 1}

    def test_rebuild_after_delete_handles_remaining_duplicates(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet", "Alpha Phone"])

        index.delete_at(1)
        matches = index.find_many("Alpha Phone", limit=None)

        assert len(matches) == 2


class TestMutableFuzzySequenceIndexBatch:
    def test_find_one_batch_preserves_order(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = index.find_one_batch(["Alpa Phone", "Coffee Grinder", "Beta Tablet"])

        first_result = require_not_none(results[0])
        assert first_result.value == "Alpha Phone"
        assert results[1] is None
        third_result = require_not_none(results[2])
        assert third_result.value == "Beta Tablet"

    def test_find_many_batch_preserves_order(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Alpha Case"])

        results = index.find_many_batch(["Alpha", "Coffee Grinder"], limit=None)

        assert len(results[0]) == 2
        assert results[1] == []

    def test_batch_after_mutation_returns_updated_results(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        index.delete_value("Alpha Phone")
        results = index.find_one_batch(["Alpha Phone", "Beta Tablet"])

        assert results[0] is None
        second_result = require_not_none(results[1])
        assert second_result.value == "Beta Tablet"


class TestScorerKwargsForwarding:
    def test_scorer_kwargs_forwarded_to_scorer(self):
        # weights=(1,1,2) — deletion costs double; changes which string matches best
        index = FuzzySequenceIndex(
            ["kitten", "sitting"],
            scorer=Levenshtein.distance,
            scorer_kwargs={"weights": (1, 1, 2)},
            scorer_type=ScorerType.DISTANCE,
            score_cutoff=None,
        )
        match = index.find_one("kitten")
        assert match is not None
        assert match.value == "kitten"


# ---------------------------------------------------------------------------
# Coverage: iter_scores / score_all with non-normalizable query
# ---------------------------------------------------------------------------


class TestFrozenIterScoresNoneQuery:
    def test_iter_scores_non_normalizable_query_yields_all_none(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = list(index.iter_scores(42))

        assert results == [None, None]

    def test_score_all_non_normalizable_query_returns_all_none(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = index.score_all(42)

        assert results == [None, None]

    def test_iter_scores_no_cutoff_yields_all_results(self):
        index = FuzzySequenceIndex(["Alpha Phone", "Beta Tablet"], score_cutoff=None)

        results = list(index.iter_scores("alpha phone"))

        assert all(r is not None for r in results)


class TestMutableIterScoresNoneQuery:
    def test_iter_scores_non_normalizable_query_yields_all_none(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = list(index.iter_scores(42))

        assert results == [None, None]

    def test_score_all_non_normalizable_query_returns_all_none(self):
        index = MutableFuzzySequenceIndex(["Alpha Phone", "Beta Tablet"])

        results = index.score_all(42)

        assert results == [None, None]


# ---------------------------------------------------------------------------
# Coverage: sparse deletion of unhashable value
# ---------------------------------------------------------------------------


class TestMutableSparseDeleteUnhashable:
    def test_delete_unhashable_value_at_sparse_position(self):
        index = MutableFuzzySequenceIndex([[1, 2, 3], "Alpha Phone Stand", "Beta Tablet Sleeve"])
        index.find_one("phone stand")

        index.delete_at(0)

        assert index.values == ("Alpha Phone Stand", "Beta Tablet Sleeve")
        match = index.find_one("Alpha Phone Stand")
        assert match is not None
        assert match.index == 0


# ---------------------------------------------------------------------------
# Coverage: dense deletion with 3+ exact duplicates (shortcut duplicate paths)
# ---------------------------------------------------------------------------


class TestMutableDenseDeleteDuplicateShortcuts:
    def test_delete_one_of_three_identical_values_keeps_remaining(self):
        index = MutableFuzzySequenceIndex(["Apple Pro", "Apple Pro", "Apple Pro", "Other Item"])
        index.find_one("Apple Pro")

        index.delete_at(0)

        assert index.values == ("Apple Pro", "Apple Pro", "Other Item")
        many = index.find_many("Apple Pro")
        assert len(many) == 2


# ---------------------------------------------------------------------------
# find_many(limit=None): exact values precede non-exact in equal-score group
# ---------------------------------------------------------------------------


def test_find_many_no_limit_exact_precede_non_exact_in_equal_score_group():
    # noinspection PyUnusedLocal
    def constant_scorer(query: str, value: str) -> int:
        return 80

    index = FuzzySequenceIndex(
        ["not1", "not2", "exact", "not3", "not4"],
        normalizer=Normalizer().isinstance_str(),
        scorer=constant_scorer,
        score_cutoff=0,
    )
    matches = index.find_many("exact", limit=None)

    assert len(matches) == 5
    exact_positions = [i for i, m in enumerate(matches) if m.value == "exact"]
    non_exact_positions = [i for i, m in enumerate(matches) if m.value != "exact"]

    assert max(exact_positions) < min(non_exact_positions)


@pytest.mark.parametrize("operation", ["insert", "replace", "replace_slice"])
def test_mutation_is_atomic_when_normalization_fails(operation):
    index = MutableFuzzySequenceIndex(["alpha", "beta"], normalizer=_reject_bad_value, score_cutoff=None)

    with pytest.raises(RuntimeError, match="normalization failed"):
        if operation == "insert":
            index.insert_at(1, "bad")
        elif operation == "replace":
            index.replace_at(0, "bad")
        else:
            index.replace_at(slice(0, 1), ["good", "bad"])

    assert index.values == ("alpha", "beta")
    assert [match.value if match is not None else None for match in index.score_all("alpha")] == ["alpha", "beta"]


def test_sort_reuses_normalized_values_without_calling_normalizer_again():
    calls = 0

    def counting_normalizer(value: object) -> str:
        nonlocal calls
        calls += 1
        if not isinstance(value, str):
            raise TypeError("test normalizer accepts strings only")
        return value

    index = MutableFuzzySequenceIndex(["beta", "alpha"], normalizer=counting_normalizer)
    assert calls == 2

    index.sort()

    assert index.values == ("alpha", "beta")
    assert calls == 2


def test_sort_is_atomic_when_key_fails():
    index = MutableFuzzySequenceIndex(["beta", "alpha"], normalizer=str)

    def bad_key(value: str) -> str:
        if value == "alpha":
            raise RuntimeError("sorting failed")
        return value

    with pytest.raises(RuntimeError, match="sorting failed"):
        index.sort(key=bad_key)

    assert index.values == ("beta", "alpha")
