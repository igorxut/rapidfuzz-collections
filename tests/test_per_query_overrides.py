"""Tests for per-query overrides of scorer, scorer_kwargs, scorer_type, score_cutoff, and score_hint.

These parameters are configurable at construction time as collection defaults
and, per method call, as keyword-only overrides that apply to that call only.
`normalizer` and `strategy` remain construction-only and are not covered here.
"""

import pytest
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import WRatio

from rapidfuzz_collections import (
    FrozenFuzzyDict,
    FrozenFuzzySet,
    FuzzyDict,
    FuzzyList,
    FuzzySet,
    FuzzyTuple,
    IndexStrategy,
    ScorerType,
)
from rapidfuzz_collections.indexes import (
    FuzzySequenceIndex,
    ImmutableFuzzyKeyedIndex,
    MutableFuzzyKeyedIndex,
    MutableFuzzySequenceIndex,
)
from tests.helpers import require_not_none

SEQUENCE_INDEX_CLASSES = [FuzzySequenceIndex, MutableFuzzySequenceIndex]
KEYED_INDEX_CLASSES = [ImmutableFuzzyKeyedIndex, MutableFuzzyKeyedIndex]


def fixed_score_scorer(_query: str, _choice: str, **_kwargs: object) -> int:
    """Custom scorer without RapidFuzz metadata; always returns a fixed score."""

    return 42


def kwarg_reading_scorer(_query: str, _choice: str, weights: tuple[int, ...] | None = None, **_kwargs: object) -> int:
    """Custom scorer that reports back the first element of `weights`."""

    return weights[0] if weights else 0


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
@pytest.mark.parametrize("parameter", ["score_cutoff", "score_hint"])
@pytest.mark.parametrize("invalid_value", [True, "invalid"])
def test_score_parameters_are_validated_at_construction(index_class, parameter, invalid_value):
    with pytest.raises(TypeError, match=parameter):
        index_class(["Alpha Phone"], **{parameter: invalid_value})


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
@pytest.mark.parametrize("parameter", ["score_cutoff", "score_hint"])
@pytest.mark.parametrize("invalid_value", [True, "invalid"])
def test_score_parameters_are_validated_per_query(index_class, parameter, invalid_value):
    index = index_class(["Alpha Phone"])

    with pytest.raises(TypeError, match=parameter):
        index.find_one("Alpha Phone", **{parameter: invalid_value})


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
@pytest.mark.parametrize("invalid_value", ["invalid", {1: 2}])
def test_scorer_kwargs_are_validated_at_construction(index_class, invalid_value):
    with pytest.raises(TypeError, match="scorer_kwargs"):
        # noinspection PyTypeChecker
        index_class(["Alpha Phone"], scorer_kwargs=invalid_value)


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
@pytest.mark.parametrize("invalid_value", ["invalid", {1: 2}])
def test_scorer_kwargs_are_validated_per_query(index_class, invalid_value):
    index = index_class(["Alpha Phone"])

    with pytest.raises(TypeError, match="scorer_kwargs"):
        # noinspection PyTypeChecker
        index.find_one("Alpha Phone", scorer_kwargs=invalid_value)


# ---------------------------------------------------------------------------
# Sequence index — score_cutoff overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_class", SEQUENCE_INDEX_CLASSES)
class TestSequenceIndexScoreCutoffOverride:
    def test_default_cutoff_still_applies_without_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert index.find_one("Alpa Phone") is None

    def test_looser_cutoff_override_accepts_previously_rejected_candidate(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        match = require_not_none(index.find_one("Alpa Phone", score_cutoff=50))

        assert match.value == "Alpha Phone"

    def test_stricter_cutoff_override_rejects_previously_accepted_candidate(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=50)

        assert index.find_one("Alpa Phone") is not None
        assert index.find_one("Alpa Phone", score_cutoff=99) is None


# ---------------------------------------------------------------------------
# Sequence index — scorer / scorer_kwargs / scorer_type overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_class", SEQUENCE_INDEX_CLASSES)
class TestSequenceIndexScorerOverride:
    def test_scorer_override_without_metadata_changes_winner(self, index_class):
        def prefer_second(_query: str, choice: str, **_kwargs: object) -> int:
            return 100 if choice == "sitting" else 0

        index = index_class(["kitten", "sitting"], score_cutoff=0)

        default_match = require_not_none(index.find_one("kitten"))
        assert default_match.value == "kitten"

        overridden_match = require_not_none(
            index.find_one(
                "kitten",
                scorer=prefer_second,
                scorer_type=ScorerType.SIMILARITY,
                score_cutoff=0,
            )
        )
        assert overridden_match.value == "sitting"

    def test_scorer_override_with_native_metadata_uses_fast_path(self, index_class):
        index = index_class(["Alpha Phone"], scorer=fixed_score_scorer, score_cutoff=0)

        default_match = require_not_none(index.find_one("Alpha Phone"))
        assert default_match.score == 42

        overridden_match = require_not_none(index.find_one("Alpha Phone", scorer=WRatio))
        assert overridden_match.score == 100.0

    def test_scorer_kwargs_override_forwarded_to_scorer(self, index_class):
        index = index_class(["Alpha Phone", "Beta Tablet"], scorer=kwarg_reading_scorer, score_cutoff=0)

        default_match = require_not_none(index.find_one("query"))
        assert default_match.score == 0

        overridden_match = require_not_none(index.find_one("query", scorer_kwargs={"weights": (42,)}))
        assert overridden_match.score == 42

    def test_scorer_kwargs_none_clears_constructor_default(self, index_class):
        index = index_class(
            ["Alpha Phone"],
            scorer=kwarg_reading_scorer,
            scorer_kwargs={"weights": (42,)},
            score_cutoff=0,
        )

        default_match = require_not_none(index.find_one("query"))
        assert default_match.score == 42

        overridden_match = require_not_none(index.find_one("query", scorer_kwargs=None))
        assert overridden_match.score == 0

    def test_scorer_type_override_flips_ranking_order(self, index_class):
        def rank_scorer(_query: str, choice: str, **_kwargs: object) -> int:
            return {"alpha": 10, "beta": 90}.get(choice, 0)

        index = index_class(
            ["Alpha", "Beta"],
            scorer=rank_scorer,
            scorer_type=ScorerType.SIMILARITY,
            score_cutoff=None,
        )

        similarity_order = index.find_many("query", limit=None)
        assert [match.value for match in similarity_order] == ["Beta", "Alpha"]

        distance_order = index.find_many("query", limit=None, scorer_type=ScorerType.DISTANCE)
        assert [match.value for match in distance_order] == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# Sequence index — exact-match shortcut must not use stale scorer metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_class", SEQUENCE_INDEX_CLASSES)
class TestSequenceIndexExactShortcutMetadataFreshness:
    def test_overriding_to_scorer_without_metadata_recomputes_exact_score(self, index_class):
        # Default scorer (WRatio) has metadata and would short-circuit with score 100.
        index = index_class(["Alpha Phone"])

        default_match = require_not_none(index.find_one("Alpha Phone"))
        assert default_match.score == 100.0

        overridden_match = require_not_none(
            index.find_one(
                "Alpha Phone",
                scorer=fixed_score_scorer,
                scorer_type=ScorerType.SIMILARITY,
                score_cutoff=0,
            )
        )
        assert overridden_match.score == 42

    def test_overriding_to_scorer_with_metadata_uses_fresh_exact_score(self, index_class):
        # Default scorer has no metadata; overriding to WRatio must use its own optimal score.
        index = index_class(["Alpha Phone"], scorer=fixed_score_scorer, score_cutoff=0)

        default_match = require_not_none(index.find_one("Alpha Phone"))
        assert default_match.score == 42

        overridden_match = require_not_none(index.find_one("Alpha Phone", scorer=WRatio))
        assert overridden_match.score == 100.0


# ---------------------------------------------------------------------------
# Sequence index — find_one_batch_cdist honors overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_class", SEQUENCE_INDEX_CLASSES)
def test_find_one_batch_cdist_override_matches_find_one_override(index_class):
    index = index_class(["Alpha Phone", "Beta Tablet"], score_cutoff=99)

    default_result = index.find_one_batch_cdist(["Alpa Phone"])
    assert default_result == [None]

    overridden_result = index.find_one_batch_cdist(["Alpa Phone"], score_cutoff=50)
    plain_override = index.find_one("Alpa Phone", score_cutoff=50)

    assert overridden_result == [plain_override]


# ---------------------------------------------------------------------------
# Sequence index — derived/convenience methods forward overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_class", SEQUENCE_INDEX_CLASSES)
class TestSequenceIndexDerivedMethodsForwardOverrides:
    def test_contains_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert index.contains("Alpa Phone") is False
        assert index.contains("Alpa Phone", score_cutoff=50) is True

    def test_score_all_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert index.score_all("Alpa Phone") == [None]
        overridden = index.score_all("Alpa Phone", score_cutoff=50)
        match = require_not_none(overridden[0])
        assert match.value == "Alpha Phone"

    def test_iter_scores_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert list(index.iter_scores("Alpa Phone")) == [None]
        overridden = list(index.iter_scores("Alpa Phone", score_cutoff=50))
        match = require_not_none(overridden[0])
        assert match.value == "Alpha Phone"

    def test_find_one_batch_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert index.find_one_batch(["Alpa Phone"]) == [None]
        overridden = index.find_one_batch(["Alpa Phone"], score_cutoff=50)
        assert overridden[0] is not None

    def test_find_many_batch_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert index.find_many_batch(["Alpa Phone"]) == [[]]
        overridden = index.find_many_batch(["Alpa Phone"], score_cutoff=50)
        assert len(overridden[0]) == 1


# ---------------------------------------------------------------------------
# Keyed index — score_cutoff, scorer, exact_match freshness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_class", KEYED_INDEX_CLASSES)
class TestKeyedIndexOverrides:
    def test_default_cutoff_still_applies_without_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        assert index.find_one("Alpa Phone") is None

    def test_looser_cutoff_override_accepts_previously_rejected_candidate(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        match = require_not_none(index.find_one("Alpa Phone", score_cutoff=50))

        assert match.value == "Alpha Phone"

    def test_score_cutoff_none_disables_constructor_default(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        match = require_not_none(index.find_one("Alpa Phone", score_cutoff=None))

        assert match.value == "Alpha Phone"

    def test_scorer_kwargs_none_clears_constructor_default(self, index_class):
        index = index_class(
            ["Alpha Phone"],
            scorer=kwarg_reading_scorer,
            scorer_kwargs={"weights": (42,)},
            score_cutoff=0,
        )

        default_match = require_not_none(index.find_one("query"))
        assert default_match.score == 42

        overridden_match = require_not_none(index.find_one("query", scorer_kwargs=None))
        assert overridden_match.score == 0

    def test_scorer_type_override_flips_ranking_order(self, index_class):
        def rank_scorer(_query: str, choice: str, **_kwargs: object) -> int:
            return {"alpha": 10, "beta": 90}.get(choice, 0)

        index = index_class(
            ["Alpha", "Beta"],
            scorer=rank_scorer,
            scorer_type=ScorerType.SIMILARITY,
            score_cutoff=None,
        )

        similarity_order = index.find_many("query", limit=None)
        assert [match.value for match in similarity_order] == ["Beta", "Alpha"]

        distance_order = index.find_many("query", limit=None, scorer_type=ScorerType.DISTANCE)
        assert [match.value for match in distance_order] == ["Alpha", "Beta"]

    def test_exact_match_recomputes_metadata_for_overridden_scorer(self, index_class):
        index = index_class(["Alpha Phone"])

        default_exact = require_not_none(index.exact_match("Alpha Phone", query="Alpha Phone"))
        assert default_exact.score == 100.0

        overridden_exact = require_not_none(
            index.exact_match(
                "Alpha Phone",
                query="Alpha Phone",
                scorer=fixed_score_scorer,
                scorer_type=ScorerType.SIMILARITY,
                score_cutoff=0,
            )
        )
        assert overridden_exact.score == 42

    def test_iter_scores_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        values = ["Alpha Phone"]
        assert list(index.iter_scores(values, "Alpa Phone")) == [None]
        overridden = list(index.iter_scores(values, "Alpa Phone", score_cutoff=50))
        assert overridden[0] is not None

    def test_score_all_forwards_score_cutoff_override(self, index_class):
        index = index_class(["Alpha Phone"], score_cutoff=99)

        values = ["Alpha Phone"]
        assert index.score_all(values, "Alpa Phone") == [None]
        overridden = index.score_all(values, "Alpa Phone", score_cutoff=50)
        assert overridden[0] is not None


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
def test_find_one_forwards_per_query_score_hint(index_class, monkeypatch):
    observed_hints: list[int | float | None] = []
    original_extract_one = process.extractOne

    def recording_extract_one(*args, **kwargs):
        observed_hints.append(kwargs["score_hint"])
        return original_extract_one(*args, **kwargs)

    monkeypatch.setattr(process, "extractOne", recording_extract_one)
    index = index_class(["Alpha Phone"], score_hint=90)

    assert index.find_one("Alpa Phone", score_hint=50) is not None
    assert observed_hints == [50]


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
def test_native_distance_scorer_type_is_inferred_per_query(index_class):
    index = index_class(["kitten", "sitting"], score_cutoff=None)

    match = require_not_none(index.find_one("sittin", scorer=Levenshtein.distance))

    assert match.value == "sitting"
    assert match.score == 1


@pytest.mark.parametrize("index_class", [*SEQUENCE_INDEX_CLASSES, *KEYED_INDEX_CLASSES])
def test_custom_scorer_override_requires_explicit_scorer_type(index_class):
    index = index_class(["Alpha Phone"])

    with pytest.raises(ValueError, match="scorer_type is required"):
        index.find_one("Alpha Phone", scorer=fixed_score_scorer)


def test_invalid_scorer_metadata_cannot_infer_query_override_type():
    def scorer(_query: str, _choice: str, **_kwargs: object) -> int:
        return 42

    def raise_type_error(**_kwargs: object) -> dict[str, int]:
        raise TypeError

    adapters = [
        {"get_scorer_flags": None},
        {"get_scorer_flags": raise_type_error},
        {"get_scorer_flags": lambda **_kwargs: []},
        {"get_scorer_flags": lambda **_kwargs: {"optimal_score": "invalid", "worst_score": 0}},
    ]
    index = FuzzySequenceIndex(["Alpha Phone"])

    for adapter in adapters:
        # noinspection PyUnresolvedReferences
        scorer._RF_ScorerPy = adapter
        with pytest.raises(ValueError, match="scorer_type is required"):
            index.find_one("Alpha Phone", scorer=scorer)


# ---------------------------------------------------------------------------
# Facade layer — one override test per facade, on the primary find method and
# on a convenience method, covering both IndexStrategy values where supported.
# ---------------------------------------------------------------------------


class TestFuzzyListOverrides:
    def test_fuzzy_find_one_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone"], score_cutoff=99)

        assert collection.fuzzy_find_one("Alpa Phone") is None
        assert collection.fuzzy_find_one("Alpa Phone", score_cutoff=50) is not None

    def test_fuzzy_contains_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone"], score_cutoff=99)

        assert collection.fuzzy_contains("Alpa Phone") is False
        assert collection.fuzzy_contains("Alpa Phone", score_cutoff=50) is True

    def test_fuzzy_get_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone"], score_cutoff=99)

        assert collection.fuzzy_get("Alpa Phone") is None
        assert collection.fuzzy_get("Alpa Phone", score_cutoff=50) == "Alpha Phone"

    def test_fuzzy_count_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone"], score_cutoff=99)

        assert collection.fuzzy_count("Alpa Phone") == 0
        assert collection.fuzzy_count("Alpa Phone", score_cutoff=50) == 1

    def test_fuzzy_find_index_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone"], score_cutoff=99)

        with pytest.raises(ValueError, match="has no fuzzy match in the collection"):
            collection.fuzzy_find_index("Alpa Phone")
        assert collection.fuzzy_find_index("Alpa Phone", score_cutoff=50) == 0

    def test_fuzzy_discard_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone"], score_cutoff=99)

        collection.fuzzy_discard("Alpa Phone")
        assert "Alpha Phone" in collection

        collection.fuzzy_discard("Alpa Phone", score_cutoff=50)
        assert "Alpha Phone" not in collection

    def test_fuzzy_discard_all_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone", "Alpha Phone Two"], score_cutoff=99)

        removed = collection.fuzzy_discard_all("Alpa Phone", score_cutoff=50)
        assert removed == 2
        assert len(collection) == 0

    def test_fuzzy_retain_all_score_cutoff_override(self):
        collection = FuzzyList(["Alpha Phone", "Beta Tablet"], score_cutoff=50)

        collection.fuzzy_retain_all("Alpa Phone", score_cutoff=99)
        assert list(collection) == []


class TestFuzzyTupleOverrides:
    def test_fuzzy_find_one_score_cutoff_override(self):
        collection = FuzzyTuple(["Alpha Phone"], score_cutoff=99)

        assert collection.fuzzy_find_one("Alpa Phone") is None
        assert collection.fuzzy_find_one("Alpa Phone", score_cutoff=50) is not None

    def test_fuzzy_get_score_cutoff_override(self):
        collection = FuzzyTuple(["Alpha Phone"], score_cutoff=99)

        assert collection.fuzzy_get("Alpa Phone") is None
        assert collection.fuzzy_get("Alpa Phone", score_cutoff=50) == "Alpha Phone"


@pytest.mark.parametrize("strategy", [IndexStrategy.SEQUENCE, IndexStrategy.KEYED])
class TestFuzzySetOverrides:
    def test_fuzzy_find_one_score_cutoff_override(self, strategy):
        collection = FuzzySet(["Alpha Phone"], score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_find_one("Alpa Phone") is None
        assert collection.fuzzy_find_one("Alpa Phone", score_cutoff=50) is not None

    def test_fuzzy_contains_score_cutoff_override(self, strategy):
        collection = FuzzySet(["Alpha Phone"], score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_contains("Alpa Phone") is False
        assert collection.fuzzy_contains("Alpa Phone", score_cutoff=50) is True


@pytest.mark.parametrize("strategy", [IndexStrategy.SEQUENCE, IndexStrategy.KEYED])
class TestFrozenFuzzySetOverrides:
    def test_fuzzy_find_one_score_cutoff_override(self, strategy):
        collection = FrozenFuzzySet(["Alpha Phone"], score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_find_one("Alpa Phone") is None
        assert collection.fuzzy_find_one("Alpa Phone", score_cutoff=50) is not None

    def test_fuzzy_get_score_cutoff_override(self, strategy):
        collection = FrozenFuzzySet(["Alpha Phone"], score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_get("Alpa Phone") is None
        assert collection.fuzzy_get("Alpa Phone", score_cutoff=50) == "Alpha Phone"


@pytest.mark.parametrize("strategy", [IndexStrategy.SEQUENCE, IndexStrategy.KEYED])
class TestFuzzyDictOverrides:
    def test_fuzzy_find_key_score_cutoff_override(self, strategy):
        collection = FuzzyDict({"Alpha Phone": 1}, score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_find_key("Alpa Phone") is None
        assert collection.fuzzy_find_key("Alpa Phone", score_cutoff=50) is not None

    def test_fuzzy_get_score_cutoff_override(self, strategy):
        collection = FuzzyDict({"Alpha Phone": 1}, score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_get("Alpa Phone") is None
        assert collection.fuzzy_get("Alpa Phone", score_cutoff=50) == 1

    def test_fuzzy_discard_score_cutoff_override(self, strategy):
        collection = FuzzyDict({"Alpha Phone": 1}, score_cutoff=99, strategy=strategy)

        collection.fuzzy_discard("Alpa Phone")
        assert "Alpha Phone" in collection

        collection.fuzzy_discard("Alpa Phone", score_cutoff=50)
        assert "Alpha Phone" not in collection


@pytest.mark.parametrize("strategy", [IndexStrategy.SEQUENCE, IndexStrategy.KEYED])
class TestFrozenFuzzyDictOverrides:
    def test_fuzzy_find_key_score_cutoff_override(self, strategy):
        collection = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_find_key("Alpa Phone") is None
        assert collection.fuzzy_find_key("Alpa Phone", score_cutoff=50) is not None

    def test_fuzzy_get_score_cutoff_override(self, strategy):
        collection = FrozenFuzzyDict({"Alpha Phone": 1}, score_cutoff=99, strategy=strategy)

        assert collection.fuzzy_get("Alpa Phone") is None
        assert collection.fuzzy_get("Alpa Phone", score_cutoff=50) == 1
