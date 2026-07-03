"""Benchmark production collection strategies and explicit architecture prototypes.

Run it directly:

    python benchmarks/index_strategy_benchmark.py
"""

import argparse
import statistics
import sys
import tempfile
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableSet
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz import process  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402
from rapidfuzz.fuzz import WRatio, ratio  # noqa: E402

from benchmarks.datasets import (  # noqa: E402
    DataProfile,
    QuerySet,
    build_queries,
    build_values,
)
from benchmarks.utils import (  # noqa: E402
    measure_peak_kib,
    measure_timings,
    non_negative_int,
    positive_int,
    result_size,
    write_benchmark_reports,
)
from rapidfuzz_collections import (  # noqa: E402
    FrozenFuzzyDict,
    FrozenFuzzySet,
    FuzzyDict,
    FuzzyList,
    FuzzySet,
    FuzzyTuple,
    IndexStrategy,
    KeyValueMatch,
    Normalizer,
    ScorerType,
    ValueMatch,
)
from rapidfuzz_collections.indexes import passes_score_cutoff  # noqa: E402

# noinspection PyProtectedMember
# The prototype uses production metadata resolution so comparisons vary the
# storage architecture rather than scorer semantics.
from rapidfuzz_collections.indexes.base import _process_scorer_metadata  # noqa: E402

SearchValue = object
Scorer = Callable[..., int | float]


class CollectionCase(StrEnum):
    """Production facades and explicitly selected prototype cases."""

    FROZEN_DICT = "frozen-dict"
    FROZEN_KEYED_DICT = "frozen-keyed-dict"
    FROZEN_KEYED_SET = "frozen-keyed-set"
    FROZEN_SET = "frozen-set"
    FUZZY_DICT_KEYED = "fuzzy-dict-keyed"
    FUZZY_DICT_SEQUENCE = "fuzzy-dict-sequence"
    FUZZY_SET_KEYED = "fuzzy-set-keyed"
    FUZZY_SET_SEQUENCE = "fuzzy-set-sequence"
    LIST = "list"
    ORDERED_DICT_PROTOTYPE = "ordered-dict"
    ORDERED_SET_PROTOTYPE = "ordered-set"
    TUPLE = "tuple"


class NormalizerProfile(StrEnum):
    """Normalizer profiles for separating storage and normalization costs."""

    DEFAULT = "default"
    PIPELINE = "pipeline"


class ScorerProfile(StrEnum):
    """RapidFuzz scorer profiles."""

    LEVENSHTEIN_DISTANCE = "levenshtein-distance"
    RATIO = "ratio"
    WRATIO = "wratio"


class WorkloadProfile(StrEnum):
    """Composite workloads for end-to-end strategy comparisons."""

    BATCH_HEAVY = "batch-heavy"
    BULK_MUTATION_HEAVY = "bulk-mutation-heavy"
    EXACT_MUTATION_HEAVY = "exact-mutation-heavy"
    LOOKUP_HEAVY = "lookup-heavy"
    READ_ONLY = "read-only"


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark row."""

    case: CollectionCase
    profile: DataProfile
    normalizer: NormalizerProfile
    scorer: ScorerProfile
    operation: str
    items: int
    repeats: int
    best_ms: float
    median_ms: float
    peak_kib: float
    result_size: int | None


def normalizer_for(profile: NormalizerProfile) -> Callable[[object], str | None] | None:
    """Return the configured normalizer for a benchmark profile."""

    if profile == NormalizerProfile.DEFAULT:
        return None
    if profile == NormalizerProfile.PIPELINE:
        return Normalizer().isinstance_str().strip().casefold().re_sub(r"\s+", " ").min_length(3)
    raise NotImplementedError(profile)


def scorer_for(profile: ScorerProfile) -> tuple[Scorer, ScorerType, int | float]:
    """Return scorer, scorer type, and default score cutoff."""

    if profile == ScorerProfile.RATIO:
        return ratio, ScorerType.SIMILARITY, 80
    if profile == ScorerProfile.WRATIO:
        return WRatio, ScorerType.SIMILARITY, 80
    if profile == ScorerProfile.LEVENSHTEIN_DISTANCE:
        return Levenshtein.distance, ScorerType.DISTANCE, 5
    raise NotImplementedError(profile)


class OrderedUniqueFuzzyIndexPrototype[T: Hashable]:
    """Benchmark-only index for unique hashable domains.

    The candidate keeps RapidFuzz choices as a dense list, matching the fast
    sequence-index lookup shape, while exact and mutation paths use hash maps.

    This remains an active architecture experiment. It is not part of the
    public package and must be selected through an ``ordered-*`` benchmark case.
    """

    __slots__ = (
        "_choice_values",
        "_normalizer",
        "_normalized_choices",
        "_normalized_values",
        "_optimal_score",
        "_scorer",
        "_scorer_kwargs",
        "_scorer_type",
        "_score_cutoff",
        "_score_hint",
        "_value_to_choice_position",
        "_value_to_position",
        "_values",
    )

    def __init__(
        self,
        values: Iterable[T] = (),
        *,
        normalizer: Callable[[object], str | None] | None = None,
        scorer: Scorer = WRatio,
        scorer_kwargs: dict[str, Any] | None = None,
        scorer_type: ScorerType = ScorerType.SIMILARITY,
        score_cutoff: int | float | None = 80,
        score_hint: int | float | None = None,
    ) -> None:
        self._normalizer = normalizer or Normalizer().isinstance_str().strip().casefold().min_length(3)
        self._scorer = scorer
        self._scorer_kwargs = dict(scorer_kwargs) if scorer_kwargs is not None else None
        self._scorer_type = scorer_type
        self._score_cutoff = score_cutoff
        self._score_hint = score_hint
        _, self._optimal_score = _process_scorer_metadata(
            scorer,
            self._scorer_kwargs,
            scorer_type,
        )
        self._values: list[T] = []
        self._normalized_values: list[str | None] = []
        self._normalized_choices: list[str] = []
        self._choice_values: list[T] = []
        self._value_to_position: dict[T, int] = {}
        self._value_to_choice_position: dict[T, int] = {}
        for value in values:
            self.add(value)

    def _exact_score(self) -> int:
        """Return the conventional exact score for the configured direction."""

        if self._scorer_type == ScorerType.DISTANCE:
            return 0
        return 100

    def _match(
        self,
        value: T,
        *,
        query: object,
        normalized_query: str | None,
        normalized_value: str | None = None,
        score: int | float | None = None,
    ) -> ValueMatch[T]:
        if normalized_value is None:
            position = self._value_to_position.get(value)
            normalized_value = None if position is None else self._normalized_values[position]
        return ValueMatch(
            value=value,
            score=self._exact_score() if score is None else score,
            query=query,
            normalized_query=normalized_query or normalized_value or "",
            normalized_value=normalized_value or "",
        )

    def _refresh_positions_from(self, start: int) -> None:
        for position in range(start, len(self._values)):
            self._value_to_position[self._values[position]] = position

    def _refresh_choice_positions_from(self, start: int) -> None:
        for position in range(start, len(self._choice_values)):
            self._value_to_choice_position[self._choice_values[position]] = position

    def add(self, value: T) -> None:
        if value in self._value_to_position:
            return
        normalized_value = self._normalizer(value)
        self._value_to_position[value] = len(self._values)
        self._values.append(value)
        self._normalized_values.append(normalized_value)
        if normalized_value is None:
            return
        self._value_to_choice_position[value] = len(self._normalized_choices)
        self._choice_values.append(value)
        self._normalized_choices.append(normalized_value)

    def batch_remove(self, values: Iterable[T]) -> None:
        removed = set(values)
        if not removed:
            return
        self.rebuild(value for value in self._values if value not in removed)

    def config_kwargs(self) -> dict[str, Any]:
        return {
            "normalizer": self._normalizer,
            "scorer": self._scorer,
            "scorer_kwargs": dict(self._scorer_kwargs) if self._scorer_kwargs is not None else None,
            "scorer_type": self._scorer_type,
            "score_cutoff": self._score_cutoff,
            "score_hint": self._score_hint,
        }

    def exact_match(self, value: T, *, query: object) -> ValueMatch[T]:
        """Return the canonical stored value equal to ``value``."""

        position = self._value_to_position[value]
        return self._match(self._values[position], query=query, normalized_query=None)

    def _exact_candidate(
        self,
        query: object,
        normalized_query: str,
    ) -> ValueMatch[T] | None:
        """Return the scored stored value equal to the query, if searchable."""

        if not isinstance(query, Hashable) or query not in self._value_to_position:
            return None
        position = self._value_to_position[query]
        value = self._values[position]
        normalized_value = self._normalized_values[position]
        if normalized_value is None:
            return None
        score = self._scorer(normalized_query, normalized_value, **(self._scorer_kwargs or {}))
        if not passes_score_cutoff(
            score,
            scorer_type=self._scorer_type,
            score_cutoff=self._score_cutoff,
        ):
            return None
        return self._match(
            value,
            query=query,
            normalized_query=normalized_query,
            normalized_value=normalized_value,
            score=score,
        )

    def _ranked_matches(
        self,
        query: object,
        normalized_query: str,
        *,
        limit: int | None,
        exact_candidate: ValueMatch[T] | None = None,
    ) -> list[ValueMatch[T]]:
        """Return matches using the production score, exact, and order policy."""

        if limit == 0:
            return []
        if limit == 1:
            # noinspection PyTypeChecker
            result = process.extractOne(
                normalized_query,
                self._normalized_choices,
                scorer=self._scorer,
                score_cutoff=self._score_cutoff,
                score_hint=self._score_hint,
                scorer_kwargs=self._scorer_kwargs,
            )
            extracted = () if result is None else (result,)
        else:
            # noinspection PyTypeChecker
            extracted = process.extract(
                normalized_query,
                self._normalized_choices,
                scorer=self._scorer,
                score_cutoff=self._score_cutoff,
                score_hint=self._score_hint,
                scorer_kwargs=self._scorer_kwargs,
                limit=limit,
            )

        matches_by_value: dict[T, ValueMatch[T]] = {}
        for normalized_value, score, choice_position in extracted:
            value = self._choice_values[cast(int, choice_position)]
            if value is not None:
                matches_by_value[value] = self._match(
                    value,
                    query=query,
                    normalized_query=normalized_query,
                    normalized_value=normalized_value,
                    score=score,
                )
        if exact_candidate is None:
            exact_candidate = self._exact_candidate(query, normalized_query)
        if exact_candidate is not None:
            matches_by_value[exact_candidate.value] = exact_candidate

        def order_key(match: ValueMatch[T]) -> tuple[int | float, bool, int]:
            ordered_score = match.score if self._scorer_type == ScorerType.DISTANCE else -match.score
            is_exact = exact_candidate is not None and match.value == exact_candidate.value
            return ordered_score, not is_exact, self._value_to_position[match.value]

        matches = sorted(matches_by_value.values(), key=order_key)
        return matches if limit is None else matches[:limit]

    def find_many(self, query: object, *, limit: int | None = 5) -> list[ValueMatch[T]]:
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return []
        return self._ranked_matches(query, normalized_query, limit=limit)

    def find_one(self, query: object) -> ValueMatch[T] | None:
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None
        exact_candidate = self._exact_candidate(query, normalized_query)
        if (
            exact_candidate is not None
            and self._optimal_score is not None
            and exact_candidate.score == self._optimal_score
        ):
            return exact_candidate
        matches = self._ranked_matches(
            query,
            normalized_query,
            limit=1,
            exact_candidate=exact_candidate,
        )
        return matches[0] if matches else None

    def iter_scores(self, query: object) -> Iterator[ValueMatch[T] | None]:
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            yield from (None for _ in self._values)
            return
        scorer_kwargs = self._scorer_kwargs or {}
        for value, normalized_value in zip(self._values, self._normalized_values, strict=True):
            if normalized_value is None:
                yield None
                continue
            score = self._scorer(normalized_query, normalized_value, **scorer_kwargs)
            if not passes_score_cutoff(score, scorer_type=self._scorer_type, score_cutoff=self._score_cutoff):
                yield None
                continue
            yield self._match(
                value,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
                score=score,
            )

    def rebuild(self, values: Iterable[T]) -> None:
        config = self.config_kwargs()
        self.__init__(values, **config)

    def remove(self, value: T) -> None:
        source_position = self._value_to_position.pop(value, None)
        if source_position is None:
            return
        normalized_value = self._normalized_values.pop(source_position)
        self._values.pop(source_position)
        self._refresh_positions_from(source_position)
        choice_position = self._value_to_choice_position.pop(value, None)
        if choice_position is None or normalized_value is None:
            return
        self._normalized_choices.pop(choice_position)
        self._choice_values.pop(choice_position)
        self._refresh_choice_positions_from(choice_position)

    def score_all(self, query: object) -> list[ValueMatch[T] | None]:
        results: list[ValueMatch[T] | None] = [None] * len(self._values)
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return results
        for normalized_value, score, choice_position in process.extract(
            normalized_query,
            self._normalized_choices,
            scorer=self._scorer,
            score_cutoff=self._score_cutoff,
            score_hint=self._score_hint,
            scorer_kwargs=self._scorer_kwargs,
            limit=None,
        ):
            value = self._choice_values[choice_position]
            source_position = self._value_to_position[value]
            results[source_position] = self._match(
                value,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
                score=score,
            )
        return results


class OrderedUniqueFuzzyDictPrototype[K: Hashable, V]:
    """Benchmark facade using the ordered unique index prototype for mapping keys."""

    __slots__ = ("_data", "_key_index")

    def __init__(
        self,
        items: Iterable[tuple[K, V]] | Mapping[K, V] = (),
        *,
        normalizer: Callable[[object], str | None] | None = None,
        scorer: Scorer = WRatio,
        scorer_kwargs: dict[str, Any] | None = None,
        scorer_type: ScorerType = ScorerType.SIMILARITY,
        score_cutoff: int | float | None = 80,
        score_hint: int | float | None = None,
    ) -> None:
        self._data: dict[K, V] = dict(items)
        self._key_index = OrderedUniqueFuzzyIndexPrototype(
            self._data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: K, value: V) -> None:
        is_new_key = key not in self._data
        self._data[key] = value
        if is_new_key:
            self._key_index.add(key)

    def _item_match(self, match: ValueMatch[K]) -> KeyValueMatch[K, V]:
        return KeyValueMatch(
            key=match.value,
            value=self._data[match.value],
            score=match.score,
            query=match.query,
            normalized_query=match.normalized_query,
            normalized_key=match.normalized_value,
        )

    def pop(self, key: K, default: V | None = None) -> V | None:
        if key not in self._data:
            return default
        value = self._data.pop(key)
        self._key_index.remove(key)
        return value

    def fuzzy_contains_key(self, query: object) -> bool:
        return self.fuzzy_find_key(query) is not None

    def fuzzy_discard(self, query: object) -> None:
        match = self.fuzzy_find_key(query)
        if match is not None:
            self.pop(match.value, None)

    def fuzzy_discard_all(self, query: object) -> int:
        matches = self.fuzzy_find_keys(query, limit=None)
        for match in matches:
            self._data.pop(match.value, None)
        self._key_index.rebuild(self._data)
        return len(matches)

    def fuzzy_find_item(self, query: object) -> KeyValueMatch[K, V] | None:
        match = self.fuzzy_find_key(query)
        return None if match is None else self._item_match(match)

    def fuzzy_find_item_batch(self, queries: Iterable[object]) -> list[KeyValueMatch[K, V] | None]:
        return [self.fuzzy_find_item(query) for query in queries]

    def fuzzy_find_key(self, query: object) -> ValueMatch[K] | None:
        return self._key_index.find_one(query)

    def fuzzy_find_key_batch(self, queries: Iterable[object]) -> list[ValueMatch[K] | None]:
        return [self.fuzzy_find_key(query) for query in queries]

    def fuzzy_find_keys(self, query: object, *, limit: int | None = 5) -> list[ValueMatch[K]]:
        return self._key_index.find_many(query, limit=limit)

    def fuzzy_get(self, query: object, default: V | None = None) -> V | None:
        match = self.fuzzy_find_item(query)
        return default if match is None else match.value

    def fuzzy_get_batch(self, queries: Iterable[object], *, default: V | None = None) -> list[V | None]:
        return [default if match is None else match.value for match in self.fuzzy_find_item_batch(queries)]

    def fuzzy_iter_scores(self, query: object) -> Iterator[KeyValueMatch[K, V] | None]:
        for match in self._key_index.iter_scores(query):
            yield None if match is None else self._item_match(match)

    def fuzzy_retain_all(self, query: object) -> int:
        retained = {match.value for match in self.fuzzy_find_keys(query, limit=None)}
        to_delete = [key for key in self._data if key not in retained]
        for key in to_delete:
            del self._data[key]
        self._key_index.rebuild(self._data)
        return len(to_delete)

    def fuzzy_score_all(self, query: object) -> list[KeyValueMatch[K, V] | None]:
        return [None if match is None else self._item_match(match) for match in self._key_index.score_all(query)]


class OrderedUniqueFuzzySetPrototype[T: Hashable](MutableSet[T]):
    """Benchmark facade using the ordered unique index prototype for set values."""

    __slots__ = ("_data", "_index")

    def __contains__(self, value: object) -> bool:
        return value in self._data

    def __init__(
        self,
        values: Iterable[T] = (),
        *,
        normalizer: Callable[[object], str | None] | None = None,
        scorer: Scorer = WRatio,
        scorer_kwargs: dict[str, Any] | None = None,
        scorer_type: ScorerType = ScorerType.SIMILARITY,
        score_cutoff: int | float | None = 80,
        score_hint: int | float | None = None,
    ) -> None:
        self._data: dict[T, None] = {}
        self._index = OrderedUniqueFuzzyIndexPrototype(
            (),
            normalizer=normalizer,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self.update(values)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, value: T) -> None:
        if value in self._data:
            return
        self._data[value] = None
        self._index.add(value)

    def discard(self, value: T) -> None:
        if value not in self._data:
            return
        del self._data[value]
        self._index.remove(value)

    def fuzzy_contains(self, query: object) -> bool:
        return self.fuzzy_find_one(query) is not None

    def fuzzy_discard(self, query: object) -> None:
        match = self.fuzzy_find_one(query)
        if match is not None:
            self.discard(match.value)

    def fuzzy_discard_all(self, query: object) -> int:
        matches = self.fuzzy_find_many(query, limit=None)
        for match in matches:
            self._data.pop(match.value, None)
        self._index.rebuild(self._data)
        return len(matches)

    def fuzzy_find_many(self, query: object, *, limit: int | None = 5) -> list[ValueMatch[T]]:
        return self._index.find_many(query, limit=limit)

    def fuzzy_find_one(self, query: object) -> ValueMatch[T] | None:
        return self._index.find_one(query)

    def fuzzy_find_one_batch(self, queries: Iterable[object]) -> list[ValueMatch[T] | None]:
        return [self.fuzzy_find_one(query) for query in queries]

    def fuzzy_get(self, query: object, default: T | None = None) -> T | None:
        match = self.fuzzy_find_one(query)
        return default if match is None else match.value

    def fuzzy_get_batch(self, queries: Iterable[object], *, default: T | None = None) -> list[T | None]:
        return [default if match is None else match.value for match in self.fuzzy_find_one_batch(queries)]

    def fuzzy_iter_scores(self, query: object) -> Iterator[ValueMatch[T] | None]:
        return self._index.iter_scores(query)

    def fuzzy_retain_all(self, query: object) -> int:
        retained = {match.value for match in self.fuzzy_find_many(query, limit=None)}
        to_delete = [value for value in self._data if value not in retained]
        for value in to_delete:
            del self._data[value]
        self._index.rebuild(self._data)
        return len(to_delete)

    def fuzzy_score_all(self, query: object) -> list[ValueMatch[T] | None]:
        return self._index.score_all(query)

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            for value in other:
                self.add(value)


def mapping_from_values(values: Iterable[SearchValue]) -> dict[SearchValue, str]:
    """Build mapping data while following ordinary dict duplicate semantics."""

    return {value: f"value-{index:06d}" for index, value in enumerate(values)}


def make_collection(
    case: CollectionCase,
    values: list[SearchValue],
    *,
    normalizer: Callable[[object], str | None] | None,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
) -> Any:
    """Construct one collection facade for benchmarking."""

    kwargs = {
        "normalizer": normalizer,
        "scorer": scorer,
        "scorer_type": scorer_type,
        "score_cutoff": score_cutoff,
    }
    if case == CollectionCase.LIST:
        return FuzzyList(values, **kwargs)
    if case == CollectionCase.TUPLE:
        return FuzzyTuple(values, **kwargs)
    if case == CollectionCase.FUZZY_SET_SEQUENCE:
        return FuzzySet(values, strategy=IndexStrategy.SEQUENCE, **kwargs)
    if case == CollectionCase.FUZZY_SET_KEYED:
        return FuzzySet(values, strategy=IndexStrategy.KEYED, **kwargs)
    if case == CollectionCase.ORDERED_SET_PROTOTYPE:
        return OrderedUniqueFuzzySetPrototype(values, **kwargs)
    if case == CollectionCase.FROZEN_KEYED_SET:
        return FrozenFuzzySet(values, strategy=IndexStrategy.KEYED, **kwargs)
    if case == CollectionCase.FROZEN_SET:
        return FrozenFuzzySet(values, **kwargs)

    mapping = mapping_from_values(values)
    if case == CollectionCase.FUZZY_DICT_SEQUENCE:
        return FuzzyDict(mapping, strategy=IndexStrategy.SEQUENCE, **kwargs)
    if case == CollectionCase.FUZZY_DICT_KEYED:
        return FuzzyDict(mapping, strategy=IndexStrategy.KEYED, **kwargs)
    if case == CollectionCase.ORDERED_DICT_PROTOTYPE:
        return OrderedUniqueFuzzyDictPrototype(mapping, **kwargs)
    if case == CollectionCase.FROZEN_KEYED_DICT:
        return FrozenFuzzyDict(mapping, strategy=IndexStrategy.KEYED, **kwargs)
    if case == CollectionCase.FROZEN_DICT:
        return FrozenFuzzyDict(mapping, **kwargs)
    raise NotImplementedError(case)


def fuzzy_get(collection: Any, query: object) -> object:
    """Call the facade's best-match value getter."""

    return collection.fuzzy_get(query)


def fuzzy_contains(collection: Any, case: CollectionCase, query: object) -> bool:
    """Call the facade's membership-like fuzzy method."""

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
        CollectionCase.FROZEN_KEYED_DICT,
        CollectionCase.FROZEN_DICT,
    }:
        return collection.fuzzy_contains_key(query)
    return collection.fuzzy_contains(query)


def fuzzy_find_one(collection: Any, case: CollectionCase, query: object) -> Any:
    """Call the facade's best-match object method."""

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
        CollectionCase.FROZEN_KEYED_DICT,
        CollectionCase.FROZEN_DICT,
    }:
        return collection.fuzzy_find_key(query)
    return collection.fuzzy_find_one(query)


def fuzzy_find_many(collection: Any, case: CollectionCase, query: object, *, limit: int | None) -> Any:
    """Call the facade's multi-match object method."""

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
        CollectionCase.FROZEN_KEYED_DICT,
        CollectionCase.FROZEN_DICT,
    }:
        return collection.fuzzy_find_keys(query, limit=limit)
    return collection.fuzzy_find_many(query, limit=limit)


def fuzzy_find_one_batch(collection: Any, case: CollectionCase, queries: Iterable[object]) -> object:
    """Call the facade's best-match batch method."""

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
        CollectionCase.FROZEN_KEYED_DICT,
        CollectionCase.FROZEN_DICT,
    }:
        return collection.fuzzy_find_key_batch(queries)
    return collection.fuzzy_find_one_batch(queries)


def supports_cdist(case: CollectionCase) -> bool:
    """Return whether a case exposes the sequence-backed cdist path."""

    return case in {
        CollectionCase.LIST,
        CollectionCase.TUPLE,
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_SET_SEQUENCE,
        CollectionCase.FROZEN_DICT,
        CollectionCase.FROZEN_SET,
    }


def fuzzy_find_one_batch_cdist(collection: Any, case: CollectionCase, queries: Iterable[object]) -> list[Any]:
    """Call the facade's cdist best-match batch method."""

    if case in {CollectionCase.FUZZY_DICT_SEQUENCE, CollectionCase.FROZEN_DICT}:
        return collection.fuzzy_find_key_batch_cdist(queries)
    return collection.fuzzy_find_one_batch_cdist(queries)


def measure(
    case: CollectionCase,
    profile: DataProfile,
    normalizer_profile: NormalizerProfile,
    scorer_profile: ScorerProfile,
    operation: str,
    items: int,
    repeats: int,
    func: Callable[[], object],
    *,
    warmup: int = 0,
    no_memory: bool = False,
) -> BenchmarkResult:
    """Measure one operation including any work inside ``func``."""

    best_ms, median_ms = measure_timings(repeats, func, warmup=warmup)
    peak_kib = 0.0 if no_memory else measure_peak_kib(func)
    size = result_size(func())
    return BenchmarkResult(
        case=case,
        profile=profile,
        normalizer=normalizer_profile,
        scorer=scorer_profile,
        operation=operation,
        items=items,
        repeats=repeats,
        best_ms=best_ms,
        median_ms=median_ms,
        peak_kib=peak_kib,
        result_size=size,
    )


def measure_after_setup(
    case: CollectionCase,
    profile: DataProfile,
    normalizer_profile: NormalizerProfile,
    scorer_profile: ScorerProfile,
    operation: str,
    items: int,
    repeats: int,
    setup: Callable[[], Any],
    func: Callable[[Any], object],
    *,
    warmup: int = 0,
    no_memory: bool = False,
) -> BenchmarkResult:
    """Measure one operation after constructing a fresh collection."""

    for _ in range(warmup):
        func(setup())

    timings_ms: list[float] = []
    for _ in range(repeats):
        collection = setup()
        best_ms_once, _ = measure_timings(1, lambda c=collection: func(c))
        timings_ms.append(best_ms_once)

    collection_for_mem = setup()
    peak_kib = 0.0 if no_memory else measure_peak_kib(lambda c=collection_for_mem: func(c))
    size = result_size(func(setup()))

    return BenchmarkResult(
        case=case,
        profile=profile,
        normalizer=normalizer_profile,
        scorer=scorer_profile,
        operation=operation,
        items=items,
        repeats=repeats,
        best_ms=min(timings_ms),
        median_ms=statistics.median(timings_ms),
        peak_kib=peak_kib,
        result_size=size,
    )


def mutation_values(values: list[SearchValue], count: int) -> list[str]:
    """Return deterministic new searchable values for mutation workloads."""

    offset = len(values) + 1_000_000
    return [f"New Device {offset + i:06d}" for i in range(count)]


def exact_delete_targets(values: list[SearchValue], count: int) -> list[SearchValue]:
    """Return deterministic existing values for exact deletion workloads."""

    unique_values = list(dict.fromkeys(values))
    return unique_values[: min(count, len(unique_values))]


def run_insert_mutation(
    collection: Any,
    case: CollectionCase,
    values: list[SearchValue],
    updates: int,
) -> int:
    """Insert deterministic new values into one mutable collection."""

    new_values = mutation_values(values, updates)
    if case == CollectionCase.LIST:
        collection.extend(new_values)
        return len(new_values)

    if case in {
        CollectionCase.FUZZY_SET_SEQUENCE,
        CollectionCase.FUZZY_SET_KEYED,
        CollectionCase.ORDERED_SET_PROTOTYPE,
    }:
        before = len(collection)
        collection.update(new_values)
        return len(collection) - before

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
    }:
        before = len(collection)
        for value in new_values:
            collection[value] = f"value-{value}"
        return len(collection) - before

    raise ValueError(f"Collection case is not mutable: {case}")


def run_exact_delete_mutation(
    collection: Any,
    case: CollectionCase,
    values: list[SearchValue],
    updates: int,
) -> int:
    """Delete deterministic exact values from one mutable collection."""

    if case == CollectionCase.LIST:
        delete_count = min(updates, len(collection))
        for _ in range(delete_count):
            del collection[0]
        return delete_count

    targets = exact_delete_targets(values, updates)
    if case in {
        CollectionCase.FUZZY_SET_SEQUENCE,
        CollectionCase.FUZZY_SET_KEYED,
        CollectionCase.ORDERED_SET_PROTOTYPE,
    }:
        before = len(collection)
        for value in targets:
            collection.discard(value)
        return before - len(collection)

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
    }:
        before = len(collection)
        for key in targets:
            collection.pop(key, None)
        return before - len(collection)

    raise ValueError(f"Collection case is not mutable: {case}")


def run_fuzzy_discard_one_mutation(collection: Any, query: object) -> int:
    """Remove one fuzzy match and return the number of removed values."""

    before = len(collection)
    collection.fuzzy_discard(query)
    return before - len(collection)


def run_fuzzy_discard_all_mutation(collection: Any, query: object) -> int:
    """Remove all fuzzy matches and return the number of removed values."""

    return collection.fuzzy_discard_all(query)


def run_fuzzy_retain_all_mutation(collection: Any, query: object) -> int:
    """Retain fuzzy matches and return the number of removed values."""

    return collection.fuzzy_retain_all(query)


def run_mutation(
    collection: Any,
    case: CollectionCase,
    values: list[SearchValue],
    queries: QuerySet,
    updates: int,
) -> None:
    """Execute a representative mutation workload for one mutable collection."""

    new_values = mutation_values(values, updates)
    if case == CollectionCase.LIST:
        collection.extend(new_values)
        for _ in range(min(updates, len(collection))):
            del collection[0]
        collection.fuzzy_discard(queries.close)
        collection.fuzzy_discard_all(queries.close)
        collection.fuzzy_retain_all(queries.close)
        return

    if case in {
        CollectionCase.FUZZY_SET_SEQUENCE,
        CollectionCase.FUZZY_SET_KEYED,
        CollectionCase.ORDERED_SET_PROTOTYPE,
    }:
        collection.update(new_values)
        for value in exact_delete_targets(values, updates):
            collection.discard(value)
        collection.fuzzy_discard(queries.close)
        collection.fuzzy_discard_all(queries.close)
        collection.fuzzy_retain_all(queries.close)
        return

    if case in {
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.ORDERED_DICT_PROTOTYPE,
    }:
        for value in new_values:
            collection[value] = f"value-{value}"
        for key in exact_delete_targets(values, updates):
            collection.pop(key, None)
        collection.fuzzy_discard(queries.close)
        collection.fuzzy_discard_all(queries.close)
        collection.fuzzy_retain_all(queries.close)
        return

    raise ValueError(f"Collection case is not mutable: {case}")


MUTABLE_CASES = {
    CollectionCase.FUZZY_DICT_KEYED,
    CollectionCase.FUZZY_DICT_SEQUENCE,
    CollectionCase.FUZZY_SET_KEYED,
    CollectionCase.FUZZY_SET_SEQUENCE,
    CollectionCase.LIST,
    CollectionCase.ORDERED_DICT_PROTOTYPE,
    CollectionCase.ORDERED_SET_PROTOTYPE,
}
"""Collection cases that support mutation workloads."""


def is_mutable_case(case: CollectionCase) -> bool:
    """Return whether a collection case supports mutation workloads."""

    return case in MUTABLE_CASES


def run_read_only_workload(collection: Any, case: CollectionCase, queries: QuerySet) -> int:
    """Run a lookup-only workload and return the operation count."""

    operations = 0
    for query in queries.batch:
        fuzzy_get(collection, query)
        operations += 1
    for query in (queries.exact, queries.normalized_exact, queries.close, queries.miss):
        fuzzy_contains(collection, case, query)
        fuzzy_find_one(collection, case, query)
        fuzzy_find_many(collection, case, query, limit=5)
        operations += 3
    return operations


def run_lookup_heavy_workload(
    collection: Any,
    case: CollectionCase,
    values: list[SearchValue],
    queries: QuerySet,
    updates: int,
) -> int:
    """Run mostly lookup operations with a small mutation tail for mutable cases."""

    operations = 0
    for _ in range(5):
        operations += run_read_only_workload(collection, case, queries)

    if is_mutable_case(case):
        operations += run_insert_mutation(collection, case, values, max(1, updates // 10))
        operations += run_exact_delete_mutation(collection, case, values, max(1, updates // 10))
    return operations


def run_batch_heavy_workload(collection: Any, case: CollectionCase, queries: QuerySet) -> int:
    """Run batch lookup operations and return the logical operation count."""

    collection.fuzzy_get_batch(queries.batch)
    fuzzy_find_one_batch(collection, case, queries.batch)
    for query in (queries.close, queries.miss):
        fuzzy_find_many(collection, case, query, limit=None)
    return (len(queries.batch) * 2) + 2


def run_exact_mutation_heavy_workload(
    collection: Any,
    case: CollectionCase,
    values: list[SearchValue],
    updates: int,
) -> int:
    """Run exact insert/delete cycles and return the affected item count."""

    if not is_mutable_case(case):
        return 0
    operations = 0
    for _ in range(3):
        operations += run_insert_mutation(collection, case, values, updates)
        operations += run_exact_delete_mutation(collection, case, values, updates)
    return operations


def run_bulk_mutation_heavy_workload(collection: Any, case: CollectionCase, queries: QuerySet) -> int:
    """Run expensive fuzzy bulk mutation paths and return affected item count."""

    if not is_mutable_case(case):
        return 0
    removed = run_fuzzy_discard_all_mutation(collection, queries.close)
    removed += run_fuzzy_retain_all_mutation(collection, queries.close)
    return removed


def run_weighted_workload(
    collection: Any,
    case: CollectionCase,
    values: list[SearchValue],
    queries: QuerySet,
    updates: int,
    workload: WorkloadProfile,
) -> int:
    """Run one composite workload for strategy-level comparison."""

    if workload == WorkloadProfile.READ_ONLY:
        return run_read_only_workload(collection, case, queries)
    if workload == WorkloadProfile.LOOKUP_HEAVY:
        return run_lookup_heavy_workload(collection, case, values, queries, updates)
    if workload == WorkloadProfile.BATCH_HEAVY:
        return run_batch_heavy_workload(collection, case, queries)
    if workload == WorkloadProfile.EXACT_MUTATION_HEAVY:
        return run_exact_mutation_heavy_workload(collection, case, values, updates)
    if workload == WorkloadProfile.BULK_MUTATION_HEAVY:
        return run_bulk_mutation_heavy_workload(collection, case, queries)
    raise NotImplementedError(workload)


def run_case(
    case: CollectionCase,
    profile: DataProfile,
    normalizer_profile: NormalizerProfile,
    scorer_profile: ScorerProfile,
    *,
    items: int,
    repeats: int,
    batch_size: int,
    updates: int,
    workloads: Iterable[WorkloadProfile] = (),
    workloads_only: bool = False,
    exact_tie_only: bool = False,
    full_result_only: bool = False,
    warmup: int = 0,
    no_memory: bool = False,
) -> list[BenchmarkResult]:
    """Run all benchmark operations for one collection/profile/scorer tuple."""

    values = build_values(items, profile)
    queries = build_queries(values, batch_size)
    normalizer = normalizer_for(normalizer_profile)
    scorer, scorer_type, score_cutoff = scorer_for(scorer_profile)

    def setup() -> object:
        return make_collection(
            case,
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    collection = setup()

    _sanity = fuzzy_get(collection, queries.exact)
    if _sanity is None:
        raise AssertionError(f"{case}/{profile}: fuzzy_get returned None for exact query {queries.exact!r}")
    if queries.normalized_collision_exact is not None:
        collision_query = queries.normalized_collision_exact
        collision_match = fuzzy_find_one(collection, case, collision_query)
        collision_matches = fuzzy_find_many(collection, case, collision_query, limit=1)
        if collision_match is None or collision_match.value != collision_query:
            raise AssertionError(f"{case}/{profile}: find-one did not prefer exact value in a normalized collision")
        if not collision_matches or collision_matches[0].value != collision_query:
            raise AssertionError(
                f"{case}/{profile}: find-many(limit=1) did not prefer exact value in a normalized collision"
            )
        if exact_tie_only and supports_cdist(case):
            cdist_matches = fuzzy_find_one_batch_cdist(collection, case, [collision_query])
            if not cdist_matches or cdist_matches[0].value != collision_query:
                raise AssertionError(f"{case}/{profile}: cdist did not prefer exact value in a normalized collision")

    m_kwargs: dict[str, Any] = {"warmup": warmup, "no_memory": no_memory}

    if full_result_only:
        return [
            measure_after_setup(
                case,
                profile,
                normalizer_profile,
                scorer_profile,
                "find-many:all",
                items,
                repeats,
                setup,
                lambda c: fuzzy_find_many(c, case, queries.close, limit=None),
                **m_kwargs,
            )
        ]

    results = [
        measure(case, profile, normalizer_profile, scorer_profile, "build", items, repeats, setup, **m_kwargs),
    ]

    if not workloads_only and not exact_tie_only:
        results.extend(
            [
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, queries.exact),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:normalized-exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, queries.normalized_exact),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:close",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:miss",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, queries.miss),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "contains:close",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_contains(c, case, queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "find-one:close",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_one(c, case, queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "find-many:5",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_many(c, case, queries.close, limit=5),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "find-many:all",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_many(c, case, queries.close, limit=None),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "batch:get",
                    items,
                    repeats,
                    setup,
                    lambda c: c.fuzzy_get_batch(queries.batch),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "batch:find-one",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_one_batch(c, case, queries.batch),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "score-all",
                    items,
                    repeats,
                    setup,
                    lambda c: c.fuzzy_score_all(queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "iter-scores:consume",
                    items,
                    repeats,
                    setup,
                    lambda c: sum(1 for _ in c.fuzzy_iter_scores(queries.close)),
                    **m_kwargs,
                ),
            ]
        )
    if exact_tie_only and not workloads_only:
        results.extend(
            [
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, queries.exact),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:normalized-exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, queries.normalized_exact),
                    **m_kwargs,
                ),
            ]
        )

    if queries.normalized_collision_exact is not None and not workloads_only:
        collision_query = queries.normalized_collision_exact
        results.extend(
            [
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "lookup:normalized-collision-exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_get(c, collision_query),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "find-one:normalized-collision-exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_one(c, case, collision_query),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "find-many:1-normalized-collision-exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_many(c, case, collision_query, limit=1),
                    **m_kwargs,
                ),
            ]
        )
        if exact_tie_only and supports_cdist(case):
            collision_batch = [collision_query] * batch_size
            results.append(
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "batch-cdist:normalized-collision-exact",
                    items,
                    repeats,
                    setup,
                    lambda c: fuzzy_find_one_batch_cdist(c, case, collision_batch),
                    **m_kwargs,
                )
            )

    if is_mutable_case(case) and not workloads_only and not exact_tie_only:
        results.extend(
            [
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    f"mutation:insert:{updates}",
                    items,
                    repeats,
                    setup,
                    lambda c: run_insert_mutation(c, case, values, updates),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    f"mutation:exact-delete:{updates}",
                    items,
                    repeats,
                    setup,
                    lambda c: run_exact_delete_mutation(c, case, values, updates),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "mutation:fuzzy-discard-one",
                    items,
                    repeats,
                    setup,
                    lambda c: run_fuzzy_discard_one_mutation(c, queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "mutation:fuzzy-discard-all",
                    items,
                    repeats,
                    setup,
                    lambda c: run_fuzzy_discard_all_mutation(c, queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    "mutation:fuzzy-retain-all",
                    items,
                    repeats,
                    setup,
                    lambda c: run_fuzzy_retain_all_mutation(c, queries.close),
                    **m_kwargs,
                ),
                measure_after_setup(
                    case,
                    profile,
                    normalizer_profile,
                    scorer_profile,
                    f"mutation:mixed:{updates}",
                    items,
                    repeats,
                    setup,
                    lambda c: run_mutation(c, case, values, queries, updates),
                    **m_kwargs,
                ),
            ]
        )
    if is_mutable_case(case) and queries.normalized_collision_exact is not None and not workloads_only:
        collision_query = queries.normalized_collision_exact
        results.append(
            measure_after_setup(
                case,
                profile,
                normalizer_profile,
                scorer_profile,
                "mutation:fuzzy-discard-normalized-collision-exact",
                items,
                repeats,
                setup,
                lambda c: run_fuzzy_discard_one_mutation(c, collision_query),
                **m_kwargs,
            )
        )

    for workload in () if exact_tie_only else workloads:
        if not is_mutable_case(case) and workload in {
            WorkloadProfile.BULK_MUTATION_HEAVY,
            WorkloadProfile.EXACT_MUTATION_HEAVY,
        }:
            continue
        results.append(
            measure_after_setup(
                case,
                profile,
                normalizer_profile,
                scorer_profile,
                f"workload:{workload}",
                items,
                repeats,
                setup,
                lambda c, selected_workload=workload: run_weighted_workload(
                    c,
                    case,
                    values,
                    queries,
                    updates,
                    selected_workload,
                ),
                **m_kwargs,
            )
        )

    # Keep one prebuilt lookup row to expose steady-state lookup without setup
    # noise in small benchmark runs.
    steady_query = (
        queries.normalized_collision_exact
        if exact_tie_only and queries.normalized_collision_exact is not None
        else queries.close
    )
    steady_operation = "steady-lookup:normalized-collision-exact" if exact_tie_only else "steady-lookup:close"
    results.append(
        measure(
            case,
            profile,
            normalizer_profile,
            scorer_profile,
            steady_operation,
            items,
            repeats,
            lambda: fuzzy_get(collection, steady_query),
            **m_kwargs,
        )
    )
    return results


def write_outputs(results: list[BenchmarkResult], output_dir: Path, *, quiet: bool = False) -> None:
    """Write raw benchmark result rows as JSON and CSV.

    Args:
        results: Benchmark measurements to write, sorted by profile,
            normalizer, scorer, operation, and case before writing.
        output_dir: Directory to write the JSON and CSV report files into.
        quiet: If ``True``, skip printing the written file paths to stdout.
            Used by the pre-flight smoke check so its temporary report paths
            do not clutter output that automation may parse.
    """

    rows = sorted(
        (asdict(result) for result in results),
        key=lambda row: (row["profile"], row["normalizer"], row["scorer"], row["operation"], row["case"]),
    )
    write_benchmark_reports(rows, output_dir, stem="index_strategy_results", quiet=quiet)


def run_smoke_check(output_dir: Path) -> None:
    """Run a small current-API matrix to verify the benchmark harness itself works.

    This is the same check exercised by
    ``tests/test_benchmark_infrastructure.py`` under pytest, exposed as a plain
    function so ``main()`` can run it as a fast pre-flight gate before a
    potentially long benchmark matrix, without depending on pytest at runtime.

    Raises:
        AssertionError: If the harness does not cover the expected operations
            or does not write the expected output files.
    """
    results: list[BenchmarkResult] = []
    for case in (
        CollectionCase.FUZZY_DICT_SEQUENCE,
        CollectionCase.FUZZY_DICT_KEYED,
        CollectionCase.FUZZY_SET_SEQUENCE,
        CollectionCase.FUZZY_SET_KEYED,
    ):
        results.extend(
            run_case(
                case,
                DataProfile.UNIQUE,
                NormalizerProfile.DEFAULT,
                ScorerProfile.RATIO,
                items=20,
                repeats=1,
                batch_size=4,
                updates=2,
                workloads=(WorkloadProfile.READ_ONLY, WorkloadProfile.BULK_MUTATION_HEAVY),
            )
        )

    expected_operations = {
        "build",
        "lookup:close",
        "mutation:exact-delete:2",
        "mutation:fuzzy-discard-all",
        "mutation:fuzzy-discard-one",
        "mutation:fuzzy-retain-all",
        "mutation:insert:2",
        "mutation:mixed:2",
        "workload:bulk-mutation-heavy",
        "workload:read-only",
    }
    actual_operations = {result.operation for result in results}
    missing_operations = expected_operations - actual_operations
    if missing_operations:
        raise AssertionError(f"benchmark smoke check is missing operations: {sorted(missing_operations)}")

    write_outputs(results, output_dir, quiet=True)
    if not (output_dir / "index_strategy_results.json").is_file():
        raise AssertionError("benchmark smoke check did not write index_strategy_results.json")
    if not (output_dir / "index_strategy_results.csv").is_file():
        raise AssertionError("benchmark smoke check did not write index_strategy_results.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=positive_int, default=1000)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=50)
    parser.add_argument("--updates", type=positive_int, default=20)
    parser.add_argument(
        "--cases",
        choices=tuple(CollectionCase),
        default=(
            CollectionCase.LIST,
            CollectionCase.TUPLE,
            CollectionCase.FUZZY_DICT_SEQUENCE,
            CollectionCase.FUZZY_DICT_KEYED,
            CollectionCase.FUZZY_SET_SEQUENCE,
            CollectionCase.FUZZY_SET_KEYED,
            CollectionCase.FROZEN_DICT,
            CollectionCase.FROZEN_KEYED_DICT,
            CollectionCase.FROZEN_SET,
            CollectionCase.FROZEN_KEYED_SET,
        ),
        nargs="+",
        type=CollectionCase,
    )
    parser.add_argument(
        "--profiles",
        choices=tuple(DataProfile),
        default=(DataProfile.UNIQUE, DataProfile.MIXED, DataProfile.COLLISION_20),
        nargs="+",
        type=DataProfile,
    )
    parser.add_argument(
        "--normalizers",
        choices=tuple(NormalizerProfile),
        default=(NormalizerProfile.DEFAULT,),
        nargs="+",
        type=NormalizerProfile,
    )
    parser.add_argument(
        "--scorers",
        choices=tuple(ScorerProfile),
        default=(ScorerProfile.RATIO, ScorerProfile.WRATIO),
        nargs="+",
        type=ScorerProfile,
    )
    parser.add_argument(
        "--workloads",
        choices=tuple(WorkloadProfile),
        default=(),
        nargs="*",
        type=WorkloadProfile,
    )
    focused_mode = parser.add_mutually_exclusive_group()
    focused_mode.add_argument(
        "--workloads-only",
        action="store_true",
        help="Measure build, steady lookup, and requested workloads without the full micro-operation matrix.",
    )
    focused_mode.add_argument(
        "--exact-tie-only",
        action="store_true",
        help="Measure only exact, normalized, and normalized-collision tie-resolution paths.",
    )
    focused_mode.add_argument(
        "--full-result-only",
        action="store_true",
        help="Measure only find-many(limit=None) to isolate full-result ranking cost.",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=None,
        help="Number of warmup repetitions before each timed measurement.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Skip tracemalloc peak memory measurement.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the fast harness smoke check that normally runs before the benchmark matrix.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/reports/index_strategy"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark matrix."""

    args = parse_args(argv)
    if not args.skip_smoke_test:
        with tempfile.TemporaryDirectory() as smoke_dir:
            run_smoke_check(Path(smoke_dir))
    results: list[BenchmarkResult] = []
    for case in args.cases:
        for profile in args.profiles:
            for normalizer_profile in args.normalizers:
                for scorer_profile in args.scorers:
                    results.extend(
                        run_case(
                            case,
                            profile,
                            normalizer_profile,
                            scorer_profile,
                            items=args.items,
                            repeats=args.repeats,
                            batch_size=args.batch_size,
                            updates=args.updates,
                            workloads=args.workloads,
                            workloads_only=args.workloads_only,
                            exact_tie_only=args.exact_tie_only,
                            full_result_only=args.full_result_only,
                            warmup=args.warmup or 0,
                            no_memory=args.no_memory,
                        )
                    )
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
