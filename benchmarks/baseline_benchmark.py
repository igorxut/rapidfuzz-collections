"""Measure baseline fuzzy lookup performance for the current implementation.

The script intentionally uses only the standard library plus RapidFuzz and the
current package. It is a lightweight baseline harness, not a statistically
complete benchmarking framework. Run it directly:

    python benchmarks/baseline_benchmark.py
"""

import argparse
import importlib.util
import statistics
import sys
from bisect import bisect_left, insort
from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Self

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import WRatio, ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.datasets import (  # noqa: E402
    DataProfile,
    QuerySet,
    build_mapping,
    build_set,
    build_values,
    build_values_with_collision_rate,
    make_typo,
)
from benchmarks.datasets import build_queries as _build_queries  # noqa: E402
from benchmarks.utils import (  # noqa: E402
    measure_peak_kib,
    measure_timings,
    non_negative_float,
    positive_int,
    result_size,
    write_benchmark_reports,
)
from benchmarks.utils import string_values as _string_values_util  # noqa: E402
from rapidfuzz_collections import (  # noqa: E402
    FrozenFuzzySet,
    FuzzyList,
    IndexStrategy,
    Normalizer,
    ScorerType,
)
from rapidfuzz_collections import FuzzyDict as UnifiedFuzzyDict  # noqa: E402
from rapidfuzz_collections import FuzzySet as UnifiedFuzzySet  # noqa: E402
from rapidfuzz_collections.indexes import (  # noqa: E402
    FuzzySequenceIndex,
    MutableFuzzySequenceIndex,
)


# noinspection PyPep8Naming
def FuzzyDict(*args: object, **kwargs: object) -> UnifiedFuzzyDict:
    """Create a sequence-strategy fuzzy dict for benchmark compatibility."""

    return UnifiedFuzzyDict(*args, strategy=IndexStrategy.SEQUENCE, **kwargs)


# noinspection PyPep8Naming
def KeyedFuzzyDict(*args: object, **kwargs: object) -> UnifiedFuzzyDict:
    """Create a keyed-strategy fuzzy dict for benchmark compatibility."""

    return UnifiedFuzzyDict(*args, strategy=IndexStrategy.KEYED, **kwargs)


# noinspection PyPep8Naming
def FuzzySet(*args: object, **kwargs: object) -> UnifiedFuzzySet:
    """Create a sequence-strategy fuzzy set for benchmark compatibility."""

    return UnifiedFuzzySet(*args, strategy=IndexStrategy.SEQUENCE, **kwargs)


# noinspection PyPep8Naming
def KeyedFuzzySet(*args: object, **kwargs: object) -> UnifiedFuzzySet:
    """Create a keyed-strategy fuzzy set for benchmark compatibility."""

    return UnifiedFuzzySet(*args, strategy=IndexStrategy.KEYED, **kwargs)


IndexValue = object
NormalizedValue = str | None
Scorer = Callable[..., int | float]


class ScorerProfile(StrEnum):
    """Supported RapidFuzz scorer profiles."""

    WRATIO = "wratio"
    RATIO = "ratio"
    LEVENSHTEIN_DISTANCE = "levenshtein-distance"
    LEVENSHTEIN_NORMALIZED_SIMILARITY = "levenshtein-normalized-similarity"


class BenchmarkSection(StrEnum):
    """Supported top-level benchmark sections."""

    ADVANCED_TOP_ONE = "advanced-top-one"
    BUILD = "build"
    SEQUENCE = "sequence"
    MAPPING = "mapping"
    SET = "set"
    KEYED_CHOICES = "keyed-choices"
    BATCH_API = "batch-api"
    MUTATION = "mutation"
    INDEX_COMPARISON = "index-comparison"
    DELETION_HEAVY = "deletion-heavy"
    REPLACEMENT_HEAVY = "replacement-heavy"
    INTERLEAVED = "interleaved"
    SCORE_HINT = "score-hint"
    COLLISION_COST = "collision-cost"


def scorer_for(profile: ScorerProfile) -> Scorer:
    """Return the RapidFuzz scorer for a scorer profile."""

    if profile == ScorerProfile.WRATIO:
        return WRatio
    if profile == ScorerProfile.RATIO:
        return ratio
    if profile == ScorerProfile.LEVENSHTEIN_DISTANCE:
        return Levenshtein.distance
    if profile == ScorerProfile.LEVENSHTEIN_NORMALIZED_SIMILARITY:
        return Levenshtein.normalized_similarity
    raise NotImplementedError(profile)


def scorer_type_for(profile: ScorerProfile) -> ScorerType:
    """Return the current wrapper scorer type for a scorer profile."""

    if profile == ScorerProfile.LEVENSHTEIN_DISTANCE:
        return ScorerType.DISTANCE
    return ScorerType.SIMILARITY


def default_score_cutoff_for(profile: ScorerProfile) -> float:
    """Return a useful default cutoff for the selected scorer profile."""

    if profile == ScorerProfile.LEVENSHTEIN_DISTANCE:
        return 5.0
    if profile == ScorerProfile.LEVENSHTEIN_NORMALIZED_SIMILARITY:
        return 0.8
    return 80.0


def score_hint_for(profile: ScorerProfile, *, high_confidence: bool) -> float:
    """Return a representative expected score for score-hint benchmarks."""

    if profile == ScorerProfile.LEVENSHTEIN_DISTANCE:
        return 0.0 if high_confidence else 5.0
    if profile == ScorerProfile.LEVENSHTEIN_NORMALIZED_SIMILARITY:
        return 1.0 if high_confidence else 0.8
    return 100.0 if high_confidence else 80.0


@dataclass(frozen=True)
class BenchmarkResult:
    """Single benchmark result."""

    name: str
    group: str
    profile: str
    items: int
    repeats: int
    best_seconds: float
    median_seconds: float
    peak_kib: float
    result_size: int | None = None
    scorer: ScorerProfile = ScorerProfile.WRATIO


@dataclass(frozen=True)
class ExplicitFuzzyIndex:
    """Experimental explicit index prototype for architecture comparison."""

    values: tuple[IndexValue, ...]
    normalized_values: tuple[NormalizedValue, ...]
    exact_first_index: dict[Hashable, int]
    normalized_first_index: dict[str, int]
    normalizer: Normalizer
    scorer: Scorer
    score_cutoff: int | float

    @classmethod
    def from_values(
        cls,
        values: list[IndexValue],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> Self:
        """Build an index with first-match lookup tables."""

        indexed_values = tuple(values)
        normalized_values = tuple(normalizer(value) for value in indexed_values)
        exact_first_index: dict[Hashable, int] = {}
        normalized_first_index: dict[str, int] = {}

        for index, value in enumerate(indexed_values):
            if isinstance(value, Hashable):
                exact_first_index.setdefault(value, index)
            normalized_value = normalized_values[index]
            if normalized_value is not None:
                normalized_first_index.setdefault(normalized_value, index)

        return cls(
            values=indexed_values,
            normalized_values=normalized_values,
            exact_first_index=exact_first_index,
            normalized_first_index=normalized_first_index,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def find_one(self, query: object) -> object | None:
        """Return the first exact, normalized exact, or fuzzy match."""

        if isinstance(query, Hashable):
            exact_index = self.exact_first_index.get(query)
            if exact_index is not None:
                return self.values[exact_index]

        normalized_query = self.normalizer(query)
        if normalized_query is not None:
            normalized_index = self.normalized_first_index.get(normalized_query)
            if normalized_index is not None:
                return self.values[normalized_index]

        result = process.extractOne(
            normalized_query,
            self.normalized_values,
            scorer=self.scorer,
            score_cutoff=self.score_cutoff,
        )
        if result is None:
            return None

        _, _, index = result
        return self.values[index]

    def contains(self, query: object) -> bool:
        """Return whether the query has an exact, normalized exact, or fuzzy match."""

        return self.find_one(query) is not None

    def scores(self, query: object) -> list[tuple[object, float | None, int]]:
        """Return scores in a shape comparable to FuzzyList output."""

        normalized_query = self.normalizer(query)
        result: list[tuple[object, float | None, int]] = []
        matched_indexes: set[int] = set()

        for _, score, index in process.extract(
            normalized_query,
            self.normalized_values,
            scorer=self.scorer,
            score_cutoff=self.score_cutoff,
            limit=None,
        ):
            result.append((self.values[index], score, index))
            matched_indexes.add(index)

        for index, value in enumerate(self.values):
            if index not in matched_indexes:
                result.append((value, None, index))

        return result

    def __len__(self) -> int:
        return len(self.values)


class MutableExplicitFuzzyIndex:
    """Experimental mutable explicit index prototype for architecture comparison."""

    def __init__(
        self,
        values: list[IndexValue],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> None:
        self.values: list[IndexValue] = []
        self.normalized_values: list[NormalizedValue] = []
        self.exact_first_index: dict[Hashable, int] = {}
        self.normalized_first_index: dict[str, int] = {}
        self.normalizer = normalizer
        self.scorer = scorer
        self.score_cutoff = score_cutoff

        self.extend(values)

    @classmethod
    def from_values(
        cls,
        values: list[IndexValue],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> Self:
        """Build a mutable index with first-match lookup tables."""

        return cls(values=values, normalizer=normalizer, scorer=scorer, score_cutoff=score_cutoff)

    def append(self, value: IndexValue) -> None:
        """Append a value and update first-match lookup tables."""

        index = len(self.values)
        normalized_value = self.normalizer(value)

        self.values.append(value)
        self.normalized_values.append(normalized_value)

        if isinstance(value, Hashable):
            self.exact_first_index.setdefault(value, index)

        if normalized_value is not None:
            self.normalized_first_index.setdefault(normalized_value, index)

    def extend(self, values: list[IndexValue]) -> None:
        """Append multiple values."""

        for value in values:
            self.append(value)

    def find_one(self, query: object) -> object | None:
        """Return the first exact, normalized exact, or fuzzy match."""

        if isinstance(query, Hashable):
            exact_index = self.exact_first_index.get(query)
            if exact_index is not None:
                return self.values[exact_index]

        normalized_query = self.normalizer(query)
        if normalized_query is not None:
            normalized_index = self.normalized_first_index.get(normalized_query)
            if normalized_index is not None:
                return self.values[normalized_index]

        result = process.extractOne(
            normalized_query,
            self.normalized_values,
            scorer=self.scorer,
            score_cutoff=self.score_cutoff,
        )
        if result is None:
            return None

        _, _, index = result
        return self.values[index]

    def contains(self, query: object) -> bool:
        """Return whether the query has an exact, normalized exact, or fuzzy match."""

        return self.find_one(query) is not None

    def scores(self, query: object) -> list[tuple[object, float | None, int]]:
        """Return scores in a shape comparable to FuzzyList output."""

        normalized_query = self.normalizer(query)
        result: list[tuple[object, float | None, int]] = []
        matched_indexes: set[int] = set()

        for _, score, index in process.extract(
            normalized_query,
            self.normalized_values,
            scorer=self.scorer,
            score_cutoff=self.score_cutoff,
            limit=None,
        ):
            result.append((self.values[index], score, index))
            matched_indexes.add(index)

        for index, value in enumerate(self.values):
            if index not in matched_indexes:
                result.append((value, None, index))

        return result

    def __len__(self) -> int:
        return len(self.values)


class TombstoneFuzzyIndex:
    """Benchmark-local soft-delete prototype for top-one lookup comparisons.

    The prototype isolates a possible deletion strategy: active searchable
    choices live in a mapping keyed by stable source positions, so deleting a
    selected value removes one mapping entry without rebuilding all normalized
    choices. A positional slot list models the translation required for
    sequence ``Match.index`` results after deletions. Exact and normalized-exact
    shortcut tables retain stable slots and skip removed entries. The prototype
    is not a complete replacement for multi-result or all-score production
    index contracts.
    """

    def __init__(
        self,
        values: list[IndexValue],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> None:
        self._active_slots = set(range(len(values)))
        self._active_choices: dict[int, str] = {}
        self._exact_duplicate_slots: dict[Hashable, list[int]] = {}
        self._exact_first_slot: dict[Hashable, int] = {}
        self._normalizer = normalizer
        self._normalized_duplicate_slots: dict[str, list[int]] = {}
        self._normalized_first_slot: dict[str, int] = {}
        self._position_slots = list(range(len(values)))
        self._scorer = scorer
        self._score_cutoff = score_cutoff
        self._values = dict(enumerate(values))

        for index, value in enumerate(values):
            if isinstance(value, Hashable):
                first_slot = self._exact_first_slot.setdefault(value, index)
                if first_slot != index:
                    self._exact_duplicate_slots.setdefault(value, []).append(index)

            normalized_value = normalizer(value)
            if normalized_value is not None:
                first_slot = self._normalized_first_slot.setdefault(normalized_value, index)
                if first_slot != index:
                    self._normalized_duplicate_slots.setdefault(normalized_value, []).append(index)
                self._active_choices[index] = normalized_value

    def _first_active_slot(
        self,
        first_slots: dict[Hashable, int] | dict[str, int],
        duplicate_slots: dict[Hashable, list[int]] | dict[str, list[int]],
        key: Hashable | str,
    ) -> int | None:
        """Return the first non-deleted shortcut slot for a key."""

        first_slot = first_slots.get(key)
        if first_slot is not None and first_slot in self._active_slots:
            return first_slot

        return next((slot for slot in duplicate_slots.get(key, ()) if slot in self._active_slots), None)

    def _find_slot(self, query: object) -> int | None:
        """Return a stable active slot for a top-one query."""

        if isinstance(query, Hashable):
            exact_slot = self._first_active_slot(self._exact_first_slot, self._exact_duplicate_slots, query)
            if exact_slot is not None:
                return exact_slot

        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None

        normalized_slot = self._first_active_slot(
            self._normalized_first_slot,
            self._normalized_duplicate_slots,
            normalized_query,
        )
        if normalized_slot is not None:
            return normalized_slot

        match = process.extractOne(
            normalized_query,
            self._active_choices,
            scorer=self._scorer,
            score_cutoff=self._score_cutoff,
        )
        if match is None:
            return None

        _, _, slot = match
        return slot

    def delete_at_positions(self, positions: set[int]) -> None:
        """Delete active positions without rebuilding remaining choices."""

        slots = [self._position_slots[position] for position in positions]
        for position in sorted(positions, reverse=True):
            del self._position_slots[position]
        for slot in slots:
            self._active_slots.remove(slot)
            self._active_choices.pop(slot, None)
            self._values.pop(slot, None)

    def discard_one(self, query: object) -> object | None:
        """Find and delete one active top-one value."""

        slot = self._find_slot(query)
        if slot is None:
            return None

        position = self._position_slots.index(slot)
        del self._position_slots[position]
        self._active_slots.remove(slot)
        self._active_choices.pop(slot, None)
        return self._values.pop(slot)

    def find_one(self, query: object) -> object | None:
        """Return one top-one match among active values."""

        slot = self._find_slot(query)
        if slot is None:
            return None

        self._position_slots.index(slot)
        return self._values[slot]


class CompactDeleteFuzzyIndex:
    """Benchmark-local compact-deletion prototype without retained slot tables.

    Deletion updates normalized choices in place. Sparse choices retain stable
    source slots and track only removed slots, testing whether translating
    result positions is cheaper than a full rebuild without an O(n) slot map.
    Exact and normalized-exact tables remain membership filters after deletion;
    source positions are resolved from current state only for shortcut hits.
    """

    def __init__(
        self,
        values: list[IndexValue],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> None:
        self._deleted_source_slots: list[int] = []
        self._dirty = False
        self._exact_first_index: dict[Hashable, int] = {}
        self._normalizer = normalizer
        self._normalized_choices: list[str] = []
        self._normalized_first_index: dict[str, int] = {}
        self._normalized_values: list[NormalizedValue] = []
        self._scorer = scorer
        self._score_cutoff = score_cutoff
        self._shortcuts_valid = True
        self._source_indexes: list[int] | None = None
        self._values = list(values)

        for index, value in enumerate(values):
            if isinstance(value, Hashable):
                self._exact_first_index.setdefault(value, index)

            normalized_value = normalizer(value)
            self._normalized_values.append(normalized_value)
            if normalized_value is None:
                continue

            self._normalized_first_index.setdefault(normalized_value, index)
            if self._source_indexes is not None:
                self._source_indexes.append(index)
            elif index != len(self._normalized_choices):
                self._source_indexes = [*range(len(self._normalized_choices)), index]
            self._normalized_choices.append(normalized_value)

    def _delete_position(self, position: int) -> object:
        """Delete one current source position using the applicable strategy."""

        value = self._values.pop(position)
        normalized_value = self._normalized_values.pop(position)
        if self._dirty:
            self._dirty = True
            return value

        if self._source_indexes is None:
            if normalized_value is None:
                self._dirty = True
                return value
            del self._normalized_choices[position]
        else:
            source_slot = self._stable_slot_from_source_position(position)
            if normalized_value is not None:
                choice_position = self._source_indexes.index(source_slot)
                del self._normalized_choices[choice_position]
                del self._source_indexes[choice_position]
            insort(self._deleted_source_slots, source_slot)
        self._shortcuts_valid = False
        return value

    def _find_position(self, query: object) -> int | None:
        """Return a current source position for a top-one query."""

        if self._dirty:
            self._rebuild()

        if isinstance(query, Hashable):
            if self._shortcuts_valid:
                exact_index = self._exact_first_index.get(query)
            elif query not in self._exact_first_index:
                exact_index = None
            else:
                try:
                    exact_index = self._values.index(query)
                except ValueError:
                    exact_index = None
            if exact_index is not None:
                return exact_index

        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None

        if self._shortcuts_valid:
            normalized_index = self._normalized_first_index.get(normalized_query)
        elif normalized_query not in self._normalized_first_index:
            normalized_index = None
        else:
            try:
                normalized_index = self._normalized_values.index(normalized_query)
            except ValueError:
                normalized_index = None
        if normalized_index is not None:
            return normalized_index

        match = process.extractOne(
            normalized_query,
            self._normalized_choices,
            scorer=self._scorer,
            score_cutoff=self._score_cutoff,
        )
        if match is None:
            return None

        _, _, choice_index = match
        if self._source_indexes is None:
            return choice_index
        return self._source_position_from_stable_slot(self._source_indexes[choice_index])

    def _rebuild(self) -> None:
        """Rebuild sparse derived state after a deletion."""

        self._exact_first_index = {}
        self._normalized_choices = []
        self._normalized_first_index = {}
        source_indexes: list[int] | None = None
        for index, (value, normalized_value) in enumerate(
            zip(self._values, self._normalized_values, strict=True),
        ):
            if isinstance(value, Hashable):
                self._exact_first_index.setdefault(value, index)
            if normalized_value is None:
                continue
            self._normalized_first_index.setdefault(normalized_value, index)
            if source_indexes is not None:
                source_indexes.append(index)
            elif index != len(self._normalized_choices):
                source_indexes = [*range(len(self._normalized_choices)), index]
            self._normalized_choices.append(normalized_value)
        self._deleted_source_slots = []
        self._dirty = False
        self._shortcuts_valid = True
        self._source_indexes = source_indexes

    def _source_position_from_stable_slot(self, source_slot: int) -> int:
        """Translate a retained sparse source slot to its current position."""

        return source_slot - bisect_left(self._deleted_source_slots, source_slot)

    def _stable_slot_from_source_position(self, position: int) -> int:
        """Translate a current sparse source position to a retained slot."""

        source_slot = position
        for deleted_slot in self._deleted_source_slots:
            if deleted_slot > source_slot:
                break
            source_slot += 1
        return source_slot

    def delete_at_positions(self, positions: set[int]) -> None:
        """Delete current positions while retaining fuzzy choice state."""

        for position in sorted(positions, reverse=True):
            self._delete_position(position)

    def discard_one(self, query: object) -> object | None:
        """Find and delete one current top-one value."""

        position = self._find_position(query)
        if position is None:
            return None
        return self._delete_position(position)

    def find_one(self, query: object) -> object | None:
        """Return one top-one match among current values."""

        position = self._find_position(query)
        if position is None:
            return None
        return self._values[position]


@dataclass(frozen=True)
class ExplicitFuzzyMappingIndex:
    """Experimental read-only mapping index prototype for fuzzy key lookup."""

    keys: tuple[IndexValue, ...]
    values: tuple[str, ...]
    normalized_keys: tuple[NormalizedValue, ...]
    exact_first_index: dict[Hashable, int]
    normalized_first_index: dict[str, int]
    normalizer: Normalizer
    scorer: Scorer
    score_cutoff: int | float

    @classmethod
    def from_mapping(
        cls,
        data: dict[IndexValue, str],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> Self:
        """Build a mapping index with first-match lookup tables."""

        keys = tuple(data.keys())
        values = tuple(data.values())
        normalized_keys = tuple(normalizer(key) for key in keys)
        exact_first_index: dict[Hashable, int] = {}
        normalized_first_index: dict[str, int] = {}

        for index, key in enumerate(keys):
            if isinstance(key, Hashable):
                exact_first_index.setdefault(key, index)
            normalized_key = normalized_keys[index]
            if normalized_key is not None:
                normalized_first_index.setdefault(normalized_key, index)

        return cls(
            keys=keys,
            values=values,
            normalized_keys=normalized_keys,
            exact_first_index=exact_first_index,
            normalized_first_index=normalized_first_index,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def find_one(self, query: object) -> tuple[object, str] | None:
        """Return the first exact, normalized exact, or fuzzy key/value match."""

        if isinstance(query, Hashable):
            exact_index = self.exact_first_index.get(query)
            if exact_index is not None:
                return self.keys[exact_index], self.values[exact_index]

        normalized_query = self.normalizer(query)
        if normalized_query is not None:
            normalized_index = self.normalized_first_index.get(normalized_query)
            if normalized_index is not None:
                return self.keys[normalized_index], self.values[normalized_index]

        result = process.extractOne(
            normalized_query,
            self.normalized_keys,
            scorer=self.scorer,
            score_cutoff=self.score_cutoff,
        )
        if result is None:
            return None

        _, _, index = result
        return self.keys[index], self.values[index]

    def contains(self, query: object) -> bool:
        """Return whether the query has an exact, normalized exact, or fuzzy key match."""

        return self.find_one(query) is not None

    def __len__(self) -> int:
        return len(self.keys)


class MutableExplicitFuzzyMappingIndex:
    """Experimental mutable mapping index prototype for fuzzy key lookup."""

    def __init__(
        self,
        data: dict[IndexValue, str],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> None:
        self.keys: list[IndexValue] = []
        self.values: list[str] = []
        self.normalized_keys: list[NormalizedValue] = []
        self.key_to_position: dict[Hashable, int] = {}
        self.exact_first_index: dict[Hashable, int] = {}
        self.normalized_first_index: dict[str, int] = {}
        self.normalizer = normalizer
        self.scorer = scorer
        self.score_cutoff = score_cutoff

        self.update(data)

    @classmethod
    def from_mapping(
        cls,
        data: dict[IndexValue, str],
        normalizer: Normalizer,
        scorer: Scorer,
        score_cutoff: int | float,
    ) -> Self:
        """Build a mutable mapping index."""

        return cls(data=data, normalizer=normalizer, scorer=scorer, score_cutoff=score_cutoff)

    def set(self, key: IndexValue, value: str) -> None:
        """Set a key/value pair and update lookup tables for new keys."""

        if isinstance(key, Hashable) and key in self.key_to_position:
            index = self.key_to_position[key]
            self.values[index] = value
            return

        index = len(self.keys)
        normalized_key = self.normalizer(key)

        self.keys.append(key)
        self.values.append(value)
        self.normalized_keys.append(normalized_key)

        if isinstance(key, Hashable):
            self.key_to_position[key] = index
            self.exact_first_index.setdefault(key, index)

        if normalized_key is not None:
            self.normalized_first_index.setdefault(normalized_key, index)

    def update(self, data: dict[IndexValue, str]) -> None:
        """Set multiple key/value pairs."""

        for key, value in data.items():
            self.set(key, value)

    def find_one(self, query: object) -> tuple[object, str] | None:
        """Return the first exact, normalized exact, or fuzzy key/value match."""

        if isinstance(query, Hashable):
            exact_index = self.exact_first_index.get(query)
            if exact_index is not None:
                return self.keys[exact_index], self.values[exact_index]

        normalized_query = self.normalizer(query)
        if normalized_query is not None:
            normalized_index = self.normalized_first_index.get(normalized_query)
            if normalized_index is not None:
                return self.keys[normalized_index], self.values[normalized_index]

        result = process.extractOne(
            normalized_query,
            self.normalized_keys,
            scorer=self.scorer,
            score_cutoff=self.score_cutoff,
        )
        if result is None:
            return None

        _, _, index = result
        return self.keys[index], self.values[index]

    def contains(self, query: object) -> bool:
        """Return whether the query has an exact, normalized exact, or fuzzy key match."""

        return self.find_one(query) is not None

    def __len__(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class BaselineData:
    """Prepared values used by query benchmarks."""

    values: list[IndexValue]
    normalized_values: list[str | None]
    collection: FuzzyList
    explicit_index: ExplicitFuzzyIndex
    mutable_explicit_index: MutableExplicitFuzzyIndex
    normalizer: Normalizer


@dataclass(frozen=True)
class MappingBaselineData:
    """Prepared mapping values used by key lookup benchmarks."""

    data: dict[IndexValue, str]
    normalized_keys: list[NormalizedValue]
    collection: UnifiedFuzzyDict
    explicit_index: ExplicitFuzzyMappingIndex
    mutable_explicit_index: MutableExplicitFuzzyMappingIndex
    normalizer: Normalizer


@dataclass(frozen=True)
class SetBaselineData:
    """Prepared set values used by membership benchmarks."""

    data: set[IndexValue]
    values: list[IndexValue]
    normalized_values: list[NormalizedValue]
    collection: UnifiedFuzzySet
    frozen_collection: FrozenFuzzySet
    explicit_index: ExplicitFuzzyIndex
    mutable_explicit_index: MutableExplicitFuzzyIndex
    normalizer: Normalizer


def string_values(values: list[IndexValue]) -> list[str]:
    """Return string values that survive the default normalizer."""

    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    return _string_values_util(values, normalizer=normalizer)


def string_keys(data: dict[IndexValue, str]) -> list[str]:
    """Return string keys that survive the default normalizer."""

    return string_values(list(data.keys()))


def build_queries(values: list[IndexValue], batch_size: int) -> QuerySet:
    """Build deterministic queries for exact, fuzzy, miss, and batch scenarios."""

    return _build_queries(values, batch_size)


def measure(
    group: str,
    name: str,
    profile: str,
    items: int,
    repeats: int,
    func: Callable[[], object],
) -> BenchmarkResult:
    """Measure elapsed time and peak traced memory for a callable."""

    best_ms, median_ms = measure_timings(repeats, func)
    peak_kib = measure_peak_kib(func)
    size = result_size(func())
    return BenchmarkResult(
        name=name,
        group=group,
        profile=profile,
        items=items,
        repeats=repeats,
        best_seconds=best_ms / 1000.0,
        median_seconds=median_ms / 1000.0,
        peak_kib=peak_kib,
        result_size=size,
    )


def measure_after_setup(
    group: str,
    name: str,
    profile: str,
    items: int,
    repeats: int,
    setup: Callable[[], object],
    func: Callable[..., object],
) -> BenchmarkResult:
    """Measure a callable while excluding setup cost from timing and memory."""

    timings_ms: list[float] = []
    for _ in range(repeats):
        target = setup()
        best_ms_once, _ = measure_timings(1, lambda t=target: func(t))
        timings_ms.append(best_ms_once)

    target_for_mem = setup()
    peak_kib = measure_peak_kib(lambda t=target_for_mem: func(t))
    size = result_size(func(setup()))

    return BenchmarkResult(
        name=name,
        group=group,
        profile=profile,
        items=items,
        repeats=repeats,
        best_seconds=min(timings_ms) / 1000.0,
        median_seconds=statistics.median(timings_ms) / 1000.0,
        peak_kib=peak_kib,
        result_size=size,
    )


def build_baseline(
    values: list[IndexValue],
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
) -> BaselineData:
    """Build current implementation baseline structures."""

    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    normalized_values = [normalizer(value) for value in values]
    collection = FuzzyList(
        values,
        normalizer=normalizer,
        scorer=scorer,
        scorer_type=scorer_type,
        score_cutoff=score_cutoff,
    )
    explicit_index = ExplicitFuzzyIndex.from_values(
        values,
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=score_cutoff,
    )
    mutable_explicit_index = MutableExplicitFuzzyIndex.from_values(
        values,
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=score_cutoff,
    )

    return BaselineData(
        values=values,
        normalized_values=normalized_values,
        collection=collection,
        explicit_index=explicit_index,
        mutable_explicit_index=mutable_explicit_index,
        normalizer=normalizer,
    )


def build_mapping_baseline(
    data: dict[IndexValue, str],
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
) -> MappingBaselineData:
    """Build current and prototype mapping baseline structures."""

    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    normalized_keys = [normalizer(key) for key in data]
    collection = FuzzyDict(
        data,
        normalizer=normalizer,
        scorer=scorer,
        scorer_type=scorer_type,
        score_cutoff=score_cutoff,
    )
    explicit_index = ExplicitFuzzyMappingIndex.from_mapping(
        data,
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=score_cutoff,
    )
    mutable_explicit_index = MutableExplicitFuzzyMappingIndex.from_mapping(
        data,
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=score_cutoff,
    )

    return MappingBaselineData(
        data=data,
        normalized_keys=normalized_keys,
        collection=collection,
        explicit_index=explicit_index,
        mutable_explicit_index=mutable_explicit_index,
        normalizer=normalizer,
    )


def build_set_baseline(
    data: set[IndexValue],
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
) -> SetBaselineData:
    """Build current and prototype set baseline structures."""

    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    values = list(data)
    normalized_values = [normalizer(value) for value in values]
    collection = FuzzySet(
        data,
        normalizer=normalizer,
        scorer=scorer,
        scorer_type=scorer_type,
        score_cutoff=score_cutoff,
    )
    frozen_collection = FrozenFuzzySet(
        data,
        normalizer=normalizer,
        scorer=scorer,
        scorer_type=scorer_type,
        score_cutoff=score_cutoff,
    )
    explicit_index = ExplicitFuzzyIndex.from_values(
        values,
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=score_cutoff,
    )
    mutable_explicit_index = MutableExplicitFuzzyIndex.from_values(
        values,
        normalizer=normalizer,
        scorer=scorer,
        score_cutoff=score_cutoff,
    )

    return SetBaselineData(
        data=data,
        values=values,
        normalized_values=normalized_values,
        collection=collection,
        frozen_collection=frozen_collection,
        explicit_index=explicit_index,
        mutable_explicit_index=mutable_explicit_index,
        normalizer=normalizer,
    )


def direct_get(
    query: object,
    values: list[IndexValue],
    normalized_values: list[str | None],
    normalizer: Normalizer,
    scorer: Scorer,
    score_cutoff: int | float,
) -> object | None:
    """Approximate current wrapper lookup using direct Python and RapidFuzz calls."""

    if query in values:
        return query

    normalized_query = normalizer(query)
    if normalized_query is not None and normalized_query in normalized_values:
        return values[normalized_values.index(normalized_query)]

    result = process.extractOne(normalized_query, normalized_values, scorer=scorer, score_cutoff=score_cutoff)
    if result is None:
        return None

    _, _, index = result
    return values[index]


def direct_mapping_get(
    query: object,
    data: dict[IndexValue, str],
    normalized_keys: list[NormalizedValue],
    normalizer: Normalizer,
    scorer: Scorer,
    score_cutoff: int | float,
) -> tuple[object, str] | None:
    """Approximate current dict wrapper lookup using direct Python and RapidFuzz calls."""

    if isinstance(query, Hashable) and query in data:
        return query, data[query]

    keys = list(data.keys())
    normalized_query = normalizer(query)
    if normalized_query is not None and normalized_query in normalized_keys:
        key = keys[normalized_keys.index(normalized_query)]
        return key, data[key]

    result = process.extractOne(normalized_query, normalized_keys, scorer=scorer, score_cutoff=score_cutoff)
    if result is None:
        return None

    _, _, index = result
    key = keys[index]
    return key, data[key]


def run_build_benchmarks(
    items: int,
    repeats: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure construction and normalization costs."""

    values = build_values(items, profile=profile)
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)

    return [
        measure(
            "build", "generate source values", profile, items, repeats, lambda: build_values(items, profile=profile)
        ),
        measure(
            "build",
            "normalize source values",
            profile,
            items,
            repeats,
            lambda: [normalizer(value) for value in values],
        ),
        measure(
            "build",
            "build FuzzyList",
            profile,
            items,
            repeats,
            lambda: FuzzyList(
                values,
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build ExplicitFuzzyIndex prototype",
            profile,
            items,
            repeats,
            lambda: ExplicitFuzzyIndex.from_values(
                values,
                normalizer=normalizer,
                scorer=scorer,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build MutableExplicitFuzzyIndex prototype",
            profile,
            items,
            repeats,
            lambda: MutableExplicitFuzzyIndex.from_values(
                values,
                normalizer=normalizer,
                scorer=scorer,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build FuzzyDict",
            profile,
            items,
            repeats,
            lambda: FuzzyDict(
                build_mapping(items, profile=profile),
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build ExplicitFuzzyMappingIndex prototype",
            profile,
            items,
            repeats,
            lambda: ExplicitFuzzyMappingIndex.from_mapping(
                build_mapping(items, profile=profile),
                normalizer=normalizer,
                scorer=scorer,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build MutableExplicitFuzzyMappingIndex prototype",
            profile,
            items,
            repeats,
            lambda: MutableExplicitFuzzyMappingIndex.from_mapping(
                build_mapping(items, profile=profile),
                normalizer=normalizer,
                scorer=scorer,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build FuzzySet",
            profile,
            items,
            repeats,
            lambda: FuzzySet(
                build_set(items, profile=profile),
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            ),
        ),
        measure(
            "build",
            "build FrozenFuzzySet",
            profile,
            items,
            repeats,
            lambda: FrozenFuzzySet(
                build_set(items, profile=profile),
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            ),
        ),
    ]


def run_query_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Run single-query and batch-query measurements."""

    values = build_values(items, profile=profile)
    baseline = build_baseline(values, scorer=scorer, scorer_type=scorer_type, score_cutoff=score_cutoff)
    queries = build_queries(values, batch_size=batch_size)

    def direct_extract_one(query: str) -> object:
        normalized_query = baseline.normalizer(query)
        return process.extractOne(
            normalized_query,
            baseline.normalized_values,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def direct_extract_all(query: str) -> object:
        normalized_query = baseline.normalizer(query)
        return process.extract(
            normalized_query,
            baseline.normalized_values,
            scorer=scorer,
            score_cutoff=score_cutoff,
            limit=None,
        )

    results: list[BenchmarkResult] = []

    for label, q in (
        ("exact", queries.exact),
        ("normalized exact", queries.normalized_exact),
        ("close", queries.close),
        ("miss", queries.miss),
    ):
        results.extend(
            [
                measure(
                    "single",
                    f"direct extractOne: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_extract_one(query),
                ),
                measure(
                    "single",
                    f"direct get with shortcuts: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_get(
                        query,
                        baseline.values,
                        baseline.normalized_values,
                        baseline.normalizer,
                        scorer,
                        score_cutoff,
                    ),
                ),
                measure(
                    "single",
                    f"FuzzyList.fuzzy_get: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_get(query),
                ),
                measure(
                    "single",
                    f"ExplicitFuzzyIndex.find_one: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.find_one(query),
                ),
                measure(
                    "single",
                    f"MutableExplicitFuzzyIndex.find_one: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.find_one(query),
                ),
                measure(
                    "single",
                    f"FuzzyList.fuzzy_contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_contains(query),
                ),
                measure(
                    "single",
                    f"ExplicitFuzzyIndex.contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.contains(query),
                ),
                measure(
                    "single",
                    f"MutableExplicitFuzzyIndex.contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.contains(query),
                ),
                measure(
                    "single",
                    f"FuzzyList.fuzzy_find_many: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_find_many(query, limit=None),
                ),
                measure(
                    "single",
                    f"ExplicitFuzzyIndex.scores: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.scores(query),
                ),
                measure(
                    "single",
                    f"MutableExplicitFuzzyIndex.scores: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.scores(query),
                ),
                measure(
                    "single",
                    f"direct extract all: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_extract_all(query),
                ),
            ]
        )

    results.extend(
        [
            measure(
                "batch",
                "direct extractOne loop",
                profile,
                items,
                repeats,
                lambda: [direct_extract_one(bq) for bq in queries.batch],
            ),
            measure(
                "batch",
                "direct get with shortcuts loop",
                profile,
                items,
                repeats,
                lambda: [
                    direct_get(
                        bq,
                        baseline.values,
                        baseline.normalized_values,
                        baseline.normalizer,
                        scorer,
                        score_cutoff,
                    )
                    for bq in queries.batch
                ],
            ),
            measure(
                "batch",
                "FuzzyList.fuzzy_get loop",
                profile,
                items,
                repeats,
                lambda: [baseline.collection.fuzzy_get(bq) for bq in queries.batch],
            ),
            measure(
                "batch",
                "FuzzyList.fuzzy_get_batch",
                profile,
                items,
                repeats,
                lambda: baseline.collection.fuzzy_get_batch(queries.batch),
            ),
            measure(
                "batch",
                "FuzzyList.fuzzy_find_one_batch",
                profile,
                items,
                repeats,
                lambda: baseline.collection.fuzzy_find_one_batch(queries.batch),
            ),
            measure(
                "batch",
                "ExplicitFuzzyIndex.find_one loop",
                profile,
                items,
                repeats,
                lambda: [baseline.explicit_index.find_one(bq) for bq in queries.batch],
            ),
            measure(
                "batch",
                "MutableExplicitFuzzyIndex.find_one loop",
                profile,
                items,
                repeats,
                lambda: [baseline.mutable_explicit_index.find_one(bq) for bq in queries.batch],
            ),
        ]
    )

    return results


def run_mapping_query_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Run mapping key lookup measurements."""

    data = build_mapping(items, profile=profile)
    baseline = build_mapping_baseline(data, scorer=scorer, scorer_type=scorer_type, score_cutoff=score_cutoff)
    queries = build_queries(list(data.keys()), batch_size=batch_size)

    def direct_extract_one(query: object) -> object:
        normalized_query = baseline.normalizer(query)
        return process.extractOne(
            normalized_query,
            baseline.normalized_keys,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    results: list[BenchmarkResult] = []

    for label, q in (
        ("exact", queries.exact),
        ("normalized exact", queries.normalized_exact),
        ("close", queries.close),
        ("miss", queries.miss),
    ):
        results.extend(
            [
                measure(
                    "mapping",
                    f"direct dict extractOne: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_extract_one(query),
                ),
                measure(
                    "mapping",
                    f"direct dict get with shortcuts: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_mapping_get(
                        query,
                        baseline.data,
                        baseline.normalized_keys,
                        baseline.normalizer,
                        scorer,
                        score_cutoff,
                    ),
                ),
                measure(
                    "mapping",
                    f"FuzzyDict.fuzzy_get: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_get(query),
                ),
                measure(
                    "mapping",
                    f"ExplicitFuzzyMappingIndex.find_one: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.find_one(query),
                ),
                measure(
                    "mapping",
                    f"MutableExplicitFuzzyMappingIndex.find_one: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.find_one(query),
                ),
                measure(
                    "mapping",
                    f"FuzzyDict.fuzzy_contains_key: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_contains_key(query),
                ),
                measure(
                    "mapping",
                    f"ExplicitFuzzyMappingIndex.contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.contains(query),
                ),
                measure(
                    "mapping",
                    f"MutableExplicitFuzzyMappingIndex.contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.contains(query),
                ),
            ]
        )

    results.extend(
        [
            measure(
                "mapping-batch",
                "FuzzyDict.fuzzy_get loop",
                profile,
                items,
                repeats,
                lambda: [baseline.collection.fuzzy_get(bq) for bq in queries.batch],
            ),
            measure(
                "mapping-batch",
                "FuzzyDict.fuzzy_get_batch",
                profile,
                items,
                repeats,
                lambda: baseline.collection.fuzzy_get_batch(queries.batch),
            ),
            measure(
                "mapping-batch",
                "FuzzyDict.fuzzy_find_item_batch",
                profile,
                items,
                repeats,
                lambda: baseline.collection.fuzzy_find_item_batch(queries.batch),
            ),
            measure(
                "mapping-batch",
                "ExplicitFuzzyMappingIndex.find_one loop",
                profile,
                items,
                repeats,
                lambda: [baseline.explicit_index.find_one(bq) for bq in queries.batch],
            ),
            measure(
                "mapping-batch",
                "MutableExplicitFuzzyMappingIndex.find_one loop",
                profile,
                items,
                repeats,
                lambda: [baseline.mutable_explicit_index.find_one(bq) for bq in queries.batch],
            ),
        ]
    )

    return results


def run_set_query_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Run set membership measurements."""

    values = build_values(items, profile=profile)
    data = set(values)
    baseline = build_set_baseline(data, scorer=scorer, scorer_type=scorer_type, score_cutoff=score_cutoff)
    queries = build_queries(values, batch_size=batch_size)

    def direct_extract_one(query: object) -> object:
        normalized_query = baseline.normalizer(query)
        return process.extractOne(
            normalized_query,
            baseline.normalized_values,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    results: list[BenchmarkResult] = []

    for label, q in (
        ("exact", queries.exact),
        ("normalized exact", queries.normalized_exact),
        ("close", queries.close),
        ("miss", queries.miss),
    ):
        results.extend(
            [
                measure(
                    "set",
                    f"direct set extractOne: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_extract_one(query),
                ),
                measure(
                    "set",
                    f"direct set get with shortcuts: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: direct_get(
                        query,
                        baseline.values,
                        baseline.normalized_values,
                        baseline.normalizer,
                        scorer,
                        score_cutoff,
                    ),
                ),
                measure(
                    "set",
                    f"FuzzySet.fuzzy_get: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_get(query),
                ),
                measure(
                    "set",
                    f"FrozenFuzzySet.fuzzy_get: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.frozen_collection.fuzzy_get(query),
                ),
                measure(
                    "set",
                    f"ExplicitFuzzyIndex.find_one from set: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.find_one(query),
                ),
                measure(
                    "set",
                    f"MutableExplicitFuzzyIndex.find_one from set: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.find_one(query),
                ),
                measure(
                    "set",
                    f"FuzzySet.fuzzy_contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.collection.fuzzy_contains(query),
                ),
                measure(
                    "set",
                    f"FrozenFuzzySet.fuzzy_contains: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.frozen_collection.fuzzy_contains(query),
                ),
                measure(
                    "set",
                    f"ExplicitFuzzyIndex.contains from set: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.explicit_index.contains(query),
                ),
                measure(
                    "set",
                    f"MutableExplicitFuzzyIndex.contains from set: {label}",
                    profile,
                    items,
                    repeats,
                    lambda query=q: baseline.mutable_explicit_index.contains(query),
                ),
            ]
        )

    results.extend(
        [
            measure(
                "set-batch",
                "FuzzySet.fuzzy_get loop",
                profile,
                items,
                repeats,
                lambda: [baseline.collection.fuzzy_get(bq) for bq in queries.batch],
            ),
            measure(
                "set-batch",
                "FuzzySet.fuzzy_get_batch",
                profile,
                items,
                repeats,
                lambda: baseline.collection.fuzzy_get_batch(queries.batch),
            ),
            measure(
                "set-batch",
                "FuzzySet.fuzzy_find_one_batch",
                profile,
                items,
                repeats,
                lambda: baseline.collection.fuzzy_find_one_batch(queries.batch),
            ),
            measure(
                "set-batch",
                "FrozenFuzzySet.fuzzy_get loop",
                profile,
                items,
                repeats,
                lambda: [baseline.frozen_collection.fuzzy_get(bq) for bq in queries.batch],
            ),
            measure(
                "set-batch",
                "FrozenFuzzySet.fuzzy_get_batch",
                profile,
                items,
                repeats,
                lambda: baseline.frozen_collection.fuzzy_get_batch(queries.batch),
            ),
            measure(
                "set-batch",
                "FrozenFuzzySet.fuzzy_find_one_batch",
                profile,
                items,
                repeats,
                lambda: baseline.frozen_collection.fuzzy_find_one_batch(queries.batch),
            ),
            measure(
                "set-batch",
                "ExplicitFuzzyIndex.find_one from set loop",
                profile,
                items,
                repeats,
                lambda: [baseline.explicit_index.find_one(bq) for bq in queries.batch],
            ),
            measure(
                "set-batch",
                "MutableExplicitFuzzyIndex.find_one from set loop",
                profile,
                items,
                repeats,
                lambda: [baseline.mutable_explicit_index.find_one(bq) for bq in queries.batch],
            ),
        ]
    )

    return results


def run_keyed_choice_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Compare positional facades with runtime keyed facades that omit indexes."""

    mapping_data = build_mapping(items, profile=profile)
    set_data = build_set(items, profile=profile)
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    queries = build_queries(list(mapping_data), batch_size=batch_size)
    mapping_updates = build_mapping_updates(updates)
    set_updates = build_updates(updates)
    removed_keys = tuple(mapping_data)[: min(updates, len(mapping_data))]
    removed_values = tuple(set_data)[: min(updates, len(set_data))]

    def make_dict() -> UnifiedFuzzyDict:
        return FuzzyDict(
            mapping_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_keyed_dict() -> UnifiedFuzzyDict:
        return KeyedFuzzyDict(
            mapping_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_set() -> UnifiedFuzzySet:
        return FuzzySet(
            set_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_keyed_set() -> UnifiedFuzzySet:
        return KeyedFuzzySet(
            set_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    current_dict = make_dict()
    keyed_dict = make_keyed_dict()
    current_set = make_set()
    keyed_set = make_keyed_set()
    for query in (queries.exact, queries.close, queries.miss, *queries.batch):
        dict_match = current_dict.fuzzy_find_key(query)
        dict_key = None if dict_match is None else dict_match.value
        keyed_dict_match = keyed_dict.fuzzy_find_key(query)
        keyed_dict_key = None if keyed_dict_match is None else keyed_dict_match.value
        if dict_key != keyed_dict_key:
            raise RuntimeError("mapping-backed choices changed FuzzyDict selected keys")
        set_match = current_set.fuzzy_find_one(query)
        set_value = None if set_match is None else set_match.value
        keyed_set_match = keyed_set.fuzzy_find_one(query)
        keyed_set_value = None if keyed_set_match is None else keyed_set_match.value
        if set_value != keyed_set_value:
            raise RuntimeError("mapping-backed choices changed FuzzySet selected values")

    results = [
        measure("keyed-build", "FuzzyDict positional index build", profile, items, repeats, make_dict),
        measure("keyed-build", "KeyedFuzzyDict build (no index)", profile, items, repeats, make_keyed_dict),
        measure("keyed-build", "FuzzySet positional index build", profile, items, repeats, make_set),
        measure("keyed-build", "KeyedFuzzySet build (no index)", profile, items, repeats, make_keyed_set),
    ]

    for label, query in (("exact", queries.exact), ("close", queries.close), ("miss", queries.miss)):
        results.extend(
            [
                measure(
                    "keyed-query",
                    f"FuzzyDict.fuzzy_find_key: {label}",
                    profile,
                    items,
                    repeats,
                    lambda q=query: current_dict.fuzzy_find_key(q),
                ),
                measure(
                    "keyed-query",
                    f"KeyedFuzzyDict.fuzzy_find_key (no index): {label}",
                    profile,
                    items,
                    repeats,
                    lambda q=query: keyed_dict.fuzzy_find_key(q),
                ),
                measure(
                    "keyed-query",
                    f"FuzzySet.fuzzy_find_one: {label}",
                    profile,
                    items,
                    repeats,
                    lambda q=query: current_set.fuzzy_find_one(q),
                ),
                measure(
                    "keyed-query",
                    f"KeyedFuzzySet.fuzzy_find_one (no index): {label}",
                    profile,
                    items,
                    repeats,
                    lambda q=query: keyed_set.fuzzy_find_one(q),
                ),
            ]
        )

    results.extend(
        [
            measure(
                "keyed-batch",
                "FuzzyDict key loop",
                profile,
                items,
                repeats,
                lambda: [current_dict.fuzzy_find_key(bq) for bq in queries.batch],
            ),
            measure(
                "keyed-batch",
                "KeyedFuzzyDict key loop (no index)",
                profile,
                items,
                repeats,
                lambda: [keyed_dict.fuzzy_find_key(bq) for bq in queries.batch],
            ),
            measure(
                "keyed-batch",
                "FuzzySet match loop",
                profile,
                items,
                repeats,
                lambda: [current_set.fuzzy_find_one(bq) for bq in queries.batch],
            ),
            measure(
                "keyed-batch",
                "KeyedFuzzySet match loop (no index)",
                profile,
                items,
                repeats,
                lambda: [keyed_set.fuzzy_find_one(bq) for bq in queries.batch],
            ),
        ]
    )
    results.extend(
        [
            measure_after_setup(
                "keyed-mutation",
                f"FuzzyDict incremental set {updates} keys",
                profile,
                items,
                repeats,
                make_dict,
                lambda collection: [collection.__setitem__(key, value) for key, value in mapping_updates.items()],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"KeyedFuzzyDict incremental set {updates} keys (no index)",
                profile,
                items,
                repeats,
                make_keyed_dict,
                lambda collection: [collection.__setitem__(key, value) for key, value in mapping_updates.items()],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"FuzzySet incremental add {updates} values",
                profile,
                items,
                repeats,
                make_set,
                lambda collection: [collection.add(value) for value in set_updates],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"KeyedFuzzySet incremental add {updates} values (no index)",
                profile,
                items,
                repeats,
                make_keyed_set,
                lambda collection: [collection.add(value) for value in set_updates],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"FuzzyDict exact delete {len(removed_keys)} keys",
                profile,
                items,
                repeats,
                make_dict,
                lambda collection: [collection.__delitem__(key) for key in removed_keys],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"KeyedFuzzyDict exact delete {len(removed_keys)} keys (no index)",
                profile,
                items,
                repeats,
                make_keyed_dict,
                lambda collection: [collection.__delitem__(key) for key in removed_keys],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"FuzzySet exact discard {len(removed_values)} values",
                profile,
                items,
                repeats,
                make_set,
                lambda collection: [collection.discard(value) for value in removed_values],
            ),
            measure_after_setup(
                "keyed-mutation",
                f"KeyedFuzzySet exact discard {len(removed_values)} values (no index)",
                profile,
                items,
                repeats,
                make_keyed_set,
                lambda collection: [collection.discard(value) for value in removed_values],
            ),
        ]
    )
    return results


def repeat_to_length(values: tuple[NormalizedValue, ...], length: int) -> tuple[NormalizedValue, ...]:
    """Repeat values until the returned tuple has the requested length."""

    if not values:
        raise ValueError("values must not be empty")

    return tuple(values[index % len(values)] for index in range(length))


def cdist_best_indexes(
    queries: tuple[NormalizedValue, ...],
    choices: tuple[NormalizedValue, ...],
    *,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
) -> object:
    """Return the best choice index per query from a RapidFuzz cdist matrix.

    This measures the native matrix path plus the minimal NumPy reduction
    needed to turn the matrix into top-1 positions.  It intentionally does not
    reproduce collection-level exact shortcuts, `Match` object construction,
    or the collection API's no-match result. In particular, reduction always
    returns an index even when every matrix score was replaced by the cutoff
    sentinel. Use this row only to compare matrix computation and reduction
    overhead, not end-to-end top-1 semantics.
    """

    import numpy as np

    scores = process.cdist(
        queries,
        choices,
        scorer=scorer,
        score_cutoff=score_cutoff,
        workers=1,
    )
    if scorer_type == ScorerType.DISTANCE:
        return np.argmin(scores, axis=1)
    return np.argmax(scores, axis=1)


def run_batch_api_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Run RapidFuzz native batch API measurements."""

    values = build_values(items, profile=profile)
    sequence_baseline = build_baseline(values, scorer=scorer, scorer_type=scorer_type, score_cutoff=score_cutoff)
    mapping_baseline = build_mapping_baseline(
        build_mapping(items, profile=profile),
        scorer=scorer,
        scorer_type=scorer_type,
        score_cutoff=score_cutoff,
    )
    set_baseline = build_set_baseline(
        build_set(items, profile=profile),
        scorer=scorer,
        scorer_type=scorer_type,
        score_cutoff=score_cutoff,
    )
    queries = build_queries(values, batch_size=batch_size)
    normalized_queries = tuple(sequence_baseline.normalizer(bq) for bq in queries.batch)
    sequence_choices = tuple(value for value in sequence_baseline.normalized_values if value is not None)
    mapping_choices = tuple(value for value in mapping_baseline.normalized_keys if value is not None)
    set_choices = tuple(value for value in set_baseline.normalized_values if value is not None)

    results = [
        measure(
            "batch-api",
            "FuzzyList.fuzzy_get loop",
            profile,
            items,
            repeats,
            lambda: [sequence_baseline.collection.fuzzy_get(bq) for bq in queries.batch],
        ),
        measure(
            "batch-api",
            "FuzzyList.fuzzy_get_batch",
            profile,
            items,
            repeats,
            lambda: sequence_baseline.collection.fuzzy_get_batch(queries.batch),
        ),
        measure(
            "batch-api",
            "FuzzyList.fuzzy_find_one_batch",
            profile,
            items,
            repeats,
            lambda: sequence_baseline.collection.fuzzy_find_one_batch(queries.batch),
        ),
        measure(
            "batch-api",
            "FuzzyDict.fuzzy_get loop",
            profile,
            items,
            repeats,
            lambda: [mapping_baseline.collection.fuzzy_get(bq) for bq in queries.batch],
        ),
        measure(
            "batch-api",
            "FuzzyDict.fuzzy_get_batch",
            profile,
            items,
            repeats,
            lambda: mapping_baseline.collection.fuzzy_get_batch(queries.batch),
        ),
        measure(
            "batch-api",
            "FuzzySet.fuzzy_get loop",
            profile,
            items,
            repeats,
            lambda: [set_baseline.collection.fuzzy_get(bq) for bq in queries.batch],
        ),
        measure(
            "batch-api",
            "FuzzySet.fuzzy_get_batch",
            profile,
            items,
            repeats,
            lambda: set_baseline.collection.fuzzy_get_batch(queries.batch),
        ),
        measure(
            "batch-api",
            "FrozenFuzzySet.fuzzy_get loop",
            profile,
            items,
            repeats,
            lambda: [set_baseline.frozen_collection.fuzzy_get(bq) for bq in queries.batch],
        ),
        measure(
            "batch-api",
            "FrozenFuzzySet.fuzzy_get_batch",
            profile,
            items,
            repeats,
            lambda: set_baseline.frozen_collection.fuzzy_get_batch(queries.batch),
        ),
    ]

    if importlib.util.find_spec("numpy") is None:
        results.append(
            BenchmarkResult(
                name="process.cdist/cpdist unavailable: numpy is not installed",
                group="batch-api",
                profile=profile,
                items=items,
                repeats=repeats,
                best_seconds=0.0,
                median_seconds=0.0,
                peak_kib=0.0,
                result_size=0,
            )
        )
        return results

    results.extend(
        [
            measure(
                "batch-api",
                "process.cdist sequence choices",
                profile,
                items,
                repeats,
                lambda: process.cdist(
                    normalized_queries,
                    sequence_choices,
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                    workers=1,
                ),
            ),
            measure(
                "batch-api",
                "process.cdist sequence top1 indexes",
                profile,
                items,
                repeats,
                lambda: cdist_best_indexes(
                    normalized_queries,
                    sequence_choices,
                    scorer=scorer,
                    scorer_type=scorer_type,
                    score_cutoff=score_cutoff,
                ),
            ),
            measure(
                "batch-api",
                "process.cpdist sequence paired choices",
                profile,
                items,
                repeats,
                lambda: process.cpdist(
                    normalized_queries,
                    repeat_to_length(sequence_choices, len(normalized_queries)),
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                    workers=1,
                ),
            ),
            measure(
                "batch-api",
                "process.cdist mapping keys",
                profile,
                items,
                repeats,
                lambda: process.cdist(
                    normalized_queries,
                    mapping_choices,
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                    workers=1,
                ),
            ),
            measure(
                "batch-api",
                "process.cdist mapping top1 indexes",
                profile,
                items,
                repeats,
                lambda: cdist_best_indexes(
                    normalized_queries,
                    mapping_choices,
                    scorer=scorer,
                    scorer_type=scorer_type,
                    score_cutoff=score_cutoff,
                ),
            ),
            measure(
                "batch-api",
                "process.cpdist mapping paired keys",
                profile,
                items,
                repeats,
                lambda: process.cpdist(
                    normalized_queries,
                    repeat_to_length(mapping_choices, len(normalized_queries)),
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                    workers=1,
                ),
            ),
            measure(
                "batch-api",
                "process.cdist set choices",
                profile,
                items,
                repeats,
                lambda: process.cdist(
                    normalized_queries,
                    set_choices,
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                    workers=1,
                ),
            ),
            measure(
                "batch-api",
                "process.cdist set top1 indexes",
                profile,
                items,
                repeats,
                lambda: cdist_best_indexes(
                    normalized_queries,
                    set_choices,
                    scorer=scorer,
                    scorer_type=scorer_type,
                    score_cutoff=score_cutoff,
                ),
            ),
            measure(
                "batch-api",
                "process.cpdist set paired choices",
                profile,
                items,
                repeats,
                lambda: process.cpdist(
                    normalized_queries,
                    repeat_to_length(set_choices, len(normalized_queries)),
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                    workers=1,
                ),
            ),
        ]
    )
    return results


def run_advanced_top_one_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure bounded opt-in cdist top-one lookup against collection lookup."""

    values = build_values(items, profile=profile)
    baseline = build_baseline(values, scorer=scorer, scorer_type=scorer_type, score_cutoff=score_cutoff)
    queries = build_queries(values, batch_size=batch_size)
    fuzzy_queries = queries.batch
    miss_queries = tuple(queries.miss for _ in range(batch_size))
    results = [
        measure(
            "advanced-top-one",
            "collection batch close",
            profile,
            items,
            repeats,
            lambda: baseline.collection.fuzzy_find_one_batch(fuzzy_queries),
        ),
        measure(
            "advanced-top-one",
            "collection batch miss",
            profile,
            items,
            repeats,
            lambda: baseline.collection.fuzzy_find_one_batch(miss_queries),
        ),
    ]
    if importlib.util.find_spec("numpy") is None:
        results.append(
            BenchmarkResult(
                name="cdist API unavailable: numpy is not installed",
                group="advanced-top-one",
                profile=profile,
                items=items,
                repeats=repeats,
                best_seconds=0.0,
                median_seconds=0.0,
                peak_kib=0.0,
                result_size=0,
            )
        )
        return results

    bounded_query_chunk_size = min(32, batch_size)
    candidate_chunk_size = min(1000, items)
    for query_label, query_values in (("close", fuzzy_queries), ("miss", miss_queries)):
        expected = baseline.collection.fuzzy_find_one_batch(query_values)
        for workers in (1, -1):

            def advanced_lookup(
                qv: tuple[str, ...] = query_values,
                w: int = workers,
            ) -> list[object]:
                """Run the opt-in bounded cdist facade API for one benchmark case."""

                return baseline.collection.fuzzy_find_one_batch_cdist(
                    qv,
                    query_chunk_size=bounded_query_chunk_size,
                    choice_chunk_size=candidate_chunk_size,
                    workers=w,
                )

            if advanced_lookup() != expected:
                raise RuntimeError(f"cdist top-one API disagrees with collection lookup: {query_label=}, {workers=}")  # noqa: E501
            results.append(
                measure(
                    "advanced-top-one",
                    f"cdist API {query_label}: workers={workers}, chunk={bounded_query_chunk_size}x{candidate_chunk_size}",  # noqa: E501
                    profile,
                    items,
                    repeats,
                    advanced_lookup,
                )
            )
    return results


def build_updates(updates: int) -> list[str]:
    """Build deterministic update values."""

    return [f"New Accessory {index:06d} Model {index % 97:02d}" for index in range(updates)]


def build_mapping_updates(updates: int) -> dict[str, str]:
    """Build deterministic mapping update values."""

    return {key: f"updated-value-{index:06d}" for index, key in enumerate(build_updates(updates))}


def run_mutation_benchmarks(
    items: int,
    repeats: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure mutable updates and explicit index rebuild costs."""

    values = build_values(items, profile=profile)
    update_values = build_updates(updates)
    mapping_data = build_mapping(items, profile=profile)
    mapping_update_values = build_mapping_updates(updates)
    set_data = build_set(items, profile=profile)
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)

    def append_to_fuzzy_list() -> FuzzyList:
        collection = FuzzyList(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )
        for value in update_values:
            collection.append(value)
        return collection

    def rebuild_explicit_index() -> ExplicitFuzzyIndex:
        return ExplicitFuzzyIndex.from_values(
            [*values, *update_values],
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def append_to_mutable_explicit_index() -> MutableExplicitFuzzyIndex:
        index = MutableExplicitFuzzyIndex.from_values(
            values,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )
        for value in update_values:
            index.append(value)
        return index

    def build_mutable_explicit_index() -> MutableExplicitFuzzyIndex:
        return MutableExplicitFuzzyIndex.from_values(
            values.copy(),
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def append_to_prebuilt_mutable_explicit_index(target: object) -> MutableExplicitFuzzyIndex:
        if not isinstance(target, MutableExplicitFuzzyIndex):
            raise TypeError(f"Expected MutableExplicitFuzzyIndex, got {type(target).__name__}")

        for value in update_values:
            target.append(value)
        return target

    def set_fuzzy_dict() -> UnifiedFuzzyDict:
        collection = FuzzyDict(
            mapping_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )
        for key, value in mapping_update_values.items():
            collection[key] = value
        return collection

    def rebuild_explicit_mapping_index() -> ExplicitFuzzyMappingIndex:
        return ExplicitFuzzyMappingIndex.from_mapping(
            mapping_data | mapping_update_values,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def append_to_mutable_mapping_index() -> MutableExplicitFuzzyMappingIndex:
        index = MutableExplicitFuzzyMappingIndex.from_mapping(
            mapping_data,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )
        index.update(mapping_update_values)
        return index

    def build_mutable_mapping_index() -> MutableExplicitFuzzyMappingIndex:
        return MutableExplicitFuzzyMappingIndex.from_mapping(
            mapping_data.copy(),
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def append_to_prebuilt_mutable_mapping_index(target: object) -> MutableExplicitFuzzyMappingIndex:
        if not isinstance(target, MutableExplicitFuzzyMappingIndex):
            raise TypeError(f"Expected MutableExplicitFuzzyMappingIndex, got {type(target).__name__}")

        target.update(mapping_update_values)
        return target

    def add_to_fuzzy_set() -> UnifiedFuzzySet:
        collection = FuzzySet(
            set_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )
        for value in update_values:
            collection.add(value)
        return collection

    def rebuild_explicit_set_index() -> ExplicitFuzzyIndex:
        return ExplicitFuzzyIndex.from_values(
            list(set_data | set(update_values)),
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def add_to_mutable_set_index() -> MutableExplicitFuzzyIndex:
        index = MutableExplicitFuzzyIndex.from_values(
            list(set_data),
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )
        for value in update_values:
            if value not in index.exact_first_index:
                index.append(value)
        return index

    def build_mutable_set_index() -> MutableExplicitFuzzyIndex:
        return MutableExplicitFuzzyIndex.from_values(
            list(set_data),
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def add_to_prebuilt_mutable_set_index(target: object) -> MutableExplicitFuzzyIndex:
        if not isinstance(target, MutableExplicitFuzzyIndex):
            raise TypeError(f"Expected MutableExplicitFuzzyIndex, got {type(target).__name__}")

        for value in update_values:
            if value not in target.exact_first_index:
                target.append(value)
        return target

    return [
        measure(
            "mutation",
            f"FuzzyList append {updates} values",
            profile,
            items,
            repeats,
            append_to_fuzzy_list,
        ),
        measure(
            "mutation",
            f"ExplicitFuzzyIndex rebuild after {updates} values",
            profile,
            items,
            repeats,
            rebuild_explicit_index,
        ),
        measure(
            "mutation",
            f"MutableExplicitFuzzyIndex append {updates} values",
            profile,
            items,
            repeats,
            append_to_mutable_explicit_index,
        ),
        measure_after_setup(
            "mutation",
            f"MutableExplicitFuzzyIndex incremental append {updates} values",
            profile,
            items,
            repeats,
            build_mutable_explicit_index,
            append_to_prebuilt_mutable_explicit_index,
        ),
        measure(
            "mapping-mutation",
            f"FuzzyDict set {updates} keys",
            profile,
            items,
            repeats,
            set_fuzzy_dict,
        ),
        measure(
            "mapping-mutation",
            f"ExplicitFuzzyMappingIndex rebuild after {updates} keys",
            profile,
            items,
            repeats,
            rebuild_explicit_mapping_index,
        ),
        measure(
            "mapping-mutation",
            f"MutableExplicitFuzzyMappingIndex set {updates} keys",
            profile,
            items,
            repeats,
            append_to_mutable_mapping_index,
        ),
        measure_after_setup(
            "mapping-mutation",
            f"MutableExplicitFuzzyMappingIndex incremental set {updates} keys",
            profile,
            items,
            repeats,
            build_mutable_mapping_index,
            append_to_prebuilt_mutable_mapping_index,
        ),
        measure(
            "set-mutation",
            f"FuzzySet add {updates} values",
            profile,
            items,
            repeats,
            add_to_fuzzy_set,
        ),
        measure(
            "set-mutation",
            f"ExplicitFuzzyIndex rebuild from set after {updates} values",
            profile,
            items,
            repeats,
            rebuild_explicit_set_index,
        ),
        measure(
            "set-mutation",
            f"MutableExplicitFuzzyIndex add from set {updates} values",
            profile,
            items,
            repeats,
            add_to_mutable_set_index,
        ),
        measure_after_setup(
            "set-mutation",
            f"MutableExplicitFuzzyIndex incremental add from set {updates} values",
            profile,
            items,
            repeats,
            build_mutable_set_index,
            add_to_prebuilt_mutable_set_index,
        ),
    ]


def run_index_comparison_benchmarks(
    items: int,
    repeats: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Compare FuzzySequenceIndex (fixed index) vs MutableFuzzySequenceIndex.

    Four scenarios:

    - ``index-build``: construction cost and peak memory.
    - ``index-query``: single query on a pre-built, clean (non-dirty) index.
    - ``index-interleaved``: K rounds of (append one value → fuzzy query).  Setup
      cost (initial build) is excluded from timing.  This is the key scenario.
    - ``index-delete``: delete D values → one fuzzy query.  Setup cost excluded.
    """

    values = build_values(items, profile=profile)
    update_values = build_updates(updates)
    delete_count = min(updates, max(1, items // 10))
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    queries = build_queries(values, batch_size=20)

    def make_frozen() -> FuzzySequenceIndex:
        return FuzzySequenceIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_mutable() -> MutableFuzzySequenceIndex:
        return MutableFuzzySequenceIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    _frozen_check = make_frozen()
    _mutable_check = make_mutable()
    _frozen_result = _frozen_check.find_one(queries.exact)
    _mutable_result = _mutable_check.find_one(queries.exact)
    assert _frozen_result is not None, f"FuzzySequenceIndex.find_one returned None for exact query {queries.exact!r}"
    assert _mutable_result is not None, (
        f"MutableFuzzySequenceIndex.find_one returned None for exact query {queries.exact!r}"
    )
    assert _frozen_result.value == _mutable_result.value, (
        f"SEQUENCE vs MUTABLE mismatch: {_frozen_result.value!r} != {_mutable_result.value!r}"
    )

    results: list[BenchmarkResult] = []

    # ── Scenario 1: Construction ──
    results.extend(
        [
            measure("index-build", "FuzzySequenceIndex build", profile, items, repeats, make_frozen),
            measure("index-build", "MutableFuzzySequenceIndex build", profile, items, repeats, make_mutable),
        ]
    )

    # ── Scenario 2: Query-only on clean index (no mutations) ──
    for label, query in (
        ("exact", queries.exact),
        ("close", queries.close),
        ("miss", queries.miss),
    ):
        results.extend(
            [
                measure_after_setup(
                    "index-query",
                    f"FuzzySequenceIndex find_one: {label}",
                    profile,
                    items,
                    repeats,
                    make_frozen,
                    lambda idx, q=query: idx.find_one(q),
                ),
                measure_after_setup(
                    "index-query",
                    f"MutableFuzzySequenceIndex find_one: {label}",
                    profile,
                    items,
                    repeats,
                    make_mutable,
                    lambda idx, q=query: idx.find_one(q),
                ),
            ]
        )

    # ── Scenario 3: Interleaved append + query (KEY scenario) ──
    # Frozen (legacy pattern): each query triggers a full rebuild of the index.
    # Setup provides a pre-copied list so initial construction is excluded.
    def frozen_interleaved_setup() -> list:
        return list(values)

    def frozen_interleaved_run(current_values: list) -> object:
        result = None
        for val in update_values:
            current_values.append(val)
            idx = FuzzySequenceIndex(
                current_values,
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            )
            result = idx.find_one(queries.close)
        return result

    # Mutable (new): O(1) append, no rebuild needed before the query.
    def mutable_interleaved_run(idx: MutableFuzzySequenceIndex) -> object:
        result = None
        for val in update_values:
            idx.append(val)
            result = idx.find_one(queries.close)
        return result

    results.extend(
        [
            measure_after_setup(
                "index-interleaved",
                f"FuzzySequenceIndex {updates}x (append + query)",
                profile,
                items,
                repeats,
                frozen_interleaved_setup,
                frozen_interleaved_run,
            ),
            measure_after_setup(
                "index-interleaved",
                f"MutableFuzzySequenceIndex {updates}x (append + query)",
                profile,
                items,
                repeats,
                make_mutable,
                mutable_interleaved_run,
            ),
        ]
    )

    # ── Scenario 4: Delete + query ──
    # Frozen (legacy pattern): delete from the raw list,then rebuild on query.
    def frozen_delete_setup() -> list:
        return list(values)

    def frozen_delete_run(current_values: list) -> object:
        del current_values[:delete_count]
        idx = FuzzySequenceIndex(
            current_values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )
        return idx.find_one(queries.close)

    # Mutable (new): delete_at_positions marks dirty; _rebuild() runs on query.
    def mutable_delete_run(idx: MutableFuzzySequenceIndex) -> object:
        idx.delete_at_positions(set(range(delete_count)))
        return idx.find_one(queries.close)

    results.extend(
        [
            measure_after_setup(
                "index-delete",
                f"FuzzySequenceIndex delete {delete_count} + query",
                profile,
                items,
                repeats,
                frozen_delete_setup,
                frozen_delete_run,
            ),
            measure_after_setup(
                "index-delete",
                f"MutableFuzzySequenceIndex delete {delete_count} + query",
                profile,
                items,
                repeats,
                make_mutable,
                mutable_delete_run,
            ),
        ]
    )

    # ── Scenario 5: Batch append → single rebuild → single query ──
    # Frozen (batch mode): accumulate all appends, rebuild once, then query.
    # This is the best-case strategy for FuzzySequenceIndex with mutations.
    def frozen_batch_setup() -> list:
        return list(values)

    def frozen_batch_run(current_values: list) -> object:
        current_values.extend(update_values)
        idx = FuzzySequenceIndex(
            current_values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )
        return idx.find_one(queries.close)

    def mutable_batch_run(idx: MutableFuzzySequenceIndex) -> object:
        for val in update_values:
            idx.append(val)
        return idx.find_one(queries.close)

    results.extend(
        [
            measure_after_setup(
                "index-batch",
                f"FuzzySequenceIndex batch {updates} appends + 1 query",
                profile,
                items,
                repeats,
                frozen_batch_setup,
                frozen_batch_run,
            ),
            measure_after_setup(
                "index-batch",
                f"MutableFuzzySequenceIndex batch {updates} appends + 1 query",
                profile,
                items,
                repeats,
                make_mutable,
                mutable_batch_run,
            ),
        ]
    )

    return results


def run_deletion_heavy_benchmarks(
    items: int,
    repeats: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure repeated fuzzy deletions and a soft-delete prototype.

    The production mutable index rebuilds its normalized lookup structures on
    the first fuzzy query after deletion. The tombstone prototype instead
    removes searchable choices from an active mapping keyed by stable source
    positions. Its fuzzy-only behavior is sufficient to test whether avoiding
    repeated rebuilds is worth pursuing in the complete index design.
    """

    values = build_values(items, profile=profile)
    mapping_data = build_mapping(items, profile=profile)
    set_data = build_set(items, profile=profile)
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    queries = build_queries(values, batch_size=20)
    candidates = string_values(values)
    delete_count = min(updates, len(candidates), max(1, items // 10))
    delete_queries = tuple(make_typo(value) for value in candidates[:delete_count])

    def make_list() -> FuzzyList:
        return FuzzyList(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_dict() -> UnifiedFuzzyDict:
        return FuzzyDict(
            mapping_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_set() -> UnifiedFuzzySet:
        return FuzzySet(
            set_data,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_mutable_index() -> MutableFuzzySequenceIndex:
        return MutableFuzzySequenceIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_tombstone_index() -> TombstoneFuzzyIndex:
        return TombstoneFuzzyIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def make_compact_delete_index() -> CompactDeleteFuzzyIndex:
        return CompactDeleteFuzzyIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            score_cutoff=score_cutoff,
        )

    def list_interleaved(collection: FuzzyList) -> object:
        for query in delete_queries:
            collection.fuzzy_discard(query)
        return collection.fuzzy_get(queries.close)

    def dict_interleaved(collection: UnifiedFuzzyDict) -> object:
        for query in delete_queries:
            collection.fuzzy_discard(query)
        return collection.fuzzy_get(queries.close)

    def set_interleaved(collection: UnifiedFuzzySet) -> object:
        for query in delete_queries:
            collection.fuzzy_discard(query)
        return collection.fuzzy_get(queries.close)

    def mutable_interleaved(index: MutableFuzzySequenceIndex) -> object:
        for query in delete_queries:
            match = index.find_one(query)
            if match is not None and match.index is not None:
                index.delete_at(match.index)
        return index.find_one(queries.close)

    def tombstone_interleaved(index: TombstoneFuzzyIndex) -> object:
        for query in delete_queries:
            index.discard_one(query)
        return index.find_one(queries.close)

    def compact_interleaved(index: CompactDeleteFuzzyIndex) -> object:
        for query in delete_queries:
            index.discard_one(query)
        return index.find_one(queries.close)

    def mutable_batch_delete(index: MutableFuzzySequenceIndex) -> object:
        index.delete_at_positions(set(range(delete_count)))
        return index.find_one(queries.close)

    def tombstone_batch_delete(index: TombstoneFuzzyIndex) -> object:
        index.delete_at_positions(set(range(delete_count)))
        return index.find_one(queries.close)

    def compact_batch_delete(index: CompactDeleteFuzzyIndex) -> object:
        index.delete_at_positions(set(range(delete_count)))
        return index.find_one(queries.close)

    return [
        measure(
            "deletion-build",
            "MutableFuzzySequenceIndex build",
            profile,
            items,
            repeats,
            make_mutable_index,
        ),
        measure(
            "deletion-build",
            "TombstoneFuzzyIndex prototype build",
            profile,
            items,
            repeats,
            make_tombstone_index,
        ),
        measure(
            "deletion-build",
            "CompactDeleteFuzzyIndex prototype build",
            profile,
            items,
            repeats,
            make_compact_delete_index,
        ),
        measure_after_setup(
            "deletion-facade",
            f"FuzzyList {delete_count}x fuzzy_discard + query",
            profile,
            items,
            repeats,
            make_list,
            list_interleaved,
        ),
        measure_after_setup(
            "deletion-facade",
            f"FuzzyDict {delete_count}x fuzzy_discard + query",
            profile,
            items,
            repeats,
            make_dict,
            dict_interleaved,
        ),
        measure_after_setup(
            "deletion-facade",
            f"FuzzySet {delete_count}x fuzzy_discard + query",
            profile,
            items,
            repeats,
            make_set,
            set_interleaved,
        ),
        measure_after_setup(
            "deletion-index",
            f"MutableFuzzySequenceIndex {delete_count}x discard + query",
            profile,
            items,
            repeats,
            make_mutable_index,
            mutable_interleaved,
        ),
        measure_after_setup(
            "deletion-index",
            f"TombstoneFuzzyIndex {delete_count}x discard + query",
            profile,
            items,
            repeats,
            make_tombstone_index,
            tombstone_interleaved,
        ),
        measure_after_setup(
            "deletion-index",
            f"CompactDeleteFuzzyIndex {delete_count}x discard + query",
            profile,
            items,
            repeats,
            make_compact_delete_index,
            compact_interleaved,
        ),
        measure_after_setup(
            "deletion-batch",
            f"MutableFuzzySequenceIndex delete {delete_count} + query",
            profile,
            items,
            repeats,
            make_mutable_index,
            mutable_batch_delete,
        ),
        measure_after_setup(
            "deletion-batch",
            f"TombstoneFuzzyIndex delete {delete_count} + query",
            profile,
            items,
            repeats,
            make_tombstone_index,
            tombstone_batch_delete,
        ),
        measure_after_setup(
            "deletion-batch",
            f"CompactDeleteFuzzyIndex delete {delete_count} + query",
            profile,
            items,
            repeats,
            make_compact_delete_index,
            compact_batch_delete,
        ),
    ]


def run_replacement_heavy_benchmarks(
    items: int,
    repeats: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure repeated positional replacements followed by a fuzzy query.

    Positional replacement always marks the index dirty.  Each fuzzy query
    after a replacement triggers a full O(n) rebuild.  Two workloads are
    compared:

    - Interleaved: K × (replace one position → query) — K rebuilds total.
    - Batch: replace K positions → one query — 1 rebuild total.
    """

    values = build_values(items, profile=profile)
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    queries = build_queries(values, batch_size=20)
    replace_count = min(updates, max(1, items // 10))
    new_values = build_updates(replace_count)

    def make_list() -> FuzzyList:
        return FuzzyList(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_mutable_index() -> MutableFuzzySequenceIndex:
        return MutableFuzzySequenceIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def list_replace_interleaved(collection: FuzzyList) -> object:
        for i, val in enumerate(new_values):
            collection[i] = val
            collection.fuzzy_get(queries.close)
        return None

    def list_replace_batch(collection: FuzzyList) -> object:
        for i, val in enumerate(new_values):
            collection[i] = val
        return collection.fuzzy_get(queries.close)

    def mutable_replace_interleaved(index: MutableFuzzySequenceIndex) -> object:
        for i, val in enumerate(new_values):
            index.replace_at(i, val)
            index.find_one(queries.close)
        return None

    def mutable_replace_batch(index: MutableFuzzySequenceIndex) -> object:
        for i, val in enumerate(new_values):
            index.replace_at(i, val)
        return index.find_one(queries.close)

    return [
        measure_after_setup(
            "replace-interleaved",
            f"FuzzyList {replace_count}x replace + query",
            profile,
            items,
            repeats,
            make_list,
            list_replace_interleaved,
        ),
        measure_after_setup(
            "replace-batch",
            f"FuzzyList replace {replace_count} + query",
            profile,
            items,
            repeats,
            make_list,
            list_replace_batch,
        ),
        measure_after_setup(
            "replace-interleaved",
            f"MutableFuzzySequenceIndex {replace_count}x replace + query",
            profile,
            items,
            repeats,
            make_mutable_index,
            mutable_replace_interleaved,
        ),
        measure_after_setup(
            "replace-batch",
            f"MutableFuzzySequenceIndex replace {replace_count} + query",
            profile,
            items,
            repeats,
            make_mutable_index,
            mutable_replace_batch,
        ),
    ]


def run_interleaved_benchmarks(
    items: int,
    repeats: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure interleaved insert, delete, and query workloads.

    Two mutation strategies are compared:

    - append + fuzzy_discard + query: append is O(1) incremental, compact
      deletion avoids a rebuild, and the query runs without rebuilding.
      Expected cost: O(K * N) total for K rounds.
    - insert-at-0 + fuzzy_discard + query: positional insert marks the index
      dirty; each query triggers a full O(n) rebuild.
      Expected cost: O(K * N_rebuild) total for K rounds.
    """

    values = build_values(items, profile=profile)
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    queries = build_queries(values, batch_size=20)
    candidates = string_values(values)
    round_count = min(updates, max(1, items // 10))
    new_values = build_updates(round_count)
    delete_queries = [make_typo(value) for value in candidates[:round_count]]
    n_delete = len(delete_queries)

    def make_list() -> FuzzyList:
        return FuzzyList(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def make_mutable_index() -> MutableFuzzySequenceIndex:
        return MutableFuzzySequenceIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
        )

    def list_append_discard_query(collection: FuzzyList) -> object:
        for i in range(round_count):
            collection.append(new_values[i])
            collection.fuzzy_discard(delete_queries[i % n_delete])
            collection.fuzzy_get(queries.close)
        return None

    def list_insert_discard_query(collection: FuzzyList) -> object:
        for i in range(round_count):
            collection.insert(0, new_values[i])
            collection.fuzzy_discard(delete_queries[i % n_delete])
            collection.fuzzy_get(queries.close)
        return None

    def mutable_append_delete_query(index: MutableFuzzySequenceIndex) -> object:
        for i in range(round_count):
            index.append(new_values[i])
            match = index.find_one(delete_queries[i % n_delete])
            if match is not None and match.index is not None:
                index.delete_at(match.index)
            index.find_one(queries.close)
        return None

    def mutable_insert_delete_query(index: MutableFuzzySequenceIndex) -> object:
        for i in range(round_count):
            index.insert_at(0, new_values[i])
            match = index.find_one(delete_queries[i % n_delete])
            if match is not None and match.index is not None:
                index.delete_at(match.index)
            index.find_one(queries.close)
        return None

    return [
        measure_after_setup(
            "interleaved",
            f"FuzzyList {round_count}x append+discard+query",
            profile,
            items,
            repeats,
            make_list,
            list_append_discard_query,
        ),
        measure_after_setup(
            "interleaved",
            f"FuzzyList {round_count}x insert+discard+query",
            profile,
            items,
            repeats,
            make_list,
            list_insert_discard_query,
        ),
        measure_after_setup(
            "interleaved",
            f"MutableFuzzySequenceIndex {round_count}x append+delete+query",
            profile,
            items,
            repeats,
            make_mutable_index,
            mutable_append_delete_query,
        ),
        measure_after_setup(
            "interleaved",
            f"MutableFuzzySequenceIndex {round_count}x insert+delete+query",
            profile,
            items,
            repeats,
            make_mutable_index,
            mutable_insert_delete_query,
        ),
    ]


def run_score_hint_benchmarks(
    items: int,
    repeats: int,
    batch_size: int,
    scorer_profile: ScorerProfile,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
    profile: DataProfile,
) -> list[BenchmarkResult]:
    """Measure RapidFuzz score-hint effects on index-compatible fuzzy paths."""

    values = build_values(items, profile=profile)
    baseline = build_baseline(values, scorer=scorer, scorer_type=scorer_type, score_cutoff=score_cutoff)
    queries = build_queries(values, batch_size=batch_size)
    choices = tuple(value for value in baseline.normalized_values if value is not None)
    normalized_close = baseline.normalizer(queries.close)
    normalized_miss = baseline.normalizer(queries.miss)
    normalized_batch = tuple(nq for nq in (baseline.normalizer(bq) for bq in queries.batch) if nq is not None)
    if normalized_close is None or normalized_miss is None:
        raise RuntimeError("score-hint benchmark queries must be searchable")

    close_query: str = normalized_close
    miss_query: str = normalized_miss
    high_hint = score_hint_for(scorer_profile, high_confidence=True)
    cutoff_hint = score_hint_for(scorer_profile, high_confidence=False)

    def extract_one(query: str, score_hint: float | None) -> object:
        return process.extractOne(
            query,
            choices,
            scorer=scorer,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

    def extract_one_batch(score_hint: float | None) -> list[object]:
        return [extract_one(query, score_hint) for query in normalized_batch]

    return [
        measure(
            "score-hint",
            "extractOne close: no hint",
            profile,
            items,
            repeats,
            lambda: extract_one(close_query, None),
        ),
        measure(
            "score-hint",
            f"extractOne close: hint={high_hint:g}",
            profile,
            items,
            repeats,
            lambda: extract_one(close_query, high_hint),
        ),
        measure(
            "score-hint",
            "extractOne miss: no hint",
            profile,
            items,
            repeats,
            lambda: extract_one(miss_query, None),
        ),
        measure(
            "score-hint",
            f"extractOne miss: hint={cutoff_hint:g}",
            profile,
            items,
            repeats,
            lambda: extract_one(miss_query, cutoff_hint),
        ),
        measure(
            "score-hint",
            "extractOne batch: no hint",
            profile,
            items,
            repeats,
            lambda: extract_one_batch(None),
        ),
        measure(
            "score-hint",
            f"extractOne batch: hint={high_hint:g}",
            profile,
            items,
            repeats,
            lambda: extract_one_batch(high_hint),
        ),
    ]


def run_collision_cost_benchmarks(
    items: int,
    repeats: int,
    updates: int,
    scorer: Scorer,
    scorer_type: ScorerType,
    score_cutoff: int | float,
) -> list[BenchmarkResult]:
    """Measure exact-deletion cost as a function of normalized collision density.

    Sweeps collision_rate over [0.0, 0.05, 0.10, 0.20, 0.50] and records the
    total time to perform ``delete_count`` exact deletions (``__delitem__`` /
    ``discard``) for FuzzyDict, KeyedFuzzyDict, FuzzySet, and KeyedFuzzySet.

    Only exact lookups are used so the measured cost reflects index-update
    overhead exclusively, without O(N·scorer) fuzzy-search cost.
    """
    collision_rates = [0.0, 0.05, 0.10, 0.20, 0.50]
    normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
    results: list[BenchmarkResult] = []

    for collision_rate in collision_rates:
        profile_label = f"col{int(collision_rate * 100):02d}pct"
        values = build_values_with_collision_rate(items, collision_rate)
        mapping_data: dict[str, str] = {v: f"val-{i:06d}" for i, v in enumerate(values)}
        set_data: set[str] = set(values)

        collision_count = int(items * collision_rate)
        collision_count -= collision_count % 2
        delete_count = min(updates, max(1, items // 10))
        n_collision_deletes = min(int(delete_count * collision_rate), collision_count // 2)
        n_unique_deletes = delete_count - n_collision_deletes

        # Canonical (non-padded) collision keys; padded variants share their normalized form.
        collision_keys = [f"Consumer Device {i:06d}" for i in range(n_collision_deletes)]
        unique_keys = [f"Discontinued Accessory {i:06d} Line {i % 97:02d}" for i in range(n_unique_deletes)]
        delete_keys = collision_keys + unique_keys
        delete_values = list(delete_keys)

        # noinspection PyDefaultArgument
        def make_dict(md: dict[str, str] = mapping_data) -> UnifiedFuzzyDict:
            return FuzzyDict(
                md,
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            )

        # noinspection PyDefaultArgument
        def make_keyed_dict(md: dict[str, str] = mapping_data) -> UnifiedFuzzyDict:
            return KeyedFuzzyDict(
                md,
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            )

        # noinspection PyDefaultArgument
        def make_set(sd: set[str] = set_data) -> UnifiedFuzzySet:
            return FuzzySet(
                sd,
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            )

        # noinspection PyDefaultArgument
        def make_keyed_set(sd: set[str] = set_data) -> UnifiedFuzzySet:
            return KeyedFuzzySet(
                sd,
                normalizer=normalizer,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            )

        # noinspection PyDefaultArgument
        def dict_delete(collection: UnifiedFuzzyDict, keys: list[str] = delete_keys) -> None:
            for key in keys:
                del collection[key]

        # noinspection PyDefaultArgument
        def keyed_dict_delete(collection: UnifiedFuzzyDict, keys: list[str] = delete_keys) -> None:
            for key in keys:
                del collection[key]

        # noinspection PyDefaultArgument
        def set_discard(collection: UnifiedFuzzySet, vals: list[str] = delete_values) -> None:
            for val in vals:
                collection.discard(val)

        # noinspection PyDefaultArgument
        def keyed_set_discard(collection: UnifiedFuzzySet, vals: list[str] = delete_values) -> None:
            for val in vals:
                collection.discard(val)

        results.extend(
            [
                measure_after_setup(
                    BenchmarkSection.COLLISION_COST,
                    "FuzzyDict delete",
                    profile_label,
                    items,
                    repeats,
                    make_dict,
                    dict_delete,
                ),
                measure_after_setup(
                    BenchmarkSection.COLLISION_COST,
                    "KeyedFuzzyDict delete",
                    profile_label,
                    items,
                    repeats,
                    make_keyed_dict,
                    keyed_dict_delete,
                ),
                measure_after_setup(
                    BenchmarkSection.COLLISION_COST,
                    "FuzzySet discard",
                    profile_label,
                    items,
                    repeats,
                    make_set,
                    set_discard,
                ),
                measure_after_setup(
                    BenchmarkSection.COLLISION_COST,
                    "KeyedFuzzySet discard",
                    profile_label,
                    items,
                    repeats,
                    make_keyed_set,
                    keyed_set_discard,
                ),
            ]
        )

    return results


def run(
    items: int,
    repeats: int,
    batch_size: int,
    updates: int,
    scorer_profile: ScorerProfile,
    score_cutoff: int | float,
    profile: DataProfile,
    sections: set[BenchmarkSection],
) -> list[BenchmarkResult]:
    """Run baseline measurements for direct RapidFuzz and current collection wrapper."""

    if items < 1:
        raise ValueError("items must be greater than 0")

    if repeats < 1:
        raise ValueError("repeats must be greater than 0")

    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    if updates < 1:
        raise ValueError("updates must be greater than 0")

    scorer = scorer_for(scorer_profile)
    scorer_type = scorer_type_for(scorer_profile)
    results: list[BenchmarkResult] = []

    if BenchmarkSection.ADVANCED_TOP_ONE in sections:
        results.extend(
            run_advanced_top_one_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.BUILD in sections:
        results.extend(
            run_build_benchmarks(
                items=items,
                repeats=repeats,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.SEQUENCE in sections:
        results.extend(
            run_query_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.MAPPING in sections:
        results.extend(
            run_mapping_query_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.SET in sections:
        results.extend(
            run_set_query_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.KEYED_CHOICES in sections:
        results.extend(
            run_keyed_choice_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.BATCH_API in sections:
        results.extend(
            run_batch_api_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.MUTATION in sections:
        results.extend(
            run_mutation_benchmarks(
                items=items,
                repeats=repeats,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.INDEX_COMPARISON in sections:
        results.extend(
            run_index_comparison_benchmarks(
                items=items,
                repeats=repeats,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.DELETION_HEAVY in sections:
        results.extend(
            run_deletion_heavy_benchmarks(
                items=items,
                repeats=repeats,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.REPLACEMENT_HEAVY in sections:
        results.extend(
            run_replacement_heavy_benchmarks(
                items=items,
                repeats=repeats,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.INTERLEAVED in sections:
        results.extend(
            run_interleaved_benchmarks(
                items=items,
                repeats=repeats,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.SCORE_HINT in sections:
        results.extend(
            run_score_hint_benchmarks(
                items=items,
                repeats=repeats,
                batch_size=batch_size,
                scorer_profile=scorer_profile,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                profile=profile,
            )
        )

    if BenchmarkSection.COLLISION_COST in sections:
        results.extend(
            run_collision_cost_benchmarks(
                items=items,
                repeats=repeats,
                updates=updates,
                scorer=scorer,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
            )
        )

    return [replace(result, scorer=scorer_profile) for result in results]


def write_outputs(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Write raw benchmark result rows as JSON and CSV."""

    rows = [
        {
            "group": result.group,
            "profile": result.profile,
            "scorer": result.scorer,
            "name": result.name,
            "items": result.items,
            "repeats": result.repeats,
            "best_ms": result.best_seconds * 1000,
            "median_ms": result.median_seconds * 1000,
            "peak_kib": result.peak_kib,
            "result_size": result.result_size,
        }
        for result in results
    ]
    write_benchmark_reports(rows, output_dir, stem="baseline_results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=positive_int, default=10_000, help="Number of generated collection values.")
    parser.add_argument("--repeats", type=positive_int, default=5, help="Number of repeated measurements.")
    parser.add_argument("--batch-size", type=positive_int, default=100, help="Number of queries in batch scenarios.")
    parser.add_argument("--updates", type=positive_int, default=100, help="Number of values in mutation scenarios.")
    parser.add_argument(
        "--score-cutoff",
        type=non_negative_float,
        default=None,
        help="RapidFuzz score cutoff used by query benchmarks. Defaults depend on --scorer.",
    )
    parser.add_argument(
        "--scorer",
        choices=tuple(ScorerProfile),
        default=ScorerProfile.WRATIO,
        type=ScorerProfile,
        help="RapidFuzz scorer used by query benchmarks.",
    )
    parser.add_argument(
        "--scorers",
        choices=tuple(ScorerProfile),
        nargs="+",
        type=ScorerProfile,
        help="RapidFuzz scorer matrix. Overrides --scorer when provided.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(DataProfile),
        default=DataProfile.UNIQUE,
        type=DataProfile,
        help="Generated data profile.",
    )
    parser.add_argument(
        "--profiles",
        choices=tuple(DataProfile),
        nargs="+",
        type=DataProfile,
        help="Generated data profile matrix. Overrides --profile when provided.",
    )
    parser.add_argument(
        "--groups",
        choices=tuple(BenchmarkSection),
        default=tuple(BenchmarkSection),
        nargs="+",
        type=BenchmarkSection,
        help="Top-level benchmark sections to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports/baseline"),
        help="Directory to write JSON and CSV results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line benchmark."""

    args = parse_args(argv)
    scorer_profiles = args.scorers if args.scorers is not None else [args.scorer]
    data_profiles = args.profiles if args.profiles is not None else [args.profile]
    sections = set(args.groups)
    results: list[BenchmarkResult] = []
    for data_profile in data_profiles:
        for scorer_profile in scorer_profiles:
            score_cutoff = default_score_cutoff_for(scorer_profile) if args.score_cutoff is None else args.score_cutoff
            results.extend(
                run(
                    items=args.items,
                    repeats=args.repeats,
                    batch_size=args.batch_size,
                    updates=args.updates,
                    scorer_profile=scorer_profile,
                    score_cutoff=score_cutoff,
                    profile=data_profile,
                    sections=sections,
                )
            )
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
