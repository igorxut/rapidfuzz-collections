from collections.abc import Callable, Hashable, Iterable, Iterator, Set
from copy import deepcopy as _deepcopy
from types import NotImplementedType
from typing import Any

from rapidfuzz.fuzz import WRatio

from ..configuration import _UNCHANGED, _apply_config_overrides, _coerce_index_strategy
from ..enums import IndexStrategy, ScorerType
from ..indexes import FuzzySequenceIndex, ImmutableFuzzyKeyedIndex, Scorer, validate_chunk_size
from ..matching import Match, ValueMatch


class FrozenFuzzySet[T: Hashable](Set[T]):
    """An immutable set-like collection with fuzzy lookup over values.

    Enforces uniqueness; the first occurrence of each value is retained and
    subsequent duplicates are discarded at construction.  Insertion order is
    preserved as the deterministic tie-breaker for equal-score fuzzy matches.

    Implements ``collections.abc.Set`` and is hashable via ``frozenset``.
    The fuzzy index is built once at construction and never rebuilt.

    ``strategy`` selects the index storage strategy used by the fuzzy index. ``SEQUENCE`` is the
    default and favors general read-heavy lookups. ``KEYED`` can improve build
    cost and can reduce frozen-index memory for unique hashable values in
    measured workloads, but point-lookup speed remains workload-dependent.
    Fuzzy results are position-free and expose ``index=None``.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = ("_data", "_index", "_is_keyed", "_strategy", "_values")

    def __and__(self, other: Set[object]) -> FrozenFuzzySet[T] | NotImplementedType:
        """Return values that exist in both sets while preserving fuzzy config.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``intersection()`` for an arbitrary iterable.

        Args:
            other: Set-like value to intersect with this set.

        Returns:
            New frozen fuzzy set containing values present in both operands, or
            ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        other_values = set(other)
        return self._new_from_values(value for value in self if value in other_values)

    def __contains__(self, value: object) -> bool:
        """Return whether the exact value exists.

        Args:
            value: Value to test for exact membership.

        Returns:
            ``True`` if ``value`` exists in the set, otherwise ``False``.
        """

        return value in self._data

    def __copy__(self) -> FrozenFuzzySet[T]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same values and fuzzy configuration.
        """

        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenFuzzySet[T]:
        """Return a deep copy preserving fuzzy configuration.

        Args:
            memo: Deepcopy memo dictionary used to preserve object identity during
                recursive copying.

        Returns:
            Deep copy with copied values and preserved fuzzy configuration.
        """

        instance = self.__new__(self.__class__)
        memo[id(self)] = instance
        instance.__init__(
            _deepcopy(self._ordered_values(), memo),
            **self._config_kwargs(deepcopy_memo=memo),
        )
        return instance

    def __hash__(self) -> int:
        """Return the order-independent set hash.

        Returns:
            Hash value derived from the stored values as a regular frozen set.
        """

        return hash(self._data)

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
        strategy: IndexStrategy | str = IndexStrategy.SEQUENCE,
    ) -> None:
        """Build the frozen set and its fuzzy index from ``values``.

        Args:
            values: Source values.  Duplicates are silently ignored; only the first occurrence is retained.
            normalizer: Callable that maps each value to a searchable string,
                or returns ``None`` to exclude it from fuzzy lookup. Excluded
                values are still members of the set. Defaults to the built-in
                normalizer, which is equivalent to ``Normalizer.default()``.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Candidates are ranked by scorer quality;
                exact equality wins equal-score ties, followed by construction
                order. Compatible RapidFuzz metadata permits an immediate
                exact result only when its score is provably optimal.
            scorer_kwargs: Additional keyword arguments forwarded to the scorer
                on every RapidFuzz call.  ``None`` passes no extra arguments.
            scorer_type: Interpretation of scorer output.  Use ``DISTANCE``
                when the scorer returns edit distances (lower = more similar);
                use ``SIMILARITY`` for percentage-like scores (higher = more similar).
            score_cutoff: Exclusion threshold for fuzzy candidates.  For
                ``SIMILARITY`` scorers, candidates below this score are
                excluded; for ``DISTANCE`` scorers, candidates above this
                distance are excluded. ``None`` disables the cutoff.
                Defaults to 80.
            score_hint: Expected score forwarded to RapidFuzz as an optional
                implementation-selection hint. Defaults to ``None``.
            strategy: Index storage strategy used by the fuzzy index. ``SEQUENCE`` is the
                default. ``KEYED`` can reduce frozen build cost and memory,
                uses keyed choices, and returns fuzzy matches with ``index=None``.

        Raises:
            TypeError: If ``normalizer``, ``scorer``, or ``scorer_kwargs`` is
                invalid, ``scorer_type`` is not a ``ScorerType`` member, or
                ``score_cutoff``/``score_hint`` is not numeric or ``None``.
            ValueError: If ``strategy`` is not a supported index strategy.

        Side Effects:
            Builds the fuzzy index eagerly.
        """
        unique_values = dict.fromkeys(values)
        self._strategy = _coerce_index_strategy(strategy)
        self._is_keyed = self._strategy == IndexStrategy.KEYED
        self._values: tuple[T, ...] | None = None
        self._data = frozenset(unique_values)
        index_kwargs = {
            "normalizer": normalizer,
            "scorer": scorer,
            "scorer_kwargs": scorer_kwargs,
            "scorer_type": scorer_type,
            "score_cutoff": score_cutoff,
            "score_hint": score_hint,
        }
        if self._is_keyed:
            keyed_values = tuple(unique_values)
            self._values = keyed_values
            self._index = ImmutableFuzzyKeyedIndex(keyed_values, **index_kwargs)
        else:
            self._index = FuzzySequenceIndex(unique_values.keys(), **index_kwargs)

    def __iter__(self) -> Iterator[T]:
        """Iterate over values in construction order.

        Returns:
            Iterator over values in construction order.
        """

        return iter(self._ordered_values())

    def __len__(self) -> int:
        """Return the number of unique values.

        Returns:
            Number of unique values stored in the set.
        """

        return len(self._data)

    def __or__[U: Hashable](self, other: Set[U]) -> FrozenFuzzySet[T | U] | NotImplementedType:
        """Return the union with ``other`` while preserving fuzzy config.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``union()`` for an arbitrary iterable.

        Args:
            other: Set-like value to union with this set.

        Returns:
            New frozen fuzzy set containing values from this set followed by values from ``other``, or
            ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        return self._new_from_values([*self, *other])

    def __rand__[U: Hashable](self, other: Set[U]) -> FrozenFuzzySet[U] | NotImplementedType:
        """Return values from ``other`` that also exist in this set.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``intersection()`` for an arbitrary iterable.

        Args:
            other: Set-like value to intersect with this set.

        Returns:
            New frozen fuzzy set containing values from ``other`` that are also present in this set, or
            ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        return self._new_from_values(value for value in other if value in self._data)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            Developer-friendly string representation of the set.
        """

        return f"{self.__class__.__name__}({self._ordered_values()!r})"

    def __ror__[U: Hashable](self, other: Set[U]) -> FrozenFuzzySet[U | T] | NotImplementedType:
        """Return the union with ``other`` using ``other`` iteration order first.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``union()`` for an arbitrary iterable.

        Args:
            other: Set-like value to union before this set's values.

        Returns:
            New frozen fuzzy set containing values from ``other`` followed by values from this set, or
            ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        return self._new_from_values([*other, *self])

    def __rsub__[U: Hashable](self, other: Set[U]) -> FrozenFuzzySet[U] | NotImplementedType:
        """Return values from ``other`` that are absent from this set.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted.

        Args:
            other: Set-like value to subtract this set from.

        Returns:
            New frozen fuzzy set containing values from ``other`` that are absent from this set, or
            ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        return self._new_from_values(value for value in other if value not in self._data)

    def __rxor__[U: Hashable](self, other: Set[U]) -> FrozenFuzzySet[U | T] | NotImplementedType:
        """Return values present in exactly one set, starting with ``other`` order.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``symmetric_difference()`` for an arbitrary iterable.

        Args:
            other: Set-like value to compute symmetric difference with.

        Returns:
            New frozen fuzzy set containing values present in exactly one operand, with
            values unique to ``other`` first, or ``NotImplemented`` when ``other`` is not
            ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        other_values = list(other)
        other_set = set(other_values)
        return self._new_from_values(
            [value for value in other_values if value not in self._data]
            + [value for value in self if value not in other_set]
        )

    def __sub__(self, other: Set[object]) -> FrozenFuzzySet[T] | NotImplementedType:
        """Return values from this set that are absent from ``other``.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``difference()`` for an arbitrary iterable.

        Args:
            other: Set-like value to subtract from this set.

        Returns:
            New frozen fuzzy set containing values from this set that are absent from
            ``other``, or ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        other_values = set(other)
        return self._new_from_values(value for value in self if value not in other_values)

    def __xor__[U: Hashable](self, other: Set[U]) -> FrozenFuzzySet[T | U] | NotImplementedType:
        """Return values present in exactly one set while preserving fuzzy config.

        This follows builtin ``set`` operator semantics: only a ``Set``-like
        operand is accepted; use ``symmetric_difference()`` for an arbitrary iterable.

        Args:
            other: Set-like value to compute symmetric difference with.

        Returns:
            New frozen fuzzy set containing values present in exactly one operand, or
            ``NotImplemented`` when ``other`` is not ``Set``-like.
        """

        if not isinstance(other, Set):
            return NotImplemented
        self_values = set(self)
        other_values = list(other)
        other_set = set(other_values)
        return self._new_from_values(
            [value for value in self if value not in other_set]
            + [value for value in other_values if value not in self_values]
        )

    def _config_kwargs(self, *, deepcopy_memo: dict[int, object] | None = None) -> dict[str, Any]:
        """Return fuzzy index configuration for constructing related sets.

        Args:
            deepcopy_memo: Optional memo dictionary used when deep-copying scorer
                keyword arguments as part of an ongoing deepcopy operation.

        Returns:
            Keyword arguments compatible with this set's constructor.
        """

        config = self._index.config_kwargs(deepcopy_memo=deepcopy_memo)
        config["strategy"] = self._strategy
        return config

    def _new_from_values[U: Hashable](self, values: Iterable[U]) -> FrozenFuzzySet[U]:
        """Build a new set with the same fuzzy configuration.

        Args:
            values: Values used to build the new set.

        Returns:
            New frozen fuzzy set containing ``values`` and preserving this set's fuzzy configuration.
        """

        return self.__class__(values, **self._config_kwargs())

    def _ordered_values(self) -> tuple[T, ...]:
        """Return stored values in deterministic fuzzy tie-breaking order.

        Returns:
            Stored values in construction order.
        """

        if self._values is not None:
            return self._values
        index = self._index
        assert isinstance(index, FuzzySequenceIndex)
        return index.values

    # noinspection PyMethodMayBeStatic
    def _value_match(self, match: Match[T] | ValueMatch[T]) -> Match[T]:
        """Return the public value-match shape for the selected strategy.

        Args:
            match: Internal sequence or keyed index match to convert.

        Returns:
            Public value match with set-level index semantics.
        """

        return Match(
            value=match.value,
            score=match.score,
            index=None,
            query=match.query,
            normalized_query=match.normalized_query,
            normalized_value=match.normalized_value,
        )

    def copy(self) -> FrozenFuzzySet[T]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same values and fuzzy configuration.
        """

        return self._new_from_values(self)

    def difference(self, *others: Iterable[object]) -> FrozenFuzzySet[T]:
        """Return values absent from all ``others`` while preserving fuzzy config.

        Args:
            *others: Iterables containing values to exclude from this set.

        Returns:
            New frozen fuzzy set containing values from this set that are absent from every iterable in ``others``.

        Raises:
            TypeError: If an operand yields an unhashable value.
        """

        excluded: set[object] = set()
        for other in others:
            excluded.update(other)
        return self._new_from_values(value for value in self if value not in excluded)

    def fuzzy_contains(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> bool:
        """Return whether any value fuzzy-matches the query above the score cutoff.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``True`` if any value fuzzy-matches ``query`` above the score cutoff, otherwise ``False``.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Candidates are ranked by scorer quality. Exact equality wins
            equal-score ties, followed by construction order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return (
            self.fuzzy_find_one(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            is not None
        )

    def fuzzy_find_many(
        self,
        query: object,
        *,
        limit: int | None = 5,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[T]]:
        """Return up to ``limit`` best fuzzy matches in scorer-defined order.

        Similarity scores are ordered from highest to lowest; distance scores
        are ordered from lowest to highest. Among equal scores, values equal
        to the query precede non-exact values, followed by construction order.

        Args:
            query: Value to search for.
            limit: Maximum number of matches. ``None`` returns all candidates above the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Fuzzy matches ordered from best to worst according to
            ``scorer_type``.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return [
            self._value_match(match)
            for match in self._index.find_many(
                query,
                limit=limit,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        ]

    def fuzzy_find_many_batch(
        self,
        queries: Iterable[object],
        *,
        limit: int | None = 5,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[list[Match[T]]]:
        """Return fuzzy match lists for multiple queries.

        Args:
            queries: Query values to search for in order.
            limit: Maximum number of matches per query. ``None`` returns all
                candidates above the score cutoff for each query.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One match list per query, preserving query order. Within each
            list, scorer quality is primary, exact equality breaks equal-score
            ties, and construction order breaks remaining ties.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return [
            self.fuzzy_find_many(
                query,
                limit=limit,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            for query in queries
        ]

    def fuzzy_find_one(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Match[T] | None:
        """Return the best fuzzy match above the score cutoff, or ``None``.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Best fuzzy match for ``query``, or ``None`` when no acceptable match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Candidates are ranked by scorer quality. Exact equality wins
            equal-score ties, followed by construction order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        match = self._index.find_one(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        return None if match is None else self._value_match(match)

    def fuzzy_find_one_batch(
        self,
        queries: Iterable[object],
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[T] | None]:
        """Return the best fuzzy match for each query.

        Args:
            queries: Query values to search for in order.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One best match or ``None`` per query, preserving query order.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Each query is ranked by scorer quality. Exact equality wins
            equal-score ties, followed by construction order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        index = self._index
        if not self._is_keyed:
            batch = index.find_one_batch(  # type: ignore[union-attr]
                queries,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            return [None if match is None else self._value_match(match) for match in batch]
        return [
            self.fuzzy_find_one(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            for query in queries
        ]

    def fuzzy_find_one_batch_cdist(
        self,
        queries: Iterable[object],
        *,
        query_chunk_size: int = 32,
        choice_chunk_size: int = 1000,
        workers: int = 1,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[T] | None]:
        """Return best value matches through bounded ``cdist`` scoring.

        Args:
            queries: Query values in result order.
            query_chunk_size: Maximum queries in one matrix block.
            choice_chunk_size: Maximum candidates in one matrix block.
            workers: RapidFuzz worker setting passed to ``process.cdist``.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One best value match or ``None`` per query.

        Raises:
            TypeError: If a matching override or chunk size has an invalid type.
            ValueError: If a chunk size is less than 1, or ``scorer`` is overridden
                without ``scorer_type`` and has no compatible RapidFuzz metadata.
            NotImplementedError: If ``IndexStrategy.KEYED`` is selected. Use
                ``IndexStrategy.SEQUENCE`` or call ``fuzzy_find_one_batch`` instead.
            ModuleNotFoundError: If matrix scoring is required but NumPy is
                not installed through ``rapidfuzz-collections[cdist]``.

        Notes:
            Each query is ranked by scorer quality. Exact equality wins
            equal-score ties, followed by construction order. Compatible
            RapidFuzz metadata permits an immediate exact result before matrix
            scoring only when its score is provably optimal.
        """

        if self._is_keyed:
            raise NotImplementedError(
                "cdist is not available for IndexStrategy.KEYED; "
                "use IndexStrategy.SEQUENCE or call fuzzy_find_one_batch() instead"
            )
        validate_chunk_size(query_chunk_size, "query_chunk_size")
        validate_chunk_size(choice_chunk_size, "choice_chunk_size")

        index = self._index
        cdist_results = index.find_one_batch_cdist(  # type: ignore[union-attr]
            queries,
            query_chunk_size=query_chunk_size,
            choice_chunk_size=choice_chunk_size,
            workers=workers,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        return [None if match is None else self._value_match(match) for match in cdist_results]

    def fuzzy_get(
        self,
        query: object,
        default: T | None = None,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> T | None:
        """Return the original value of the best fuzzy match, or ``default``.

        Args:
            query: Value to search for.
            default: Value to return when no match meets the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Original value of the best fuzzy match, or ``default`` when no acceptable match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Candidates are ranked by scorer quality. Exact equality wins
            equal-score ties, followed by construction order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        match = self.fuzzy_find_one(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        if match is None:
            return default
        return match.value

    def fuzzy_get_batch(
        self,
        queries: Iterable[object],
        *,
        default: T | None = None,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[T | None]:
        """Return original values for the best fuzzy matches for multiple queries.

        Args:
            queries: Query values to search for in order.
            default: Value to return for each query with no match.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One matched value or ``default`` per query, preserving query order.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Each query is ranked by scorer quality. Exact equality wins
            equal-score ties, followed by construction order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return [
            default if match is None else match.value
            for match in self.fuzzy_find_one_batch(
                queries,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        ]

    def fuzzy_iter_scores(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Iterator[Match[T] | None]:
        """Yield one scorer result per stored value without allocating a result list.

        This performs fuzzy scoring for every searchable value and does not
        apply exact-match shortcuts. Position order matches construction order.

        Args:
            query: Value to score against every stored value.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Yields:
            ``Match`` for values above the score cutoff and ``None`` for
            unsearchable or rejected positions.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        index = self._index
        if self._is_keyed:
            matches = index.iter_scores(
                self._ordered_values(),
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        else:
            matches = index.iter_scores(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        return (None if match is None else self._value_match(match) for match in matches)

    def fuzzy_score_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[T] | None]:
        """Return one result per stored value, ``None`` for non-matching positions.

        Unlike ``fuzzy_find_many``, the result length always equals ``len(self)``.
        Position ``i`` corresponds to the value at insertion-order position ``i``.
        Exact shortcuts are not applied; all searchable values are scored with
        the configured scorer. Scorers without compatible RapidFuzz metadata use
        a direct position-aligned pass without ranking.

        Args:
            query: Value to score against every stored value.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``Match`` at position ``i`` if the value scored above the score
            cutoff, ``None`` if it did not match or is unsearchable.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        index = self._index
        if self._is_keyed:
            matches = index.score_all(
                self._ordered_values(),
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        else:
            matches = index.score_all(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        return [None if match is None else self._value_match(match) for match in matches]

    def intersection(self, *others: Iterable[object]) -> FrozenFuzzySet[T]:
        """Return values found in every ``other`` while preserving fuzzy config.

        Args:
            *others: Iterables containing values that must also contain each retained value.

        Returns:
            New frozen fuzzy set containing values from this set that are present in every iterable in ``others``.

        Raises:
            TypeError: If an operand yields an unhashable value.
        """

        required = [set(other) for other in others]
        if not required:
            return self.copy()
        return self._new_from_values(value for value in self if all(value in other for other in required))

    def isdisjoint(self, other: Iterable[object]) -> bool:
        """Return whether this set has no values in common with ``other``.

        Args:
            other: Iterable of candidate values.

        Returns:
            ``True`` if the operands have no values in common, otherwise ``False``.

        Raises:
            TypeError: If ``other`` yields an unhashable value.
        """

        for value in other:
            hash(value)
            if value in self._data:
                return False
        return True

    def issubset(self, other: Iterable[object]) -> bool:
        """Return whether every value in this set is present in ``other``.

        Args:
            other: Iterable of candidate values.

        Returns:
            ``True`` if this set is a subset of ``other``, otherwise ``False``.

        Raises:
            TypeError: If an operand yields an unhashable value.
        """

        other_values = set(other)
        return all(value in other_values for value in self)

    def issuperset(self, other: Iterable[object]) -> bool:
        """Return whether every value in ``other`` is present in this set.

        Args:
            other: Iterable of candidate values.

        Returns:
            ``True`` if this set is a superset of ``other``, otherwise ``False``.

        Raises:
            TypeError: If an operand yields an unhashable value.
        """

        for value in other:
            hash(value)
            if value not in self._data:
                return False
        return True

    def symmetric_difference[U: Hashable](self, other: Iterable[U]) -> FrozenFuzzySet[T | U]:
        """Return values present in exactly one operand while preserving fuzzy config.

        Args:
            other: Values to compute symmetric difference with.

        Returns:
            New frozen fuzzy set containing values present in exactly one operand.

        Raises:
            TypeError: If an operand yields an unhashable value.
        """

        self_values = set(self)
        other_values = list(other)
        other_set = set(other_values)
        return self._new_from_values(
            [value for value in self if value not in other_set]
            + [value for value in other_values if value not in self_values]
        )

    def union[U: Hashable](self, *others: Iterable[U]) -> FrozenFuzzySet[T | U]:
        """Return the union with all ``others`` while preserving fuzzy config.

        Args:
            *others: Iterables containing values to add after this set's values.

        Returns:
            New frozen fuzzy set containing values from this set followed by values from every iterable in ``others``.

        Raises:
            TypeError: If an operand yields an unhashable value.
        """

        values: list[T] = list(self)
        for other in others:
            values.extend(other)
        return self._new_from_values(values)

    def with_config(
        self,
        *,
        normalizer: Callable[[object], str | None] | None = _UNCHANGED,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
        strategy: IndexStrategy | str = _UNCHANGED,
    ) -> FrozenFuzzySet[T]:
        """Return a new frozen set with updated matching configuration.

        Passing ``None`` restores the built-in default normalizer, clears
        scorer keyword arguments, or disables cutoff filtering, according to
        the parameter selected. It also clears an existing ``score_hint``.
        Omitted parameters preserve current settings.

        Args:
            normalizer: Replacement normalizer, ``None`` to restore the default
                normalizer, or ``_UNCHANGED`` to preserve the current normalizer.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Replacement RapidFuzz scorer callable, or ``_UNCHANGED`` to
                preserve the current scorer.
            scorer_kwargs: Replacement scorer keyword arguments, ``None`` to clear
                them, or ``_UNCHANGED`` to preserve the current scorer keyword arguments.
            scorer_type: Replacement scorer type, or ``_UNCHANGED`` to preserve the current scorer type.
            score_cutoff: Replacement score cutoff, ``None`` to disable cutoff
                filtering, or ``_UNCHANGED`` to preserve the current cutoff.
            score_hint: Replacement score hint, ``None`` to clear it, or
                ``_UNCHANGED`` to preserve the current score hint.
            strategy: Replacement index strategy, or ``_UNCHANGED`` to preserve the current strategy.

        Returns:
            New frozen set containing the same values under the resulting matching configuration.

        Raises:
            TypeError: If an overridden configuration value fails constructor validation.
            ValueError: If ``strategy`` is replaced with an unsupported index strategy.

        Notes:
            This method leaves the source set unchanged and constructs a new
            fuzzy index from its construction order.
        """

        return self.__class__(
            self,
            **_apply_config_overrides(
                self._config_kwargs(),
                normalizer=normalizer,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
                strategy=strategy,
            ),
        )
