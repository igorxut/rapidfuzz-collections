from collections.abc import Callable, Iterable, Iterator, MutableSequence
from copy import deepcopy as _deepcopy
from reprlib import recursive_repr
from types import NotImplementedType
from typing import Any, Self, SupportsIndex, overload

from rapidfuzz.fuzz import WRatio

from ..configuration import _UNCHANGED, _apply_config_overrides
from ..enums import ScorerType
from ..indexes import MutableFuzzySequenceIndex, Scorer
from ..matching import Match


class FuzzyList[T](MutableSequence[T]):
    """A list-like mutable collection with fuzzy lookup over values.

    Values are stored unchanged in insertion order; duplicates are allowed.
    Standard list operations (indexing, slicing, insertion, deletion, and
    assignment) behave identically to a regular Python list.

    Appending values updates the fuzzy index incrementally. Deletions use
    incremental dense or sparse paths when practical, while mutations that
    replace or reorder existing positions mark the index dirty for a lazy
    rebuild before the next fuzzy query.

    Notes:
        Index maintenance selects between incremental updates and lazy full
        rebuilds according to the mutation shape and affected positions.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = ("_index",)

    def __add__[U](self, other: list[U] | FuzzyList[U]) -> FuzzyList[T | U] | NotImplementedType:
        """Return a new fuzzy list concatenated with a list-like value.

        This follows builtin ``list`` operator semantics: use ``extend`` or
        ``+=`` when appending values from an arbitrary iterable.

        Args:
            other: List-like value to concatenate after this list.

        Returns:
            New fuzzy list containing values from this list followed by ``other``, or
            ``NotImplemented`` when ``other`` is not list-like.
        """

        if not isinstance(other, (list, FuzzyList)):
            return NotImplemented
        return self.__class__([*self, *other], **self._config_kwargs())

    def __copy__(self) -> FuzzyList[T]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same values and fuzzy configuration.
        """

        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> FuzzyList[T]:
        """Return a deep copy preserving fuzzy configuration.

        Args:
            memo: Deepcopy memo dictionary used to preserve object identity during recursive copying.

        Returns:
            Deep copy with copied values and preserved fuzzy configuration.
        """

        instance = self.__new__(self.__class__)
        memo[id(self)] = instance
        instance.__init__(
            _deepcopy(self._index.values, memo),
            **self._config_kwargs(deepcopy_memo=memo),
        )
        return instance

    def __delitem__(self, index: int | slice) -> None:
        """Delete a value or slice and update the fuzzy index.

        Args:
            index: Index or slice to delete.

        Side Effects:
            Removes value(s) from the list and updates the fuzzy index.
        """

        self._index.delete_at(index)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        """Compare stored values using builtin list equality semantics."""

        if isinstance(other, FuzzyList):
            return self._index.values == other._index.values
        if isinstance(other, list):
            return list(self._index) == other
        return NotImplemented

    def __ge__(self, other: object) -> bool | NotImplementedType:
        """Compare stored values using builtin list ordering semantics."""

        if isinstance(other, FuzzyList):
            return self._index.values >= other._index.values
        if isinstance(other, list):
            return list(self._index) >= other
        return NotImplemented

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        """Return a value or slice.

        Args:
            index: Index or slice to retrieve.

        Returns:
            Value for an integer index, or a list of values for a slice.
        """

        if isinstance(index, slice):
            return self._index[index]
        return self._index[index]

    def __gt__(self, other: object) -> bool | NotImplementedType:
        """Compare stored values using builtin list ordering semantics."""

        if isinstance(other, FuzzyList):
            return self._index.values > other._index.values
        if isinstance(other, list):
            return list(self._index) > other
        return NotImplemented

    def __iadd__(self, other: Iterable[T]) -> Self:
        """Append values from ``other`` in place.

        Args:
            other: Values to append.

        Returns:
            This list after appending ``other``.

        Side Effects:
            Appends values to this list and updates the fuzzy index.
        """

        self.extend(other)
        return self

    def __imul__(self, value: SupportsIndex) -> Self:
        """Repeat the list in place and mark the fuzzy index dirty.

        Args:
            value: Repetition count.

        Returns:
            This list after in-place repetition.

        Side Effects:
            Replaces the stored values with their repetition and marks the fuzzy
            index dirty for lazy rebuilding.
        """

        repeated_values = self._index.values * value
        self._index.replace_at(slice(None), repeated_values)
        return self

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
        """Build the list and its fuzzy index from ``values``.

        Args:
            values: Initial values.  Stored in order; duplicates are preserved.
            normalizer: Callable that maps each value to a searchable string,
                or returns ``None`` to exclude it from fuzzy lookup.  Excluded
                values are still accessible by list index. Defaults to the
                built-in normalizer, which is equivalent to ``Normalizer.default()``.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Candidates are ranked by scorer quality;
                hashable exact equality wins equal-score ties, followed by source
                position. Compatible RapidFuzz metadata permits an immediate
                exact result only when its score is provably optimal.
            scorer_kwargs: Additional keyword arguments forwarded to the scorer
                on every RapidFuzz call.  Useful for scorers with extra
                parameters such as ``Levenshtein.distance`` with custom
                ``weights``. ``None`` passes no extra arguments.
            scorer_type: Interpretation of scorer output.  Use ``DISTANCE``
                when the scorer returns edit distances (lower = more similar);
                use ``SIMILARITY`` for percentage-like scores (higher = more similar).
            score_cutoff: Exclusion threshold for fuzzy candidates.  For
                ``SIMILARITY`` scorers, candidates below this score are
                excluded; for ``DISTANCE`` scorers, candidates above this
                distance are excluded. ``None`` disables the cutoff. Defaults to 80.
            score_hint: Expected score forwarded to RapidFuzz as an optional
                implementation-selection hint. Defaults to ``None``.

        Raises:
            TypeError: If ``normalizer``, ``scorer``, or ``scorer_kwargs`` is
                invalid, ``scorer_type`` is not a ``ScorerType`` member, or
                ``score_cutoff``/``score_hint`` is not numeric or ``None``.

        Side Effects:
            Builds the fuzzy index eagerly from ``values``.
        """
        self._index: MutableFuzzySequenceIndex[T] = MutableFuzzySequenceIndex(
            values,
            normalizer=normalizer,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

    def __iter__(self) -> Iterator[T]:
        """Iterate over values.

        Returns:
            Iterator over stored values in list order.
        """

        return iter(self._index)

    def __le__(self, other: object) -> bool | NotImplementedType:
        """Compare stored values using builtin list ordering semantics."""

        if isinstance(other, FuzzyList):
            return self._index.values <= other._index.values
        if isinstance(other, list):
            return list(self._index) <= other
        return NotImplemented

    def __len__(self) -> int:
        """Return the number of values.

        Returns:
            Number of values stored in the list.
        """

        return len(self._index)

    def __lt__(self, other: object) -> bool | NotImplementedType:
        """Compare stored values using builtin list ordering semantics."""

        if isinstance(other, FuzzyList):
            return self._index.values < other._index.values
        if isinstance(other, list):
            return list(self._index) < other
        return NotImplemented

    def __mul__(self, value: SupportsIndex) -> FuzzyList[T]:
        """Return a repeated fuzzy list preserving fuzzy configuration.

        Args:
            value: Repetition count.

        Returns:
            New fuzzy list containing repeated values and preserving fuzzy configuration.
        """

        return self.__class__(self._index.values * value, **self._config_kwargs())

    def __radd__[U](self, other: list[U] | FuzzyList[U]) -> FuzzyList[U | T] | NotImplementedType:
        """Return a fuzzy list with ``other`` values before this list.

        Args:
            other: List-like value to concatenate before this list.

        Returns:
            New fuzzy list preserving this list's fuzzy configuration, or
            ``NotImplemented`` when ``other`` is not list-like.
        """

        if not isinstance(other, (list, FuzzyList)):
            return NotImplemented
        return self.__class__([*other, *self], **self._config_kwargs())

    @recursive_repr()
    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            Developer-friendly string representation of the list.
        """

        return f"{self.__class__.__name__}({list(self._index)!r})"

    def __reversed__(self) -> Iterator[T]:
        """Iterate over values in reverse order.

        Returns:
            Iterator over stored values in reverse list order.
        """

        return reversed(self._index)

    def __rmul__(self, value: SupportsIndex) -> FuzzyList[T]:
        """Return a repeated fuzzy list preserving fuzzy configuration.

        Args:
            value: Repetition count.

        Returns:
            New fuzzy list containing repeated values and preserving fuzzy configuration.
        """

        return self * value

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        """Set a value or slice and mark the fuzzy index dirty.

        Args:
            index: Index or slice to assign.
            value: Replacement value for an integer index, or replacement values for a slice.

        Side Effects:
            Updates stored value(s) and marks the fuzzy index dirty.
        """

        self._index.replace_at(index, value)

    def _config_kwargs(self, *, deepcopy_memo: dict[int, object] | None = None) -> dict[str, Any]:
        """Return fuzzy index configuration for constructing related lists.

        Args:
            deepcopy_memo: Optional memo dictionary used when deep-copying scorer
                keyword arguments as part of an ongoing deepcopy operation.

        Returns:
            Keyword arguments compatible with this list's constructor.
        """

        return self._index.config_kwargs(deepcopy_memo=deepcopy_memo)

    def append(self, value: T) -> None:
        """Append ``value`` to the end of the list.

        The fuzzy index is updated incrementally in O(1) time; no full rebuild is triggered.

        Args:
            value: Value to append.

        Side Effects:
            Adds the value to the list and updates the fuzzy index without triggering a full rebuild.
        """

        self._index.append(value)

    def copy(self) -> FuzzyList[T]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same values and fuzzy configuration.
        """

        return self.__class__(self._index, **self._config_kwargs())

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
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return self._index.contains(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

    def fuzzy_count(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> int:
        """Return how many values fuzzy-match the query above the score cutoff.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Number of values that fuzzy-match ``query`` above the score cutoff.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return len(
            self._index.find_many(
                query,
                limit=None,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        )

    def fuzzy_discard(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> None:
        """Remove the best fuzzy match above the score cutoff; no-op if no match.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes the matched value and updates the fuzzy index. Supported
            compact deletion paths avoid a full rebuild.

        Notes:
            The removed value is selected by scorer quality. Hashable exact equality
            wins equal-score ties, followed by source position. Compatible
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
        if match is None or match.index is None:
            return
        self._index.delete_at(match.index)

    def fuzzy_discard_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> int:
        """Remove every value that fuzzy-matches the query above the score cutoff.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Number of removed values.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes all matched values and updates the fuzzy index. Large or
            complex positional deletions may defer one rebuild until the next fuzzy query.
        """

        matches = self._index.find_many(
            query,
            limit=None,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        if not matches:
            return 0
        self._index.delete_at_positions({m.index for m in matches if m.index is not None})
        return len(matches)

    def fuzzy_find_index(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> int:
        """Return the source index of the best fuzzy match, or raise ``ValueError``.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Source index of the best fuzzy match.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If no value matches, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
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
        if match is None or match.index is None:
            # noinspection PyStringConversionWithoutDunderMethod
            raise ValueError(f"{repr(query)} has no fuzzy match in the collection")
        return match.index

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
        are ordered from lowest to highest. Among equal scores, hashable values
        equal to the query precede non-exact values, followed by source position.

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

        return self._index.find_many(
            query,
            limit=limit,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

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
            list, scorer quality is primary, hashable exact equality breaks equal-score
            ties, and source position breaks remaining ties.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return self._index.find_many_batch(
            queries,
            limit=limit,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

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
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return self._index.find_one(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

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
            Each query is ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return self._index.find_one_batch(
            queries,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

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
            ModuleNotFoundError: If matrix scoring is required but NumPy is
                not installed through ``rapidfuzz-collections[cdist]``.

        Notes:
            Each query is ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result before matrix
            scoring only when its score is provably optimal.
        """

        return self._index.find_one_batch_cdist(
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
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
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
            Each query is ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return [
            default if m is None else m.value
            for m in self._index.find_one_batch(
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
        apply exact-match shortcuts. Position order matches list order.

        Args:
            query: Value to score against every stored value.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Yields:
            ``Match`` for values above the score cutoff and ``None`` for unsearchable or rejected positions.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return self._index.iter_scores(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

    def fuzzy_remove(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> None:
        """Remove the best fuzzy match above the score cutoff, or raise ``ValueError``.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If no value matches, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes the matched value and updates the fuzzy index. Supported
            compact deletion paths avoid a full rebuild.

        Notes:
            The removed value is selected by scorer quality. Hashable exact equality
            wins equal-score ties, followed by source position. Compatible
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
        if match is None or match.index is None:
            # noinspection PyStringConversionWithoutDunderMethod
            raise ValueError(f"{repr(query)} has no fuzzy match in the collection")
        self._index.delete_at(match.index)

    def fuzzy_retain_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> int:
        """Remove every value that does not fuzzy-match the query above the score cutoff.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Number of removed values.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes all non-matching values and updates the fuzzy index.
            Retention currently defers one rebuild when values are removed.
        """

        matches = self._index.find_many(
            query,
            limit=None,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        return self._index.keep_at_positions({m.index for m in matches if m.index is not None})

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
        Position ``i`` corresponds to value at index ``i``. Exact shortcuts
        are not applied; all searchable values are scored with the configured
        scorer. Scorers without compatible RapidFuzz metadata use a direct
        position-aligned pass without ranking.

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

        return self._index.score_all(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )

    def insert(self, index: int, value: T) -> None:
        """Insert a value and mark the fuzzy index dirty.

        Args:
            index: Position before which to insert ``value``.
            value: Value to insert.

        Side Effects:
            Inserts ``value`` into this list and marks the fuzzy index dirty for
            lazy rebuilding.
        """

        self._index.insert_at(index, value)

    def sort(
        self,
        *,
        key: Callable[[T], Any] | None = None,
        reverse: bool = False,
    ) -> None:
        """Sort the list in place and rebuild the fuzzy index lazily.

        Args:
            key: Optional one-argument ordering function.
            reverse: Whether to sort in descending order.

        Side Effects:
            Reorders stored values and marks the fuzzy index dirty.
        """

        self._index.sort(key=key, reverse=reverse)

    def with_config(
        self,
        *,
        normalizer: Callable[[object], str | None] | None = _UNCHANGED,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> FuzzyList[T]:
        """Return a new fuzzy list with explicitly updated matching configuration.

        Passing ``None`` restores the built-in default normalizer, clears
        scorer keyword arguments, or disables cutoff filtering, according to
        the parameter selected. It also clears an existing ``score_hint``.
        Omitted parameters preserve current settings.

        Args:
            normalizer: Replacement normalizer, ``None`` to restore the default
                normalizer, or ``_UNCHANGED`` to preserve the current normalizer.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Replacement RapidFuzz scorer callable, or ``_UNCHANGED`` to preserve the current scorer.
            scorer_kwargs: Replacement scorer keyword arguments, ``None`` to clear
                them, or ``_UNCHANGED`` to preserve the current scorer keyword arguments.
            scorer_type: Replacement scorer type, or ``_UNCHANGED`` to preserve the current scorer type.
            score_cutoff: Replacement score cutoff, ``None`` to disable cutoff
                filtering, or ``_UNCHANGED`` to preserve the current cutoff.
            score_hint: Replacement score hint, ``None`` to clear it, or
                ``_UNCHANGED`` to preserve the current score hint.

        Returns:
            New list containing the same value objects under the resulting matching configuration.

        Raises:
            TypeError: If an overridden configuration value fails constructor validation.

        Notes:
            This method leaves the source list unchanged and constructs a new fuzzy index from its current values.
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
            ),
        )
