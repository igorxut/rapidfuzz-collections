from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from copy import deepcopy as _deepcopy
from types import NotImplementedType
from typing import Any

from rapidfuzz.fuzz import WRatio

from ..configuration import _UNCHANGED, _apply_config_overrides, _coerce_index_strategy
from ..enums import IndexStrategy, ScorerType
from ..indexes import FuzzySequenceIndex, ImmutableFuzzyKeyedIndex, Scorer, validate_chunk_size
from ..matching import MappingMatch, Match, ValueMatch


class FrozenFuzzyDict[K, V](Mapping[K, V]):
    """An immutable dict-like collection with fuzzy lookup over keys.

    Keys and values are stored in insertion order.  Standard mapping read
    operations behave identically to a regular Python dict.  Fuzzy lookup
    searches over keys; the associated value is returned alongside the matched
    key.

    The fuzzy key index is built once at construction and never rebuilt.
    Duplicate keys follow standard dict semantics — the last value for each key
    wins and only one key entry is indexed.

    ``strategy`` selects the index storage strategy used by the fuzzy index. ``SEQUENCE`` is the
    default and favors general read-heavy lookups. ``KEYED`` can improve build
    cost and can reduce frozen-index memory for unique hashable keys in
    measured workloads, but point-lookup speed remains workload-dependent.
    Fuzzy results are position-free and expose ``index=None``.

    Notes:
        Choose ``FuzzyDict`` for mutable mappings. Choose this class when the
        mapping is loaded once and then queried many times.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = ("_data", "_is_keyed", "_key_index", "_strategy")

    def __contains__(self, key: object) -> bool:
        """Return whether the exact key exists.

        Args:
            key: Key to test for exact membership.

        Returns:
            ``True`` if ``key`` exists in the mapping, otherwise ``False``.
        """

        return key in self._data

    def __copy__(self) -> FrozenFuzzyDict[K, V]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same items and fuzzy key configuration.
        """

        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenFuzzyDict[K, V]:
        """Return a deep copy preserving fuzzy configuration.

        Args:
            memo: Deepcopy memo dictionary used to preserve object identity during recursive copying.

        Returns:
            Deep copy with copied items and preserved fuzzy key configuration.
        """

        instance = self.__new__(self.__class__)
        memo[id(self)] = instance
        instance.__init__(
            _deepcopy(self._data, memo),
            **self._config_kwargs(deepcopy_memo=memo),
        )
        return instance

    def __getitem__(self, key: K) -> V:
        """Return the value for an exact key.

        Args:
            key: Exact key to look up.

        Returns:
            Value associated with ``key``.
        """

        return self._data[key]

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
        strategy: IndexStrategy | str = IndexStrategy.SEQUENCE,
    ) -> None:
        """Build the frozen mapping and its fuzzy key index from ``items``.

        Args:
            items: Initial key-value pairs as an iterable of 2-tuples or a
                mapping.  Duplicate keys follow standard dict semantics (last write wins).
            normalizer: Callable that maps each key to a searchable string,
                or returns ``None`` to exclude it from fuzzy lookup.  Excluded
                keys are still accessible by exact lookup. Defaults to the
                built-in normalizer, which is equivalent to ``Normalizer.default()``.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Keys are ranked by scorer quality; exact
                equality wins equal-score ties, followed by insertion order.
                Compatible RapidFuzz metadata permits an immediate exact result
                only when its score is provably optimal.
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
                uses keyed choices, and returns fuzzy matches with
                ``index=None``.

        Raises:
            TypeError: If ``normalizer``, ``scorer``, or ``scorer_kwargs`` is
                invalid, ``scorer_type`` is not a ``ScorerType`` member, or
                ``score_cutoff``/``score_hint`` is not numeric or ``None``.
            ValueError: If ``strategy`` is not a supported index strategy.

        Side Effects:
            Builds the fuzzy key index eagerly.
        """
        self._data: dict[K, V] = dict(items)
        self._strategy = _coerce_index_strategy(strategy)
        self._is_keyed = self._strategy == IndexStrategy.KEYED
        index_kwargs = {
            "normalizer": normalizer,
            "scorer": scorer,
            "scorer_kwargs": scorer_kwargs,
            "scorer_type": scorer_type,
            "score_cutoff": score_cutoff,
            "score_hint": score_hint,
        }
        if self._is_keyed:
            self._key_index = ImmutableFuzzyKeyedIndex(self._data.keys(), **index_kwargs)
        else:
            self._key_index = FuzzySequenceIndex(self._data.keys(), **index_kwargs)

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys in insertion order.

        Returns:
            Iterator over keys in insertion order.
        """

        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items.

        Returns:
            Number of key-value pairs stored in the mapping.
        """

        return len(self._data)

    def __or__[K2, V2](
        self,
        items: Mapping[K2, V2],
    ) -> FrozenFuzzyDict[K | K2, V | V2] | NotImplementedType:
        """Return a merged mapping preserving fuzzy configuration.

        Only mapping operands are accepted; use the constructor to merge an
        arbitrary iterable of pairs.

        Args:
            items: Mapping to merge after this mapping.

        Returns:
            New frozen fuzzy mapping containing this mapping's items updated with ``items``, or
            ``NotImplemented`` when ``items`` is not a mapping.
        """

        if not isinstance(items, Mapping):
            return NotImplemented
        data = dict(self._data)
        data.update(items)
        return self.__class__(data, **self._config_kwargs())

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            Developer-friendly string representation of the mapping.
        """

        return f"{self.__class__.__name__}({self._data!r})"

    def __reversed__(self) -> Iterator[K]:
        """Iterate over keys in reverse insertion order.

        Returns:
            Iterator over keys in reverse insertion order.
        """

        return reversed(self._data)

    def __ror__[K2, V2](
        self,
        items: Mapping[K2, V2],
    ) -> FrozenFuzzyDict[K | K2, V | V2] | NotImplementedType:
        """Return a merged mapping with ``items`` ordered before this mapping.

        Only mapping operands are accepted; use the constructor to merge an
        arbitrary iterable of pairs.

        Args:
            items: Mapping to merge before this mapping.

        Returns:
            New frozen fuzzy mapping containing ``items`` updated with this mapping's
            items, or ``NotImplemented`` when ``items`` is not a mapping.
        """

        if not isinstance(items, Mapping):
            return NotImplemented
        data = dict(items)
        data.update(self._data)
        return self.__class__(data, **self._config_kwargs())

    def _config_kwargs(self, *, deepcopy_memo: dict[int, object] | None = None) -> dict[str, Any]:
        """Return fuzzy key-index configuration for constructing related dicts.

        Args:
            deepcopy_memo: Optional memo dictionary used when deep-copying scorer
                keyword arguments as part of an ongoing deepcopy operation.

        Returns:
            Keyword arguments compatible with this mapping's constructor.
        """

        config = self._key_index.config_kwargs(deepcopy_memo=deepcopy_memo)
        config["strategy"] = self._strategy
        return config

    # noinspection PyMethodMayBeStatic
    def _key_match(self, match: Match[K] | ValueMatch[K]) -> Match[K]:
        """Return the public key-match shape for the selected strategy.

        Args:
            match: Internal sequence or keyed index match to convert.

        Returns:
            Public key match with mapping-level index semantics.
        """

        return Match(
            value=match.value,
            score=match.score,
            index=None,
            query=match.query,
            normalized_query=match.normalized_query,
            normalized_value=match.normalized_value,
        )

    def _mapping_match_from_key_match(self, match: Match[K]) -> MappingMatch[K, V]:
        """Build a mapping match from a fuzzy key match.

        Args:
            match: Fuzzy key match to convert.

        Returns:
            Mapping match containing the matched key and associated value.
        """

        return MappingMatch(
            key=match.value,
            value=self._data[match.value],
            score=match.score,
            index=None,
            query=match.query,
            normalized_query=match.normalized_query,
            normalized_key=match.normalized_value,
        )

    def copy(self) -> FrozenFuzzyDict[K, V]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same items and fuzzy key configuration.
        """

        return self.__class__(self._data, **self._config_kwargs())

    @classmethod
    def fromkeys(
        cls,
        keys: Iterable[K],
        value: V | None = None,
        *,
        normalizer: Callable[[object], str | None] | None = None,
        scorer: Scorer = WRatio,
        scorer_kwargs: dict[str, Any] | None = None,
        scorer_type: ScorerType = ScorerType.SIMILARITY,
        score_cutoff: int | float | None = 80,
        score_hint: int | float | None = None,
        strategy: IndexStrategy | str = IndexStrategy.SEQUENCE,
    ) -> FrozenFuzzyDict[K, V | None]:
        """Build a frozen fuzzy mapping from keys that all share one value.

        Args:
            keys: Keys to store in insertion order. Duplicate keys follow standard dict semantics.
            value: Value assigned to every key.
            normalizer: Callable that maps each key to a searchable string,
                or returns ``None`` to exclude it from fuzzy lookup.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Keys are ranked by scorer quality; exact
                equality wins equal-score ties, followed by insertion order.
                Compatible RapidFuzz metadata permits an immediate exact result
                only when its score is provably optimal.
            scorer_kwargs: Additional keyword arguments forwarded to the scorer.
            scorer_type: Interpretation of scorer output as distance or similarity.
            score_cutoff: Exclusion threshold for fuzzy key candidates.
            score_hint: Expected key score forwarded to RapidFuzz as an optional implementation-selection hint.
            strategy: Index storage strategy used by the fuzzy index for the new mapping.

        Returns:
            New frozen fuzzy mapping with the supplied key lookup configuration.

        Raises:
            TypeError: If ``normalizer``, ``scorer``, or ``scorer_kwargs`` is
                invalid, ``scorer_type`` is not a ``ScorerType`` member, or
                ``score_cutoff``/``score_hint`` is not numeric or ``None``.
            ValueError: If ``strategy`` is not a supported index strategy.
        """

        return cls(
            ((key, value) for key in keys),
            normalizer=normalizer,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
            strategy=strategy,
        )

    def fuzzy_contains_key(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> bool:
        """Return whether any key fuzzy-matches the query above the score cutoff.

        Args:
            query: Value to match against keys.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``True`` if any key fuzzy-matches ``query`` above the score cutoff, otherwise ``False``.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Keys are ranked by scorer quality. Exact equality wins equal-score
            ties, followed by insertion order. Compatible RapidFuzz metadata
            permits an immediate exact result only when its score is provably
            optimal.
        """

        return (
            self.fuzzy_find_key(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            is not None
        )

    def fuzzy_find_item(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> MappingMatch[K, V] | None:
        """Return the best fuzzy key/value match above the score cutoff, or ``None``.

        Args:
            query: Value to match against keys.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Best key/value match for ``query``, or ``None`` when no acceptable key match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Keys are ranked by scorer quality. Exact equality wins equal-score
            ties, followed by insertion order. Compatible RapidFuzz metadata
            permits an immediate exact result only when its score is provably
            optimal.
        """

        match = self.fuzzy_find_key(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        if match is None:
            return None
        return self._mapping_match_from_key_match(match)

    def fuzzy_find_item_batch(
        self,
        queries: Iterable[object],
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[MappingMatch[K, V] | None]:
        """Return the best fuzzy key/value match for each query.

        Args:
            queries: Query values to match against keys in order.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One mapping match or ``None`` per query, preserving query order.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            For each query, keys are ranked by scorer quality. Exact equality
            wins equal-score ties, followed by insertion order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return [
            None if match is None else self._mapping_match_from_key_match(match)
            for match in self.fuzzy_find_key_batch(
                queries,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        ]

    def fuzzy_find_item_batch_cdist(
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
    ) -> list[MappingMatch[K, V] | None]:
        """Return best key/value matches through bounded ``cdist`` scoring.

        Args:
            queries: Query values to match against keys in result order.
            query_chunk_size: Maximum queries in one matrix block.
            choice_chunk_size: Maximum keys in one matrix block.
            workers: RapidFuzz worker setting passed to ``process.cdist``.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One mapping match or ``None`` per query.

        Raises:
            TypeError: If a matching override or chunk size has an invalid type.
            ValueError: If a chunk size is less than 1, or ``scorer`` is overridden
                without ``scorer_type`` and has no compatible RapidFuzz metadata.
            NotImplementedError: If ``IndexStrategy.KEYED`` is selected. Use
                ``IndexStrategy.SEQUENCE`` or call ``fuzzy_find_item_batch`` instead.
            ModuleNotFoundError: If matrix scoring is required but NumPy is
                not installed through ``rapidfuzz-collections[cdist]``.

        Notes:
            For each query, keys are ranked by scorer quality. Exact equality
            wins equal-score ties, followed by insertion order. Compatible
            RapidFuzz metadata permits an immediate exact result before matrix
            scoring only when its score is provably optimal.
        """

        return [
            None if match is None else self._mapping_match_from_key_match(match)
            for match in self.fuzzy_find_key_batch_cdist(
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
        ]

    def fuzzy_find_items(
        self,
        query: object,
        *,
        limit: int | None = 5,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[MappingMatch[K, V]]:
        """Return up to ``limit`` key/value pairs in scorer-defined order.

        Similarity scores are ordered from highest to lowest; distance scores
        are ordered from lowest to highest. Among equal scores, keys equal to
        the query precede non-exact keys, followed by insertion order.

        Args:
            query: Value to match against keys.
            limit: Maximum number of results. ``None`` returns all candidates above the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Key/value matches ordered from best to worst according to
            ``scorer_type``.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return [
            self._mapping_match_from_key_match(match)
            for match in self.fuzzy_find_keys(
                query,
                limit=limit,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        ]

    def fuzzy_find_items_batch(
        self,
        queries: Iterable[object],
        *,
        limit: int | None = 5,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[list[MappingMatch[K, V]]]:
        """Return fuzzy key/value match lists for multiple queries.

        Args:
            queries: Query values to match against keys in order.
            limit: Maximum number of matches per query. ``None`` returns all key candidates
            above the score cutoff for each query.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One mapping-match list per query, preserving query order. Within
            each list, scorer quality is primary, exact key equality breaks
            equal-score ties, and insertion order breaks remaining ties.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return [
            [self._mapping_match_from_key_match(match) for match in matches]
            for matches in self.fuzzy_find_keys_batch(
                queries,
                limit=limit,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        ]

    def fuzzy_find_key(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Match[K] | None:
        """Return the best fuzzy key match above the score cutoff, or ``None``.

        Args:
            query: Value to match against keys.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Best fuzzy key match for ``query``, or ``None`` when no acceptable key match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Keys are ranked by scorer quality. Exact equality wins equal-score
            ties, followed by insertion order. Compatible RapidFuzz metadata
            permits an immediate exact result only when its score is provably
            optimal.
        """

        match = self._key_index.find_one(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        return None if match is None else self._key_match(match)

    def fuzzy_find_key_batch(
        self,
        queries: Iterable[object],
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[K] | None]:
        """Return the best fuzzy key match for each query.

        Args:
            queries: Query values to match against keys in order.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One key match or ``None`` per query, preserving query order.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            For each query, keys are ranked by scorer quality. Exact equality
            wins equal-score ties, followed by insertion order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        index = self._key_index
        if not self._is_keyed:
            batch = index.find_one_batch(  # type: ignore[union-attr]
                queries,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            return [None if match is None else self._key_match(match) for match in batch]
        return [
            self.fuzzy_find_key(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            for query in queries
        ]

    def fuzzy_find_key_batch_cdist(
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
    ) -> list[Match[K] | None]:
        """Return best key matches through bounded ``cdist`` scoring.

        Requires ``IndexStrategy.SEQUENCE``.

        Args:
            queries: Query values to match against keys in result order.
            query_chunk_size: Maximum queries in one matrix block.
            choice_chunk_size: Maximum keys in one matrix block.
            workers: RapidFuzz worker setting passed to ``process.cdist``.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One key match or ``None`` per query, preserving query order.

        Raises:
            TypeError: If a matching override or chunk size has an invalid type.
            ValueError: If a chunk size is less than 1, or ``scorer`` is overridden
                without ``scorer_type`` and has no compatible RapidFuzz metadata.
            NotImplementedError: If ``IndexStrategy.KEYED`` is selected. Use
                ``IndexStrategy.SEQUENCE`` or call ``fuzzy_find_key_batch`` instead.
            ModuleNotFoundError: If matrix scoring is required but NumPy is
                not installed through ``rapidfuzz-collections[cdist]``.

        Notes:
            For each query, keys are ranked by scorer quality. Exact equality
            wins equal-score ties, followed by insertion order. Compatible
            RapidFuzz metadata permits an immediate exact result before matrix
            scoring only when its score is provably optimal.
        """

        if self._is_keyed:
            raise NotImplementedError(
                "cdist is not available for IndexStrategy.KEYED; "
                "use IndexStrategy.SEQUENCE or call fuzzy_find_key_batch() instead"
            )
        validate_chunk_size(query_chunk_size, "query_chunk_size")
        validate_chunk_size(choice_chunk_size, "choice_chunk_size")

        index = self._key_index
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
        return [None if match is None else self._key_match(match) for match in cdist_results]

    def fuzzy_find_keys(
        self,
        query: object,
        *,
        limit: int | None = 5,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[K]]:
        """Return up to ``limit`` fuzzy key matches in scorer-defined order.

        Similarity scores are ordered from highest to lowest; distance scores
        are ordered from lowest to highest. Among equal scores, keys equal to
        the query precede non-exact keys, followed by insertion order.

        Args:
            query: Value to match against keys.
            limit: Maximum number of matches. ``None`` returns all candidates above the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Fuzzy key matches ordered from best to worst according to
            ``scorer_type``.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return [
            self._key_match(match)
            for match in self._key_index.find_many(
                query,
                limit=limit,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
        ]

    def fuzzy_find_keys_batch(
        self,
        queries: Iterable[object],
        *,
        limit: int | None = 5,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[list[Match[K]]]:
        """Return fuzzy key match lists for multiple queries.

        Args:
            queries: Query values to match against keys in order.
            limit: Maximum number of key matches per query.  ``None`` returns
                all key candidates above the score cutoff for each query.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One key-match list per query, preserving query order. Within each
            list, scorer quality is primary, exact equality breaks equal-score
            ties, and insertion order breaks remaining ties.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        return [
            self.fuzzy_find_keys(
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

    def fuzzy_get(
        self,
        query: object,
        default: V | None = None,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> V | None:
        """Return the value for the best fuzzy-matched key, or ``default``.

        Args:
            query: Value to match against keys.
            default: Value to return when no key meets the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Value associated with the best fuzzy-matched key, or ``default`` when no
            acceptable key match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Keys are ranked by scorer quality. Exact equality wins equal-score
            ties, followed by insertion order. Compatible RapidFuzz metadata
            permits an immediate exact result only when its score is provably
            optimal.
        """

        match = self.fuzzy_find_item(
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
        default: V | None = None,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[V | None]:
        """Return values for the best fuzzy key matches for multiple queries.

        Args:
            queries: Query values to match against keys in order.
            default: Value to return for each query with no key match.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One mapping value or ``default`` per query, preserving query order.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            For each query, keys are ranked by scorer quality. Exact equality
            wins equal-score ties, followed by insertion order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return [
            default if match is None else match.value
            for match in self.fuzzy_find_item_batch(
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
    ) -> Iterator[MappingMatch[K, V] | None]:
        """Yield one key-scoring result per item without allocating a result list.

        This performs fuzzy scoring for every searchable key and does not
        apply exact-match shortcuts. Position order matches key insertion order.

        Args:
            query: Value to score against every stored key.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Yields:
            ``MappingMatch`` for keys above the score cutoff and ``None`` for unsearchable or rejected key positions.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        index = self._key_index
        if self._is_keyed:
            matches = index.iter_scores(
                self._data,
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
        for match in matches:
            yield None if match is None else self._mapping_match_from_key_match(self._key_match(match))

    def fuzzy_score_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[MappingMatch[K, V] | None]:
        """Return one item-scoring result per stored key, ``None`` for rejected positions.

        Unlike ``fuzzy_find_keys``, this always returns a list equal in length to
        the number of stored keys. Position ``i`` corresponds to the key at
        insertion-order position ``i``. Exact shortcuts are not applied; all
        searchable keys are scored with the configured scorer. Scorers without
        compatible RapidFuzz metadata use a direct position-aligned pass without
        ranking.

        Args:
            query: Value to score against every stored key.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``MappingMatch`` at position ``i`` if the key scored above the
            score cutoff, ``None`` if it did not match or is unsearchable.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        index = self._key_index
        if self._is_keyed:
            matches = index.score_all(
                self._data,
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
        return [
            None if match is None else self._mapping_match_from_key_match(self._key_match(match)) for match in matches
        ]

    def items(self) -> ItemsView[K, V]:
        """Return a dict-style, live view of key-value pairs in insertion order.

        Returns:
            Real ``dict_items`` view backed by the underlying mapping, reversible and
            exposing ``.mapping`` like a builtin ``dict`` items view.
        """

        return self._data.items()

    def keys(self) -> KeysView[K]:
        """Return a dict-style, live view of keys in insertion order.

        Returns:
            Real ``dict_keys`` view backed by the underlying mapping, reversible and
            exposing ``.mapping`` like a builtin ``dict`` keys view.
        """

        return self._data.keys()

    def values(self) -> ValuesView[V]:
        """Return a dict-style, live view of values in insertion order.

        Returns:
            Real ``dict_values`` view backed by the underlying mapping, reversible like
            a builtin ``dict`` values view.
        """

        return self._data.values()

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
    ) -> FrozenFuzzyDict[K, V]:
        """Return a new frozen mapping with updated key matching configuration.

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
            strategy: Replacement index strategy, or ``_UNCHANGED`` to preserve the current strategy.

        Returns:
            New frozen mapping containing the same key and value objects under the resulting fuzzy key configuration.

        Raises:
            TypeError: If an overridden configuration value fails constructor validation.
            ValueError: If ``strategy`` is replaced with an unsupported index strategy.

        Notes:
            This method leaves the source mapping unchanged and constructs a new fuzzy key index from its keys.
        """

        return self.__class__(
            self._data,
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
