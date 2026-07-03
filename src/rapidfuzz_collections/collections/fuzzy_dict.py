from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, MutableMapping, ValuesView
from copy import deepcopy as _deepcopy
from types import NotImplementedType
from typing import Any, Self

from rapidfuzz.fuzz import WRatio

from ..configuration import _UNCHANGED, _apply_config_overrides, _coerce_index_strategy
from ..enums import IndexStrategy, ScorerType
from ..indexes import MutableFuzzyKeyedIndex, MutableFuzzySequenceIndex, Scorer, validate_chunk_size
from ..matching import MappingMatch, Match, ValueMatch

_KEYED_RETAIN_REBUILD_THRESHOLD: float = 0.3
"""Fraction of keyed retain deletions at which full index rebuild is cheaper."""


class FuzzyDict[K, V](MutableMapping[K, V]):
    """A dict-like mutable collection with strategy-selectable fuzzy key lookup.

    Exact mapping behavior follows ``dict``: keys and values are stored
    unchanged, duplicate construction keys follow last-write-wins semantics,
    and exact reads, writes, deletion, iteration, and membership are not fuzzy.

    Fuzzy methods search keys through RapidFuzz using cached normalized key
    data. ``IndexStrategy.SEQUENCE`` is the default because benchmarks show it
    is the strongest general read-heavy strategy. ``IndexStrategy.KEYED`` uses
    keyed choices over unique hashable keys and can be better for build-heavy,
    selected bulk-mutation, or normalized-collision-heavy workloads. Its
    canonical exact-key registry generally uses more memory than the mutable
    sequence strategy.

    Both strategies return the same public fuzzy result classes. Dict/set
    fuzzy results are intentionally position-free, so ``Match.index`` and
    ``MappingMatch.index`` are always ``None`` for this facade.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = ("_data", "_is_keyed", "_key_index", "_strategy")

    def __copy__(self) -> FuzzyDict[K, V]:
        """Return a shallow copy preserving fuzzy configuration.

        Returns:
            Shallow copy with the same items and fuzzy key configuration.
        """

        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> FuzzyDict[K, V]:
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

    def __delitem__(self, key: K) -> None:
        """Delete an exact key and update the fuzzy key index.

        Args:
            key: Exact key to delete.

        Side Effects:
            Removes ``key`` from the mapping and updates the fuzzy key index.
        """

        del self._data[key]
        if self._is_keyed:
            self._key_index.remove(key)  # type: ignore[union-attr]
        else:
            self._key_index.delete_value(key)  # type: ignore[union-attr]

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
        """Build the mapping and its selected fuzzy key index.

        Args:
            items: Initial key-value pairs as a mapping or iterable of pairs.
            normalizer: Callable that maps each key to a searchable string, or
                returns ``None`` to exclude it from fuzzy lookup.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Keys are ranked by scorer quality; exact
                equality wins equal-score ties, followed by insertion order.
                Compatible RapidFuzz metadata permits an immediate exact result
                only when its score is provably optimal.
            scorer_kwargs: Keyword arguments forwarded to the scorer.
            scorer_type: Whether scorer output is a similarity or distance.
            score_cutoff: Fuzzy acceptance threshold. ``None`` disables it.
            score_hint: Optional RapidFuzz implementation-selection hint.
            strategy: Index storage strategy used by the fuzzy index. Use ``SEQUENCE`` for
                the best general read-heavy behavior. Try ``KEYED`` when keys
                are unique hashable values and build cost, selected bulk
                mutation, or normalized collisions dominate. ``KEYED``
                generally increases mutable-index memory use.

        Raises:
            TypeError: If ``normalizer``, ``scorer``, or ``scorer_kwargs`` is
                invalid, ``scorer_type`` is not a ``ScorerType`` member, or
                ``score_cutoff``/``score_hint`` is not numeric or ``None``.
            ValueError: If ``strategy`` is not a supported index strategy.

        Side Effects:
            Builds the selected fuzzy key index eagerly.
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
            self._key_index = MutableFuzzyKeyedIndex(self._data, **index_kwargs)
        else:
            self._key_index = MutableFuzzySequenceIndex(self._data.keys(), **index_kwargs)

    def __ior__(self, items: Iterable[tuple[K, V]] | Mapping[K, V]) -> Self:
        """Update the mapping in place and keep the fuzzy key index synchronized.

        Args:
            items: Key-value pairs or mapping to merge into this mapping.

        Returns:
            This mapping after applying ``items``.

        Side Effects:
            Updates the mapping and synchronizes the fuzzy key index for new keys.
        """

        self.update(items)
        return self

    def __iter__(self) -> Iterator[K]:
        """Iterate over exact keys in insertion order.

        Returns:
            Iterator over exact keys in insertion order.
        """

        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of mapping items.

        Returns:
            Number of key-value pairs stored in the mapping.
        """

        return len(self._data)

    def __or__[K2, V2](
        self,
        items: Mapping[K2, V2],
    ) -> FuzzyDict[K | K2, V | V2] | NotImplementedType:
        """Return a merged mapping preserving fuzzy configuration.

        Only mapping operands are accepted; use ``update()`` to merge an
        arbitrary iterable of pairs.

        Args:
            items: Mapping to merge after this mapping.

        Returns:
            New fuzzy mapping containing this mapping's items updated with ``items``, or
            ``NotImplemented`` when ``items`` is not a mapping.
        """

        if not isinstance(items, Mapping):
            return NotImplemented
        result = self.copy()
        result.update(items)
        return result

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            Developer-friendly string representation of the mapping.
        """

        return f"{self.__class__.__name__}({self._data!r})"

    def __reversed__(self) -> Iterator[K]:
        """Iterate over exact keys in reverse insertion order.

        Returns:
            Iterator over exact keys in reverse insertion order.
        """

        return reversed(self._data)

    def __ror__[K2, V2](
        self,
        items: Mapping[K2, V2],
    ) -> FuzzyDict[K | K2, V | V2] | NotImplementedType:
        """Return a merged mapping with ``items`` ordered before this mapping.

        Only mapping operands are accepted; use ``update()`` to merge an
        arbitrary iterable of pairs.

        Args:
            items: Mapping to merge before this mapping.

        Returns:
            New fuzzy mapping containing ``items`` updated with this mapping's items, or
            ``NotImplemented`` when ``items`` is not a mapping.
        """

        if not isinstance(items, Mapping):
            return NotImplemented
        data = dict(items)
        data.update(self._data)
        return self.__class__(data, **self._config_kwargs())

    def __setitem__(self, key: K, value: V) -> None:
        """Set a value and update the fuzzy key index when the key is new.

        Args:
            key: Exact key to set.
            value: Value to associate with ``key``.

        Side Effects:
            Updates the mapping. If ``key`` is new, adds it to the fuzzy key index.
        """

        is_new_key = key not in self._data
        self._data[key] = value
        if not is_new_key:
            return
        try:
            if self._is_keyed:
                self._key_index.add(key)  # type: ignore[union-attr]
            else:
                self._key_index.append(key)  # type: ignore[union-attr]
        except BaseException:
            del self._data[key]
            raise

    def _config_kwargs(self, *, deepcopy_memo: dict[int, object] | None = None) -> dict[str, Any]:
        """Return fuzzy configuration for constructing related mappings.

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
        """Return a position-free public key match.

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
        """Build a position-free mapping match from a public key match.

        Args:
            match: Public key match to convert.

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

    def copy(self) -> FuzzyDict[K, V]:
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
    ) -> FuzzyDict[K, V | None]:
        """Build a fuzzy mapping from keys that all share one value.

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
            New fuzzy mapping with the supplied key lookup configuration.

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
            ``True`` if any searchable key matches ``query`` above the score
            cutoff, otherwise ``False``.

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
        """Remove the best fuzzy key match, or do nothing on miss.

        Args:
            query: Value used to find the best fuzzy key to remove.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes the best fuzzy-matched key from the mapping and updates the
            fuzzy key index when a match is found.

        Notes:
            The removed key is selected by scorer quality. Exact equality wins
            equal-score ties, followed by insertion order. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        index = self._key_index
        if self._is_keyed:
            match = self.fuzzy_find_key(
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            if match is not None:
                del self[match.value]
            return

        match = index.find_one(
            query,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        if match is None or match.index is None:  # type: ignore[union-attr]
            return
        del self._data[match.value]
        index.delete_at(match.index)  # type: ignore[union-attr]

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
        """Remove every item whose key fuzzy-matches the query.

        Args:
            query: Value to match against keys for removal.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Number of removed key-value pairs.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes matching items from the mapping and updates the fuzzy key index.
        """

        index = self._key_index
        if self._is_keyed:
            matches = index.find_many(
                query,
                limit=None,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            keys = [match.value for match in matches]
            for key in keys:
                del self._data[key]
            index.batch_remove(keys)  # type: ignore[union-attr]
            return len(keys)

        matches = index.find_many(
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
        for match in matches:
            del self._data[match.value]
        index.delete_at_positions({m.index for m in matches if m.index is not None})  # type: ignore[union-attr]
        return len(matches)

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
        """Return the best fuzzy key/value match, or ``None``.

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
        return None if match is None else self._mapping_match_from_key_match(match)

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

        Requires ``IndexStrategy.SEQUENCE``. Delegates to
        ``fuzzy_find_key_batch_cdist`` internally.

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
            One mapping match or ``None`` per query, preserving query order.

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
        """Return up to ``limit`` fuzzy key/value matches.

        Similarity scores are ordered from highest to lowest; distance scores
        are ordered from lowest to highest. Among equal scores, keys equal to
        the query precede non-exact keys, followed by insertion order.

        Args:
            query: Value to match against keys.
            limit: Maximum number of matches. ``None`` returns all key candidates above the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            List of key/value matches for ``query`` ordered by scorer quality, exact-equality tie-break,
            and insertion order.

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
            limit: Maximum number of matches per query.  ``None`` returns all
                key candidates above the score cutoff for each query.
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
        """Return the best fuzzy key match, or ``None``.

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
        """Return best key matches through bounded ``cdist`` when available.

        The sequence strategy uses RapidFuzz ``cdist``. The keyed strategy is
        not supported by this method; call ``fuzzy_find_key_batch`` instead.

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
        """Return up to ``limit`` fuzzy key matches.

        Similarity scores are ordered from highest to lowest; distance scores
        are ordered from lowest to highest. Among equal scores, keys equal to
        the query precede non-exact keys, followed by insertion order.

        Args:
            query: Value to match against keys.
            limit: Maximum number of matches. ``None`` returns all key candidates above the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            List of fuzzy key matches for ``query`` ordered by scorer quality, exact-equality tie-break,
            and insertion order.

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
            limit: Maximum number of matches per query.  ``None`` returns all
                key candidates above the score cutoff for each query.
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
            Value associated with the best fuzzy-matched searchable key, or
            ``default`` when no acceptable key match is found.

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
        return default if match is None else match.value

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
        """Return matched values for multiple fuzzy key queries.

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
        """Retain only items whose keys fuzzy-match the query above the score cutoff.

        Args:
            query: Value to match against keys.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Number of removed key-value pairs.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Side Effects:
            Removes keys that do not match ``query`` and updates or rebuilds the
            fuzzy key index.
        """

        index = self._key_index
        if self._is_keyed:
            retained = {
                match.value
                for match in index.find_many(
                    query,
                    limit=None,
                    scorer=scorer,
                    scorer_kwargs=scorer_kwargs,
                    scorer_type=scorer_type,
                    score_cutoff=score_cutoff,
                    score_hint=score_hint,
                )
            }
            to_delete = [key for key in self._data if key not in retained]
            if not to_delete:
                return 0
            delete_count = len(to_delete)
            live_count = len(self._data)
            for key in to_delete:
                del self._data[key]
            if delete_count < live_count * _KEYED_RETAIN_REBUILD_THRESHOLD:
                index.batch_remove(to_delete)  # type: ignore[union-attr]
            else:
                self._key_index = MutableFuzzyKeyedIndex(self._data, **index.config_kwargs())
            return delete_count

        matches = index.find_many(
            query,
            limit=None,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        keep_positions = {m.index for m in matches if m.index is not None}
        keys_to_keep = {match.value for match in matches}
        keys_to_remove = [key for key in self._data if key not in keys_to_keep]
        if not keys_to_remove:
            return 0
        for key in keys_to_remove:
            del self._data[key]
        index.keep_at_positions(keep_positions)  # type: ignore[union-attr]
        return len(keys_to_remove)

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

        Unlike ``fuzzy_find_keys``, the result length always equals the number
        of stored keys.  Position ``i`` corresponds to the key at
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

    def popitem(self) -> tuple[K, V]:
        """Remove and return the last inserted key-value pair (LIFO order, matching ``dict``).

        Returns:
            Last inserted key-value pair.

        Raises:
            KeyError: If the mapping is empty.

        Side Effects:
            Removes the last inserted key from the mapping and updates the fuzzy key index.
        """

        if not self._data:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self._data))
        value = self._data[key]
        del self[key]
        return key, value

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
    ) -> FuzzyDict[K, V]:
        """Return a new fuzzy mapping with updated matching configuration.

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
            New fuzzy mapping containing the same key and value objects under the resulting fuzzy key configuration.

        Raises:
            TypeError: If an overridden configuration value fails constructor validation.
            ValueError: If ``strategy`` is replaced with an unsupported index strategy.
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
