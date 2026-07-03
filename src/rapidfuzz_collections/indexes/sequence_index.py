from bisect import bisect_left, insort
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from itertools import groupby
from typing import Any, overload

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from ..configuration import _UNCHANGED
from ..enums import ScorerType
from ..matching import Match
from ..normalization import _default_normalizer
from .base import (
    Scorer,
    _is_hashable,
    _MatchConfig,
    _process_scorer_metadata,
    _resolve_match_config,
    copy_scorer_kwargs,
    index_config_kwargs,
    validate_chunk_size,
    validate_limit,
    validate_normalizer,
    validate_optional_score,
    validate_scorer,
    validate_scorer_type,
)

_DENSE_INCREMENTAL_DELETE_LIMIT = 128
"""Maximum dense batch size deleted in place instead of by list filtering."""

_SPARSE_INCREMENTAL_DELETE_LIMIT = 1024
"""Maximum sparse batch size deleted incrementally instead of by rebuild."""

type _NormalizedChoices = list[str] | tuple[str, ...]


def passes_score_cutoff(
    score: int | float,
    *,
    scorer_type: ScorerType,
    score_cutoff: int | float | None,
) -> bool:
    """Return whether a scorer result satisfies the configured cutoff.

    Args:
        score: The scorer result to evaluate.
        scorer_type: The scorer type that defines whether lower or higher scores are better.
        score_cutoff: Optional minimum similarity score or maximum distance score to accept.

    Returns:
        ``True`` if ``score`` satisfies ``score_cutoff`` or no cutoff is configured.
    """

    if score_cutoff is None:
        return True
    if scorer_type == ScorerType.DISTANCE:
        return score <= score_cutoff
    return score >= score_cutoff


def _find_one_batch_cdist[T](
    values: Sequence[T],
    normalized_choices: _NormalizedChoices,
    queries: Iterable[object],
    *,
    resolve_query: Callable[[object], tuple[Match[T] | None, str | None]],
    exact_source_indexes: Callable[[object], Sequence[int]],
    normalized_value_from_source: Callable[[int], str | None],
    source_index_from_choice: Callable[[int], int],
    scorer: Scorer,
    scorer_kwargs: dict[str, Any] | None,
    scorer_type: ScorerType,
    score_cutoff: int | float | None,
    score_hint: int | float | None,
    process_scorer_type_matches: bool,
    query_chunk_size: int,
    choice_chunk_size: int,
    workers: int,
) -> list[Match[T] | None]:
    """Find best matches through bounded RapidFuzz ``cdist`` score matrices.

    This specialized path is opt-in because it evaluates score matrices and
    may be slower or consume more memory than ``find_one_batch`` depending on
    the scorer and workload.

    Args:
        values: Source values returned in produced matches.
        normalized_choices: Normalized values compared against normalized queries.
        queries: Query values to resolve and match.
        resolve_query: Callable that resolves a query before matrix scoring.
            It may return an immediate match, a normalized query, or no match.
        exact_source_indexes: Callable returning exact source positions for a query.
        normalized_value_from_source: Callable returning a source value's
            cached normalized form.
        source_index_from_choice: Callable that maps a normalized choice index
            back to the corresponding source value index.
        scorer: RapidFuzz scorer used to compare normalized strings.
        scorer_kwargs: Optional keyword arguments passed to ``scorer``.
        scorer_type: The scorer type that defines whether lower or higher scores are better.
        score_cutoff: Optional minimum similarity score or maximum distance score to accept.
        score_hint: Optional expected score passed to RapidFuzz for optimization.
        process_scorer_type_matches: Whether RapidFuzz recognizes the configured
            scorer direction and can apply cutoff and hint directly.
        query_chunk_size: Maximum number of pending queries per ``cdist`` call.
        choice_chunk_size: Maximum number of normalized choices per ``cdist`` call.
        workers: Number of RapidFuzz worker threads used by ``cdist``.

    Returns:
        A list aligned with ``queries`` containing the best match for each query,
        or ``None`` when no acceptable match is found.

    Raises:
        TypeError: If a chunk size is not an integer, or is a boolean.
        ValueError: If ``query_chunk_size`` or ``choice_chunk_size`` is less than 1.
        ModuleNotFoundError: If matrix scoring is required but NumPy is not installed.
    """

    validate_chunk_size(query_chunk_size, "query_chunk_size")
    validate_chunk_size(choice_chunk_size, "choice_chunk_size")

    results: list[Match[T] | None] = []
    pending: list[tuple[int, object, str]] = []
    for query in queries:
        result_index = len(results)
        results.append(None)
        match, normalized_query = resolve_query(query)
        if match is not None:
            results[result_index] = match
        elif normalized_query is not None:
            pending.append((result_index, query, normalized_query))

    if not pending or not normalized_choices:
        return results

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("find_one_batch_cdist requires NumPy. Install rapidfuzz-collections[cdist].") from exc

    direct_scorer_kwargs = scorer_kwargs or {}
    cdist_scorer = scorer
    cdist_scorer_kwargs = scorer_kwargs
    if not process_scorer_type_matches:

        def cdist_scorer(left: str, right: str, **_process_kwargs: object) -> int | float:
            return scorer(left, right, **direct_scorer_kwargs)

        cdist_scorer_kwargs = None

    for query_start in range(0, len(pending), query_chunk_size):
        query_block = pending[query_start : (query_start + query_chunk_size)]  # noqa: E203
        normalized_queries = tuple(normalized_query for _, _, normalized_query in query_block)
        best_scores: list[int | float | None] = [None] * len(query_block)
        best_choice_indexes: list[int | None] = [None] * len(query_block)

        for choice_start in range(0, len(normalized_choices), choice_chunk_size):
            score_block = process.cdist(  # type: ignore[call-overload]
                normalized_queries,
                normalized_choices[choice_start : (choice_start + choice_chunk_size)],  # noqa: E203
                scorer=cdist_scorer,
                score_cutoff=score_cutoff if process_scorer_type_matches else None,
                score_hint=score_hint if process_scorer_type_matches else None,
                dtype=np.float64,
                workers=workers,
                scorer_kwargs=cdist_scorer_kwargs,
            )
            best_positions = (
                np.argmin(score_block, axis=1) if scorer_type == ScorerType.DISTANCE else np.argmax(score_block, axis=1)
            )
            for offset, (row, best_position) in enumerate(zip(score_block, best_positions, strict=True)):
                position = int(best_position)
                score = row[position].item()
                if not passes_score_cutoff(score, scorer_type=scorer_type, score_cutoff=score_cutoff):
                    continue
                previous_score = best_scores[offset]
                improves_result = previous_score is None or (
                    score < previous_score if scorer_type == ScorerType.DISTANCE else score > previous_score
                )
                if improves_result:
                    best_scores[offset] = score
                    best_choice_indexes[offset] = choice_start + position

        for offset, ((result_index, query, normalized_query), choice_index) in enumerate(
            zip(query_block, best_choice_indexes, strict=True)
        ):
            selected_score = best_scores[offset]
            selected_source_index: int | None = None
            selected_normalized_value: str | None = None
            exact_indexes = exact_source_indexes(query)
            exact_index_set = set(exact_indexes)

            if choice_index is not None:
                selected_choice_index: int = choice_index
                if isinstance(normalized_choices, list):
                    selected_normalized_value = normalized_choices[selected_choice_index]
                else:
                    selected_normalized_value = normalized_choices[selected_choice_index]
                selected_score = scorer(normalized_query, selected_normalized_value, **direct_scorer_kwargs)
                selected_source_index = source_index_from_choice(selected_choice_index)

            selected_is_exact = selected_source_index in exact_index_set
            for exact_source_index in exact_indexes:
                exact_normalized_value = normalized_value_from_source(exact_source_index)
                if exact_normalized_value is None:
                    continue
                exact_score = scorer(normalized_query, exact_normalized_value, **direct_scorer_kwargs)
                if not passes_score_cutoff(
                    exact_score,
                    scorer_type=scorer_type,
                    score_cutoff=score_cutoff,
                ):
                    continue
                improves_result = selected_score is None or (
                    exact_score < selected_score if scorer_type == ScorerType.DISTANCE else exact_score > selected_score
                )
                wins_tie = exact_score == selected_score and not selected_is_exact
                if improves_result or wins_tie:
                    selected_score = exact_score
                    selected_source_index = exact_source_index
                    selected_normalized_value = exact_normalized_value
                    selected_is_exact = True

            if selected_source_index is None or selected_normalized_value is None or selected_score is None:
                continue
            results[result_index] = Match(
                value=values[selected_source_index],
                score=selected_score,
                index=selected_source_index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=selected_normalized_value,
            )

    return results


class _SequenceIndexCommon[T]:
    """Share non-hot-path sequence index API between concrete index variants."""

    __slots__ = ()
    _values: Sequence[T]
    _normalizer: Callable[[object], str | None]
    _normalized_choices: _NormalizedChoices
    _optimal_score: int | float | None
    _process_scorer_type_matches: bool
    _scorer: Scorer
    _scorer_kwargs: dict[str, Any] | None
    _scorer_type: ScorerType
    _score_cutoff: int | float | None
    _score_hint: int | float | None
    _default_config: _MatchConfig

    def __iter__(self) -> Iterator[T]:
        """Iterate over source values.

        Returns:
            Iterator over source values in source order.
        """

        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of source values.

        Returns:
            Number of source values stored in the index.
        """

        return len(self._values)

    def __reversed__(self) -> Iterator[T]:
        """Iterate over source values in reverse order.

        Returns:
            Iterator over source values from last to first.
        """

        return reversed(self._values)

    @property
    def normalizer(self) -> Callable[[object], str | None]:
        """Callable that maps values to searchable strings.

        Treat the returned callable as immutable while the index is in use.
        Mutating a mutable or stateful normalizer can make later query
        normalization inconsistent with cached choices.

        Returns:
            Configured normalizer callable.
        """

        return self._normalizer

    @property
    def normalized_choices(self) -> tuple[str, ...]:
        """Immutable snapshot of normalized searchable choices.

        Returns:
            Tuple containing normalized searchable choices in choice order.
        """

        return tuple(self._normalized_choices)

    @property
    def score_cutoff(self) -> int | float | None:
        """Minimum score or maximum distance threshold for a match.

        Returns:
            Configured score cutoff, or ``None`` when cutoff filtering is disabled.
        """

        return self._score_cutoff

    @property
    def score_hint(self) -> int | float | None:
        """Expected score hint for RapidFuzz process and matrix operations.

        The hint may help RapidFuzz select an implementation on extraction and
        ``cdist`` paths. It is not passed to direct scorer calls.

        Returns:
            Configured score hint, or ``None`` when no hint is configured.
        """

        return self._score_hint

    @property
    def scorer(self) -> Callable[..., int | float]:
        """Scorer callable used to compare normalized strings.

        Returns:
            Configured scorer callable.
        """

        return self._scorer

    @property
    def scorer_kwargs(self) -> dict[str, Any] | None:
        """Independent copy of keyword arguments forwarded to the scorer.

        Returns:
            Copy of configured scorer keyword arguments, or ``None`` when no extra
            scorer keyword arguments are configured.
        """

        return copy_scorer_kwargs(self._scorer_kwargs)

    @property
    def scorer_type(self) -> ScorerType:
        """Whether the scorer returns similarity or distance values.

        Returns:
            Configured scorer type.
        """

        return self._scorer_type

    @property
    def values(self) -> tuple[T, ...]:
        """Immutable snapshot of source values in source order.

        Returns:
            Tuple containing source values in source order.
        """

        return tuple(self._values)

    def _exact_matches(
        self,
        query: object,
        normalized_query: str,
        exact_indexes: Sequence[int],
        *,
        config: _MatchConfig,
    ) -> dict[int, Match[T]]:
        """Score searchable exact candidates and return them by source position."""

        matches: dict[int, Match[T]] = {}
        for source_index in exact_indexes:
            normalized_value = self._normalized_value_from_source(source_index)  # type: ignore[attr-defined]
            if normalized_value is None:
                continue
            score = self._score_pair(normalized_query, normalized_value, config=config)
            if score is None:
                continue
            matches[source_index] = Match(
                value=self._values[source_index],
                score=score,
                index=source_index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )
        return matches

    def _exact_source_indexes(self, query: object) -> tuple[int, ...]:
        """Return source positions whose stored values equal ``query``.

        Side Effects:
            Lazily rebuilds invalid exact-value shortcuts without rebuilding
            normalized fuzzy choices.
        """

        if not _is_hashable(query):
            return ()
        if not getattr(self, "_shortcuts_valid", True):
            self._rebuild_exact_shortcuts()  # type: ignore[attr-defined]
        first_index = self._exact_first_index.get(query)  # type: ignore[attr-defined]
        if first_index is None:
            return ()
        duplicate_indexes = self._exact_duplicate_indexes.get(query, ())  # type: ignore[attr-defined]
        return first_index, *duplicate_indexes

    def _extract(
        self,
        normalized_query: str,
        *,
        limit: int | None,
        config: _MatchConfig,
    ) -> list[tuple[str, int | float, int]]:
        """Return matches for a normalized query.

        ``config`` is used to compare the normalized query against stored
        normalized choices. Returned matches are ordered according to the
        scorer type: similarity scorers are sorted from the highest score to lowest,
        while distance scorers are sorted from the lowest score to highest.

        Args:
            normalized_query: Normalized query value to compare with stored choices.
            limit: Maximum number of matches to return. If ``None``, all matches
                passing the configured score cutoff are returned.
            config: Resolved matching configuration for this call.

        Returns:
            A list of tuples containing the normalized matched value, the score,
            and the index of the original choice.
        """

        if config.process_scorer_type_matches:
            return list(
                process.extract(
                    normalized_query,
                    self._normalized_choices,
                    scorer=config.scorer,
                    score_cutoff=config.score_cutoff,
                    score_hint=config.score_hint,
                    scorer_kwargs=config.scorer_kwargs,
                    limit=limit,
                )
            )

        scorer_kwargs = config.scorer_kwargs or {}
        matches = [
            (normalized_value, score, choice_index)
            for choice_index, normalized_value in enumerate(self._normalized_choices)
            if passes_score_cutoff(
                score := config.scorer(normalized_query, normalized_value, **scorer_kwargs),
                scorer_type=config.scorer_type,
                score_cutoff=config.score_cutoff,
            )
        ]
        matches.sort(key=lambda result: result[1], reverse=config.scorer_type == ScorerType.SIMILARITY)
        return matches if limit is None else matches[:limit]

    def _extract_one(self, normalized_query: str, *, config: _MatchConfig) -> tuple[str, int | float, int] | None:
        """Return the highest-ranked candidate for a normalized query."""

        if config.process_scorer_type_matches:
            return process.extractOne(  # type: ignore[call-overload]
                normalized_query,
                self._normalized_choices,
                scorer=config.scorer,
                score_cutoff=config.score_cutoff,
                score_hint=config.score_hint,
                scorer_kwargs=config.scorer_kwargs,
            )
        matches = self._extract(normalized_query, limit=1, config=config)
        return matches[0] if matches else None

    def _ranked_matches(
        self,
        query: object,
        normalized_query: str,
        *,
        limit: int | None,
        exact_indexes: Sequence[int],
        exact_matches: dict[int, Match[T]] | None = None,
        config: _MatchConfig,
    ) -> list[Match[T]]:
        """Return matches ordered by score, hashable exact equality, and source position."""

        if limit == 0:
            return []

        if limit == 1:
            result = self._extract_one(normalized_query, config=config)
            extracted = () if result is None else (result,)
        else:
            extracted = self._extract(normalized_query, limit=limit, config=config)

        matches_by_index: dict[int, Match[T]] = {}
        for normalized_value, score, choice_index in extracted:
            source_index = self._source_index_from_choice(choice_index)  # type: ignore[attr-defined]
            matches_by_index[source_index] = Match(
                value=self._values[source_index],
                score=score,
                index=source_index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )

        if exact_matches is None:
            exact_matches = self._exact_matches(query, normalized_query, exact_indexes, config=config)
        matches_by_index.update(exact_matches)
        exact_index_set = set(exact_matches)

        if limit is None:
            # _extract already ranked every accepted choice. Only equal-score
            # exact candidates need promotion, so avoid sorting the full result
            # again when an unbounded lookup returns many matches.
            matches = list(matches_by_index.values())
            if not exact_index_set:
                return matches

            ranked: list[Match[T]] = []
            for _, score_group_iter in groupby(matches, key=lambda match: match.score):
                score_group = list(score_group_iter)
                ranked.extend(match for match in score_group if match.index in exact_index_set)
                ranked.extend(match for match in score_group if match.index not in exact_index_set)
            return ranked

        def order_key(match: Match[T]) -> tuple[int | float, bool, int]:
            m_score = match.score if config.scorer_type == ScorerType.DISTANCE else -match.score
            assert match.index is not None
            return m_score, match.index not in exact_index_set, match.index

        matches = sorted(matches_by_index.values(), key=order_key)
        return matches[:limit]

    # noinspection PyMethodMayBeStatic
    def _score_pair(
        self,
        normalized_query: str,
        normalized_value: str,
        *,
        config: _MatchConfig,
    ) -> int | float | None:
        """Score one normalized pair and apply the configured cutoff."""

        score = config.scorer(normalized_query, normalized_value, **(config.scorer_kwargs or {}))
        if not passes_score_cutoff(
            score,
            scorer_type=config.scorer_type,
            score_cutoff=config.score_cutoff,
        ):
            return None
        return score

    def config_kwargs(self, *, deepcopy_memo: dict[int, object] | None = None) -> dict[str, Any]:
        """Return constructor keyword arguments that reproduce this index's configuration.

        Args:
            deepcopy_memo: When provided, ``scorer_kwargs`` is deep-copied through
                this memo dict to share object identity with an ongoing deepcopy operation.

        Returns:
            Keyword arguments compatible with the index constructor.

        Notes:
            The returned scorer keyword arguments are independent of this index's stored configuration.
        """

        return index_config_kwargs(
            normalizer=self._normalizer,
            scorer=self._scorer,
            scorer_kwargs=self._scorer_kwargs,
            scorer_type=self._scorer_type,
            score_cutoff=self._score_cutoff,
            score_hint=self._score_hint,
            deepcopy_memo=deepcopy_memo,
        )

    def contains(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> bool:
        """Return whether any indexed value matches the query.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``True`` if any indexed value matches ``query``, otherwise ``False``.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result only when its
            score is provably optimal.
        """

        return (
            self.find_one(  # type: ignore[attr-defined]
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            is not None
        )

    def find_many_batch(
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
            limit: Maximum number of matches per query.
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

        Notes:
            Overrides apply uniformly to every query in the batch.
        """

        validate_limit(limit)
        return [
            self.find_many(  # type: ignore[attr-defined]
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

    def find_one_batch(
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
            score is provably optimal. Overrides apply uniformly to every query
            in the batch.
        """

        return [
            self.find_one(  # type: ignore[attr-defined]
                query,
                scorer=scorer,
                scorer_kwargs=scorer_kwargs,
                scorer_type=scorer_type,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
            )
            for query in queries
        ]

    def normalize(self, value: object) -> str | None:
        """Apply the configured normalizer to ``value``.

        Args:
            value: Value to normalize.

        Returns:
            Normalized string, or ``None`` if the normalizer rejects the value.
        """

        return self._normalizer(value)


class FuzzySequenceIndex[T](_SequenceIndexCommon[T]):
    """Read-only fuzzy index over an ordered sequence of values.

    Lookup first normalizes the query. Matches are ordered by scorer result,
    hashable exact equality with the query, and source position. Compatible RapidFuzz
    metadata permits an optimal exact candidate to return without scanning all
    normalized choices.

    The index is immutable after construction.  Values that the normalizer
    rejects (returns ``None``) are stored and accessible as source values but
    excluded from fuzzy comparison.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = (
        "_default_config",
        "_exact_duplicate_indexes",
        "_exact_first_index",
        "_normalizer",
        "_normalized_choices",
        "_optimal_score",
        "_process_scorer_type_matches",
        "_scorer",
        "_scorer_kwargs",
        "_scorer_type",
        "_score_cutoff",
        "_score_hint",
        "_source_indexes",
        "_values",
    )

    @overload
    def __getitem__(self, position: int) -> T: ...

    @overload
    def __getitem__(self, position: slice) -> tuple[T, ...]: ...

    def __getitem__(self, position: int | slice) -> T | tuple[T, ...]:
        """Return a source value or an immutable source-value slice.

        Args:
            position: Source index or slice.

        Returns:
            Source value for an integer index, or an immutable tuple of source values for a slice.
        """

        if isinstance(position, slice):
            return self._values[position]
        return self._values[position]

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
        """Build the index from ``values`` and configure the matching pipeline.

        Args:
            values: Source values to index.  Stored in the order provided.
            normalizer: Callable that maps a value to a searchable string, or
                returns ``None`` to exclude the value from fuzzy lookup.
                Defaults to the built-in normalizer, which is equivalent to
                ``Normalizer.default()``.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Candidates are ranked by scorer quality;
                hashable exact equality wins equal-score ties, followed by source
                position. Compatible RapidFuzz metadata permits an immediate
                exact result only when its score is provably optimal.
            scorer_kwargs: Additional keyword arguments forwarded to the scorer
                on every RapidFuzz call.  Useful for scorers with extra
                parameters, such as ``Levenshtein.distance`` with custom
                ``weights``. ``None`` passes no extra arguments.
            scorer_type: Interpretation of scorer output.  Use ``DISTANCE``
                when the scorer returns edit distances (lower = more similar);
                use ``SIMILARITY`` for percentage-like scores (higher = more similar).
            score_cutoff: Exclusion threshold applied by RapidFuzz.  For
                ``SIMILARITY`` scorers, candidates below this score are
                excluded; for ``DISTANCE`` scorers, candidates above this
                distance are excluded.  ``None`` disables the cutoff.
            score_hint: Expected score passed to RapidFuzz so it may select a
                faster implementation. ``None`` disables this optimization.

        Raises:
            TypeError: If any matching configuration argument has an invalid type.

        Side Effects:
            Normalizes all values and eagerly builds the exact-value registry
            and RapidFuzz choice index.
        """
        validate_normalizer(normalizer)
        validate_optional_score(score_cutoff, "score_cutoff")
        validate_optional_score(score_hint, "score_hint")
        validate_scorer(scorer)
        self._values = tuple(values)
        self._normalizer = normalizer if normalizer is not None else _default_normalizer
        self._scorer = scorer
        self._scorer_kwargs = copy_scorer_kwargs(scorer_kwargs) if scorer_kwargs is not None else None
        self._scorer_type = validate_scorer_type(scorer_type)
        self._score_cutoff = score_cutoff
        self._score_hint = score_hint
        self._process_scorer_type_matches, self._optimal_score = _process_scorer_metadata(
            scorer,
            self._scorer_kwargs,
            self._scorer_type,
        )
        self._default_config = _MatchConfig(
            scorer=self._scorer,
            scorer_kwargs=self._scorer_kwargs,
            scorer_type=self._scorer_type,
            score_cutoff=self._score_cutoff,
            score_hint=self._score_hint,
            process_scorer_type_matches=self._process_scorer_type_matches,
            optimal_score=self._optimal_score,
        )
        self._exact_first_index: dict[Hashable, int] = {}

        exact_duplicate_indexes: dict[Hashable, list[int]] = {}
        normalized_choices: list[str] = []
        source_indexes: list[int] | None = None

        for index, value in enumerate(self._values):
            if _is_hashable(value):
                first_index = self._exact_first_index.setdefault(value, index)
                if first_index != index:
                    exact_duplicate_indexes.setdefault(value, []).append(index)

            normalized_value = self._normalizer(value)
            if normalized_value is None:
                continue

            if source_indexes is not None:
                source_indexes.append(index)
            elif index != len(normalized_choices):
                source_indexes = [*range(len(normalized_choices)), index]
            normalized_choices.append(normalized_value)

        self._normalized_choices = tuple(normalized_choices)
        # If any values are unsearchable (normalizer returned None), _source_indexes must
        # track the mapping from choice positions to source positions.  The loop above only
        # creates source_indexes when a mid-sequence gap is detected; unsearchable elements
        # at the tail never trigger that branch, so we must handle them here.
        if source_indexes is None and len(normalized_choices) != len(self._values):
            source_indexes = list(range(len(normalized_choices)))
        self._source_indexes = None if source_indexes is None else tuple(source_indexes)
        self._exact_duplicate_indexes = {value: tuple(indexes) for value, indexes in exact_duplicate_indexes.items()}

    def _normalized_value_from_source(self, source_index: int) -> str | None:
        """Return the cached normalized value for a source position."""

        if self._source_indexes is None:
            return self._normalized_choices[source_index]
        choice_index = bisect_left(self._source_indexes, source_index)
        if choice_index == len(self._source_indexes) or self._source_indexes[choice_index] != source_index:
            return None
        return self._normalized_choices[choice_index]

    def _resolve_query(
        self,
        query: object,
        *,
        config: _MatchConfig | None = None,
    ) -> tuple[Match[T] | None, str | None]:
        """Resolve an optimal exact candidate before matrix scoring.

        Args:
            query: Query value to resolve.
            config: Resolved matching configuration for this call. The index
                defaults are used when omitted.

        Returns:
            ``(match, None)`` when an exact candidate has the scorer's optimal
            result. ``(None, normalized_query)`` when matrix scoring is needed.
            ``(None, None)`` when the query cannot be normalized.
        """

        if config is None:
            config = self._default_config
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None, None
        exact_indexes = self._exact_source_indexes(query)
        exact_matches = self._exact_matches(query, normalized_query, exact_indexes, config=config)
        if config.optimal_score is not None:
            for source_index in exact_indexes:
                match = exact_matches.get(source_index)
                if match is not None and match.score == config.optimal_score:
                    return match, None
        return None, normalized_query

    def _source_index_from_choice(self, choice_index: int) -> int:
        """Return the source index represented by a RapidFuzz choice index.

        Args:
            choice_index: RapidFuzz choice index.

        Returns:
            Source value index represented by ``choice_index``.
        """

        if self._source_indexes is None:
            return choice_index
        return self._source_indexes[choice_index]

    def find_many(
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
        """Return up to ``limit`` best matches in scorer-defined order.

        The query is normalized and passed to RapidFuzz for fuzzy matching.
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
            Matches ordered from best to worst according to ``scorer_type``.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.
        """

        validate_limit(limit)
        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return []
        exact_indexes = self._exact_source_indexes(query)
        return self._ranked_matches(
            query,
            normalized_query,
            limit=limit,
            exact_indexes=exact_indexes,
            config=config,
        )

    def find_one(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Match[T] | None:
        """Return the best match above the score cutoff, or ``None``.

        Candidates are ordered by scorer result, then hashable exact equality with the
        query, then source position. An exact candidate returns immediately
        only when compatible RapidFuzz metadata proves that its score is
        optimal.

        Args:
            query: Value to normalize and search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Best match for ``query``, or ``None`` when no acceptable match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Custom scorers without compatible RapidFuzz metadata evaluate all
            searchable values before hashable exact equality is used to break score
            ties. Overriding ``scorer``, ``scorer_kwargs``, or ``scorer_type``
            recomputes RapidFuzz scorer metadata for this call instead of
            reusing the collection's cached metadata.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None
        exact_indexes = self._exact_source_indexes(query)
        exact_matches = self._exact_matches(query, normalized_query, exact_indexes, config=config)
        if config.optimal_score is not None:
            for source_index in exact_indexes:
                match = exact_matches.get(source_index)
                if match is not None and match.score == config.optimal_score:
                    return match
        matches = self._ranked_matches(
            query,
            normalized_query,
            limit=1,
            exact_indexes=exact_indexes,
            exact_matches=exact_matches,
            config=config,
        )
        return matches[0] if matches else None

    def find_one_batch_cdist(
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
        """Return best matches through bounded RapidFuzz ``cdist`` scoring.

        Exact candidates with a proven optimal score resolve before matrix
        evaluation. Other candidates are ordered by score, hashable exact equality,
        and source position.

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
            One best match or ``None`` per query, preserving query order.

        Raises:
            TypeError: If a matching override or chunk size has an invalid type.
            ValueError: If a chunk size is less than 1, or ``scorer`` is overridden
                without ``scorer_type`` and has no compatible RapidFuzz metadata.
            ModuleNotFoundError: If matrix scoring is required but NumPy is
                not installed through ``rapidfuzz-collections[cdist]``.

        Notes:
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result before matrix
            scoring only when its score is provably optimal. This specialized
            path may allocate more memory or run slower than
            ``find_one_batch``.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        return _find_one_batch_cdist(
            self._values,
            self._normalized_choices,
            queries,
            resolve_query=lambda resolve_target: self._resolve_query(resolve_target, config=config),
            exact_source_indexes=self._exact_source_indexes,
            normalized_value_from_source=self._normalized_value_from_source,
            source_index_from_choice=self._source_index_from_choice,
            scorer=config.scorer,
            scorer_kwargs=config.scorer_kwargs,
            scorer_type=config.scorer_type,
            score_cutoff=config.score_cutoff,
            score_hint=config.score_hint,
            process_scorer_type_matches=config.process_scorer_type_matches,
            query_chunk_size=query_chunk_size,
            choice_chunk_size=choice_chunk_size,
            workers=workers,
        )

    def iter_scores(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Iterator[Match[T] | None]:
        """Yield one fuzzy scoring result per source value in source order.

        Top-one exact shortcuts are not applied; all searchable values are
        scored through the configured scorer. The extraction-only
        ``score_hint`` optimization is not applied by this streaming path.
        This is the memory-efficient counterpart to ``score_all`` and does not
        allocate a result list.

        Args:
            query: Value to score against all indexed values.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Yields:
            ``Match`` for values satisfying the score cutoff and ``None`` for rejected or unsearchable positions.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            This streaming operation calls the scorer once per searchable
            value. Use ``score_all`` when a materialized, position-aligned list
            is required; this iterator minimizes output memory.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            yield from (None for _ in self._values)
            return

        scorer_kwargs_resolved = config.scorer_kwargs or {}
        choice_index = 0
        for index, value in enumerate(self._values):
            if self._source_indexes is None:
                normalized_value = self._normalized_choices[index]
            elif choice_index >= len(self._source_indexes) or self._source_indexes[choice_index] != index:
                yield None
                continue
            else:
                normalized_value = self._normalized_choices[choice_index]
                choice_index += 1
            score = config.scorer(normalized_query, normalized_value, **scorer_kwargs_resolved)
            if not passes_score_cutoff(
                score,
                scorer_type=config.scorer_type,
                score_cutoff=config.score_cutoff,
            ):
                yield None
                continue
            yield Match(
                value=value,
                score=score,
                index=index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )

    def score_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[T] | None]:
        """Return one result per source value, ``None`` for non-matching positions.

        Unlike ``find_many``, the result length always equals the number of
        source values.  Position ``i`` corresponds to source value ``i``.
        Top-one exact shortcuts are not applied; all searchable values are
        scored with the configured scorer. Scorers with compatible RapidFuzz
        metadata use ranked extraction; scorers without it use a direct
        position-aligned pass without unnecessary ranking.

        Args:
            query: Value to score against all indexed values.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``Match`` at position ``i`` if source value ``i`` scored above the
            score cutoff, ``None`` otherwise.  Positions occupied by
            unsearchable values (normalizer returned ``None``) always yield ``None``.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        results: list[Match[T] | None] = [None] * len(self._values)
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return results
        if not config.process_scorer_type_matches:
            scorer_kwargs_resolved = config.scorer_kwargs or {}
            choice_index = 0
            for index, value in enumerate(self._values):
                if self._source_indexes is None:
                    normalized_value = self._normalized_choices[index]
                elif choice_index >= len(self._source_indexes) or self._source_indexes[choice_index] != index:
                    continue
                else:
                    normalized_value = self._normalized_choices[choice_index]
                    choice_index += 1
                score = config.scorer(normalized_query, normalized_value, **scorer_kwargs_resolved)
                if not passes_score_cutoff(
                    score,
                    scorer_type=config.scorer_type,
                    score_cutoff=config.score_cutoff,
                ):
                    continue
                results[index] = Match(
                    value=value,
                    score=score,
                    index=index,
                    query=query,
                    normalized_query=normalized_query,
                    normalized_value=normalized_value,
                )
            return results
        for normalized_value, score, choice_index in self._extract(normalized_query, limit=None, config=config):
            source_index = self._source_index_from_choice(choice_index)
            results[source_index] = Match(
                value=self._values[source_index],
                score=score,
                index=source_index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )
        return results


class MutableFuzzySequenceIndex[T](_SequenceIndexCommon[T]):
    """Fuzzy index over a mutable ordered sequence of values.

    Supports O(1) incremental appends without full index rebuild. Deletions
    update fuzzy choices in place; sparse choices retain stable source slots
    and a compact journal of removed slots to translate result positions.
    Exact-value positions are rebuilt lazily after incremental deletion without
    rebuilding fuzzy choices. Other structural mutations mark the index dirty
    and the next fuzzy query triggers one full rebuild.

    Lookup first normalizes the query. Candidates are ranked by scorer quality,
    then hashable exact equality, then source position. Compatible RapidFuzz metadata
    permits an exact candidate to return without scanning all choices only
    when its score is provably optimal.

    Values that the normalizer rejects (returns ``None``) are stored and
    accessible as source values but excluded from fuzzy comparison.

    Notes:
        ``append`` and incremental deletion avoid a full fuzzy-index rebuild.
        The first fuzzy query after an incremental deletion rebuilds exact-value
        shortcuts in O(n) time; later exact lookups reuse those shortcuts.
        Normalized fuzzy choices remain intact during this recovery.
        Sparse deletion retains one integer per removed source slot until a
        later rebuild. Insert, replacement, retention, slices, and sparse
        batches larger than 1024 positions defer one rebuild to the next
        fuzzy query.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = (
        "_default_config",
        "_deleted_source_slots",
        "_dirty",
        "_exact_duplicate_indexes",
        "_exact_first_index",
        "_normalizer",
        "_normalized_choices",
        "_normalized_values",
        "_optimal_score",
        "_process_scorer_type_matches",
        "_scorer",
        "_scorer_kwargs",
        "_scorer_type",
        "_score_cutoff",
        "_score_hint",
        "_shortcuts_valid",
        "_source_indexes",
        "_values",
    )

    @property
    def normalized_choices(self) -> tuple[str, ...]:
        """Return a current snapshot of normalized searchable choices.

        Returns:
            Tuple containing normalized searchable choices in choice order.

        Side Effects:
            Rebuilds deferred fuzzy index state before creating the snapshot.
        """

        self._ensure_built()
        return tuple(self._normalized_choices)

    @overload
    def __getitem__(self, position: int) -> T: ...

    @overload
    def __getitem__(self, position: slice) -> list[T]: ...

    def __getitem__(self, position: int | slice) -> T | list[T]:
        """Return a source value or an independent source-value slice.

        Args:
            position: Source index or slice.

        Returns:
            Source value for an integer index, or an independent list of source values for a slice.
        """

        if isinstance(position, slice):
            return self._values[position]
        return self._values[position]

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
        """Build the index from ``values`` and configure the matching pipeline.

        Args:
            values: Source values to index. Stored in the order provided.
            normalizer: Callable that maps a value to a searchable string, or
                returns ``None`` to exclude the value from fuzzy lookup.
                Defaults to the built-in normalizer, which is equivalent to
                ``Normalizer.default()``.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Candidates are ranked by scorer quality;
                hashable exact equality wins equal-score ties, followed by source
                position. Compatible RapidFuzz metadata permits an immediate
                exact result only when its score is provably optimal.
            scorer_kwargs: Additional keyword arguments forwarded to the scorer
                on every RapidFuzz call. ``None`` passes no extra arguments.
            scorer_type: Interpretation of scorer output. Use ``DISTANCE``
                when the scorer returns edit distances (lower = more similar);
                use ``SIMILARITY`` for percentage-like scores (higher = more similar).
            score_cutoff: Exclusion threshold applied by RapidFuzz. ``None`` disables the cutoff.
            score_hint: Expected score passed to RapidFuzz so it may select a
                faster implementation. ``None`` disables this optimization.

        Raises:
            TypeError: If any matching configuration argument has an invalid type.

        Side Effects:
            Normalizes all values and builds the index structures eagerly.
        """

        validate_normalizer(normalizer)
        validate_optional_score(score_cutoff, "score_cutoff")
        validate_optional_score(score_hint, "score_hint")
        validate_scorer(scorer)
        self._normalizer = normalizer if normalizer is not None else _default_normalizer
        self._scorer = scorer
        self._scorer_kwargs = copy_scorer_kwargs(scorer_kwargs) if scorer_kwargs is not None else None
        self._scorer_type = validate_scorer_type(scorer_type)
        self._score_cutoff = score_cutoff
        self._score_hint = score_hint
        self._process_scorer_type_matches, self._optimal_score = _process_scorer_metadata(
            scorer,
            self._scorer_kwargs,
            self._scorer_type,
        )
        self._default_config = _MatchConfig(
            scorer=self._scorer,
            scorer_kwargs=self._scorer_kwargs,
            scorer_type=self._scorer_type,
            score_cutoff=self._score_cutoff,
            score_hint=self._score_hint,
            process_scorer_type_matches=self._process_scorer_type_matches,
            optimal_score=self._optimal_score,
        )
        self._values: list[T] = []
        self._normalized_values: list[str | None] = []
        self._exact_duplicate_indexes: dict[Hashable, list[int]] = {}
        self._exact_first_index: dict[Hashable, int] = {}
        self._normalized_choices: list[str] = []
        self._source_indexes: list[int] | None = None
        # Removed stable source slots retained for sparse result translation.
        self._deleted_source_slots: list[int] = []
        self._dirty = False
        # Whether shortcut dictionary values are valid source positions.
        self._shortcuts_valid = True

        for value in values:
            self.append(value)

    @property
    def is_dirty(self) -> bool:
        """Whether the index needs a rebuild before the next fuzzy query.

        Returns:
            ``True`` when derived lookup state is stale and must be rebuilt before
            the next fuzzy query, otherwise ``False``.
        """

        return self._dirty

    def _discard_deleted_shortcuts(self, value: T) -> None:
        """Remove deleted membership while invalidating shortcut positions.

        Args:
            value: Deleted source value.

        Side Effects:
            Updates exact membership metadata and marks shortcut positions as
            stale.
        """

        if _is_hashable(value):
            duplicate_indexes = self._exact_duplicate_indexes.get(value)
            if duplicate_indexes:
                duplicate_indexes.pop()
                if not duplicate_indexes:
                    del self._exact_duplicate_indexes[value]
            else:
                self._exact_first_index.pop(value, None)

        self._shortcuts_valid = False

    def _ensure_built(self) -> None:
        """Rebuild derived index state if structural mutations made it stale.

        Side Effects:
            Rebuilds cached lookup structures and clears the dirty flag when
            structural mutations made the derived state stale.
        """

        if self._dirty:
            self._rebuild()

    def _normalized_value_from_source(self, source_index: int) -> str | None:
        """Return the cached normalized value for a current source position."""

        return self._normalized_values[source_index]

    def _rebuild(self) -> None:
        """Rebuild all index structures from the current values list.

        Side Effects:
            Replaces all derived lookup structures, clears deleted-slot
            metadata, and marks the index clean with valid shortcut positions.
        """

        self._exact_duplicate_indexes = {}
        self._exact_first_index = {}
        self._normalized_choices = []
        self._deleted_source_slots = []
        source_indexes: list[int] | None = None

        for index, value in enumerate(self._values):
            if _is_hashable(value):
                first_index = self._exact_first_index.setdefault(value, index)
                if first_index != index:
                    self._exact_duplicate_indexes.setdefault(value, []).append(index)

            normalized_value = self._normalized_values[index]
            if normalized_value is None:
                continue

            if source_indexes is not None:
                source_indexes.append(index)
            elif index != len(self._normalized_choices):
                source_indexes = [*range(len(self._normalized_choices)), index]
            self._normalized_choices.append(normalized_value)

        self._source_indexes = source_indexes
        if self._source_indexes is None and len(self._normalized_choices) != len(self._values):
            self._source_indexes = list(range(len(self._normalized_choices)))
        self._dirty = False
        self._shortcuts_valid = True

    def _rebuild_exact_shortcuts(self) -> None:
        """Rebuild exact-value positions without rebuilding fuzzy choices.

        Side Effects:
            Replaces exact-value shortcut state and marks its source positions
            valid for the current values.
        """

        exact_first_index: dict[Hashable, int] = {}
        exact_duplicate_indexes: dict[Hashable, list[int]] = {}
        for index, value in enumerate(self._values):
            if not _is_hashable(value):
                continue
            first_index = exact_first_index.setdefault(value, index)
            if first_index != index:
                exact_duplicate_indexes.setdefault(value, []).append(index)

        self._exact_first_index = exact_first_index
        self._exact_duplicate_indexes = exact_duplicate_indexes
        self._shortcuts_valid = True

    def _resolve_query(
        self,
        query: object,
        *,
        config: _MatchConfig | None = None,
    ) -> tuple[Match[T] | None, str | None]:
        """Resolve an optimal exact candidate before matrix scoring.

        Args:
            query: Query value to resolve.
            config: Resolved matching configuration for this call. The index
                defaults are used when omitted.

        Returns:
            ``(match, None)`` when an exact candidate has the scorer's optimal
            result. ``(None, normalized_query)`` when matrix scoring is needed.
            ``(None, None)`` when the query cannot be normalized.
        """

        if config is None:
            config = self._default_config
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None, None
        exact_indexes = self._exact_source_indexes(query)
        exact_matches = self._exact_matches(query, normalized_query, exact_indexes, config=config)
        if config.optimal_score is not None:
            for source_index in exact_indexes:
                match = exact_matches.get(source_index)
                if match is not None and match.score == config.optimal_score:
                    return match, None
        return None, normalized_query

    def _source_index_from_choice(self, choice_index: int) -> int:
        """Return the source index represented by a RapidFuzz choice index.

        Args:
            choice_index: RapidFuzz choice index.

        Returns:
            Current source value index represented by ``choice_index``.
        """

        if self._source_indexes is None:
            return choice_index
        source_slot = self._source_indexes[choice_index]
        return source_slot - bisect_left(self._deleted_source_slots, source_slot)

    def _source_slot_from_position(self, position: int) -> int:
        """Translate a current sparse source position to its retained slot.

        Args:
            position: Current source value position.

        Returns:
            Retained sparse source slot corresponding to ``position``.
        """

        source_slot = position
        for deleted_slot in self._deleted_source_slots:
            if deleted_slot > source_slot:
                break
            source_slot += 1
        return source_slot

    def append(self, value: T) -> None:
        """Append ``value`` and update index structures incrementally.

        O(1) when the index is in a consistent state. When dirty, only the
        raw storage is updated; index structures are rebuilt on the next query.

        Args:
            value: Value to append.

        Side Effects:
            Appends to the internal values list. Updates lookup tables and
            RapidFuzz choice lists when the index is not dirty.
        """

        source_index = len(self._values)
        normalized_value = self._normalizer(value)

        self._values.append(value)
        self._normalized_values.append(normalized_value)

        if self._dirty:
            return

        if _is_hashable(value):
            first_index = self._exact_first_index.setdefault(value, source_index)
            if first_index != source_index:
                self._exact_duplicate_indexes.setdefault(value, []).append(source_index)

        if normalized_value is None:
            if self._source_indexes is None:
                self._source_indexes = list(range(len(self._normalized_choices)))
            return

        source_indexes = self._source_indexes
        if source_indexes is not None:
            if self._deleted_source_slots:
                source_index += len(self._deleted_source_slots)
            source_indexes.append(source_index)
        self._normalized_choices.append(normalized_value)

    def delete_at(self, position: int | slice) -> None:
        """Delete value(s) at ``position`` and update indexed state.

        Args:
            position: Index or slice to delete.

        Side Effects:
            Updates the internal values list. Single-value deletion updates
            dense or sparse fuzzy choice state immediately. Slice deletion
            retains the rebuild path when sparse source mapping is present.
        """

        dense_delete = not self._dirty and self._source_indexes is None
        sparse_delete = not self._dirty and self._source_indexes is not None and not isinstance(position, slice)
        source_slot: int | None = None
        if isinstance(position, slice):
            removed_values = self._values[position]
            removed_normalized = self._normalized_values[position]
            dense_delete = dense_delete and all(value is not None for value in removed_normalized)
        else:
            removed_values = [self._values[position]]
            removed_normalized = self._normalized_values[position]
            dense_delete = dense_delete and removed_normalized is not None
            removed_normalized = [removed_normalized]
            if position < 0:
                position += len(self._values)
            if sparse_delete:
                source_slot = self._source_slot_from_position(position)

        del self._values[position]
        del self._normalized_values[position]
        if dense_delete:
            del self._normalized_choices[position]
            for value in removed_values:
                self._discard_deleted_shortcuts(value)
            return
        if sparse_delete:
            assert source_slot is not None
            assert self._source_indexes is not None
            normalized_value = removed_normalized[0]
            if normalized_value is not None:
                choice_position = self._source_indexes.index(source_slot)
                del self._normalized_choices[choice_position]
                del self._source_indexes[choice_position]
            insort(self._deleted_source_slots, source_slot)
            self._discard_deleted_shortcuts(removed_values[0])
            return
        self._dirty = True

    def delete_at_positions(self, positions: set[int]) -> None:
        """Delete values at the given source positions and update indexed state.

        Args:
            positions: Set of source indexes to remove. Positions outside the current value range are ignored.

        Side Effects:
            Updates the internal values list. Small dense or sparse deletion
            sets update fuzzy choice state immediately; larger sparse sets retain the rebuild path.
        """

        valid_positions = {position for position in positions if 0 <= position < len(self._values)}
        if not valid_positions:
            return
        dense_delete = (
            not self._dirty
            and self._source_indexes is None
            and all(self._normalized_values[position] is not None for position in valid_positions)
        )
        if dense_delete:
            removed_values = [
                (self._values[position], self._normalized_values[position]) for position in valid_positions
            ]
            if len(valid_positions) <= _DENSE_INCREMENTAL_DELETE_LIMIT:
                for position in sorted(valid_positions, reverse=True):
                    del self._values[position]
                    del self._normalized_values[position]
                    del self._normalized_choices[position]
            else:
                self._values = [value for index, value in enumerate(self._values) if index not in valid_positions]
                self._normalized_values = [
                    normalized_value
                    for index, normalized_value in enumerate(self._normalized_values)
                    if index not in valid_positions
                ]
                self._normalized_choices = [
                    value for index, value in enumerate(self._normalized_choices) if index not in valid_positions
                ]
            for value, _normalized_value in removed_values:
                self._discard_deleted_shortcuts(value)
            return
        if (
            not self._dirty
            and self._source_indexes is not None
            and len(valid_positions) <= _SPARSE_INCREMENTAL_DELETE_LIMIT
        ):
            for position in sorted(valid_positions, reverse=True):
                self.delete_at(position)
            return
        self._values = [value for index, value in enumerate(self._values) if index not in valid_positions]
        self._normalized_values = [
            normalized_value
            for index, normalized_value in enumerate(self._normalized_values)
            if index not in valid_positions
        ]
        self._dirty = True

    def delete_value(self, value: T) -> bool:
        """Remove the first occurrence of ``value`` and update indexed state.

        Args:
            value: Value to remove.

        Returns:
            ``True`` if the value was found and removed, ``False`` otherwise.

        Side Effects:
            Updates the internal values list when the value is found. A
            single dense or sparse deletion updates fuzzy choice state immediately.
        """

        try:
            idx = self._values.index(value)
        except ValueError:
            return False
        self.delete_at(idx)
        return True

    def find_many(
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
        """Return up to ``limit`` best matches in scorer-defined order.

        The query is normalized and passed to RapidFuzz for fuzzy matching.
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
            Matches ordered from best to worst according to ``scorer_type``.

        Raises:
            TypeError: If a matching override or ``limit`` has an invalid type.
            ValueError: If ``limit`` is negative, or ``scorer`` is overridden without
                ``scorer_type`` and has no compatible RapidFuzz metadata.
        """

        validate_limit(limit)
        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self._ensure_built()

        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return []
        exact_indexes = self._exact_source_indexes(query)
        return self._ranked_matches(
            query,
            normalized_query,
            limit=limit,
            exact_indexes=exact_indexes,
            config=config,
        )

    def find_one(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Match[T] | None:
        """Return the best match above the score cutoff, or ``None``.

        Candidates are ordered by scorer result, then hashable exact equality with the
        query, then source position. An exact candidate returns immediately
        only when compatible RapidFuzz metadata proves that its score is
        optimal.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Best match for ``query``, or ``None`` when no acceptable match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Custom scorers without compatible RapidFuzz metadata evaluate all
            searchable values before hashable exact equality is used to break score
            ties. Overriding ``scorer``, ``scorer_kwargs``, or ``scorer_type``
            recomputes RapidFuzz scorer metadata for this call instead of
            reusing the collection's cached metadata.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self._ensure_built()

        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return None
        exact_indexes = self._exact_source_indexes(query)
        exact_matches = self._exact_matches(query, normalized_query, exact_indexes, config=config)
        if config.optimal_score is not None:
            for source_index in exact_indexes:
                match = exact_matches.get(source_index)
                if match is not None and match.score == config.optimal_score:
                    return match
        matches = self._ranked_matches(
            query,
            normalized_query,
            limit=1,
            exact_indexes=exact_indexes,
            exact_matches=exact_matches,
            config=config,
        )
        return matches[0] if matches else None

    def find_one_batch_cdist(
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
        """Return best matches through bounded RapidFuzz ``cdist`` scoring.

        Dirty state is rebuilt before scoring. Exact candidates with a proven
        optimal score resolve before matrix evaluation. Other candidates are
        ordered by score, hashable exact equality, and source position.

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
            One best match or ``None`` per query, preserving query order.

        Raises:
            TypeError: If a matching override or chunk size has an invalid type.
            ValueError: If a chunk size is less than 1, or ``scorer`` is overridden
                without ``scorer_type`` and has no compatible RapidFuzz metadata.
            ModuleNotFoundError: If matrix scoring is required but NumPy is
                not installed through ``rapidfuzz-collections[cdist]``.

        Notes:
            Candidates are ranked by scorer quality. Hashable exact equality wins
            equal-score ties, followed by source position. Compatible
            RapidFuzz metadata permits an immediate exact result before matrix
            scoring only when its score is provably optimal. This specialized
            path may allocate more memory or run slower than
            ``find_one_batch``.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self._ensure_built()
        return _find_one_batch_cdist(
            self._values,
            self._normalized_choices,
            queries,
            resolve_query=lambda resolve_target: self._resolve_query(resolve_target, config=config),
            exact_source_indexes=self._exact_source_indexes,
            normalized_value_from_source=self._normalized_value_from_source,
            source_index_from_choice=self._source_index_from_choice,
            scorer=config.scorer,
            scorer_kwargs=config.scorer_kwargs,
            scorer_type=config.scorer_type,
            score_cutoff=config.score_cutoff,
            score_hint=config.score_hint,
            process_scorer_type_matches=config.process_scorer_type_matches,
            query_chunk_size=query_chunk_size,
            choice_chunk_size=choice_chunk_size,
            workers=workers,
        )

    def insert_at(self, position: int, value: T) -> None:
        """Insert ``value`` at ``position`` and mark the index dirty.

        Args:
            position: Insertion position.
            value: Value to insert.

        Side Effects:
            Updates the internal values list. Marks the index dirty; the next fuzzy query will trigger a full rebuild.
        """

        normalized_value = self._normalizer(value)
        self._values.insert(position, value)
        self._normalized_values.insert(position, normalized_value)
        self._dirty = True

    def iter_scores(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Iterator[Match[T] | None]:
        """Yield one fuzzy scoring result per source value in source order.

        Top-one exact shortcuts are not applied; all searchable values are
        scored through the configured scorer. The extraction-only
        ``score_hint`` optimization is not applied by this streaming path.
        This is the memory-efficient counterpart to ``score_all`` and does not
        allocate a result list.

        Args:
            query: Value to score against all indexed values.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Yields:
            ``Match`` for values satisfying the score cutoff and ``None`` for rejected or unsearchable positions.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            This streaming operation calls the scorer once per searchable
            value. Use ``score_all`` when a materialized, position-aligned list
            is required; this iterator minimizes output memory.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self._ensure_built()
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            yield from (None for _ in self._values)
            return

        scorer_kwargs_resolved = config.scorer_kwargs or {}
        for index, (value, normalized_value) in enumerate(zip(self._values, self._normalized_values, strict=True)):
            if normalized_value is None:
                yield None
                continue
            score = config.scorer(normalized_query, normalized_value, **scorer_kwargs_resolved)
            if not passes_score_cutoff(
                score,
                scorer_type=config.scorer_type,
                score_cutoff=config.score_cutoff,
            ):
                yield None
                continue
            yield Match(
                value=value,
                score=score,
                index=index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )

    def keep_at_positions(self, positions: set[int]) -> int:
        """Retain only values at ``positions``; remove all others.

        Args:
            positions: Set of source indexes to retain. Positions outside the current value range are ignored.

        Returns:
            Number of removed values.

        Side Effects:
            Updates the internal values list when values are removed. Marks
            the index dirty; the next fuzzy query will trigger a full rebuild.
        """

        valid_positions = {position for position in positions if 0 <= position < len(self._values)}
        removed = len(self._values) - len(valid_positions)
        if removed == 0:
            return 0
        self._values = [value for index, value in enumerate(self._values) if index in valid_positions]
        self._normalized_values = [
            normalized_value
            for index, normalized_value in enumerate(self._normalized_values)
            if index in valid_positions
        ]
        self._dirty = True
        return removed

    def replace_at(self, position: int | slice, value: T | Iterable[T]) -> None:
        """Replace value(s) at ``position`` and mark the index dirty.

        Args:
            position: Index or slice to replace.
            value: Replacement value or values.

        Side Effects:
            Updates the internal values list. Marks the index dirty; the next fuzzy query will trigger a full rebuild.
        """

        if isinstance(position, slice):
            new_values: list[T] = list(value)
            new_normalized_values = [self._normalizer(new_value) for new_value in new_values]
            self._values[position] = new_values
            self._normalized_values[position] = new_normalized_values
        else:
            normalized_value = self._normalizer(value)
            self._values[position] = value
            self._normalized_values[position] = normalized_value
        self._dirty = True

    def score_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[T] | None]:
        """Return one result per source value, ``None`` for non-matching positions.

        Unlike ``find_many``, the result length always equals the number of
        source values. Position ``i`` corresponds to source value ``i``.
        Top-one exact shortcuts are not applied; all searchable values are
        scored with the configured scorer. Scorers with compatible RapidFuzz
        metadata use ranked extraction; scorers without it use a direct
        position-aligned pass without unnecessary ranking.

        Args:
            query: Value to score against all indexed values.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            ``Match`` at position ``i`` if source value ``i`` scored above the
            score cutoff, ``None`` otherwise. Positions occupied by
            unsearchable values (normalizer returned ``None``) always yield ``None``.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self._ensure_built()
        results: list[Match[T] | None] = [None] * len(self._values)
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return results
        if not config.process_scorer_type_matches:
            scorer_kwargs_resolved = config.scorer_kwargs or {}
            for index, (value, normalized_value) in enumerate(zip(self._values, self._normalized_values, strict=True)):
                if normalized_value is None:
                    continue
                score = config.scorer(normalized_query, normalized_value, **scorer_kwargs_resolved)
                if not passes_score_cutoff(
                    score,
                    scorer_type=config.scorer_type,
                    score_cutoff=config.score_cutoff,
                ):
                    continue
                results[index] = Match(
                    value=value,
                    score=score,
                    index=index,
                    query=query,
                    normalized_query=normalized_query,
                    normalized_value=normalized_value,
                )
            return results
        for normalized_value, score, choice_index in self._extract(normalized_query, limit=None, config=config):
            source_index = self._source_index_from_choice(choice_index)
            results[source_index] = Match(
                value=self._values[source_index],
                score=score,
                index=source_index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )
        return results

    def sort(self, *, key: Callable[[T], Any] | None = None, reverse: bool = False) -> None:
        """Sort source values in place and invalidate derived lookup state.

        Args:
            key: Optional callable used to extract comparison keys.
            reverse: Whether to sort in descending order.

        Side Effects:
            Reorders source values and marks derived fuzzy lookup structures for rebuild before the next lookup.
        """

        pairs = list(zip(self._values, self._normalized_values, strict=True))
        if key is None:
            pairs.sort(key=lambda pair: pair[0], reverse=reverse)
        else:
            pairs.sort(key=lambda pair: key(pair[0]), reverse=reverse)
        self._values = [value for value, _ in pairs]
        self._normalized_values = [normalized_value for _, normalized_value in pairs]
        self._dirty = True
