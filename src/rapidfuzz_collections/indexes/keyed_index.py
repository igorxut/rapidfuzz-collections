from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from ..configuration import _UNCHANGED
from ..enums import ScorerType
from ..matching import ValueMatch
from ..normalization import _default_normalizer
from .base import (
    Scorer,
    _is_hashable,
    _MatchConfig,
    _process_scorer_metadata,
    _resolve_match_config,
    copy_scorer_kwargs,
    index_config_kwargs,
    validate_limit,
    validate_normalizer,
    validate_optional_score,
    validate_scorer,
    validate_scorer_type,
)
from .sequence_index import passes_score_cutoff


class _BaseFuzzyKeyedIndex[T: Hashable]:
    """Share keyed fuzzy lookup logic between concrete index variants."""

    __slots__ = (
        "_choices",
        "_default_config",
        "_exact_values",
        "_normalizer",
        "_optimal_score",
        "_process_scorer_type_matches",
        "_scorer",
        "_scorer_kwargs",
        "_scorer_type",
        "_score_cutoff",
        "_score_hint",
    )

    def _add_choice(self, value: T) -> str | None:
        """Add a normalized RapidFuzz choice for ``value``.

        Args:
            value: Value to add to searchable choices.

        Returns:
            Normalized form of ``value`` if it was added, otherwise ``None``.

        Side Effects:
            Adds a previously absent value to the exact registry and, when
            searchable, to the RapidFuzz choices.
        """

        if value in self._exact_values:
            return None
        normalized_value = self._normalizer(value)
        self._exact_values[value] = value
        if normalized_value is None:
            return None
        self._choices[value] = normalized_value
        return normalized_value

    def _exact_candidate(
        self,
        query: object,
        normalized_query: str,
        *,
        config: _MatchConfig,
    ) -> tuple[str, int | float, T] | None:
        """Return the scored canonical value equal to ``query``, if searchable."""

        if not _is_hashable(query) or query not in self._exact_values:
            return None
        value = self._exact_values[query]
        normalized_value = self._choices.get(value)
        if normalized_value is None:
            return None
        score = self._score_pair(normalized_query, normalized_value, config=config)
        if score is None:
            return None
        return normalized_value, score, value

    def _extract(
        self,
        normalized_query: str,
        *,
        limit: int | None,
        config: _MatchConfig,
    ) -> list[tuple[str, int | float, T]]:
        """Return matches for a normalized query.

        The configured scorer is used to compare the normalized query against
        stored normalized choices. Returned matches are ordered according to the
        scorer type: similarity scorers are sorted from the highest score to lowest,
        while distance scorers are sorted from the lowest score to highest.

        Args:
            normalized_query: Normalized query value to compare with stored choices.
            limit: Maximum number of matches to return. If ``None``, all matches
                passing the configured score cutoff are returned.
            config: Resolved matching configuration for this call.

        Returns:
            A list of tuples containing the normalized matched value, the score,
            and the original stored value.
        """

        if config.process_scorer_type_matches:
            return list(
                process.extract(
                    normalized_query,
                    self._choices,
                    scorer=config.scorer,
                    score_cutoff=config.score_cutoff,
                    score_hint=config.score_hint,
                    scorer_kwargs=config.scorer_kwargs,
                    limit=limit,
                )
            )

        scorer_kwargs = config.scorer_kwargs or {}
        matches = [
            (normalized_value, score, value)
            for value, normalized_value in self._choices.items()
            if passes_score_cutoff(
                score := config.scorer(normalized_query, normalized_value, **scorer_kwargs),
                scorer_type=config.scorer_type,
                score_cutoff=config.score_cutoff,
            )
        ]
        matches.sort(key=lambda result: result[1], reverse=config.scorer_type == ScorerType.SIMILARITY)
        return matches if limit is None else matches[:limit]

    def _extract_one(self, normalized_query: str, *, config: _MatchConfig) -> tuple[str, int | float, T] | None:
        """Return the highest-ranked candidate for a normalized query."""

        if config.process_scorer_type_matches:
            return process.extractOne(  # type: ignore[call-overload]
                normalized_query,
                self._choices,
                scorer=config.scorer,
                score_cutoff=config.score_cutoff,
                score_hint=config.score_hint,
                scorer_kwargs=config.scorer_kwargs,
            )
        matches = self._extract(normalized_query, limit=1, config=config)
        return matches[0] if matches else None

    def _init_base(
        self,
        *,
        normalizer: Callable[[object], str | None] | None,
        scorer: Scorer,
        scorer_kwargs: dict[str, Any] | None,
        scorer_type: ScorerType,
        score_cutoff: int | float | None,
        score_hint: int | float | None,
    ) -> None:
        """Initialize shared keyed fuzzy lookup state.

        Args:
            normalizer: Callable that maps values to searchable strings, or ``None`` to use the default normalizer.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: RapidFuzz scorer callable used for fuzzy search.
            scorer_kwargs: Additional keyword arguments forwarded to the scorer.
            scorer_type: Interpretation of scorer output.
            score_cutoff: Optional minimum similarity score or maximum distance score to accept.
            score_hint: Optional expected score passed to RapidFuzz for optimization.

        Raises:
            TypeError: If any matching configuration argument has an invalid type.

        Side Effects:
            Initializes this index's matching configuration and empty lookup state.
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
        self._choices: dict[T, str] = {}
        self._exact_values: dict[T, T] = {}

    # noinspection PyMethodMayBeStatic
    def _match(
        self,
        value: T,
        *,
        query: object,
        normalized_query: str,
        normalized_value: str,
        score: int | float,
    ) -> ValueMatch[T]:
        """Construct a position-free match for one indexed value.

        Args:
            value: Indexed value used to build the match.
            query: Original query value.
            normalized_query: Normalized query value.
            normalized_value: Normalized indexed value.
            score: Configured scorer result for the normalized pair.

        Returns:
            Position-free match for ``value``.
        """

        return ValueMatch(
            value=value,
            score=score,
            query=query,
            normalized_query=normalized_query,
            normalized_value=normalized_value,
        )

    def _ranked_extract(
        self,
        query: object,
        normalized_query: str,
        *,
        limit: int | None,
        exact_candidate: tuple[str, int | float, T] | None = None,
        config: _MatchConfig,
    ) -> list[tuple[str, int | float, T]]:
        """Return candidates ordered by score, exact equality, and insertion order."""

        if limit == 0:
            return []

        if limit == 1:
            result = self._extract_one(normalized_query, config=config)
            candidates = [] if result is None else [result]
        else:
            candidates = self._extract(normalized_query, limit=limit, config=config)
        if exact_candidate is None:
            exact_candidate = self._exact_candidate(query, normalized_query, config=config)

        candidates_by_value = {value: (normalized_value, score, value) for normalized_value, score, value in candidates}
        if exact_candidate is not None:
            candidates_by_value[exact_candidate[2]] = exact_candidate

        if limit is None:
            # _extract already ranked every accepted choice. Moving the single
            # exact keyed candidate to the front of its score group preserves
            # the contract without sorting the full result a second time.
            ranked = list(candidates_by_value.values())
            if exact_candidate is None:
                return ranked

            exact_value = exact_candidate[2]
            exact_position = next(position for position, candidate in enumerate(ranked) if candidate[2] == exact_value)
            exact_score = ranked[exact_position][1]
            score_group_start = exact_position
            while score_group_start > 0 and ranked[score_group_start - 1][1] == exact_score:
                score_group_start -= 1
            if score_group_start != exact_position:
                ranked.insert(score_group_start, ranked.pop(exact_position))
            return ranked

        def order_key(res: tuple[str, int | float, T]) -> tuple[int | float, bool]:
            _, score, value = res
            ordered_score = score if config.scorer_type == ScorerType.DISTANCE else -score
            is_exact = exact_candidate is not None and value == exact_candidate[2]
            return ordered_score, not is_exact

        # Both RapidFuzz's mapping extraction and the direct fallback preserve
        # _choices insertion order. Python's stable sort therefore supplies the
        # final documented tie-break without another per-value rank mapping.
        ranked = sorted(candidates_by_value.values(), key=order_key)
        return ranked[:limit]

    # noinspection PyMethodMayBeStatic
    def _score_pair(self, normalized_query: str, normalized_value: str, *, config: _MatchConfig) -> int | float | None:
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
        """Return independent constructor configuration for another keyed index.

        Args:
            deepcopy_memo: Optional memo dictionary used when deep-copying scorer
                keyword arguments as part of an ongoing deepcopy operation.

        Returns:
            Keyword arguments compatible with keyed index constructors.
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

    def exact_match(
        self,
        value: T,
        *,
        query: object,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> ValueMatch[T] | None:
        """Return a scored exact match for the canonical stored value.

        Args:
            value: Value equal to an indexed value. The result contains the original object added to the index.
            query: Original query value.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: RapidFuzz extraction hint override for this call. It has
                no effect because this operation calls the scorer directly.

        Returns:
            Scored exact match for the stored value equal to ``value``, or
            ``None`` when the query or stored value is unsearchable or the
            scorer result does not satisfy the cutoff.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.
            KeyError: If no indexed value equals ``value``.

        Notes:
            This explicit operation evaluates only the requested exact value.
            It does not search for a higher-scoring non-exact candidate or
            pass ``score_hint`` to the scorer.
        """

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        stored_value = self._exact_values[value]
        normalized_query = self._normalizer(query)
        normalized_value = self._choices.get(stored_value)
        if normalized_query is None or normalized_value is None:
            return None
        score = self._score_pair(normalized_query, normalized_value, config=config)
        if score is None:
            return None
        return self._match(
            stored_value,
            query=query,
            normalized_query=normalized_query,
            normalized_value=normalized_value,
            score=score,
        )

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
    ) -> list[ValueMatch[T]]:
        """Return fuzzy matches in scorer-defined order.

        Similarity scores are ordered from highest to lowest; distance scores
        are ordered from lowest to highest. Among equal scores, values equal
        to the query precede non-exact values, followed by insertion order.

        Args:
            query: Value to search for.
            limit: Maximum number of matches. ``None`` returns all candidates above the score cutoff.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            List of fuzzy matches for ``query`` ordered from best to worst.

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
        return [
            self._match(
                value,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
                score=score,
            )
            for normalized_value, score, value in self._ranked_extract(
                query,
                normalized_query,
                limit=limit,
                config=config,
            )
        ]

    def find_one(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> ValueMatch[T] | None:
        """Return the best searchable match above the score cutoff.

        Candidates are ordered by scorer result, then exact equality with the
        query, then choice insertion order. An exact candidate returns
        immediately only when compatible RapidFuzz metadata proves that its
        score is optimal.

        Args:
            query: Value to search for.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            Best match for ``query``, or ``None`` when the query is
            unsearchable or no acceptable match is found.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            Custom scorers without compatible RapidFuzz metadata evaluate all
            searchable values before exact equality is used to break score
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
        exact_candidate = self._exact_candidate(query, normalized_query, config=config)
        if (
            exact_candidate is not None
            and config.optimal_score is not None
            and exact_candidate[1] == config.optimal_score
        ):
            normalized_value, score, value = exact_candidate
            return self._match(
                value,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
                score=score,
            )
        results = self._ranked_extract(
            query,
            normalized_query,
            limit=1,
            exact_candidate=exact_candidate,
            config=config,
        )
        if not results:
            return None
        normalized_value, score, value = results[0]
        return self._match(
            value,
            query=query,
            normalized_query=normalized_query,
            normalized_value=normalized_value,
            score=score,
        )

    def iter_scores(
        self,
        values: Iterable[T],
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> Iterator[ValueMatch[T] | None]:
        """Yield one scorer result per source value in supplied order.

        Args:
            values: Source values to score in iteration order.
            query: Value to score against every source value.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: RapidFuzz extraction hint override for this call. It has
                no effect because this operation calls the scorer directly.

        Yields:
            ``ValueMatch`` for values above the score cutoff and ``None`` for unsearchable or rejected values.

        Raises:
            TypeError: If a matching override has an invalid type.
            ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.

        Notes:
            ``score_hint`` is accepted for configuration consistency but is
            not passed to direct scorer calls on this streaming path.

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
            yield from (None for _ in values)
            return
        scorer_kwargs_resolved = config.scorer_kwargs or {}
        for value in values:
            normalized_value = self._choices.get(value)
            if normalized_value is None:
                yield None
                continue
            score = config.scorer(normalized_query, normalized_value, **scorer_kwargs_resolved)
            if not passes_score_cutoff(score, scorer_type=config.scorer_type, score_cutoff=config.score_cutoff):
                yield None
                continue
            yield self._match(
                value,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
                score=score,
            )

    def score_all(
        self,
        values: Iterable[T],
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[ValueMatch[T] | None]:
        """Return materialized scoring results aligned with supplied values.

        Scorers with compatible RapidFuzz metadata use ranked mapping
        extraction. Scorers without it use a direct position-aligned pass
        without unnecessary ranking.

        Args:
            values: Source values to score in iteration order.
            query: Value to score against every source value.
            scorer: Override for this call only; omit to use the collection's default scorer.
            scorer_kwargs: Override for this call only; omit to use the collection's default scorer keyword arguments.
            scorer_type: Override for this call only; omit to use the collection's default scorer type.
            score_cutoff: Override for this call only; omit to use the collection's default score cutoff.
            score_hint: Override for this call only; omit to use the collection's default score hint.

        Returns:
            One result per source value in supplied order. ``None`` at
            position ``i`` if the value did not match or is unsearchable.

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
        source_values = list(values)
        results: list[ValueMatch[T] | None] = [None] * len(source_values)
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return results

        if not config.process_scorer_type_matches:
            scorer_kwargs_resolved = config.scorer_kwargs or {}
            for position, value in enumerate(source_values):
                normalized_value = self._choices.get(value)
                if normalized_value is None:
                    continue
                score = config.scorer(normalized_query, normalized_value, **scorer_kwargs_resolved)
                if not passes_score_cutoff(
                    score,
                    scorer_type=config.scorer_type,
                    score_cutoff=config.score_cutoff,
                ):
                    continue
                results[position] = self._match(
                    value,
                    query=query,
                    normalized_query=normalized_query,
                    normalized_value=normalized_value,
                    score=score,
                )
            return results

        positions = {value: index for index, value in enumerate(source_values)}
        duplicate_positions: dict[T, list[int]] | None = None
        if len(positions) != len(source_values):
            duplicate_positions = {}
            for index, value in enumerate(source_values):
                duplicate_positions.setdefault(value, []).append(index)  # type: ignore[call-overload]
        for normalized_value, score, value in self._extract(normalized_query, limit=None, config=config):
            position = positions.get(value)
            if position is None:
                continue
            if duplicate_positions is None:
                results[position] = self._match(
                    source_values[position],
                    query=query,
                    normalized_query=normalized_query,
                    normalized_value=normalized_value,
                    score=score,
                )
            else:
                for duplicate_position in duplicate_positions[value]:
                    results[duplicate_position] = self._match(
                        source_values[duplicate_position],
                        query=query,
                        normalized_query=normalized_query,
                        normalized_value=normalized_value,
                        score=score,
                    )
        return results


class ImmutableFuzzyKeyedIndex[T: Hashable](_BaseFuzzyKeyedIndex[T]):
    """Read-only fuzzy index over unique hashable values.

    This index stores exact values and normalized RapidFuzz mapping choices,
    but no mutation-only reverse state. Candidates are ranked by scorer
    quality, then exact equality, then insertion order.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = ()

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
        """Build keyed fuzzy choices from unique hashable values.

        Args:
            values: Values in deterministic tie-breaking order.
            normalizer: Callable returning searchable strings or ``None`` for values excluded from fuzzy comparison.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Candidates are ranked by scorer quality;
                exact equality wins equal-score ties, followed by insertion
                order. Compatible RapidFuzz metadata permits an immediate
                exact result only when its score is provably optimal.
            scorer_kwargs: Keyword arguments forwarded to the scorer.
            scorer_type: Whether lower or higher scorer output is preferable.
            score_cutoff: Configured fuzzy-match acceptance threshold.
            score_hint: Optional RapidFuzz implementation-selection hint.

        Raises:
            TypeError: If any matching configuration argument has an invalid type.

        Side Effects:
            Normalizes ``values`` and eagerly builds exact-value and RapidFuzz
            keyed lookup state.
        """

        self._init_base(
            normalizer=normalizer,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        for value in values:
            self._add_choice(value)


class MutableFuzzyKeyedIndex[T: Hashable](_BaseFuzzyKeyedIndex[T]):
    """Maintain normalized RapidFuzz mapping choices keyed by original values.

    The index deliberately stores no source-position array. An exact-value
    registry preserves the original object for equal searchable lookup values.
    Candidates are ranked by scorer quality, then exact equality, then
    insertion order.
    Values rejected by the normalizer remain stored but do not participate in
    fuzzy lookup. A reverse index
    ``_normalized_to_values`` maps each normalized form to its ordered list of
    original values, enabling O(K_collision) removal without scanning all
    choices.

    Exceptions raised by user-supplied normalizers, scorers, or other
    callbacks propagate unchanged.
    """

    __slots__ = ("_normalized_to_values",)

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
        """Build keyed fuzzy choices from unique hashable values.

        Args:
            values: Values in deterministic tie-breaking order.
            normalizer: Callable returning searchable strings or ``None`` for values excluded from fuzzy comparison.
                Supplied mutable or stateful callables must be fully configured before
                use and must not change afterward; callers are responsible for this invariant.
            scorer: Scorer callable. Candidates are ranked by scorer quality;
                exact equality wins equal-score ties, followed by insertion
                order. Compatible RapidFuzz metadata permits an immediate
                exact result only when its score is provably optimal.
            scorer_kwargs: Keyword arguments forwarded to the scorer.
            scorer_type: Whether lower or higher scorer output is preferable.
            score_cutoff: Configured fuzzy-match acceptance threshold.
            score_hint: Optional RapidFuzz implementation-selection hint.

        Raises:
            TypeError: If any matching configuration argument has an invalid type.

        Side Effects:
            Normalizes ``values`` and eagerly builds mutable exact-value,
            reverse, and RapidFuzz keyed lookup state.
        """

        self._init_base(
            normalizer=normalizer,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        self._normalized_to_values: dict[str, list[T]] = {}
        for value in values:
            self.add(value)

    def add(self, value: T) -> None:
        """Add a value to exact and fuzzy lookup state.

        Args:
            value: Value to add. Values rejected by the normalizer remain
                stored but are excluded from fuzzy lookup.

        Side Effects:
            Adds a previously absent value to collection-tracking state and
            updates fuzzy and reverse lookup state when the value is searchable.
        """

        normalized_value = self._add_choice(value)
        if normalized_value is None:
            return
        if normalized_value in self._normalized_to_values:
            self._normalized_to_values[normalized_value].append(value)
        else:
            self._normalized_to_values[normalized_value] = [value]

    def batch_remove(self, values: Iterable[T]) -> None:
        """Remove multiple values with one grouped update per normalized form.

        Collects all deletions per normalized form first, then rebuilds each
        affected group's list once instead of calling ``list.remove`` per deleted value.

        Args:
            values: Values to remove. Values not in the index are skipped.

        Side Effects:
            Updates ``_choices`` and ``_normalized_to_values``.
        """
        by_norm: dict[str, list[T]] = {}

        for value in values:
            self._exact_values.pop(value, None)
            normalized = self._choices.pop(value, None)
            if normalized is None:
                continue
            by_norm.setdefault(normalized, []).append(value)

        for normalized, deleted in by_norm.items():
            values_list = self._normalized_to_values.get(normalized)
            if values_list is None:
                continue
            deleted_set = set(deleted)
            new_list = [v for v in values_list if v not in deleted_set]
            if not new_list:
                del self._normalized_to_values[normalized]
            else:
                self._normalized_to_values[normalized] = new_list

    def remove(self, value: T) -> None:
        """Remove a value from exact and fuzzy lookup state if present.

        Args:
            value: Value to remove.

        Notes:
            A missing value is ignored. This idempotent behavior supports
            synchronization after the owning collection has already removed
            the value.

        Side Effects:
            If ``value`` is present, updates exact, fuzzy, and reverse lookup
            state in O(K_collision) without scanning all choices.
        """

        self._exact_values.pop(value, None)
        normalized_value = self._choices.pop(value, None)
        if normalized_value is None:
            return
        values_list = self._normalized_to_values.get(normalized_value)
        if values_list is None:
            return
        values_list.remove(value)
        if not values_list:
            del self._normalized_to_values[normalized_value]
