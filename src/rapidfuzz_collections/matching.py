from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Match[T]:
    """A single fuzzy or exact match result.

    Attributes:
        value: Original (un-normalized) collection value.
        score: Scorer-dependent score or distance.  For SIMILARITY scorers,
            higher is better; for DISTANCE scorers, lower is better.
        index: Zero-based source position when the collection exposes
            positional fuzzy results. ``None`` means the selected strategy is
            intentionally position-free.
        query: Original query as provided by the caller, before normalization.
        normalized_query: Normalized form of the query used for the fuzzy comparison.
        normalized_value: Normalized form of the matched collection value.
    """

    value: T
    score: float | int
    index: int | None
    query: object
    normalized_query: str
    normalized_value: str


@dataclass(frozen=True, slots=True)
class MappingMatch[K, V]:
    """A single fuzzy or exact match result for a mapping key.

    Attributes:
        key: Original (un-normalized) mapping key that matched the query.
        value: Value stored under the matched key.
        score: Scorer-dependent score or distance.  For SIMILARITY scorers,
            higher is better; for DISTANCE scorers, lower is better.
        index: Zero-based source position of the matched key when the
            collection exposes positional fuzzy results. ``None`` means the
            selected strategy is intentionally position-free.
        query: Original query as provided by the caller, before normalization.
        normalized_query: Normalized form of the query used for the fuzzy comparison.
        normalized_key: Normalized form of the matched key.
    """

    key: K
    value: V
    score: float | int
    index: int | None
    query: object
    normalized_query: str
    normalized_key: str


@dataclass(frozen=True, slots=True)
class ValueMatch[T]:
    """A position-free fuzzy or exact match result for keyed indexes.

    Keyed indexes do not track sequence positions. Collection facades adapt this
    result to ``Match`` or ``MappingMatch`` and expose ``index=None`` where the
    public result type includes an index field.

    Attributes:
        value: Original collection value.
        score: Scorer-dependent score or distance. For similarity scorers,
            higher is better; for distance scorers, lower is better.
        query: Original query as provided by the caller, before normalization.
        normalized_query: Normalized form of the query used for fuzzy comparison.
        normalized_value: Normalized form of the matched value.
    """

    value: T
    score: float | int
    query: object
    normalized_query: str
    normalized_value: str


@dataclass(frozen=True, slots=True)
class KeyValueMatch[K, V]:
    """A fuzzy or exact mapping match without positional semantics.

    Attributes:
        key: Original mapping key that matched the query.
        value: Value stored under the matched key.
        score: Scorer-dependent score or distance.
        query: Original query as provided by the caller.
        normalized_query: Normalized query used for fuzzy comparison.
        normalized_key: Normalized matched key.
    """

    key: K
    value: V
    score: float | int
    query: object
    normalized_query: str
    normalized_key: str
