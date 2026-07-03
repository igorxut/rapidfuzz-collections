from enum import Enum


class IndexStrategy(Enum):
    """Fuzzy-index storage strategy for dict-like and set-like collections.

    Attributes:
        SEQUENCE: Store normalized choices in sequence order. This is the
            default because benchmarks show it is the strongest general
            read-heavy strategy.
        KEYED: Store normalized choices keyed by each unique hashable value.
            This can reduce build overhead and selected mutation costs for
            unique hashable domains, especially when normalized collisions are
            common. Mutable keyed indexes trade additional memory for an exact
            registry that preserves canonical stored objects.
    """

    SEQUENCE = "sequence"
    KEYED = "keyed"


class ScorerType(Enum):
    """Score interpretation mode for a RapidFuzz scorer.

    Attributes:
        DISTANCE: Lower score means greater similarity (e.g., Levenshtein distance).
        SIMILARITY: Higher score means greater similarity (e.g., WRatio).
    """

    DISTANCE = 0
    SIMILARITY = 1
