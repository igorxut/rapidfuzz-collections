"""Small helpers shared by tests."""

from inspect import Signature, getmembers

from rapidfuzz_collections import MappingMatch, Match, ValueMatch


class HashableCycleNode:
    """Hold a reference while retaining identity-based hashing."""

    def __init__(self) -> None:
        self.owner: object | None = None


class SearchableEqualityKey:
    """Represent equal keys whose normalization eligibility can differ."""

    def __init__(self, identity: str, normalized: str | None) -> None:
        self.identity = identity
        self.normalized = normalized

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchableEqualityKey):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)


def casefold_string(value: object) -> str | None:
    """Case-fold string values and reject other objects."""

    return value.casefold() if isinstance(value, str) else None


def normalize_equality_key(value: object) -> str | None:
    """Return the configured normalized form for an equality test key."""

    if not isinstance(value, SearchableEqualityKey):
        return None
    return value.normalized


def normalize_boolean_one(value: object) -> str:
    """Normalize equal boolean and integer values to the same searchable text."""

    # noinspection PyStringConversionWithoutDunderMethod
    return "boolean-one" if value in (True, 1) else str(value)


def require_not_none[T](value: T | None) -> T:
    """Return a value after asserting it is not `None`."""

    assert value is not None
    return value


def public_methods(cls: type[object]) -> set[str]:
    """Return callable public attributes visible on a class, including inherited ones."""

    return {name for name, member in getmembers(cls) if callable(member) and not name.startswith("_")}


def signature_text(signature: Signature) -> str:
    """Return a stable text representation of a function signature."""

    return signature.format()


def mapping_match_signature[K, V](
    match: MappingMatch[K, V] | None,
) -> tuple[K, V, int | float, int | None, object, str, str] | None:
    """Return stable fields for mapping fuzzy match assertions."""

    if match is None:
        return None
    return (
        match.key,
        match.value,
        match.score,
        match.index,
        match.query,
        match.normalized_query,
        match.normalized_key,
    )


def keyed_value_match_signature[T](
    match: ValueMatch[T] | None,
) -> tuple[T, int | float, object, str, str] | None:
    """Return stable fields for keyed-index fuzzy match assertions."""

    if match is None:
        return None
    return (
        match.value,
        match.score,
        match.query,
        match.normalized_query,
        match.normalized_value,
    )


def positioned_value_match_signature[T](
    match: Match[T] | None,
) -> tuple[T, int | float, int | None, object, str, str] | None:
    """Return stable fields for sequence fuzzy match assertions."""

    if match is None:
        return None
    return (
        match.value,
        match.score,
        match.index,
        match.query,
        match.normalized_query,
        match.normalized_value,
    )
