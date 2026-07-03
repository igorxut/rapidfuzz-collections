from collections.abc import Callable
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from typing import Any

from ..configuration import _UNCHANGED
from ..enums import ScorerType

Scorer = Callable[..., int | float]


def _is_hashable(value: object) -> bool:
    """Return whether ``value`` can actually be hashed."""

    try:
        hash(value)
    except TypeError:
        return False
    return True


def validate_scorer_type(scorer_type: object) -> ScorerType:
    """Return a validated scorer interpretation mode.

    Args:
        scorer_type: Value to validate.

    Returns:
        Validated scorer type.

    Raises:
        TypeError: If ``scorer_type`` is not a ``ScorerType`` member.
    """

    if not isinstance(scorer_type, ScorerType):
        raise TypeError("scorer_type must be ScorerType.DISTANCE or ScorerType.SIMILARITY")
    return scorer_type


def copy_scorer_kwargs(
    scorer_kwargs: dict[str, Any] | None,
    *,
    deepcopy_memo: dict[int, object] | None = None,
) -> dict[str, Any] | None:
    """Return an independent scorer keyword-argument mapping.

    Args:
        scorer_kwargs: The scorer keyword arguments to copy.
        deepcopy_memo: Optional memo mapping used by ``deepcopy`` during recursive copying.

    Returns:
        A deep copy of ``scorer_kwargs``, or ``None`` if ``scorer_kwargs`` is ``None``.
    """

    validate_scorer_kwargs(scorer_kwargs)
    if deepcopy_memo is None:
        return _deepcopy(scorer_kwargs)
    return _deepcopy(scorer_kwargs, deepcopy_memo)


def _process_scorer_metadata(
    scorer: Scorer,
    scorer_kwargs: dict[str, Any] | None,
    scorer_type: ScorerType,
) -> tuple[bool, int | float | None]:
    """Return RapidFuzz process compatibility and the optimal scorer result.

    RapidFuzz treats third-party scorers as similarities. Native scorers expose
    direction metadata through their Python scorer adapter; this helper keeps
    that compatibility check isolated and falls back conservatively for
    callables without metadata.

    Args:
        scorer: Scorer callable passed to RapidFuzz.
        scorer_kwargs: Keyword arguments used to resolve scorer metadata.
        scorer_type: Direction required by the collection configuration.

    Returns:
        A pair containing whether RapidFuzz process APIs can apply the
        configured direction directly and the scorer's optimal result. The
        optimal result is ``None`` when compatible metadata is unavailable.
    """

    scorer_adapter = getattr(scorer, "_RF_ScorerPy", None)
    if scorer_adapter is None:
        return False, None
    if not isinstance(scorer_adapter, dict):
        return False, None
    get_scorer_flags = scorer_adapter.get("get_scorer_flags")
    if not callable(get_scorer_flags):
        return False, None
    try:
        flags = get_scorer_flags(**(scorer_kwargs or {}))
    except TypeError:
        return False, None
    if not isinstance(flags, dict):
        return False, None
    optimal_score = flags.get("optimal_score")
    worst_score = flags.get("worst_score")
    if not isinstance(optimal_score, int | float) or not isinstance(worst_score, int | float):
        return False, None
    scorer_is_distance = optimal_score < worst_score
    if scorer_is_distance != (scorer_type == ScorerType.DISTANCE):
        return False, None
    return True, optimal_score


def _scorer_type_from_metadata(
    scorer: Scorer,
    scorer_kwargs: dict[str, Any] | None,
) -> ScorerType | None:
    """Infer score direction from compatible RapidFuzz scorer metadata."""

    scorer_adapter = getattr(scorer, "_RF_ScorerPy", None)
    if not isinstance(scorer_adapter, dict):
        return None
    get_scorer_flags = scorer_adapter.get("get_scorer_flags")
    if not callable(get_scorer_flags):
        return None
    try:
        flags = get_scorer_flags(**(scorer_kwargs or {}))
    except TypeError:
        return None
    if not isinstance(flags, dict):
        return None
    optimal_score = flags.get("optimal_score")
    worst_score = flags.get("worst_score")
    if not isinstance(optimal_score, int | float) or not isinstance(worst_score, int | float):
        return None
    return ScorerType.DISTANCE if optimal_score < worst_score else ScorerType.SIMILARITY


@dataclass(frozen=True, slots=True)
class _MatchConfig:
    """Fully-resolved matching configuration used for one query.

    Bundles the resolved scorer settings together with the RapidFuzz
    compatibility metadata (``process_scorer_type_matches``, ``optimal_score``)
    computed for that exact combination, so callers never mix an overridden
    scorer with metadata cached for a different scorer.
    """

    scorer: Scorer
    scorer_kwargs: dict[str, Any] | None
    scorer_type: ScorerType
    score_cutoff: int | float | None
    score_hint: int | float | None
    process_scorer_type_matches: bool
    optimal_score: int | float | None


def _resolve_match_config(
    defaults: _MatchConfig,
    *,
    scorer: Scorer = _UNCHANGED,
    scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
    scorer_type: ScorerType = _UNCHANGED,
    score_cutoff: int | float | None = _UNCHANGED,
    score_hint: int | float | None = _UNCHANGED,
) -> _MatchConfig:
    """Return a per-query matching configuration with overrides applied.

    Omitting an argument preserves the corresponding value from ``defaults``.
    Passing ``None`` is an explicit override for parameters that accept it.

    Returns:
        Resolved configuration. RapidFuzz compatibility metadata is
        recomputed when ``scorer``, ``scorer_kwargs``, or ``scorer_type`` is
        overridden; otherwise ``defaults`` metadata is reused unchanged.

    Raises:
        ValueError: If ``scorer`` is overridden without ``scorer_type`` and has no compatible RapidFuzz metadata.
        TypeError: If an overridden ``scorer`` is not callable,
            ``scorer_kwargs`` is invalid, ``scorer_type`` is not a
            ``ScorerType`` member, or
            ``score_cutoff``/``score_hint`` is not numeric or ``None``.
    """

    if scorer is not _UNCHANGED:
        validate_scorer(scorer)
    if scorer_kwargs is not _UNCHANGED:
        validate_scorer_kwargs(scorer_kwargs)
    if score_cutoff is not _UNCHANGED:
        validate_optional_score(score_cutoff, "score_cutoff")
    if score_hint is not _UNCHANGED:
        validate_optional_score(score_hint, "score_hint")

    resolved_scorer = defaults.scorer if scorer is _UNCHANGED else scorer
    resolved_scorer_kwargs = defaults.scorer_kwargs if scorer_kwargs is _UNCHANGED else scorer_kwargs
    if scorer_type is not _UNCHANGED:
        resolved_scorer_type = validate_scorer_type(scorer_type)
    elif scorer is _UNCHANGED:
        resolved_scorer_type = defaults.scorer_type
    else:
        inferred_scorer_type = _scorer_type_from_metadata(resolved_scorer, resolved_scorer_kwargs)
        if inferred_scorer_type is None:
            raise ValueError("scorer_type is required when an overridden scorer has no compatible metadata")
        resolved_scorer_type = inferred_scorer_type
    resolved_score_cutoff = defaults.score_cutoff if score_cutoff is _UNCHANGED else score_cutoff
    resolved_score_hint = defaults.score_hint if score_hint is _UNCHANGED else score_hint

    if scorer is _UNCHANGED and scorer_kwargs is _UNCHANGED and scorer_type is _UNCHANGED:
        return _MatchConfig(
            scorer=resolved_scorer,
            scorer_kwargs=defaults.scorer_kwargs,
            scorer_type=defaults.scorer_type,
            score_cutoff=resolved_score_cutoff,
            score_hint=resolved_score_hint,
            process_scorer_type_matches=defaults.process_scorer_type_matches,
            optimal_score=defaults.optimal_score,
        )

    process_scorer_type_matches, optimal_score = _process_scorer_metadata(
        resolved_scorer,
        resolved_scorer_kwargs,
        resolved_scorer_type,
    )
    return _MatchConfig(
        scorer=resolved_scorer,
        scorer_kwargs=resolved_scorer_kwargs,
        scorer_type=resolved_scorer_type,
        score_cutoff=resolved_score_cutoff,
        score_hint=resolved_score_hint,
        process_scorer_type_matches=process_scorer_type_matches,
        optimal_score=optimal_score,
    )


def validate_limit(limit: int | None) -> None:
    """Validate a maximum match count.

    Args:
        limit: Maximum number of matches, or ``None`` for no limit.

    Raises:
        TypeError: If ``limit`` is not an integer or is a boolean.
        ValueError: If ``limit`` is negative.
    """

    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"limit must be an integer, got {type(limit).__name__!r}")
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")


def validate_chunk_size(size: int, name: str) -> None:
    """Validate a chunk size parameter for batch cdist operations.

    Args:
        size: Chunk size value.
        name: Parameter name used in error messages.

    Raises:
        TypeError: If ``size`` is not an integer or is a boolean.
        ValueError: If ``size`` is less than 1.
    """

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"{name} must be an integer, got {type(size).__name__!r}")
    if size < 1:
        raise ValueError(f"{name} must be greater than 0")


def validate_normalizer(normalizer: object) -> None:
    """Validate a normalizer callable.

    Args:
        normalizer: Value to validate. ``None`` is accepted and passes validation.

    Raises:
        TypeError: If ``normalizer`` is not ``None`` and is not callable.
    """

    if normalizer is not None and not callable(normalizer):
        raise TypeError(f"normalizer must be callable, got {type(normalizer).__name__!r}")


def validate_optional_score(value: object, name: str) -> None:
    """Validate an optional scorer threshold or hint.

    Args:
        value: Value to validate.
        name: Parameter name used in the error message.

    Raises:
        TypeError: If the value is not an integer, float, or ``None``, or is a boolean.
    """

    if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
        raise TypeError(f"{name} must be an integer, float, or None, got {type(value).__name__!r}")


def validate_scorer(scorer: object) -> None:
    """Validate a scorer callable.

    Args:
        scorer: Value to validate.

    Raises:
        TypeError: If ``scorer`` is not callable.
    """

    if not callable(scorer):
        raise TypeError(f"scorer must be callable, got {type(scorer).__name__!r}")


def validate_scorer_kwargs(scorer_kwargs: object) -> None:
    """Validate keyword arguments forwarded to a scorer.

    Args:
        scorer_kwargs: Value to validate.

    Raises:
        TypeError: If the value is not a dictionary or ``None``, or contains a non-string key.
    """

    if scorer_kwargs is None:
        return
    if not isinstance(scorer_kwargs, dict):
        raise TypeError(f"scorer_kwargs must be a dictionary or None, got {type(scorer_kwargs).__name__!r}")
    if not all(isinstance(key, str) for key in scorer_kwargs):
        raise TypeError("scorer_kwargs keys must be strings")


def index_config_kwargs(
    *,
    normalizer: Callable[[object], str | None],
    scorer: Scorer,
    scorer_kwargs: dict[str, Any] | None,
    scorer_type: ScorerType,
    score_cutoff: int | float | None,
    score_hint: int | float | None,
    deepcopy_memo: dict[int, object] | None = None,
) -> dict[str, Any]:
    """Return constructor keyword arguments for another index instance."""

    return {
        "normalizer": normalizer,
        "scorer": scorer,
        "scorer_kwargs": copy_scorer_kwargs(scorer_kwargs, deepcopy_memo=deepcopy_memo),
        "scorer_type": scorer_type,
        "score_cutoff": score_cutoff,
        "score_hint": score_hint,
    }
