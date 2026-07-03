from collections.abc import Callable
from typing import Any

from .enums import IndexStrategy, ScorerType

_UNCHANGED: Any = object()


def _apply_config_overrides(
    config: dict[str, Any],
    *,
    normalizer: Callable[[object], str | None] | None = _UNCHANGED,
    scorer: Callable[..., int | float] = _UNCHANGED,
    scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
    scorer_type: ScorerType = _UNCHANGED,
    score_cutoff: int | float | None = _UNCHANGED,
    score_hint: int | float | None = _UNCHANGED,
    strategy: IndexStrategy | str = _UNCHANGED,
) -> dict[str, Any]:
    """Return construction configuration with explicit overrides applied.

    Omitting an argument preserves the source collection configuration.
    Passing ``None`` is an explicit override for parameters that accept it.

    Returns:
        Copy of ``config`` with all non-sentinel overrides applied.
    """

    updated = config.copy()
    for name, value in (
        ("normalizer", normalizer),
        ("scorer", scorer),
        ("scorer_kwargs", scorer_kwargs),
        ("scorer_type", scorer_type),
        ("score_cutoff", score_cutoff),
        ("score_hint", score_hint),
        ("strategy", strategy),
    ):
        if value is not _UNCHANGED:
            updated[name] = value
    return updated


def _coerce_index_strategy(strategy: IndexStrategy | str) -> IndexStrategy:
    """Return a validated index strategy enum value."""

    if isinstance(strategy, IndexStrategy):
        return strategy
    try:
        return IndexStrategy(strategy)
    except ValueError as exc:
        raise ValueError("strategy must be IndexStrategy.SEQUENCE or IndexStrategy.KEYED") from exc
