"""Smoke tests for benchmark infrastructure."""

from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.index_strategy_benchmark import (
    CollectionCase,
    OrderedUniqueFuzzyIndexPrototype,
    parse_args,
    run_smoke_check,
)


def test_default_benchmark_cases_exclude_prototypes() -> None:
    """Keep the default strategy benchmark limited to production classes."""
    cases = set(parse_args([]).cases)

    assert CollectionCase.ORDERED_DICT_PROTOTYPE not in cases
    assert CollectionCase.ORDERED_SET_PROTOTYPE not in cases


def test_index_strategy_benchmark_smoke() -> None:
    """Verify that a small current-API strategy matrix completes.

    This exercises the same pre-flight check that ``main()`` runs before the
    full benchmark matrix.
    """
    with TemporaryDirectory() as temporary_directory:
        run_smoke_check(Path(temporary_directory))


def test_ordered_unique_prototype_returns_canonical_exact_value() -> None:
    """Keep the active ordered prototype aligned with production exact lookup."""
    ordered = OrderedUniqueFuzzyIndexPrototype([True])

    match = ordered.exact_match(1, query=1)
    assert match is not None
    assert match.value is True
