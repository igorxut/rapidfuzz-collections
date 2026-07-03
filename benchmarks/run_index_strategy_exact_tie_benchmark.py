"""Pytest entry point for exact tie-resolution strategy measurements."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import FROZEN_STRATEGY_CASES, MUTABLE_STRATEGY_CASES, build_runner_args


def test_run_exact_tie_strategy_matrix():
    """Measure exact tie resolution across mutable and frozen strategies."""

    main(
        build_runner_args(
            items=10_000,
            repeats=5,
            cases=(*MUTABLE_STRATEGY_CASES, *FROZEN_STRATEGY_CASES),
            profiles=("duplicates", "collision-50"),
            normalizers=("default",),
            scorers=("ratio", "wratio", "levenshtein-distance"),
            exact_tie_only=True,
            output_dir="benchmarks/reports/index_strategy_exact_tie_10000",
        )
    )
