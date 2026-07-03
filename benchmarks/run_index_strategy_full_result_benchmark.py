"""Pytest entry point for full-result ranking measurements."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import MUTABLE_STRATEGY_CASES, build_runner_args


def test_run_full_result_strategy_matrix():
    """Measure unbounded matching across collision densities and scorer directions."""

    main(
        build_runner_args(
            items=100_000,
            repeats=3,
            cases=MUTABLE_STRATEGY_CASES,
            profiles=("collision-0", "collision-5", "collision-50"),
            normalizers=("default",),
            scorers=("ratio", "levenshtein-distance"),
            full_result_only=True,
            output_dir="benchmarks/reports/index_strategy_full_result_100000",
        )
    )
