"""Pytest entry point for the historical index-strategy matrix.

This file is outside the configured pytest ``testpaths`` and is intended to be
run manually:

    python -m pytest benchmarks/run_index_strategy_benchmark.py -q -s
"""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import MUTABLE_STRATEGY_CASES, ORDERED_PROTOTYPE_CASES, build_runner_args


def test_run_index_strategy_benchmark_matrix():
    """Run the historical strategy and ordered-prototype matrix."""

    main(
        build_runner_args(
            items=500,
            repeats=3,
            updates=10,
            cases=(*MUTABLE_STRATEGY_CASES, *ORDERED_PROTOTYPE_CASES),
            profiles=("unique", "mixed", "collision-0", "collision-5", "collision-20", "collision-50"),
            normalizers=("default", "pipeline"),
            scorers=("ratio", "wratio", "levenshtein-distance"),
            output_dir="benchmarks/reports/index_strategy",
        )
    )
