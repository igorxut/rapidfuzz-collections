"""Pytest entry point for sequence and frozen collection benchmark rows."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import FROZEN_STRATEGY_CASES, build_runner_args


def test_run_index_strategy_sequence_and_frozen_matrix():
    """Run sequence collection and frozen strategy measurements."""

    main(
        build_runner_args(
            items=500,
            repeats=3,
            updates=10,
            cases=("list", "tuple", *FROZEN_STRATEGY_CASES),
            profiles=("unique", "mixed", "duplicates"),
            normalizers=("default", "pipeline"),
            scorers=("ratio", "wratio", "levenshtein-distance"),
            output_dir="benchmarks/reports/index_strategy_sequences",
        )
    )
