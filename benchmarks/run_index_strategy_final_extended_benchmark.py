"""Pytest entry point for the extended final strategy matrix."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import FINAL_WORKLOADS, FROZEN_STRATEGY_CASES, MUTABLE_STRATEGY_CASES, build_runner_args


def test_run_final_extended_mutable_strategy_matrix():
    """Run the extended mutable strategy matrix."""

    main(
        build_runner_args(
            items=3000,
            repeats=2,
            cases=MUTABLE_STRATEGY_CASES,
            profiles=("unique", "mixed", "collision-50"),
            normalizers=("default", "pipeline"),
            scorers=("ratio", "wratio", "levenshtein-distance"),
            workloads=FINAL_WORKLOADS,
            workloads_only=True,
            output_dir="benchmarks/reports/index_strategy_final_extended_mutable_3000",
        )
    )


def test_run_final_extended_frozen_strategy_matrix():
    """Run the extended frozen strategy matrix."""

    main(
        build_runner_args(
            items=3000,
            repeats=2,
            cases=FROZEN_STRATEGY_CASES,
            profiles=("unique", "mixed"),
            normalizers=("default", "pipeline"),
            scorers=("ratio", "wratio", "levenshtein-distance"),
            workloads=("read-only", "batch-heavy"),
            workloads_only=True,
            output_dir="benchmarks/reports/index_strategy_final_extended_frozen_3000",
        )
    )
