"""Pytest entry point for the narrow final strategy matrix."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import FINAL_WORKLOADS, FROZEN_STRATEGY_CASES, MUTABLE_STRATEGY_CASES, build_runner_args


def test_run_final_narrow_mutable_strategy_matrix():
    """Run the narrow mutable strategy matrices at two sizes."""

    main(
        build_runner_args(
            items=1000,
            repeats=5,
            cases=MUTABLE_STRATEGY_CASES,
            profiles=("unique", "mixed", "collision-50"),
            normalizers=("default",),
            scorers=("ratio", "wratio"),
            workloads=FINAL_WORKLOADS,
            workloads_only=True,
            output_dir="benchmarks/reports/index_strategy_final_narrow_mutable_1000",
        )
    )
    main(
        build_runner_args(
            items=10000,
            repeats=3,
            cases=MUTABLE_STRATEGY_CASES,
            profiles=("unique", "mixed", "collision-50"),
            normalizers=("default",),
            scorers=("ratio", "wratio"),
            workloads=FINAL_WORKLOADS,
            workloads_only=True,
            output_dir="benchmarks/reports/index_strategy_final_narrow_mutable_10000",
        )
    )


def test_run_final_narrow_frozen_strategy_matrix():
    """Run the narrow frozen strategy matrix."""

    main(
        build_runner_args(
            items=10000,
            repeats=5,
            cases=FROZEN_STRATEGY_CASES,
            profiles=("unique", "mixed"),
            normalizers=("default",),
            scorers=("ratio", "wratio"),
            workloads=("read-only", "batch-heavy"),
            workloads_only=True,
            output_dir="benchmarks/reports/index_strategy_final_narrow_frozen_10000",
        )
    )
