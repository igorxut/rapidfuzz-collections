"""Pytest entry point for the mutable strategy smoke matrix."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import FINAL_WORKLOADS, MUTABLE_STRATEGY_CASES, build_runner_args


def test_run_mutable_facade_strategy_smoke_matrix():
    """Run a small production mutable-strategy workload matrix."""

    main(
        build_runner_args(
            items=1000,
            repeats=2,
            cases=MUTABLE_STRATEGY_CASES,
            profiles=("unique", "mixed", "collision-50"),
            normalizers=("default",),
            scorers=("ratio",),
            workloads=FINAL_WORKLOADS,
            workloads_only=True,
            output_dir="benchmarks/reports/index_strategy_facade_smoke_1000",
        )
    )
