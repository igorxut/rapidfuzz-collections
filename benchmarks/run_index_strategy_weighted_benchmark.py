"""Pytest entry point for composite workload strategy benchmarks."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import (
    FINAL_WORKLOADS,
    MUTABLE_STRATEGY_CASES,
    ORDERED_PROTOTYPE_CASES,
    build_runner_args,
)


def test_run_weighted_dict_set_strategy_matrix():
    """Run composite production and ordered-prototype workloads."""

    main(
        build_runner_args(
            items=3000,
            repeats=2,
            cases=(*MUTABLE_STRATEGY_CASES, *ORDERED_PROTOTYPE_CASES),
            profiles=("unique", "mixed", "collision-50"),
            normalizers=("default",),
            scorers=("ratio", "wratio"),
            workloads=FINAL_WORKLOADS,
            output_dir="benchmarks/reports/index_strategy_weighted",
        )
    )
