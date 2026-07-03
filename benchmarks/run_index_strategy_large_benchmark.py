"""Pytest entry point for the larger dict/set strategy matrix."""

from benchmarks.index_strategy_benchmark import main
from benchmarks.runner_args import MUTABLE_STRATEGY_CASES, ORDERED_PROTOTYPE_CASES, build_runner_args


def test_run_large_dict_set_strategy_matrix():
    """Run the larger production and ordered-prototype strategy matrix."""

    main(
        build_runner_args(
            items=3000,
            repeats=2,
            cases=(*MUTABLE_STRATEGY_CASES, *ORDERED_PROTOTYPE_CASES),
            profiles=("unique", "mixed", "collision-50"),
            normalizers=("default",),
            scorers=("ratio", "wratio"),
            output_dir="benchmarks/reports/index_strategy_large",
        )
    )
