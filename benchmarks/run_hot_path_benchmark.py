"""Pytest entry point for focused index hot-path measurements."""

from benchmarks.hot_path_benchmark import main


def test_run_index_hot_path_benchmark():
    """Measure shortcut recovery and custom score materialization at scale."""

    main(
        [
            "--items",
            "1000",
            "10000",
            "100000",
            "--repeats",
            "5",
            "--lookup-counts",
            "1",
            "2",
            "3",
            "10",
            "100",
            "--output-dir",
            "benchmarks/reports/index_hot_paths",
        ]
    )
