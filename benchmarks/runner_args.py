"""Argument presets used by the benchmark runner tests."""

from collections.abc import Iterable

MUTABLE_STRATEGY_CASES = (
    "fuzzy-dict-sequence",
    "fuzzy-dict-keyed",
    "fuzzy-set-sequence",
    "fuzzy-set-keyed",
)

ORDERED_PROTOTYPE_CASES = (
    "ordered-dict",
    "ordered-set",
)
"""Active ordered-storage prototype cases retained for architecture comparisons."""

FROZEN_STRATEGY_CASES = (
    "frozen-dict",
    "frozen-keyed-dict",
    "frozen-set",
    "frozen-keyed-set",
)

FINAL_WORKLOADS = (
    "read-only",
    "lookup-heavy",
    "batch-heavy",
    "exact-mutation-heavy",
    "bulk-mutation-heavy",
)


def build_runner_args(
    *,
    items: int,
    repeats: int,
    cases: Iterable[str],
    profiles: Iterable[str],
    normalizers: Iterable[str],
    scorers: Iterable[str],
    output_dir: str,
    batch_size: int = 32,
    updates: int = 20,
    workloads: Iterable[str] = (),
    workloads_only: bool = False,
    exact_tie_only: bool = False,
    full_result_only: bool = False,
    warmup: int | None = None,
    include_memory: bool = True,
) -> list[str]:
    """Build CLI arguments for the index-strategy benchmark.

    Args:
        items: Number of generated collection values.
        repeats: Number of repeated measurements per operation.
        cases: Collection facade identifiers to benchmark.
        profiles: Data distribution profiles to benchmark.
        normalizers: Normalizer profiles to use.
        scorers: RapidFuzz scorer profiles to use.
        output_dir: Directory to write JSON and CSV results.
        batch_size: Number of queries in batch scenarios.
        updates: Number of values in mutation scenarios.
        workloads: Composite workload identifiers to include.
        workloads_only: When True, skip the full micro-operation matrix.
        exact_tie_only: When True, measure only exact and normalized-collision
            tie-resolution paths.
        full_result_only: When True, measure only unbounded multi-match lookup.
        warmup: Optional number of warmup repetitions before measurement.
        include_memory: When False, skip tracemalloc peak measurement.

    Returns:
        List of CLI argument strings for ``index_strategy_benchmark.main``.
    """
    args = [
        "--items",
        str(items),
        "--repeats",
        str(repeats),
        "--batch-size",
        str(batch_size),
        "--updates",
        str(updates),
        "--cases",
        *cases,
        "--profiles",
        *profiles,
        "--normalizers",
        *normalizers,
        "--scorers",
        *scorers,
    ]
    if workloads:
        args.extend(["--workloads", *workloads])
    if workloads_only:
        args.append("--workloads-only")
    if exact_tie_only:
        args.append("--exact-tie-only")
    if full_result_only:
        args.append("--full-result-only")
    if warmup is not None:
        args.extend(["--warmup", str(warmup)])
    if not include_memory:
        args.append("--no-memory")
    args.extend(["--output-dir", output_dir])
    return args
