"""Low-level timing and memory primitives shared by all benchmark harnesses."""

import argparse
import csv
import gc
import json
import platform
import statistics
import time
import tomllib
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import rapidfuzz

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument.

    Raises:
        argparse.ArgumentTypeError: If the parsed value is not a valid
            integer or is less than 1.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument.

    Raises:
        argparse.ArgumentTypeError: If the parsed value is not a valid
            integer or is negative.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def non_negative_float(value: str) -> float:
    """Parse a non-negative float CLI argument.

    Raises:
        argparse.ArgumentTypeError: If the parsed value is not a valid
            float or is negative.
    """
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _pyproject_version() -> str | None:
    """Return the version declared in ``pyproject.toml``, if it can be read."""
    try:
        with _PYPROJECT_PATH.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError, tomllib.TOMLDecodeError:
        return None
    return data.get("project", {}).get("version")


def environment_metadata() -> dict[str, str]:
    """Return interpreter, platform, and dependency version metadata.

    Notes:
        Benchmark timings are only comparable across runs captured under the
        same environment. This snapshot lets a report reader tell whether two
        result sets were produced under matching conditions.
    """
    try:
        rapidfuzz_collections_version = version("rapidfuzz-collections")
    except PackageNotFoundError:
        rapidfuzz_collections_version = _pyproject_version() or "unknown (not installed as a package)"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rapidfuzz_version": rapidfuzz.__version__,
        "rapidfuzz_collections_version": rapidfuzz_collections_version,
    }


def write_benchmark_reports(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    stem: str,
    quiet: bool = False,
) -> None:
    """Write benchmark result rows as ``<stem>.json`` and ``<stem>.csv``.

    This is the shared reporting format for all benchmark scripts: every run
    writes exactly one JSON file (result rows plus :func:`environment_metadata`)
    and one CSV file (result rows only, no environment metadata) into
    ``output_dir``, which is created if missing.

    Args:
        rows: Benchmark result rows, one mapping per measurement. All rows
            must share the same keys.
        output_dir: Directory to write the two report files into. Created,
            along with any missing parents, if it does not already exist.
        stem: Filename stem shared by both output files.
        quiet: If ``True``, skip printing the written file paths to stdout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps({"environment": environment_metadata(), "results": list(rows)}, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not quiet:
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")


def measure_timings(
    repeats: int,
    fn: Callable[[], object],
    warmup: int = 0,
) -> tuple[float, float]:
    """Run fn exactly repeats times and return (best_ms, median_ms).

    Args:
        repeats: Number of timed repetitions.
        fn: Callable to time.
        warmup: Number of untimed warmup calls before the timed loop.

    Returns:
        Tuple of best elapsed time in ms and median elapsed time in ms.
    """
    for _ in range(warmup):
        gc.collect()
        fn()
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return min(timings), statistics.median(timings)


def measure_peak_kib(fn: Callable[[], object]) -> float:
    """Run fn once under tracemalloc and return peak traced allocation in KiB.

    Returns:
        Peak traced memory in KiB.

    Notes:
        Safe to call when tracemalloc is already active: resets only the peak
        counter and does not stop the outer tracing session on exit.
    """
    gc.collect()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    return peak / 1024.0


def result_size(value: object) -> int | None:
    """Return a lightweight coarse size signal for a benchmark result value.

    Returns:
        0 for None, len() for sized types, None if the value has no len().
    """
    if value is None:
        return 0
    if isinstance(value, (str, bytes)):
        return len(value)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def string_values(
    values: list[object],
    normalizer: Callable[[object], object] | None = None,
) -> list[str]:
    """Return string values that survive the given normalizer.

    Args:
        values: Input values of any type.
        normalizer: Optional callable returning None to reject a value. When
            absent, values with stripped length < 3 are excluded.

    Returns:
        Strings that the normalizer accepts (or that pass the default length check).
    """
    result: list[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        if normalizer is None:
            if len(v.strip()) >= 3:
                result.append(v)
        elif normalizer(v) is not None:
            result.append(v)
    return result
