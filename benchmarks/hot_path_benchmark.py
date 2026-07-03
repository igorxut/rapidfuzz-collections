"""Benchmark exact-shortcut recovery and custom-scorer materialization paths.

Run it directly:

    python benchmarks/hot_path_benchmark.py
"""

import argparse
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Hashable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz.fuzz import ratio  # noqa: E402

from benchmarks.datasets import DataProfile, build_queries, build_values  # noqa: E402
from benchmarks.utils import (  # noqa: E402
    measure_peak_kib,
    measure_timings,
    positive_int,
    write_benchmark_reports,
)
from rapidfuzz_collections import Match, ScorerType, ValueMatch  # noqa: E402

# noinspection PyProtectedMember
# Private helpers are intentional here: the legacy subclasses must resolve
# configuration exactly like production so the benchmark isolates ranking cost.
from rapidfuzz_collections.configuration import _UNCHANGED  # noqa: E402
from rapidfuzz_collections.indexes import (  # noqa: E402
    FuzzySequenceIndex,
    ImmutableFuzzyKeyedIndex,
    MutableFuzzySequenceIndex,
    Scorer,
)

# noinspection PyProtectedMember
from rapidfuzz_collections.indexes.base import _resolve_match_config  # noqa: E402

type CustomScorer = Callable[[str, str], int | float]


@dataclass(frozen=True)
class HotPathResult:
    """One focused hot-path benchmark measurement."""

    suite: str
    implementation: str
    items: int
    workload: str
    repeats: int
    best_ms: float
    median_ms: float
    peak_kib: float
    result_size: int


class LegacyScanningMutableSequenceIndex(MutableFuzzySequenceIndex[str]):
    """Preserve the pre-optimization exact scan after incremental deletion."""

    def _exact_source_indexes(self, query: object) -> tuple[int, ...]:
        if not isinstance(query, Hashable):
            return ()
        if self._shortcuts_valid:
            first_index = self._exact_first_index.get(query)
            if first_index is None:
                return ()
            return first_index, *self._exact_duplicate_indexes.get(query, ())
        return tuple(
            index for index, value in enumerate(self._values) if isinstance(value, Hashable) and value == query
        )


class LegacyRankedSequenceScoreAll(FuzzySequenceIndex[str]):
    """Preserve ranked extraction for custom-scorer score materialization."""

    def score_all(
        self,
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[Match[str] | None]:
        """Materialize scores through the legacy ranked sequence path."""

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        results: list[Match[str] | None] = [None] * len(self._values)
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return results
        for normalized_value, score, choice_index in self._extract(normalized_query, limit=None, config=config):
            source_index = self._source_index_from_choice(choice_index)
            results[source_index] = Match(
                value=self._values[source_index],
                score=score,
                index=source_index,
                query=query,
                normalized_query=normalized_query,
                normalized_value=normalized_value,
            )
        return results


class LegacyRankedKeyedScoreAll(ImmutableFuzzyKeyedIndex[str]):
    """Preserve ranked keyed extraction for custom-scorer score materialization."""

    def score_all(
        self,
        values: Iterable[str],
        query: object,
        *,
        scorer: Scorer = _UNCHANGED,
        scorer_kwargs: dict[str, Any] | None = _UNCHANGED,
        scorer_type: ScorerType = _UNCHANGED,
        score_cutoff: int | float | None = _UNCHANGED,
        score_hint: int | float | None = _UNCHANGED,
    ) -> list[ValueMatch[str] | None]:
        """Materialize scores through the legacy ranked keyed path."""

        config = _resolve_match_config(
            self._default_config,
            scorer=scorer,
            scorer_kwargs=scorer_kwargs,
            scorer_type=scorer_type,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
        )
        source_values = list(values)
        results: list[ValueMatch[str] | None] = [None] * len(source_values)
        normalized_query = self._normalizer(query)
        if normalized_query is None:
            return results

        positions = {value: index for index, value in enumerate(source_values)}
        duplicate_positions: dict[str, list[int]] | None = None
        if len(positions) != len(source_values):
            dup: dict[str, list[int]] = {}
            for index, value in enumerate(source_values):
                dup.setdefault(value, []).append(index)
            duplicate_positions = dup
        for normalized_value, score, value in self._extract(normalized_query, limit=None, config=config):
            position = positions.get(value)
            if position is None:
                continue
            target_positions = [position] if duplicate_positions is None else duplicate_positions[value]
            for target_position in target_positions:
                results[target_position] = self._match(
                    source_values[target_position],
                    query=query,
                    normalized_query=normalized_query,
                    normalized_value=normalized_value,
                    score=score,
                )
        return results


def constant_similarity(_left: str, _right: str) -> int:
    """Return one score to isolate traversal and materialization overhead."""

    return 70


def wrapped_ratio(left: str, right: str) -> float:
    """Expose RapidFuzz ratio as a custom scorer without scorer metadata."""

    return ratio(left, right)


def measure_paired_after_setup[S, R](
    repeats: int,
    setups: tuple[tuple[str, Callable[[], S]], ...],
    operation: Callable[[S], R],
) -> dict[str, tuple[float, float, float, R]]:
    """Measure competing mutating implementations in alternating order."""

    timings = {name: [] for name, _ in setups}
    results: dict[str, R] = {}
    for repeat in range(repeats):
        ordered_setups = setups if repeat % 2 == 0 else tuple(reversed(setups))
        for name, setup in ordered_setups:
            subject = setup()
            start = time.perf_counter()
            results[name] = operation(subject)
            timings[name].append((time.perf_counter() - start) * 1000.0)

    measurements: dict[str, tuple[float, float, float, R]] = {}
    for name, setup in setups:
        subject = setup()
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            results[name] = operation(subject)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not already_tracing:
                tracemalloc.stop()
        measurements[name] = (
            min(timings[name]),
            statistics.median(timings[name]),
            peak / 1024.0,
            results[name],
        )
    return measurements


def result_length(result: object) -> int:
    """Return a stable coarse result size for benchmark sanity checks."""

    if isinstance(result, list | tuple):
        return len(result)
    return int(result is not None)


def benchmark_shortcut_recovery(items: int, repeats: int, lookup_counts: Iterable[int]) -> list[HotPathResult]:
    """Compare legacy repeated scans with lazy exact-shortcut recovery."""

    values = build_values(items, DataProfile.COLLISION_0)
    string_values = [value for value in values if isinstance(value, str)]
    exact_query = string_values[-1]
    close_query = build_queries(string_values, 32).close
    results: list[HotPathResult] = []

    implementations = (
        ("legacy-repeated-scan", LegacyScanningMutableSequenceIndex),
        ("production-lazy-rebuild", MutableFuzzySequenceIndex),
    )
    query_profiles = (
        ("exact", exact_query, tuple(lookup_counts)),
        ("close", close_query, tuple(count for count in lookup_counts if count in {1, 10})),
    )
    for query_name, query, profile_lookup_counts in query_profiles:
        for lookup_count in profile_lookup_counts:

            def make_setup(index_class: type[MutableFuzzySequenceIndex[str]]) -> Callable[[], object]:
                def setup() -> MutableFuzzySequenceIndex[str]:
                    index = index_class(string_values)
                    index.delete_at(0)
                    return index

                return setup

            def make_operation(n: int, q: str) -> Callable[[object], Match[str] | None]:
                def operation(index: object) -> Match[str] | None:
                    assert isinstance(index, MutableFuzzySequenceIndex)
                    match = None
                    for _ in range(n):
                        match = index.find_one(q)
                    return match

                return operation

            setups = tuple((name, make_setup(index_class)) for name, index_class in implementations)
            measurements = measure_paired_after_setup(repeats, setups, make_operation(lookup_count, query))
            for implementation, (best_ms, median_ms, peak_kib, result) in measurements.items():
                if result is None:
                    raise AssertionError(f"{query_name} lookup failed after incremental deletion")
                if query_name == "exact" and result.value != query:
                    raise AssertionError("exact lookup returned the wrong value")
                results.append(
                    HotPathResult(
                        suite="shortcut-recovery",
                        implementation=implementation,
                        items=items,
                        workload=f"delete+{lookup_count}-{query_name}-lookups",
                        repeats=repeats,
                        best_ms=best_ms,
                        median_ms=median_ms,
                        peak_kib=peak_kib,
                        result_size=1,
                    )
                )
    return results


def benchmark_custom_score_all(items: int, repeats: int) -> list[HotPathResult]:
    """Compare ranked, direct, and streaming custom-scorer materialization."""

    values = [value for value in build_values(items, DataProfile.COLLISION_0) if isinstance(value, str)]
    query = build_queries(values, 32).close
    results: list[HotPathResult] = []

    for scorer_name, scorer in (("constant", constant_similarity), ("wrapped-ratio", wrapped_ratio)):
        sequence_legacy = LegacyRankedSequenceScoreAll(values, scorer=scorer, score_cutoff=0)
        sequence_production = FuzzySequenceIndex(values, scorer=scorer, score_cutoff=0)
        keyed_legacy = LegacyRankedKeyedScoreAll(values, scorer=scorer, score_cutoff=0)
        keyed_production = ImmutableFuzzyKeyedIndex(values, scorer=scorer, score_cutoff=0)

        operations: tuple[tuple[str, Callable[[], object]], ...] = (
            ("sequence-legacy-ranked", lambda index=sequence_legacy: index.score_all(query)),
            ("sequence-production-direct", lambda index=sequence_production: index.score_all(query)),
            (
                "sequence-materialized-iterator",
                lambda index=sequence_production: list(index.iter_scores(query)),
            ),
            ("keyed-legacy-ranked", lambda index=keyed_legacy: index.score_all(values, query)),
            ("keyed-production-direct", lambda index=keyed_production: index.score_all(values, query)),
            (
                "keyed-materialized-iterator",
                lambda index=keyed_production: list(index.iter_scores(values, query)),
            ),
        )

        expected = sequence_legacy.score_all(query)
        if sequence_production.score_all(query) != expected or list(sequence_production.iter_scores(query)) != expected:
            raise AssertionError("sequence score_all implementations disagree")
        expected_keyed = keyed_legacy.score_all(values, query)
        if keyed_production.score_all(values, query) != expected_keyed:
            raise AssertionError("keyed score_all implementations disagree")
        if list(keyed_production.iter_scores(values, query)) != expected_keyed:
            raise AssertionError("keyed iter_scores disagrees with score_all")

        for implementation, operation in operations:
            best_ms, median_ms = measure_timings(repeats, operation)
            peak_kib = measure_peak_kib(operation)
            result = operation()
            results.append(
                HotPathResult(
                    suite="custom-score-all",
                    implementation=implementation,
                    items=items,
                    workload=scorer_name,
                    repeats=repeats,
                    best_ms=best_ms,
                    median_ms=median_ms,
                    peak_kib=peak_kib,
                    result_size=result_length(result),
                )
            )
    return results


def write_outputs(results: list[HotPathResult], output_dir: Path) -> None:
    """Write raw benchmark result rows as JSON and CSV."""

    rows = sorted(
        (asdict(result) for result in results),
        key=lambda row: (row["suite"], row["items"], row["workload"], row["implementation"]),
    )
    write_benchmark_reports(rows, output_dir, stem="hot_path_results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse focused hot-path benchmark arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=positive_int, nargs="+", default=(1_000, 10_000, 100_000))
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--lookup-counts", type=positive_int, nargs="+", default=(1, 2, 3, 10, 100))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/reports/index_hot_paths"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run focused index hot-path benchmarks."""

    args = parse_args(argv)
    results: list[HotPathResult] = []
    for items in args.items:
        results.extend(benchmark_shortcut_recovery(items, args.repeats, args.lookup_counts))
        results.extend(benchmark_custom_score_all(items, args.repeats))
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
