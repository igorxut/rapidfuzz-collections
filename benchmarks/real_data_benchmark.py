"""Benchmark fuzzy collection builds and lookups against real entity matching datasets.

This script is opt-in and is not picked up by pytest automatically (no ``test_`` prefix
and not in ``testpaths``).  Run it directly:

    python benchmarks/real_data_benchmark.py

Datasets must be present under ``examples/data/``.  They are excluded from the
package distribution.
"""

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.datasets import make_typo  # noqa: E402
from benchmarks.utils import measure_peak_kib, measure_timings, write_benchmark_reports  # noqa: E402
from rapidfuzz_collections import (  # noqa: E402
    FrozenFuzzyDict,
    FrozenFuzzySet,
    FuzzyList,
    FuzzySequenceIndex,
    FuzzySet,
    FuzzyTuple,
    ImmutableFuzzyKeyedIndex,
    Normalizer,
)

_DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

FODORS_ZAGATS_DIR = _DATA_DIR / "structured_fodors_zagats"
AMAZON_GOOGLE_DIR = _DATA_DIR / "structured_amazon_google"
DBLP_ACM_DIR = _DATA_DIR / "dirty_dblp_acm"


@dataclass(frozen=True)
class RealDataResult:
    """One collection build/lookup measurement against a real dataset."""

    dataset: str
    collection: str
    items: int
    build_ms: float
    find_ms: float
    peak_kib: float


def _load_records_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {row["id"]: row for row in csv.DictReader(file)}


def _get_positive_pairs(test_path: Path) -> list[tuple[str, str]]:
    with test_path.open(newline="", encoding="utf-8") as file:
        return [(row["table1.id"], row["table2.id"]) for row in csv.DictReader(file) if row.get("label") == "1"]


def _join_fields(record: dict[str, str], fields: list[str]) -> str:
    return " ".join(record[f].strip() for f in fields if record.get(f, "").strip())


_REPEATS = 5
_QUERY_REPEATS = 100

_FZ_FIELDS = ["name", "city", "type"]
_AG_FIELDS = ["title", "manufacturer"]
_DA_FIELDS = ["title", "authors"]


def bench_frozen_fuzzy_dict_fodors_zagats() -> RealDataResult:
    """Benchmark FrozenFuzzyDict on Fodors-Zagats restaurants."""
    table_a = _load_records_by_id(FODORS_ZAGATS_DIR / "tableA.csv")
    table_b = _load_records_by_id(FODORS_ZAGATS_DIR / "tableB.csv")
    pairs = _get_positive_pairs(FODORS_ZAGATS_DIR / "test.csv")

    mapping = {_join_fields(rec, _FZ_FIELDS): rec for rec in table_b.values()}
    query_text = _join_fields(table_a[pairs[0][0]], _FZ_FIELDS)

    def build() -> FrozenFuzzyDict[str, dict[str, str]]:
        return FrozenFuzzyDict(mapping, score_cutoff=60)

    build_best, _ = measure_timings(_REPEATS, build)
    index = build()
    peak_kib = measure_peak_kib(build)
    find_best, _ = measure_timings(_QUERY_REPEATS, lambda: index.fuzzy_find_item(query_text))

    return RealDataResult(
        dataset="fodors-zagats",
        collection="FrozenFuzzyDict",
        items=len(index),
        build_ms=build_best,
        find_ms=find_best,
        peak_kib=peak_kib,
    )


def bench_fuzzy_sequence_index_amazon_google() -> RealDataResult:
    """Benchmark FuzzySequenceIndex on Amazon-Google products."""
    table_a = _load_records_by_id(AMAZON_GOOGLE_DIR / "tableA.csv")
    table_b = _load_records_by_id(AMAZON_GOOGLE_DIR / "tableB.csv")
    pairs = _get_positive_pairs(AMAZON_GOOGLE_DIR / "test.csv")

    b_records = list(table_b.values())
    query_text = _join_fields(table_a[pairs[0][0]], _AG_FIELDS)

    normalizer = (
        Normalizer()
        .custom(lambda v: _join_fields(v, _AG_FIELDS) if isinstance(v, dict) else v)
        .isinstance_str()
        .strip()
        .casefold()
        .min_length(3)
    )

    def build() -> FuzzySequenceIndex[dict[str, str]]:
        return FuzzySequenceIndex(b_records, normalizer=normalizer, score_cutoff=60)

    build_best, _ = measure_timings(_REPEATS, build)
    index = build()
    peak_kib = measure_peak_kib(build)
    find_best, _ = measure_timings(_QUERY_REPEATS, lambda: index.find_one(query_text))

    return RealDataResult(
        dataset="amazon-google",
        collection="FuzzySequenceIndex",
        items=len(index),
        build_ms=build_best,
        find_ms=find_best,
        peak_kib=peak_kib,
    )


def bench_immutable_fuzzy_keyed_index_dblp_acm() -> RealDataResult:
    """Benchmark ImmutableFuzzyKeyedIndex on dirty DBLP-ACM publications."""
    table_a = _load_records_by_id(DBLP_ACM_DIR / "tableA.csv")
    table_b = _load_records_by_id(DBLP_ACM_DIR / "tableB.csv")
    pairs = _get_positive_pairs(DBLP_ACM_DIR / "test.csv")

    text_by_id = {rid: _join_fields(rec, _DA_FIELDS) or None for rid, rec in table_b.items()}
    query_text = _join_fields(table_a[pairs[0][0]], _DA_FIELDS)

    normalizer = (
        Normalizer()
        .custom(lambda v: text_by_id.get(v, v) if isinstance(v, str) else None)
        .isinstance_str()
        .strip()
        .casefold()
        .min_length(3)
    )

    def build() -> ImmutableFuzzyKeyedIndex[str]:
        return ImmutableFuzzyKeyedIndex(table_b.keys(), normalizer=normalizer, score_cutoff=60)

    build_best, _ = measure_timings(_REPEATS, build)
    index = build()
    peak_kib = measure_peak_kib(build)
    find_best, _ = measure_timings(_QUERY_REPEATS, lambda: index.find_one(query_text))

    return RealDataResult(
        dataset="dblp-acm",
        collection="ImmutableFuzzyKeyedIndex",
        items=len(table_b),
        build_ms=build_best,
        find_ms=find_best,
        peak_kib=peak_kib,
    )


def bench_smoke_collections() -> list[RealDataResult]:
    """Benchmark smoke: FuzzyList, FuzzyTuple, FuzzySet, FrozenFuzzySet."""
    table_b = _load_records_by_id(FODORS_ZAGATS_DIR / "tableB.csv")
    names = [rec["name"] for rec in table_b.values() if rec.get("name", "").strip()]
    query = make_typo(names[0])

    def build_list() -> FuzzyList[str]:
        return FuzzyList(names, score_cutoff=60)

    def build_tuple() -> FuzzyTuple[str]:
        return FuzzyTuple(names, score_cutoff=60)

    fl = build_list()
    ft = build_tuple()

    build_best_l, _ = measure_timings(_REPEATS, build_list)
    find_best_l, _ = measure_timings(_QUERY_REPEATS, lambda: fl.fuzzy_find_one(query))
    peak_l = measure_peak_kib(build_list)

    build_best_t, _ = measure_timings(_REPEATS, build_tuple)
    find_best_t, _ = measure_timings(_QUERY_REPEATS, lambda: ft.fuzzy_find_one(query))
    peak_t = measure_peak_kib(build_tuple)

    table_b_ag = _load_records_by_id(AMAZON_GOOGLE_DIR / "tableB.csv")
    titles = sorted(rec["title"] for rec in table_b_ag.values() if rec.get("title", "").strip())
    query_ag = make_typo(titles[0])

    def build_set() -> FuzzySet[str]:
        return FuzzySet(titles, score_cutoff=60)

    def build_frozen_set() -> FrozenFuzzySet[str]:
        return FrozenFuzzySet(titles, score_cutoff=60)

    fs = build_set()
    ffs = build_frozen_set()

    build_best_s, _ = measure_timings(_REPEATS, build_set)
    find_best_s, _ = measure_timings(_QUERY_REPEATS, lambda: fs.fuzzy_find_one(query_ag))
    peak_s = measure_peak_kib(build_set)

    build_best_fs, _ = measure_timings(_REPEATS, build_frozen_set)
    find_best_fs, _ = measure_timings(_QUERY_REPEATS, lambda: ffs.fuzzy_find_one(query_ag))
    peak_fs = measure_peak_kib(build_frozen_set)

    return [
        RealDataResult(
            dataset="fodors-zagats",
            collection="FuzzyList",
            items=len(fl),
            build_ms=build_best_l,
            find_ms=find_best_l,
            peak_kib=peak_l,
        ),
        RealDataResult(
            dataset="fodors-zagats",
            collection="FuzzyTuple",
            items=len(ft),
            build_ms=build_best_t,
            find_ms=find_best_t,
            peak_kib=peak_t,
        ),
        RealDataResult(
            dataset="amazon-google",
            collection="FuzzySet",
            items=len(fs),
            build_ms=build_best_s,
            find_ms=find_best_s,
            peak_kib=peak_s,
        ),
        RealDataResult(
            dataset="amazon-google",
            collection="FrozenFuzzySet",
            items=len(ffs),
            build_ms=build_best_fs,
            find_ms=find_best_fs,
            peak_kib=peak_fs,
        ),
    ]


def write_outputs(results: list[RealDataResult], output_dir: Path) -> None:
    """Write raw benchmark result rows as JSON and CSV."""

    rows = [asdict(result) for result in results]
    write_benchmark_reports(rows, output_dir, stem="real_data_results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse real-data benchmark arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports/real_data"),
        help="Directory to write JSON and CSV results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run all real-data benchmarks and write JSON and CSV reports."""

    args = parse_args(argv)
    results = [
        bench_frozen_fuzzy_dict_fodors_zagats(),
        bench_fuzzy_sequence_index_amazon_google(),
        bench_immutable_fuzzy_keyed_index_dblp_acm(),
        *bench_smoke_collections(),
    ]
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
