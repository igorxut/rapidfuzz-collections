"""Lookup imported restaurant rows in a canonical catalog with FrozenFuzzyDict.

The fuzzy dictionary maps a compact searchable restaurant key to the full Zagat
record. A match therefore gives both the matched dictionary key and the
attached payload record, including the canonical ``id`` and all original
fields.

A compact key is useful when it is unique enough for the domain. If duplicate
or near-duplicate keys are common, prefer a sequence or keyed index that keeps
all records independently.

This example intentionally separates fuzzy lookup from final business decisions:
clear matches can be accepted automatically, while ambiguous or medium-score
matches remain review candidates.
"""

from collections.abc import Sequence

from dataset_utils import (
    FODORS_ZAGATS_DIR,
    first_existing_pairs,
    get_positive_pairs,
    join_fields,
    load_records_by_id,
    print_decision,
    print_record,
)

from rapidfuzz_collections import FrozenFuzzyDict, MappingMatch

_KEY_FIELDS = ["name"]
_DISPLAY_FIELDS = ["name", "addr", "city", "phone", "type"]
_PREFERRED_PAIRS = [("74", "292"), ("37", "255"), ("22", "240")]
_TOP_K = 3


def _print_candidates(matches: Sequence[MappingMatch[str, dict[str, str]]], expected_id: str) -> None:
    """Print a compact candidate list returned by the fuzzy dictionary."""
    print("  top candidates:")
    for rank, candidate in enumerate(matches, start=1):
        marker = "*" if candidate.value["id"] == expected_id else " "
        print(f"    {marker}{rank}. tableB.id={candidate.value['id']}, score={candidate.score:.1f}")


def main() -> None:
    """Run the catalog lookup example."""
    table_a = load_records_by_id(FODORS_ZAGATS_DIR / "tableA.csv")
    table_b = load_records_by_id(FODORS_ZAGATS_DIR / "tableB.csv")
    pairs = first_existing_pairs(
        get_positive_pairs(FODORS_ZAGATS_DIR / "test.csv"),
        table_a,
        table_b,
        _PREFERRED_PAIRS,
        limit=3,
    )

    catalog: FrozenFuzzyDict[str, dict[str, str]] = FrozenFuzzyDict(
        {join_fields(record, _KEY_FIELDS): record for record in table_b.values()},
        score_cutoff=60,
    )

    print(f"Canonical Zagat records indexed: {len(catalog)}")
    print("Use case: compact fuzzy key -> reviewable canonical candidates + payload")
    print("The key is used for fuzzy lookup; the value keeps the full canonical record.")
    print()

    for query_id, expected_id in pairs:
        query_record = table_a[query_id]
        query_text = join_fields(query_record, _KEY_FIELDS)
        matches = catalog.fuzzy_find_items(query_text, limit=_TOP_K)
        best = matches[0] if matches else None
        next_score = matches[1].score if len(matches) > 1 else None

        print(f"tableA.id={query_id}, expected tableB.id={expected_id}")
        print_record("  imported", query_record, _DISPLAY_FIELDS)
        print(f"  search key: {query_text}")
        if best is None:
            print("  best candidate: none")
            print_decision(None)
            print()
            continue

        status = "OK" if best.value["id"] == expected_id else "candidate for review"
        print(f"  best candidate: tableB.id={best.value['id']} ({status}), score={best.score:.1f}")
        print_record("  candidate payload", best.value, _DISPLAY_FIELDS)
        print(f"  matched dictionary key: {best.key}")
        _print_candidates(matches, expected_id)
        print_decision(best.score, next_score=next_score)
        print()


if __name__ == "__main__":
    main()
