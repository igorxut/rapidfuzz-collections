"""Update a mutable fuzzy reference collection at runtime.

This example uses ``FuzzyDict`` when canonical records are added while the
process is running. The collection updates its fuzzy index as mapping keys are
inserted or replaced, so the same query can resolve to newly added reference
data without rebuilding the collection.

The strict cutoff keeps unrelated pre-update records out of the result. After
the canonical record is added, the same query resolves immediately without
rebuilding the collection.
"""

from collections.abc import Sequence

from dataset_utils import AMAZON_GOOGLE_DIR, join_fields, load_records_by_id, print_decision, print_record

from rapidfuzz_collections import FuzzyDict, MappingMatch

_SEARCH_FIELDS = ["title", "manufacturer", "price"]
_TOP_K = 2


def _print_lookup(label: str, matches: Sequence[MappingMatch[str, dict[str, str]]]) -> None:
    """Print one mutable-catalog lookup result and its operational decision."""
    print(label)
    best = matches[0] if matches else None
    next_score = matches[1].score if len(matches) > 1 else None
    if best is None:
        print("  No match")
        print_decision(None)
        return

    print(f"  Matched key: {best.key}")
    print(f"  Matched record id: {best.value['id']}, score={best.score:.1f}")
    print_record("  Matched record", best.value, _SEARCH_FIELDS)
    print_decision(best.score, next_score=next_score)


def main() -> None:
    """Run the mutable reference data example."""
    table_b = load_records_by_id(AMAZON_GOOGLE_DIR / "tableB.csv")
    seed_records = list(table_b.values())[:25]

    catalog: FuzzyDict[str, dict[str, str]] = FuzzyDict(
        {join_fields(record, _SEARCH_FIELDS): record for record in seed_records},
        score_cutoff=90,
    )

    new_record = {
        "id": "local-001",
        "title": "Adobe Photoshop CS3 for Mac",
        "manufacturer": "Adobe",
        "price": "609.99",
    }
    new_key = join_fields(new_record, _SEARCH_FIELDS)
    query = new_key

    print(f"Initial mutable catalog size: {len(catalog)}")
    print(f"Query: {query}")
    _print_lookup(
        "Before adding the local canonical record (existing catalog only):",
        catalog.fuzzy_find_items(query, limit=_TOP_K),
    )
    print()

    print("Adding one local canonical record...")
    catalog[new_key] = new_record
    print(f"Updated mutable catalog size: {len(catalog)}")
    _print_lookup(
        "After adding the local canonical record (same query, updated collection):",
        catalog.fuzzy_find_items(query, limit=_TOP_K),
    )


if __name__ == "__main__":
    main()
