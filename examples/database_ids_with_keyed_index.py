"""Return stable database-style IDs with ImmutableFuzzyKeyedIndex.

The index stores ACM publication IDs as values. A custom normalizer resolves each
ID to searchable publication text. Search results therefore return stable IDs
that can be used to retrieve full records from an external mapping.
"""

from dataset_utils import (
    DBLP_ACM_DIR,
    find_expected_rank,
    first_existing_pairs,
    get_positive_pairs,
    join_fields,
    load_records_by_id,
    print_decision,
    print_record,
)
from rapidfuzz.fuzz import token_set_ratio

from rapidfuzz_collections import ImmutableFuzzyKeyedIndex, Normalizer

_SEARCH_FIELDS = ["title", "authors", "venue", "year"]
_PREFERRED_PAIRS = [("1683", "1341"), ("1157", "2244"), ("1538", "873")]
_TOP_K = 5


def _id_normalizer(text_by_id: dict[str, str | None]) -> Normalizer:
    """Build a normalizer that resolves stored IDs to searchable text."""
    return (
        Normalizer()
        .custom(lambda value: text_by_id.get(value, value) if isinstance(value, str) else None)
        .isinstance_str()
        .strip()
        .casefold()
        .min_length(3)
    )


def main() -> None:
    """Run the keyed-index lookup example."""
    table_a = load_records_by_id(DBLP_ACM_DIR / "tableA.csv")
    table_b = load_records_by_id(DBLP_ACM_DIR / "tableB.csv")
    pairs = first_existing_pairs(
        get_positive_pairs(DBLP_ACM_DIR / "test.csv"),
        table_a,
        table_b,
        _PREFERRED_PAIRS,
        limit=3,
    )
    text_by_id = {record_id: join_fields(record, _SEARCH_FIELDS) or None for record_id, record in table_b.items()}

    index: ImmutableFuzzyKeyedIndex[str] = ImmutableFuzzyKeyedIndex(
        table_b.keys(),
        normalizer=_id_normalizer(text_by_id),
        scorer=token_set_ratio,
        score_cutoff=40,
    )

    print(f"Canonical ACM publication IDs indexed: {len(table_b)}")
    print("Use case: fuzzy lookup returns stable IDs for external record storage")
    print()

    for query_id, expected_id in pairs:
        query_record = table_a[query_id]
        query_text = join_fields(query_record, _SEARCH_FIELDS)
        matches = index.find_many(query_text, limit=_TOP_K)
        best = matches[0] if matches else None
        next_score = matches[1].score if len(matches) > 1 else None
        expected_rank = find_expected_rank(matches, expected_id, lambda match: match.value)

        print(f"tableA.id={query_id}, expected tableB.id={expected_id}")
        print_record("  imported", query_record, _SEARCH_FIELDS)
        if best is None:
            print("  best id: no match")
            print_decision(None)
            print()
            continue

        selected_record = table_b[best.value]
        status = "OK" if best.value == expected_id else "candidate for review"
        print(f"  best id: tableB.id={best.value} ({status}), score={best.score:.1f}")
        print_record("  selected", selected_record, _SEARCH_FIELDS)
        print(f"  expected rank in top-{_TOP_K}: {expected_rank if expected_rank is not None else 'not retrieved'}")
        print_decision(best.score, next_score=next_score)
        print()


if __name__ == "__main__":
    main()
