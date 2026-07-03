"""Lookup product records with FuzzySequenceIndex without reshaping data.

The source data remains a list of record dictionaries. The index uses a
normalizer to extract searchable text from each record, and returns the original
record object as ``match.value``.

The example prints top-K candidates and uses a conservative decision policy:
high scores are accepted only when the best candidate is clearly separated from
the next candidate. It also demonstrates a per-query ``score_cutoff`` override:
each query additionally checks, just for itself, how many candidates would
clear a stricter cutoff, without changing the index's own default cutoff.
"""

from dataset_utils import (
    AMAZON_GOOGLE_DIR,
    find_expected_rank,
    first_existing_pairs,
    get_positive_pairs,
    join_fields,
    load_records_by_id,
    print_decision,
    print_record,
)
from rapidfuzz.fuzz import token_set_ratio

from rapidfuzz_collections import FuzzySequenceIndex, Normalizer

_SEARCH_FIELDS = ["title", "manufacturer", "price"]
_DISPLAY_FIELDS = ["title", "manufacturer", "price"]
_PREFERRED_PAIRS = [("1191", "567"), ("276", "3022"), ("517", "986")]
_TOP_K = 5


def _record_normalizer() -> Normalizer:
    """Build a normalizer that accepts record dicts and query strings."""
    return (
        Normalizer()
        .custom(lambda value: join_fields(value, _SEARCH_FIELDS) if isinstance(value, dict) else value)
        .isinstance_str()
        .strip()
        .casefold()
        .min_length(3)
    )


def main() -> None:
    """Run the sequence-index product lookup example."""
    table_a = load_records_by_id(AMAZON_GOOGLE_DIR / "tableA.csv")
    table_b = load_records_by_id(AMAZON_GOOGLE_DIR / "tableB.csv")
    pairs = first_existing_pairs(
        get_positive_pairs(AMAZON_GOOGLE_DIR / "test.csv"),
        table_a,
        table_b,
        _PREFERRED_PAIRS,
        limit=3,
    )

    records = list(table_b.values())
    index: FuzzySequenceIndex[dict[str, str]] = FuzzySequenceIndex(
        records,
        normalizer=_record_normalizer(),
        scorer=token_set_ratio,
        score_cutoff=40,
    )

    print(f"Canonical Google product records indexed: {len(index)}")
    print("Use case: keep records as a list, add a fuzzy search layer over them")
    print()

    for query_id, expected_id in pairs:
        query_record = table_a[query_id]
        query_text = join_fields(query_record, _SEARCH_FIELDS)
        matches = index.find_many(query_text, limit=_TOP_K)
        best = matches[0] if matches else None
        next_score = matches[1].score if len(matches) > 1 else None
        expected_rank = find_expected_rank(matches, expected_id, lambda match: match.value["id"])

        print(f"tableA.id={query_id}, expected tableB.id={expected_id}")
        print_record("  imported", query_record, _DISPLAY_FIELDS)
        if best is None:
            print("  top candidates: none")
            print_decision(None)
            print()
            continue

        status = "OK" if best.value["id"] == expected_id else "candidate for review"
        # Per-query override: check how many candidates would clear a stricter,
        # auto-accept-level cutoff for this one query, without changing the index's
        # own default (score_cutoff=40) for any other query.
        strict_matches = index.find_many(query_text, limit=_TOP_K, score_cutoff=90)
        print(f"  candidates clearing a stricter score_cutoff=90 (this query only): {len(strict_matches)}")
        print(f"  best candidate: tableB.id={best.value['id']} ({status}), score={best.score:.1f}, index={best.index}")
        print_record("  candidate record", best.value, _DISPLAY_FIELDS)
        print(f"  expected rank in top-{_TOP_K}: {expected_rank if expected_rank is not None else 'not retrieved'}")
        print("  top candidates:")
        for rank, candidate in enumerate(matches, start=1):
            marker = "*" if candidate.value["id"] == expected_id else " "
            print(f"    {marker}{rank}. tableB.id={candidate.value['id']}, score={candidate.score:.1f}")
        print_decision(best.score, next_score=next_score)
        print()


if __name__ == "__main__":
    main()
