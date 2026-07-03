"""Resolve dirty publication records through bounded cdist batch lookup.

This advanced example compares dirty DBLP records with an immutable tuple of
canonical ACM records and evaluates top-one results against known positive
pairs. Try the ordinary batch API first. Use cdist only after measuring the
target scorer, cutoff, query distribution, batch size, and chunk sizes.

Install the optional dependency before running this script:

    pip install -e ".[cdist]"
"""

from dataset_utils import DBLP_ACM_DIR, get_positive_pairs, join_fields, load_records_by_id, truncate
from rapidfuzz.fuzz import token_set_ratio

from rapidfuzz_collections import FuzzyTuple, Match, Normalizer

_CHOICE_CHUNK_SIZE = 1000
_DISPLAY_ROWS = 5
_PUBLICATION_FIELDS = ("title", "authors", "venue", "year")
_QUERY_CHUNK_SIZE = 32

Publication = dict[str, str]


def _publication_normalizer() -> Normalizer:
    """Build a normalizer for publication records and query strings."""
    return (
        Normalizer()
        .custom(lambda value: join_fields(value, _PUBLICATION_FIELDS) if isinstance(value, dict) else value)
        .isinstance_str()
        .strip()
        .casefold()
        .min_length(3)
    )


def _publication_text(record: Publication) -> str:
    """Return searchable text built from common publication fields."""
    return join_fields(record, _PUBLICATION_FIELDS)


def _format_result(query_id: str, expected_id: str, match: Match[Publication] | None) -> str:
    """Format one compact, deterministic resolution row."""
    if match is None:
        return f"DBLP {query_id:>4} -> no candidate (expected ACM {expected_id})"
    predicted_id = match.value["id"]
    status = "correct" if predicted_id == expected_id else "mismatch"
    title = truncate(match.value.get("title", ""), width=64)
    return (
        f"DBLP {query_id:>4} -> ACM {predicted_id:>4} "
        f"(expected {expected_id:>4}, {status}, score={match.score:5.1f}) {title}"
    )


def main() -> None:
    """Run bounded batch resolution and print a compact quality summary."""
    dirty_records = load_records_by_id(DBLP_ACM_DIR / "tableA.csv")
    canonical_records = load_records_by_id(DBLP_ACM_DIR / "tableB.csv")
    positive_pairs = [
        pair
        for pair in get_positive_pairs(DBLP_ACM_DIR / "test.csv")
        if pair[0] in dirty_records and pair[1] in canonical_records
    ]

    catalog: FuzzyTuple[Publication] = FuzzyTuple(
        canonical_records.values(),
        normalizer=_publication_normalizer(),
        scorer=token_set_ratio,
        score_cutoff=60,
    )
    queries = [_publication_text(dirty_records[query_id]) for query_id, _ in positive_pairs]

    # This workflow can also be implemented with:
    #
    #     catalog.fuzzy_find_one_batch(queries)
    #
    # The cdist variant has the same top-one lookup semantics, but uses bounded
    # query-by-choice matrix scoring internally. Try the ordinary batch method
    # first and switch to cdist only after measuring this workload.

    matches = catalog.fuzzy_find_one_batch_cdist(
        queries,
        query_chunk_size=_QUERY_CHUNK_SIZE,
        choice_chunk_size=_CHOICE_CHUNK_SIZE,
        workers=1,
    )

    correct = sum(
        match is not None and match.value["id"] == expected_id
        for match, (_, expected_id) in zip(matches, positive_pairs, strict=True)
    )
    no_candidate = sum(match is None for match in matches)
    mismatches = len(matches) - correct - no_candidate

    print(f"Canonical ACM records indexed: {len(catalog)}")
    print(f"Dirty DBLP records evaluated: {len(queries)}")
    print(f"Known positives resolved at top-1: {correct}")
    print(f"Top-1 mismatches: {mismatches}")
    print(f"No candidate above cutoff: {no_candidate}")
    print()
    print(f"First {_DISPLAY_ROWS} results in dataset order:")
    for (query_id, expected_id), match in zip(positive_pairs[:_DISPLAY_ROWS], matches[:_DISPLAY_ROWS], strict=True):
        print(f"  {_format_result(query_id, expected_id, match)}")


if __name__ == "__main__":
    main()
