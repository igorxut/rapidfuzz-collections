"""Use FrozenFuzzySet for typo-tolerant vocabulary lookup.

Sets are useful when the result itself is the value: a command, category, label,
or allowed vocabulary term. This example uses restaurant type labels from the
Fodors-Zagats dataset.
"""

from dataset_utils import FODORS_ZAGATS_DIR, load_records_by_id

from rapidfuzz_collections import FrozenFuzzySet

_QUERIES = ["americn", "calfornian", "sea food", "french bisttro"]


def main() -> None:
    """Run the fuzzy vocabulary lookup example."""
    table_b = load_records_by_id(FODORS_ZAGATS_DIR / "tableB.csv")
    labels = sorted({record["type"].strip(" `'") for record in table_b.values() if record.get("type", "").strip()})
    vocabulary: FrozenFuzzySet[str] = FrozenFuzzySet(labels, score_cutoff=60)

    print(f"Known labels indexed: {len(vocabulary)}")
    print("Use case: typo-tolerant lookup over allowed category labels")
    print("A set stores only the matched value; use a dict or index when you need payload.")
    print()

    for query in _QUERIES:
        match = vocabulary.fuzzy_find_one(query)
        if match is None:
            print(f"{query!r} -> no accepted label")
            continue
        print(f"{query!r} -> {match.value!r}, score={match.score:.1f}")


if __name__ == "__main__":
    main()
