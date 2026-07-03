"""Clean duplicate ordered labels with FuzzyList.

Lists preserve source order and duplicate positions, so they are useful when
fuzzy matching augments an ordered import rather than replacing it with a set.
This example finds a typo-tolerant label, reports positional matches, and then
removes every exact duplicate through the fuzzy API.
"""

from rapidfuzz_collections import FuzzyList


def main() -> None:
    """Run the ordered-label cleanup example."""
    labels: FuzzyList[str] = FuzzyList(
        [
            "Account Settings",
            "Customer Support",
            "Billing",
            "Customer Support",
            "Shipping",
        ],
        score_cutoff=80,
    )
    query = "customer suport"
    matches = labels.fuzzy_find_many(query, limit=None)

    print(f"Original ordered labels: {list(labels)!r}")
    print(f"Query: {query!r}")
    print(f"Matching positions: {[match.index for match in matches]}")
    print(f"Best source index: {labels.fuzzy_find_index(query)}")
    print(f"Fuzzy match count: {labels.fuzzy_count(query)}")

    removed = labels.fuzzy_discard_all("Customer Support", score_cutoff=100)
    print(f"Exact duplicates removed through fuzzy lookup: {removed}")
    print(f"Remaining ordered labels: {list(labels)!r}")


if __name__ == "__main__":
    main()
