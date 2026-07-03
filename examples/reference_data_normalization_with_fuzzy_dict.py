"""Normalize external reference-data aliases to canonical IDs with FrozenFuzzyDict.

Real-world ETL pipelines often receive country, region, product, or category
names from noisy external sources. A plain dictionary is enough for exact alias
lookup, but it cannot handle punctuation variants and misspellings.

This example uses ``FrozenFuzzyDict`` as an immutable alias table:

* dictionary keys are accepted aliases from a controlled reference table;
* dictionary values are canonical IDs used by downstream code;
* ``None`` values represent known aliases that should be ignored;
* missing matches are collected for manual review;
* an explicit per-query ``score_cutoff=None`` maintenance lookup records the
  nearest rejected candidate without changing the alias table's configured
  cutoff.

The same pattern can be used for country classifiers, product catalogs,
internal category vocabularies, or any other reference data loaded once and
queried many times.
"""

from rapidfuzz_collections import FrozenFuzzyDict, MappingMatch, Normalizer

# High enough to auto-resolve common typos; medium scores go to manual review.
_AUTO_ACCEPT_SCORE = 94

# Intentionally lower than auto-accept: keeps plausible candidates visible.
_REVIEW_CANDIDATE_SCORE_CUTOFF = 80

CountryId = str | None


REFERENCE_ALIASES: dict[str, CountryId] = {
    "United States": "USA",
    "United States of America": "USA",
    "USA": "USA",
    "US": "USA",
    "United Kingdom": "GBR",
    "Great Britain": "GBR",
    "Britain": "GBR",
    "UK": "GBR",
    "South Korea": "KOR",
    "Republic of Korea": "KOR",
    "Korea Republic": "KOR",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Africa": None,
    "Europe": None,
    "Middle East": None,
}

IMPORTED_VALUES = [
    "United States",
    "Unted States",
    "U.S.A.",
    "Great Britan",
    "Republic of Korea",
    "Czech Rep.",
    "Africa",
    "Atlantis",
]


def _build_alias_normalizer() -> Normalizer:
    """Return a normalizer for compact reference-data alias keys."""
    return Normalizer().isinstance_str().strip().re_sub(r"[^A-Za-z]", "").casefold().min_length(2)


def _describe_match(match: MappingMatch[str, CountryId]) -> str:
    """Return a compact human-readable match description."""
    value = "ignored" if match.value is None else match.value
    return f"{value:<7} matched={match.key!r:<27} score={match.score:5.1f}"


def main() -> None:
    """Run the reference-data normalization example."""
    aliases: FrozenFuzzyDict[str, CountryId] = FrozenFuzzyDict(
        REFERENCE_ALIASES,
        normalizer=_build_alias_normalizer(),
        score_cutoff=_REVIEW_CANDIDATE_SCORE_CUTOFF,
    )

    resolved: list[tuple[str, MappingMatch[str, CountryId]]] = []
    ignored: list[tuple[str, MappingMatch[str, CountryId]]] = []
    review: list[tuple[str, MappingMatch[str, CountryId]]] = []
    unknown: list[tuple[str, MappingMatch[str, CountryId] | None]] = []

    for raw_value in IMPORTED_VALUES:
        match = aliases.fuzzy_find_item(raw_value)
        if match is None:
            nearest_rejected = aliases.fuzzy_find_item(raw_value, score_cutoff=None)
            unknown.append((raw_value, nearest_rejected))
        elif match.value is None:
            ignored.append((raw_value, match))
        elif match.score >= _AUTO_ACCEPT_SCORE:
            resolved.append((raw_value, match))
        else:
            review.append((raw_value, match))

    print(f"Reference aliases indexed: {len(aliases)}")
    print("Use case: dirty external values -> canonical IDs / ignored / review")
    print()

    print("Resolved automatically:")
    for raw_value, match in resolved:
        print(f"  {raw_value!r:<22} -> {_describe_match(match)}")
    print()

    print("Known values ignored intentionally:")
    for raw_value, match in ignored:
        print(f"  {raw_value!r:<22} -> {_describe_match(match)}")
    print()

    print("Review candidates:")
    for raw_value, match in review:
        print(f"  {raw_value!r:<22} -> {_describe_match(match)}")
    print()

    print("Unknown values for reference-data maintenance:")
    for raw_value, nearest_rejected in unknown:
        if nearest_rejected is None:
            print(f"  {raw_value!r:<22} -> no searchable candidate")
        else:
            print(f"  {raw_value!r:<22} -> nearest rejected: {_describe_match(nearest_rejected)}")


if __name__ == "__main__":
    main()
