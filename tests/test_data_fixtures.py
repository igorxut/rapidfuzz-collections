from rapidfuzz_collections import FuzzyDict, FuzzyList, FuzzySet
from tests.data import (
    DUPLICATE_PRODUCT_NAMES,
    MAPPING_VALUES,
    MIXED_SEARCH_VALUES,
    NORMALIZED_COLLISION_VALUES,
    PRODUCT_NAMES,
    PRODUCT_PRICES,
    PRODUCT_QUERIES,
    UNICODE_PRODUCT_NAMES,
)
from tests.helpers import require_not_none


def test_product_data_works_for_sequence_and_mapping_lookup():
    products = FuzzyList(PRODUCT_NAMES)
    prices = FuzzyDict(PRODUCT_PRICES)

    for query, expected_name in PRODUCT_QUERIES.items():
        assert products.fuzzy_get(query) == expected_name
        assert prices.fuzzy_get(query) == PRODUCT_PRICES[expected_name]

    assert products.fuzzy_get("Coffee Grinder") is None
    assert prices.fuzzy_get("Coffee Grinder") is None


def test_unicode_product_names_are_real_unicode_values():
    products = FuzzySet(UNICODE_PRODUCT_NAMES)

    match = require_not_none(products.fuzzy_find_one("M\u00dcNCHEN ADAPTER"))
    assert match.value == "M\u00fcnchen Adapter"
    assert match.normalized_query == "m\u00fcnchen adapter"
    assert match.normalized_value == "m\u00fcnchen adapter"


def test_normalized_collisions_keep_insertion_order():
    values = FuzzyList(NORMALIZED_COLLISION_VALUES)

    match = require_not_none(values.fuzzy_find_one("alpha phone"))
    matches = values.fuzzy_find_many("alpha phone", limit=None)

    assert match.value == "Alpha Phone"
    assert [result.value for result in matches] == list(NORMALIZED_COLLISION_VALUES)


def test_duplicate_data_keeps_sequence_duplicates():
    values = FuzzyList(DUPLICATE_PRODUCT_NAMES)

    matches = values.fuzzy_find_many("Alpha Phone 128GB", limit=None)

    assert [match.index for match in matches] == [0, 1]
    assert [match.value for match in matches] == ["Alpha Phone 128GB", "Alpha Phone 128GB"]


def test_mixed_data_excludes_unsearchable_values_from_fuzzy_paths():
    values = FuzzyList(MIXED_SEARCH_VALUES)

    assert values.fuzzy_get("xy") is None
    assert values.fuzzy_count("xy") == 0
    assert values[2] == "xy"
    assert values.fuzzy_get("Alpa Phone 128") == "Alpha Phone 128GB"


def test_mapping_data_searches_keys_and_returns_payloads():
    values = FuzzyDict(MAPPING_VALUES)

    match = require_not_none(values.fuzzy_find_item("Gama Camera"))
    assert match.key == "Gamma Camera Pro"
    assert match.value == "GC-PRO"
