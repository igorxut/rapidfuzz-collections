"""Reusable test data for fuzzy collection behavior."""

PRODUCT_CATALOG = {
    "Alpha Phone 128GB": {"sku": "AP-128", "price": 499},
    "Alpha Phone Case": {"sku": "AP-CASE", "price": 29},
    "Beta Tablet 11": {"sku": "BT-11", "price": 799},
    "Gamma Camera Pro": {"sku": "GC-PRO", "price": 1199},
    "Delta USB-C Cable": {"sku": "DC-CABLE", "price": 19},
}

PRODUCT_NAMES = tuple(PRODUCT_CATALOG)
PRODUCT_PRICES = {name: data["price"] for name, data in PRODUCT_CATALOG.items()}
PRODUCT_QUERIES = {
    "Alpa Phone 128": "Alpha Phone 128GB",
    "Bta Tablet": "Beta Tablet 11",
    "Gama Camera": "Gamma Camera Pro",
    "usb cable": "Delta USB-C Cable",
}

UNICODE_PRODUCT_NAMES = (
    "M\u00fcnchen Adapter",
    "S\u00e3o Paulo Router",
    "Z\u00fcrich Sensor",
    "\u6771\u4eac Camera",
)

NORMALIZED_COLLISION_VALUES = (
    "Alpha Phone",
    "  alpha phone  ",
    "ALPHA PHONE",
)

DUPLICATE_PRODUCT_NAMES = (
    "Alpha Phone 128GB",
    "Alpha Phone 128GB",
    "Beta Tablet 11",
)

MIXED_SEARCH_VALUES = (
    None,
    1,
    "xy",
    "Alpha Phone 128GB",
    "Beta Tablet 11",
)

MAPPING_VALUES = {
    "Alpha Phone 128GB": "AP-128",
    "Beta Tablet 11": "BT-11",
    "Gamma Camera Pro": "GC-PRO",
}
