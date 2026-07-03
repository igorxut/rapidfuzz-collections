"""Private utility module for loading datasets used in examples."""

import csv
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "examples" / "data"

FODORS_ZAGATS_DIR: Path = DATA_DIR / "structured_fodors_zagats"
AMAZON_GOOGLE_DIR: Path = DATA_DIR / "structured_amazon_google"
DBLP_ACM_DIR: Path = DATA_DIR / "dirty_dblp_acm"

Record = dict[str, str]
RecordById = dict[str, Record]
Pair = tuple[str, str]


def load_csv(path: Path) -> list[Record]:
    """Load a CSV file and return its rows as a list of dicts.

    Args:
        path: Path to the CSV file.

    Returns:
        List of row dicts with string keys and string values.
    """
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_records_by_id(path: Path, id_field: str = "id") -> RecordById:
    """Load a CSV file and return a dict mapping each record ID to its row dict.

    Args:
        path: Path to the CSV file.
        id_field: Name of the column to use as the record identifier.

    Returns:
        Dict mapping record ID strings to their row dicts.

    Raises:
        KeyError: If a row does not contain ``id_field``.
    """
    return {row[id_field]: row for row in load_csv(path)}


def get_positive_pairs(test_path: Path) -> list[Pair]:
    """Return positive matching pairs from a test CSV file.

    A pair is positive when the ``label`` column equals ``"1"``.

    Args:
        test_path: Path to the test.csv file containing ``label``, ``table1.id``,
            and ``table2.id`` columns.

    Returns:
        List of ``(table1_id, table2_id)`` tuples for positive-label rows.
    """
    return [(row["table1.id"], row["table2.id"]) for row in load_csv(test_path) if row.get("label") == "1"]


def join_fields(record: Record, fields: Sequence[str]) -> str:
    """Join non-empty field values from a record dict into a single string.

    Args:
        record: Row dict from a CSV file.
        fields: Ordered field names to include.

    Returns:
        Space-joined string of stripped non-empty field values.
    """
    return " ".join(record[field].strip() for field in fields if record.get(field, "").strip())


def first_existing_pairs(
    pairs: Iterable[Pair],
    table_a: RecordById,
    table_b: RecordById,
    preferred: Sequence[Pair] = (),
    *,
    limit: int = 3,
) -> list[Pair]:
    """Return deterministic positive pairs present in both tables.

    Preferred pairs are yielded first when available. The original pair order
    then fills the remaining slots.

    Args:
        pairs: Candidate ``(tableA_id, tableB_id)`` pairs.
        table_a: Loaded left-side records keyed by ID.
        table_b: Loaded right-side records keyed by ID.
        preferred: Optional pairs that should be selected first when present.
        limit: Maximum number of pairs to return.

    Returns:
        Existing pairs in deterministic order.
    """
    selected: list[Pair] = []
    seen: set[Pair] = set()

    def add(pair: Pair) -> None:
        if pair in seen:
            return
        a_id, b_id = pair
        if a_id in table_a and b_id in table_b:
            selected.append(pair)
            seen.add(pair)

    for _pair in preferred:
        add(_pair)
        if len(selected) >= limit:
            return selected

    for _pair in pairs:
        add(_pair)
        if len(selected) >= limit:
            return selected

    return selected


def truncate(text: str, *, width: int = 96) -> str:
    """Return a single-line display string no wider than ``width``."""
    compact = " ".join(text.split())
    if len(compact) <= width:
        return compact
    return f"{compact[: max(0, width - 3)]}..."


def print_record(label: str, record: Record, fields: Sequence[str]) -> None:
    """Print a compact display of selected record fields."""
    print(f"{label}: {truncate(join_fields(record, fields))}")


def find_expected_rank[M](
    matches: Sequence[M],
    expected_id: str,
    get_record_id: Callable[[M], str | None],
) -> int | None:
    """Return one-based rank of ``expected_id`` in ``matches`` or ``None``."""
    for rank, match in enumerate(matches, start=1):
        if get_record_id(match) == expected_id:
            return rank
    return None


def print_decision(
    score: float | int | None,
    *,
    accept_cutoff: float = 90.0,
    review_cutoff: float = 75.0,
    next_score: float | int | None = None,
    min_accept_gap: float = 10.0,
) -> None:
    """Print a conservative operational decision for a fuzzy candidate.

    A high score alone is not always enough for automatic acceptance. When the
    next candidate score is available, the best candidate must also be clearly
    separated from it. This keeps the examples honest on datasets with
    near-duplicate products, restaurants, or publications.
    """
    if score is None:
        print("Decision: no candidate; send to manual review")
        return

    score_value = float(score)
    if score_value >= accept_cutoff:
        if next_score is None:
            print(
                "Decision: accept automatically "
                f"(score {score_value:.2f} >= {accept_cutoff:.2f}; no competing candidate retrieved)"
            )
            return

        gap = score_value - float(next_score)
        if gap + 1e-9 >= min_accept_gap:
            print(
                "Decision: accept automatically "
                f"(score {score_value:.2f} >= {accept_cutoff:.2f}, "
                f"gap {gap:.2f} >= {min_accept_gap:.2f})"
            )
        else:
            print(
                "Decision: review manually "
                f"(score {score_value:.2f} is high, but gap {gap:.2f} "
                f"< required {min_accept_gap:.2f})"
            )
        return

    if score_value >= review_cutoff:
        print(
            "Decision: review manually "
            f"(score {score_value:.2f} is between review threshold "
            f"{review_cutoff:.2f} and accept threshold {accept_cutoff:.2f})"
        )
    else:
        print(
            "Decision: no reliable candidate "
            f"(score {score_value:.2f} < review threshold {review_cutoff:.2f}); "
            "send to manual review"
        )
