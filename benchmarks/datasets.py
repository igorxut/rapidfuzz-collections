"""Deterministic dataset factories for reproducible benchmark runs.

All value generators are seeded and deterministic so benchmark results
are comparable across runs.
"""

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import cast


class DataProfile(StrEnum):
    """Supported generated data profiles shared across all benchmark harnesses."""

    COLLISION_0 = "collision-0"
    COLLISION_5 = "collision-5"
    COLLISION_20 = "collision-20"
    COLLISION_50 = "collision-50"
    DUPLICATES = "duplicates"
    MIXED = "mixed"
    UNIQUE = "unique"


@dataclass(frozen=True)
class QuerySet:
    """Representative fuzzy lookup queries for a generated dataset."""

    exact: str
    normalized_exact: str
    normalized_collision_exact: str | None
    close: str
    miss: str
    batch: tuple[str, ...]


def _collision_rate(profile: DataProfile) -> float | None:
    if profile == DataProfile.COLLISION_0:
        return 0.0
    if profile == DataProfile.COLLISION_5:
        return 0.05
    if profile == DataProfile.COLLISION_20:
        return 0.20
    if profile == DataProfile.COLLISION_50:
        return 0.50
    return None


def make_typo(s: str) -> str:
    """Return s with a character transposition in the first long alphabetic word.

    Args:
        s: Input string.

    Returns:
        String with one transposition, or s + "x" if no eligible word is found.
    """
    for word in s.split():
        if word.isalpha() and len(word) > 3:
            i = len(word) // 2
            swapped = word[:i] + word[i + 1] + word[i] + word[i + 2 :]  # noqa: E203
            if swapped != word:
                return s.replace(word, swapped, 1)
    return s + "x"


def build_values(items: int, profile: DataProfile) -> list[object]:
    """Build deterministic values for repeatable benchmarks.

    Domain names:
    - UNIQUE: ``Alpha Phone {i:06d} Model {i % 97:02d}``
    - MIXED: every 7th → int, 7th+1 → None, 7th+2 → "XS" (too short for normalizer),
      rest → ``Beta Tablet {i:06d} Series {i % 97:02d}``
    - DUPLICATES: repeated normalized groups cycling through canonical,
      padded-lowercase, and uppercase variants of
      ``Coffee Grinder {group:06d} Type {group % 97:02d}``
    - COLLISION_*: collision pairs ``Consumer Device {i:06d}`` / padded lower,
      unique fill ``Discontinued Accessory {i:06d} Line {i % 97:02d}``

    Args:
        items: Number of values to generate. Must be >= 1.
        profile: Distribution profile.

    Returns:
        List of generated values (str, int, or None depending on profile).

    Raises:
        ValueError: If items < 1.
        NotImplementedError: If profile is unsupported.
    """
    if items < 1:
        raise ValueError("items must be greater than 0")

    if profile == DataProfile.UNIQUE:
        return [f"Alpha Phone {i:06d} Model {i % 97:02d}" for i in range(items)]

    if profile == DataProfile.MIXED:
        values: list[object] = []
        for i in range(items):
            if i % 7 == 0:
                values.append(i)
            elif i % 7 == 1:
                values.append(None)
            elif i % 7 == 2:
                values.append("XS")
            else:
                values.append(f"Beta Tablet {i:06d} Series {i % 97:02d}")
        return values

    if profile == DataProfile.DUPLICATES:
        group_count = max(1, items // 10)
        result: list[object] = []
        for i in range(items):
            group = i % group_count
            if i % 3 == 0:
                result.append(f"Coffee Grinder {group:06d} Type {group % 97:02d}")
            elif i % 3 == 1:
                result.append(f"  coffee grinder {group:06d} type {group % 97:02d}  ")
            else:
                result.append(f"COFFEE GRINDER {group:06d} TYPE {group % 97:02d}")
        return result

    collision_rate = _collision_rate(profile)
    if collision_rate is None:
        raise NotImplementedError(profile)

    return build_values_with_collision_rate(items, collision_rate)


def build_values_with_collision_rate(items: int, collision_rate: float) -> list[str]:
    """Build string values with a controlled fraction of normalized collisions.

    Each collision pair consists of a canonical form ``Consumer Device {i:06d}``
    and a padded-lowercase variant that normalizes to the same string. Unique
    fill values use the ``Discontinued Accessory`` template.

    Args:
        items: Total number of values to generate.
        collision_rate: Fraction of values sharing a normalized form. Rounded
            down to the nearest even count to ensure complete pairs.

    Returns:
        Shuffled list with the requested collision density (seed 42).
    """
    collision_count = int(items * collision_rate)
    collision_count -= collision_count % 2

    unique_count = items - collision_count
    values: list[str] = [f"Discontinued Accessory {i:06d} Line {i % 97:02d}" for i in range(unique_count)]
    for i in range(collision_count // 2):
        canonical = f"Consumer Device {i:06d}"
        values.append(canonical)
        values.append(f"  consumer device {i:06d}  ")

    rng = random.Random(42)
    rng.shuffle(values)
    return values


def build_queries(values: list[object], batch_size: int) -> QuerySet:
    """Build deterministic queries for exact, collision, miss, and batch scenarios.

    Args:
        values: Domain values for the benchmark.
        batch_size: Number of entries in the returned batch tuple.

    Returns:
        QuerySet covering exact, normalized-exact, normalized-collision exact,
        close (with typo), miss, and a batch of typo-ified candidates.

    Raises:
        ValueError: If no searchable string (stripped length >= 3) is found.
    """
    candidates = cast(list[str], [v for v in values if isinstance(v, str) and len(v.strip()) >= 3])
    if not candidates:
        raise ValueError("value profile does not contain searchable strings")

    exact = candidates[len(candidates) // 3]
    normalized_exact = f"   {exact.upper()}   "
    first_by_normalized: dict[str, str] = {}
    normalized_collision_exact: str | None = None
    for candidate in candidates:
        normalized_candidate = candidate.strip().casefold()
        first_candidate = first_by_normalized.setdefault(normalized_candidate, candidate)
        if first_candidate != candidate:
            normalized_collision_exact = candidate
            break
    close = make_typo(candidates[len(candidates) // 2])
    miss = "Wireless Earbuds 999999"
    batch = tuple(make_typo(candidates[i % len(candidates)]) for i in range(batch_size))
    return QuerySet(
        exact=exact,
        normalized_exact=normalized_exact,
        normalized_collision_exact=normalized_collision_exact,
        close=close,
        miss=miss,
        batch=batch,
    )


def build_mapping(items: int, profile: DataProfile) -> dict[object, str]:
    """Build deterministic mapping data for key lookup benchmarks.

    Returns:
        Dict mapping generated values to sequential ``value-{i:06d}`` strings.
    """
    return {key: f"value-{i:06d}" for i, key in enumerate(build_values(items, profile=profile))}


def build_set(items: int, profile: DataProfile) -> set[object]:
    """Build deterministic set data for membership benchmarks.

    Returns:
        Set of generated values (hashable entries only).
    """
    return set(build_values(items, profile=profile))
