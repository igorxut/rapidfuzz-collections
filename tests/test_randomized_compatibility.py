"""Randomized compatibility checks between Fuzzy collections and Python builtins.

These tests replay long pseudo-random operation sequences against a Fuzzy
collection and an equivalent builtin collection in lockstep, asserting that
observable state stays identical and exact fuzzy lookup remains synchronized
after every step. They complement the
deterministic compatibility tests (``test_fuzzy_*_compatibility.py``) by
exercising interaction effects between operations that handwritten cases are
unlikely to cover, using ``random`` with fixed seeds so failures reproduce
deterministically.
"""

import random
from collections.abc import Callable

import pytest

from rapidfuzz_collections import FuzzyDict, FuzzyList, FuzzySet

_POOL = [f"item-{i}" for i in range(12)]
_SEEDS = [0, 1, 2, 3, 4]
_STEPS = 200


def _random_value(rng: random.Random) -> str:
    return rng.choice(_POOL)


def _run_fuzzy_set_sequence(rng: random.Random) -> None:
    fuzzy = FuzzySet[str]()
    reference: set[str] = set()

    def do_add() -> None:
        value = _random_value(rng)
        fuzzy.add(value)
        reference.add(value)

    def do_discard() -> None:
        value = _random_value(rng)
        fuzzy.discard(value)
        reference.discard(value)

    def do_pop() -> None:
        if not reference:
            return
        popped = fuzzy.pop()
        reference.remove(popped)

    def do_update() -> None:
        values = [_random_value(rng) for _ in range(rng.randint(1, 3))]
        fuzzy.update(values)
        reference.update(values)

    actions: list[Callable[[], None]] = [do_add, do_discard, do_pop, do_update]
    for _ in range(_STEPS):
        rng.choice(actions)()
        assert set(fuzzy) == reference
        if reference:
            expected = min(reference)
            assert fuzzy.fuzzy_get(expected) == expected


def _run_fuzzy_list_sequence(rng: random.Random) -> None:
    fuzzy = FuzzyList[str]()
    reference: list[str] = []

    def do_append() -> None:
        value = _random_value(rng)
        fuzzy.append(value)
        reference.append(value)

    def do_insert() -> None:
        value = _random_value(rng)
        index = rng.randint(0, len(reference))
        fuzzy.insert(index, value)
        reference.insert(index, value)

    def do_remove() -> None:
        if not reference:
            return
        value = rng.choice(reference)
        fuzzy.remove(value)
        reference.remove(value)

    def do_pop() -> None:
        if not reference:
            return
        index = rng.randrange(len(reference))
        assert fuzzy.pop(index) == reference.pop(index)

    def do_setitem() -> None:
        if not reference:
            return
        index = rng.randrange(len(reference))
        value = _random_value(rng)
        fuzzy[index] = value
        reference[index] = value

    def do_delitem() -> None:
        if not reference:
            return
        index = rng.randrange(len(reference))
        del fuzzy[index]
        del reference[index]

    def do_extend() -> None:
        values = [_random_value(rng) for _ in range(rng.randint(1, 3))]
        fuzzy.extend(values)
        reference.extend(values)

    def do_reverse() -> None:
        fuzzy.reverse()
        reference.reverse()

    def do_sort() -> None:
        fuzzy.sort()
        reference.sort()

    actions: list[Callable[[], None]] = [
        do_append,
        do_insert,
        do_remove,
        do_pop,
        do_setitem,
        do_delitem,
        do_extend,
        do_reverse,
        do_sort,
    ]
    for _ in range(_STEPS):
        rng.choice(actions)()
        assert list(fuzzy) == reference
        if reference:
            expected = reference[0]
            assert fuzzy.fuzzy_get(expected) == expected


def _run_fuzzy_dict_sequence(rng: random.Random) -> None:
    fuzzy = FuzzyDict[str, str]()
    reference: dict[str, str] = {}

    def do_setitem() -> None:
        key = _random_value(rng)
        value = _random_value(rng)
        fuzzy[key] = value
        reference[key] = value

    def do_delitem() -> None:
        if not reference:
            return
        key = rng.choice(list(reference))
        del fuzzy[key]
        del reference[key]

    def do_pop() -> None:
        if not reference:
            return
        key = rng.choice(list(reference))
        assert fuzzy.pop(key) == reference.pop(key)

    def do_setdefault() -> None:
        key = _random_value(rng)
        default = _random_value(rng)
        assert fuzzy.setdefault(key, default) == reference.setdefault(key, default)

    def do_update() -> None:
        pairs = {_random_value(rng): _random_value(rng) for _ in range(rng.randint(1, 3))}
        fuzzy.update(pairs)
        reference.update(pairs)

    def do_popitem() -> None:
        if not reference:
            return
        assert fuzzy.popitem() == reference.popitem()

    actions: list[Callable[[], None]] = [
        do_setitem,
        do_delitem,
        do_pop,
        do_setdefault,
        do_update,
        do_popitem,
    ]
    for _ in range(_STEPS):
        rng.choice(actions)()
        assert dict(fuzzy) == reference
        assert list(fuzzy) == list(reference)
        if reference:
            expected_key = next(iter(reference))
            assert fuzzy.fuzzy_get(expected_key) == reference[expected_key]


@pytest.mark.parametrize("seed", _SEEDS)
def test_fuzzy_set_random_operation_sequence_matches_builtin_set(seed: int) -> None:
    _run_fuzzy_set_sequence(random.Random(seed))


@pytest.mark.parametrize("seed", _SEEDS)
def test_fuzzy_list_random_operation_sequence_matches_builtin_list(seed: int) -> None:
    _run_fuzzy_list_sequence(random.Random(seed))


@pytest.mark.parametrize("seed", _SEEDS)
def test_fuzzy_dict_random_operation_sequence_matches_builtin_dict(seed: int) -> None:
    _run_fuzzy_dict_sequence(random.Random(seed))
