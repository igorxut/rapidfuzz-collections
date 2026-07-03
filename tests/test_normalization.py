import re
from copy import copy, deepcopy

import pytest

from rapidfuzz_collections import Normalizer

# noinspection PyProtectedMember
from rapidfuzz_collections.normalization import _default_normalizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NON_STR_VALUES = (None, 123, 1.5, ["a"], {"a"}, ("a",), {"a": "b"}, object())


def _non_str_all_none(normalizer: Normalizer) -> None:
    for value in NON_STR_VALUES:
        assert normalizer(value) is None


# ---------------------------------------------------------------------------
# Default normalizer
# ---------------------------------------------------------------------------


def test_normalizer_default_matches_pipeline_and_function():
    normalizer1 = Normalizer.default()
    normalizer2 = Normalizer().isinstance_str().strip().casefold().min_length(3)

    for value in (
        None,
        123,
        123.123,
        ["test"],
        {"test"},
        ("test",),
        {"test": "test"},
        "test",
        "  TEST  ",
        "\u2003Stra\u00dfe\u2003",
        "\u00a0ABC\u00a0",
        "xy",
    ):
        assert normalizer1(value) == normalizer2(value)
        assert _default_normalizer(value) == normalizer2(value)


def test_normalizer_capitalize_transforms_str():
    normalizer = Normalizer().capitalize()
    for source, target in (
        (None, None),
        (123, None),
        (123.123, None),
        (["test"], None),
        ({"test"}, None),
        (("test",), None),
        ({"test": "test"}, None),
        ("  teSt1 2 tesT3 ", "  test1 2 test3 "),
        ("teSt1 2 tesT3 ", "Test1 2 test3 "),
    ):
        assert normalizer(source) == target


def test_normalizer_casefold_lowercases_str():
    normalizer = Normalizer().casefold()
    for source, target in (
        (None, None),
        (123, None),
        (123.123, None),
        (["test"], None),
        ({"test"}, None),
        (("test",), None),
        ({"test": "test"}, None),
        ("  teSt1 2 tesT3 ", "  test1 2 test3 "),
        ("teSt1 2 tesT3 ", "test1 2 test3 "),
    ):
        assert normalizer(source) == target


def test_normalizer_endswith_filters_by_suffix():
    normalizer = Normalizer().endswith("suffix")

    assert normalizer("value-suffix") == "value-suffix"
    assert normalizer("value-prefix") is None


# ---------------------------------------------------------------------------
# String-transform steps
# ---------------------------------------------------------------------------


def test_normalizer_lower_transforms_str():
    n = Normalizer().lower()
    assert n("HELLO World") == "hello world"
    _non_str_all_none(n)


def test_normalizer_upper_transforms_str():
    n = Normalizer().upper()
    assert n("hello World") == "HELLO WORLD"
    _non_str_all_none(n)


def test_normalizer_strip_removes_whitespace():
    n = Normalizer().strip()
    assert n("  hello  ") == "hello"
    assert n("hello") == "hello"
    _non_str_all_none(n)


def test_normalizer_strip_with_chars():
    n = Normalizer().strip(chars="x")
    assert n("xxhelloxx") == "hello"
    assert n("hello") == "hello"
    _non_str_all_none(n)


def test_normalizer_strip_invalid_chars_raises():
    with pytest.raises(TypeError):
        Normalizer().strip(chars=123)  # type: ignore[arg-type]


def test_normalizer_lstrip_removes_left_whitespace():
    n = Normalizer().lstrip()
    assert n("  hello  ") == "hello  "
    _non_str_all_none(n)


def test_normalizer_lstrip_with_chars():
    n = Normalizer().lstrip(chars="x")
    assert n("xxhello") == "hello"
    _non_str_all_none(n)


def test_normalizer_lstrip_invalid_chars_raises():
    with pytest.raises(TypeError):
        Normalizer().lstrip(chars=42)  # type: ignore[arg-type]


def test_normalizer_rstrip_removes_right_whitespace():
    n = Normalizer().rstrip()
    assert n("  hello  ") == "  hello"
    _non_str_all_none(n)


def test_normalizer_rstrip_with_chars():
    n = Normalizer().rstrip(chars="x")
    assert n("helloxx") == "hello"
    _non_str_all_none(n)


def test_normalizer_rstrip_invalid_chars_raises():
    with pytest.raises(TypeError):
        Normalizer().rstrip(chars=42)  # type: ignore[arg-type]


def test_normalizer_removeprefix_removes_matching_prefix():
    n = Normalizer().removeprefix("pre_")
    assert n("pre_value") == "value"
    assert n("other") == "other"
    _non_str_all_none(n)


def test_normalizer_removeprefix_invalid_arg_raises():
    with pytest.raises(TypeError):
        Normalizer().removeprefix(123)  # type: ignore[arg-type]


def test_normalizer_removesuffix_removes_matching_suffix():
    n = Normalizer().removesuffix("_suf")
    assert n("value_suf") == "value"
    assert n("other") == "other"
    _non_str_all_none(n)


def test_normalizer_removesuffix_invalid_arg_raises():
    with pytest.raises(TypeError):
        Normalizer().removesuffix(42)  # type: ignore[arg-type]


def test_normalizer_replace_replaces_all_occurrences():
    n = Normalizer().replace("a", "b")
    assert n("banana") == "bbnbnb"
    _non_str_all_none(n)


def test_normalizer_replace_with_count():
    n = Normalizer().replace("a", "b", count=1)
    assert n("banana") == "bbnana"


def test_normalizer_replace_invalid_old_raises():
    with pytest.raises(TypeError):
        Normalizer().replace(1, "b")  # type: ignore[arg-type]


def test_normalizer_replace_invalid_new_raises():
    with pytest.raises(TypeError):
        Normalizer().replace("a", 2)  # type: ignore[arg-type]


def test_normalizer_replace_invalid_count_raises():
    with pytest.raises(TypeError):
        Normalizer().replace("a", "b", count="x")  # type: ignore[arg-type]


def test_normalizer_re_sub_replaces_pattern():
    n = Normalizer().re_sub(r"\d+", "NUM")
    assert n("abc123def456") == "abcNUMdefNUM"
    _non_str_all_none(n)


def test_normalizer_re_sub_with_callable_repl():
    n = Normalizer().re_sub(r"\d+", lambda m: str(int(m.group()) * 2))
    assert n("abc3def5") == "abc6def10"


def test_normalizer_re_sub_with_flags():
    n = Normalizer().re_sub(r"hello", "hi", flags=re.IGNORECASE)
    assert n("HELLO world") == "hi world"


@pytest.mark.parametrize(
    ("kwargs", "exception_type"),
    [
        ({"pattern": 1, "repl": "x"}, TypeError),
        ({"pattern": re.compile(b"x"), "repl": "y"}, TypeError),
        ({"pattern": "x", "repl": 1}, TypeError),
        ({"pattern": "x", "repl": "y", "count": True}, TypeError),
        ({"pattern": "x", "repl": "y", "count": -1}, ValueError),
        ({"pattern": "x", "repl": "y", "flags": "IGNORECASE"}, TypeError),
        ({"pattern": re.compile("x"), "repl": "y", "flags": re.IGNORECASE}, ValueError),
    ],
)
def test_normalizer_re_sub_validates_arguments(kwargs, exception_type):
    with pytest.raises(exception_type):
        Normalizer().re_sub(**kwargs)


# ---------------------------------------------------------------------------
# String-filter steps (is* predicates)
# ---------------------------------------------------------------------------


def test_normalizer_isinstance_str_passes_str_rejects_others():
    n = Normalizer().isinstance_str()
    assert n("hello") == "hello"
    _non_str_all_none(n)


def test_normalizer_not_empty_str_rejects_empty():
    n = Normalizer().not_empty_str()
    assert n("hello") == "hello"
    assert n("") is None
    _non_str_all_none(n)


def test_normalizer_isalnum_passes_alphanumeric():
    n = Normalizer().isalnum()
    assert n("abc123") == "abc123"
    assert n("abc 123") is None
    _non_str_all_none(n)


def test_normalizer_isalpha_passes_letters_only():
    n = Normalizer().isalpha()
    assert n("hello") == "hello"
    assert n("hello1") is None
    _non_str_all_none(n)


def test_normalizer_isascii_passes_ascii():
    n = Normalizer().isascii()
    assert n("hello") == "hello"
    assert n("héllo") is None
    _non_str_all_none(n)


def test_normalizer_isdecimal_passes_decimal():
    n = Normalizer().isdecimal()
    assert n("123") == "123"
    assert n("12.3") is None
    _non_str_all_none(n)


def test_normalizer_isdigit_passes_digits():
    n = Normalizer().isdigit()
    assert n("123") == "123"
    assert n("12a") is None
    _non_str_all_none(n)


def test_normalizer_isidentifier_passes_valid_identifier():
    n = Normalizer().isidentifier()
    assert n("hello_world") == "hello_world"
    assert n("123abc") is None
    _non_str_all_none(n)


def test_normalizer_islower_passes_all_lowercase():
    n = Normalizer().islower()
    assert n("hello") == "hello"
    assert n("Hello") is None
    _non_str_all_none(n)


def test_normalizer_isnumeric_passes_numeric():
    n = Normalizer().isnumeric()
    assert n("123") == "123"
    assert n("12.3") is None
    _non_str_all_none(n)


def test_normalizer_isprintable_passes_printable():
    n = Normalizer().isprintable()
    assert n("hello") == "hello"
    assert n("hello\n") is None
    _non_str_all_none(n)


def test_normalizer_isspace_passes_whitespace_only():
    n = Normalizer().isspace()
    assert n("   ") == "   "
    assert n("  a ") is None
    _non_str_all_none(n)


def test_normalizer_istitle_passes_titlecase():
    n = Normalizer().istitle()
    assert n("Hello World") == "Hello World"
    assert n("Hello world") is None
    _non_str_all_none(n)


def test_normalizer_isupper_passes_all_uppercase():
    n = Normalizer().isupper()
    assert n("HELLO") == "HELLO"
    assert n("Hello") is None
    _non_str_all_none(n)


# ---------------------------------------------------------------------------
# Length filter steps
# ---------------------------------------------------------------------------


def test_normalizer_min_length_passes_long_enough():
    n = Normalizer().min_length(3)
    assert n("abc") == "abc"
    assert n("ab") is None
    _non_str_all_none(n)


def test_normalizer_min_length_invalid_type_raises():
    with pytest.raises(TypeError):
        Normalizer().min_length("3")  # type: ignore[arg-type]


def test_normalizer_min_length_zero_raises():
    with pytest.raises(ValueError):
        Normalizer().min_length(0)


def test_normalizer_max_length_passes_short_enough():
    n = Normalizer().max_length(5)
    assert n("hello") == "hello"
    assert n("toolong") is None
    _non_str_all_none(n)


def test_normalizer_max_length_invalid_type_raises():
    with pytest.raises(TypeError):
        Normalizer().max_length(1.5)  # type: ignore[arg-type]


def test_normalizer_max_length_zero_raises():
    with pytest.raises(ValueError):
        Normalizer().max_length(0)


def test_normalizer_exact_length_passes_matching_length():
    n = Normalizer().exact_length(5)
    assert n("hello") == "hello"
    assert n("hi") is None
    assert n("toolong") is None
    _non_str_all_none(n)


def test_normalizer_exact_length_invalid_type_raises():
    with pytest.raises(TypeError):
        Normalizer().exact_length("5")  # type: ignore[arg-type]


def test_normalizer_exact_length_zero_raises():
    with pytest.raises(ValueError):
        Normalizer().exact_length(0)


def test_normalizer_exact_length_bool_raises():
    with pytest.raises(TypeError):
        Normalizer().exact_length(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# startswith / endswith
# ---------------------------------------------------------------------------


def test_normalizer_startswith_str_prefix():
    n = Normalizer().startswith("pre_")
    assert n("pre_value") == "pre_value"
    assert n("other") is None
    _non_str_all_none(n)


def test_normalizer_startswith_tuple_of_prefixes():
    n = Normalizer().startswith(("pre_", "alt_"))
    assert n("pre_value") == "pre_value"
    assert n("alt_value") == "alt_value"
    assert n("other") is None


def test_normalizer_startswith_with_start_end():
    n = Normalizer().startswith("bc", start=1, end=3)
    assert n("abcdef") == "abcdef"
    assert n("aXcdef") is None


def test_normalizer_startswith_with_start_only():
    n = Normalizer().startswith("Phone", start=6)
    assert n("Alpha Phone") == "Alpha Phone"
    assert n("Phone Alpha") is None


def test_normalizer_startswith_invalid_prefix_raises():
    with pytest.raises(TypeError):
        Normalizer().startswith(123)  # type: ignore[arg-type]


def test_normalizer_startswith_invalid_tuple_element_raises():
    with pytest.raises(TypeError):
        Normalizer().startswith(("ok", 123))  # type: ignore[arg-type]


def test_normalizer_startswith_invalid_start_raises():
    with pytest.raises(TypeError):
        Normalizer().startswith("pre", start="0")  # type: ignore[arg-type]


def test_normalizer_startswith_invalid_end_raises():
    with pytest.raises(TypeError):
        Normalizer().startswith("pre", end=1.5)  # type: ignore[arg-type]


def test_normalizer_endswith_str_suffix():
    n = Normalizer().endswith("_suf")
    assert n("value_suf") == "value_suf"
    assert n("other") is None
    _non_str_all_none(n)


def test_normalizer_endswith_tuple_of_suffixes():
    n = Normalizer().endswith(("_suf", "_end"))
    assert n("value_suf") == "value_suf"
    assert n("value_end") == "value_end"
    assert n("other") is None


def test_normalizer_endswith_with_start_end():
    n = Normalizer().endswith("bc", start=1, end=3)
    assert n("abcdef") == "abcdef"
    assert n("aXcdef") is None


def test_normalizer_endswith_with_start_only():
    n = Normalizer().endswith("Phone", start=6)
    assert n("Alpha Phone") == "Alpha Phone"
    assert n("Beta Phone") is None


def test_normalizer_endswith_invalid_suffix_raises():
    with pytest.raises(TypeError):
        Normalizer().endswith(42)  # type: ignore[arg-type]


def test_normalizer_endswith_invalid_tuple_element_raises():
    with pytest.raises(TypeError):
        Normalizer().endswith(("ok", 42))  # type: ignore[arg-type]


def test_normalizer_endswith_invalid_start_raises():
    with pytest.raises(TypeError):
        Normalizer().endswith("suf", start="1")  # type: ignore[arg-type]


def test_normalizer_endswith_invalid_end_raises():
    with pytest.raises(TypeError):
        Normalizer().endswith("suf", end=2.0)  # type: ignore[arg-type]


def test_normalizer_startswith_end_only_does_not_shift_to_start():
    # `end` alone should not be treated as `start`.
    n = Normalizer().startswith("he", end=4)
    assert n("hello") == "hello"  # "hello"[0:4] = "hell", startswith("he") → True
    assert n("world") is None


def test_normalizer_endswith_end_only_does_not_shift_to_start():
    # `end` alone should not be treated as `start`.
    n = Normalizer().endswith("lo", end=5)
    assert n("hello") == "hello"  # "hello"[0:5] = "hello", endswith("lo") → True
    assert n("world") is None


# ---------------------------------------------------------------------------
# custom step
# ---------------------------------------------------------------------------


def test_normalizer_custom_function():
    n = Normalizer().custom(lambda v: v.upper() if isinstance(v, str) else None)
    assert n("hello") == "HELLO"
    assert n(42) is None


def test_normalizer_custom_function_with_args():
    def truncate(v, length):
        return v[:length] if isinstance(v, str) else None

    n = Normalizer().custom(truncate, 3)
    assert n("hello") == "hel"


def test_normalizer_custom_function_with_kwargs():
    def pad(v, *, width):
        return v.ljust(width) if isinstance(v, str) else None

    n = Normalizer().custom(pad, width=10)
    assert n("hi") == "hi        "


def test_normalizer_custom_non_callable_raises():
    with pytest.raises(TypeError):
        Normalizer().custom("not_callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# steps property and chaining
# ---------------------------------------------------------------------------


def test_normalizer_steps_property_returns_tuple():
    n = Normalizer().lower().strip()
    assert len(n.steps) == 2


def test_normalizer_chaining_applies_steps_in_order():
    n = Normalizer().strip().lower().min_length(3)
    assert n("  HELLO  ") == "hello"
    assert n("  HI  ") is None


def test_normalizer_call_raises_on_non_str_final_result():
    n = Normalizer().custom(lambda v: 42)
    with pytest.raises(ValueError):
        n("hello")


def test_normalizer_custom_allows_non_string_intermediate_result():
    normalizer = Normalizer().custom(lambda value: len(value)).custom(lambda length: "x" * length)

    assert normalizer("abc") == "xxx"


# ---------------------------------------------------------------------------
# copy and deepcopy
# ---------------------------------------------------------------------------


def test_normalizer_copy_produces_independent_instance():
    n1 = Normalizer().lower()
    n2 = copy(n1)

    n2.strip()

    assert len(n1.steps) == 1
    assert len(n2.steps) == 2


def test_normalizer_deepcopy_produces_independent_instance():
    n1 = Normalizer().lower()
    n2 = deepcopy(n1)

    n2.strip()

    assert len(n1.steps) == 1
    assert len(n2.steps) == 2


def test_normalizer_deepcopy_preserves_recursive_step_reference():
    class RecursiveStep:
        def __init__(self) -> None:
            self.normalizer: Normalizer | None = None

        def __call__(self, value: object) -> str | None:
            return value if isinstance(value, str) else None

    normalizer = Normalizer()
    step = RecursiveStep()
    step.normalizer = normalizer
    normalizer.custom(step)

    copied = deepcopy(normalizer)
    copied_step = copied.steps[0]

    assert isinstance(copied_step, RecursiveStep)
    assert copied_step.normalizer is copied


def test_normalizer_deepcopy_preserves_shared_identity_in_outer_graph():
    class SharedStep:
        def __call__(self, value: object) -> str | None:
            return value if isinstance(value, str) else None

    step = SharedStep()
    normalizer = Normalizer().custom(step)

    copied_normalizer, copied_step = deepcopy((normalizer, step))

    assert copied_normalizer.steps[0] is copied_step
