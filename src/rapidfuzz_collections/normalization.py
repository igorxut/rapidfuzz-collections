from collections.abc import Callable
from copy import copy, deepcopy
from functools import partial
from re import Match as ReMatch
from re import Pattern, RegexFlag, sub
from typing import Any, Self


def _step_capitalize(value: Any) -> str | None:
    return value.capitalize() if isinstance(value, str) else None


def _step_casefold(value: Any) -> str | None:
    return value.casefold() if isinstance(value, str) else None


def _step_isalnum(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isalnum() else None


def _step_isalpha(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isalpha() else None


def _step_isascii(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isascii() else None


def _step_isdecimal(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isdecimal() else None


def _step_isdigit(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isdigit() else None


def _step_isidentifier(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isidentifier() else None


def _step_isinstance_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _step_islower(value: Any) -> str | None:
    return value if isinstance(value, str) and value.islower() else None


def _step_isnumeric(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isnumeric() else None


def _step_isprintable(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isprintable() else None


def _step_isspace(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isspace() else None


def _step_istitle(value: Any) -> str | None:
    return value if isinstance(value, str) and value.istitle() else None


def _step_isupper(value: Any) -> str | None:
    return value if isinstance(value, str) and value.isupper() else None


def _step_lower(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) else None


def _step_lstrip(value: Any) -> str | None:
    return value.lstrip() if isinstance(value, str) else None


def _step_not_empty_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _step_rstrip(value: Any) -> str | None:
    return value.rstrip() if isinstance(value, str) else None


def _step_strip(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) else None


def _step_upper(value: Any) -> str | None:
    return value.upper() if isinstance(value, str) else None


def _step_custom_with_args(
    value: Any,
    *,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    return func(value, *args, **kwargs)


def _step_endswith(value: Any, *, suffix: str | tuple[str, ...], start: int | None, end: int | None) -> str | None:
    return value if isinstance(value, str) and value.endswith(suffix, start, end) else None


def _step_exact_length(value: Any, *, length: int) -> str | None:
    return value if isinstance(value, str) and len(value) == length else None


def _step_lstrip_chars(value: Any, *, chars: str) -> str | None:
    return value.lstrip(chars) if isinstance(value, str) else None


def _step_max_length(value: Any, *, length: int) -> str | None:
    return value if isinstance(value, str) and len(value) <= length else None


def _step_min_length(value: Any, *, length: int) -> str | None:
    return value if isinstance(value, str) and len(value) >= length else None


def _step_re_sub(
    value: Any,
    *,
    pattern: str | Pattern[str],
    repl: str | Callable[[ReMatch[str]], str],
    count: int,
    flags: int | RegexFlag,
) -> str | None:
    return sub(pattern, repl, value, count=count, flags=flags) if isinstance(value, str) else None


def _step_removeprefix(value: Any, *, prefix: str) -> str | None:
    return value.removeprefix(prefix) if isinstance(value, str) else None


def _step_removesuffix(value: Any, *, suffix: str) -> str | None:
    return value.removesuffix(suffix) if isinstance(value, str) else None


def _step_replace(value: Any, *, old: str, new: str, count: int | None) -> str | None:
    if not isinstance(value, str):
        return None
    if count is None:
        return value.replace(old, new)
    return value.replace(old, new, count)


def _step_rstrip_chars(value: Any, *, chars: str) -> str | None:
    return value.rstrip(chars) if isinstance(value, str) else None


def _step_startswith(value: Any, *, prefix: str | tuple[str, ...], start: int | None, end: int | None) -> str | None:
    return value if isinstance(value, str) and value.startswith(prefix, start, end) else None


def _step_strip_chars(value: Any, *, chars: str) -> str | None:
    return value.strip(chars) if isinstance(value, str) else None


def _default_normalizer(value: object) -> str | None:
    """Normalize values with the built-in default fuzzy-search pipeline.

    This function is behaviorally equivalent to
    `Normalizer().isinstance_str().strip().casefold().min_length(3)`, but it
    avoids per-step Python callable dispatch in the default runtime path.

    Args:
        value: Value to normalize.

    Returns:
        Normalized string, or `None` when the value is not searchable by the default rules.
    """

    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if len(normalized) < 3:
        return None
    return normalized


class Normalizer:
    """Utility for converting and normalizing collection values or keys.

    Operations are chained by calling methods in sequence. Each call appends
    a step; the normalizer is applied by calling the instance directly.

    Notes:
        Builder methods mutate this instance. Collections and indexes retain a
        supplied normalizer rather than taking a snapshot of its pipeline.
        Fully configure an instance before passing it to them, and do not
        mutate it afterward. Otherwise, cached choices and later queries can
        be normalized by different pipelines. The caller is responsible for
        keeping the normalizer's behavior stable while it is in use.
        Exceptions raised by custom pipeline steps propagate unchanged.

    Examples:
        normalizer = Normalizer().isinstance_str().strip().casefold().min_length(3)
        normalizer('  Hello  ')  # 'hello'
        normalizer(42)           # None
        normalizer("Hi")         # None  (too short after strip)
    """

    __slots__ = ("_steps",)

    def __call__(self, value: Any) -> str | None:
        """Apply all steps in order and return the normalized string or None.

        Args:
            value: Value to normalize.

        Returns:
            Normalized string, or None if any step rejects the value.

        Raises:
            ValueError: If the final result is neither str nor None.
        """
        result: Any = value
        for step in self._steps:
            result = step(result)
            if result is None:
                return None
        if not isinstance(result, str):
            raise ValueError(f"Need: `None` or `str`. Got: value=`{result!r}` type=`{type(result).__name__}`")
        return result

    def __copy__(self) -> Self:
        """Return a shallow copy."""
        instance = self.__new__(self.__class__)
        instance._steps = copy(self._steps)
        return instance

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Return a deep copy.

        Args:
            memo: Deepcopy memo dictionary used to preserve object identity
                and handle recursive references.

        Returns:
            Independent normalizer with deeply copied pipeline steps.
        """
        instance = self.__new__(self.__class__)
        memo[id(self)] = instance
        instance._steps = deepcopy(self._steps, memo)
        return instance

    def __init__(self) -> None:
        self._steps: list[Callable[[Any], str | None]] = []

    @property
    def steps(self) -> tuple[Callable[[Any], str | None], ...]:
        """Normalizer steps in application order."""
        return tuple(self._steps)

    def _add(self, step: Callable[[Any], str | None]) -> Self:
        """Append a step to this normalizer.

        Side Effects:
            Mutates this normalizer by appending ``step`` to its pipeline.
        """

        self._steps.append(step)
        return self

    def capitalize(self) -> Self:
        """Append a step that capitalizes the string; non-strings yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_capitalize)

    def casefold(self) -> Self:
        """Append a step that case-folds the string; non-strings yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_casefold)

    def custom(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Self:
        """Append a custom callable step.

        Args:
            func: Callable that accepts the current intermediate value as its
                first argument and returns the value for the next pipeline
                step. The final pipeline result must be ``str`` or ``None``.
            *args: Extra positional arguments forwarded to func after the value.
            **kwargs: Extra keyword arguments forwarded to func.

        Raises:
            TypeError: If func is not callable.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if not callable(func):
            raise TypeError(f"Need: `Callable`. Got: `{func!r}` type=`{type(func).__name__}`")
        if args or kwargs:
            return self._add(partial(_step_custom_with_args, func=func, args=args, kwargs=kwargs))
        return self._add(func)

    @classmethod
    def default(cls) -> Normalizer:
        """Return the default normalizer: isinstance_str → strip → casefold → min_length(3)."""
        return cls().isinstance_str().strip().casefold().min_length(3)

    def endswith(
        self,
        suffix: str | tuple[str, ...],
        start: int | None = None,
        end: int | None = None,
    ) -> Self:
        """Append a step that passes strings matching the given suffix; others yield ``None``.

        Args:
            suffix: Suffix string or tuple of suffix strings to test.
            start: Slice start position for the suffix check.
            end: Slice end position for the suffix check.

        Raises:
            TypeError: If suffix contains non-string elements, or if start or
                end are not integers or are booleans.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if isinstance(suffix, tuple):
            for s in suffix:
                if not isinstance(s, str):
                    raise TypeError(f"Need: `str`. Got: `{s!r}` type=`{type(s).__name__}`")
        elif not isinstance(suffix, str):
            raise TypeError(f"Need: `str` or `tuple[str, ...]`. Got: `{suffix!r}` type=`{type(suffix).__name__}`")
        if start is not None and (isinstance(start, bool) or not isinstance(start, int)):
            raise TypeError(f"Need: `int`. Got: `{start!r}` type=`{type(start).__name__}`")
        if end is not None and (isinstance(end, bool) or not isinstance(end, int)):
            raise TypeError(f"Need: `int`. Got: `{end!r}` type=`{type(end).__name__}`")
        return self._add(partial(_step_endswith, suffix=suffix, start=start, end=end))

    def exact_length(self, length: int) -> Self:
        """Append a step that passes strings of exactly ``length`` characters; others yield ``None``.

        Args:
            length: length of the string.

        Raises:
            TypeError: If length is not an integer or is a boolean.
            ValueError: If length is less than 1.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"Need: `int`. Got: `{length!r}` type=`{type(length).__name__}`")
        if length < 1:
            raise ValueError(f"Need: value greater than 0. Got: `{length}`")
        return self._add(partial(_step_exact_length, length=length))

    def isalnum(self) -> Self:
        """Append a step that passes alphanumeric strings; non-strings and others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isalnum)

    def isalpha(self) -> Self:
        """Append a step that passes strings with only alphabetic characters; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isalpha)

    def isascii(self) -> Self:
        """Append a step that passes strings with only ASCII characters; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isascii)

    def isdecimal(self) -> Self:
        """Append a step that passes decimal strings; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isdecimal)

    def isdigit(self) -> Self:
        """Append a step that passes digit-only strings; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isdigit)

    def isidentifier(self) -> Self:
        """Append a step that passes valid Python identifier strings; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isidentifier)

    def isinstance_str(self) -> Self:
        """Append a step that passes string values unchanged; non-strings yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isinstance_str)

    def islower(self) -> Self:
        """Append a step that passes strings with all cased characters lowercase; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_islower)

    def isnumeric(self) -> Self:
        """Append a step that passes numeric strings; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isnumeric)

    def isprintable(self) -> Self:
        """Append a step that passes strings with all printable characters; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isprintable)

    def isspace(self) -> Self:
        """Append a step that passes whitespace-only strings; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isspace)

    def istitle(self) -> Self:
        """Append a step that passes titlecase strings; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_istitle)

    def isupper(self) -> Self:
        """Append a step that passes strings with all cased characters uppercase; others yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_isupper)

    def lower(self) -> Self:
        """Append a step that lowercases the string; non-strings yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_lower)

    def lstrip(self, chars: str | None = None) -> Self:
        """Append a step that strips leading characters from the string; non-strings yield ``None``.

        Args:
            chars: Characters to strip from the left. ``None`` strips Unicode
                whitespace recognized by ``str.lstrip``.

        Raises:
            TypeError: If chars is not a string or None.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if chars is not None and not isinstance(chars, str):
            raise TypeError(f"Need: `None` or `str`. Got: `{chars!r}` type=`{type(chars).__name__}`")
        if chars is None:
            return self._add(_step_lstrip)
        return self._add(partial(_step_lstrip_chars, chars=chars))

    def max_length(self, length: int) -> Self:
        """Append a step that passes strings of at most ``length`` characters; others yield ``None``.

        Args:
            length: length of the string.

        Raises:
            TypeError: If length is not an integer or is a boolean.
            ValueError: If length is less than 1.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"Need: `int`. Got: `{length!r}` type=`{type(length).__name__}`")
        if length < 1:
            raise ValueError(f"Need: value greater than 0. Got: `{length}`")
        return self._add(partial(_step_max_length, length=length))

    def min_length(self, length: int) -> Self:
        """Append a step that passes strings of at least ``length`` characters; others yield ``None``.

        Args:
            length: length of the string.

        Raises:
            TypeError: If length is not an integer or is a boolean.
            ValueError: If length is less than 1.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"Need: `int`. Got: `{length!r}` type=`{type(length).__name__}`")
        if length < 1:
            raise ValueError(f"Need: value greater than 0. Got: `{length}`")
        return self._add(partial(_step_min_length, length=length))

    def not_empty_str(self) -> Self:
        """Append a step that passes non-empty strings; empty strings and non-strings yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_not_empty_str)

    def re_sub(
        self,
        pattern: str | Pattern[str],
        repl: str | Callable[[ReMatch[str]], str],
        count: int = 0,
        flags: int | RegexFlag = 0,
    ) -> Self:
        """Append a step that applies ``re.sub``; non-strings yield ``None``.

        Args:
            pattern: Regular expression pattern to search for.
            repl: Replacement string or callable receiving the match object.
            count: Maximum number of substitutions; 0 means replace all occurrences.
            flags: Regular expression flags passed to ``re.sub``.

        Raises:
            TypeError: If an argument has an incompatible type.
            ValueError: If ``count`` is negative, or flags are provided with a compiled pattern.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        if not isinstance(pattern, str | Pattern):
            raise TypeError(f"pattern must be a string or compiled pattern, got {type(pattern).__name__!r}")
        if isinstance(pattern, Pattern) and not isinstance(pattern.pattern, str):
            raise TypeError("pattern must be a string pattern, not a bytes pattern")
        if not isinstance(repl, str) and not callable(repl):
            raise TypeError(f"repl must be a string or callable, got {type(repl).__name__!r}")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"count must be an integer, got {type(count).__name__!r}")
        if count < 0:
            raise ValueError("count must be greater than or equal to 0")
        if isinstance(flags, bool) or not isinstance(flags, int):
            raise TypeError(f"flags must be an integer or RegexFlag, got {type(flags).__name__!r}")
        if isinstance(pattern, Pattern) and flags:
            raise ValueError("flags cannot be used with a compiled pattern")
        return self._add(partial(_step_re_sub, pattern=pattern, repl=repl, count=count, flags=flags))

    def removeprefix(self, prefix: str) -> Self:
        """Append a step that removes a prefix from the string; non-strings yield ``None``.

        Args:
            prefix: The string prefix to remove from the beginning of each string value.

        Raises:
            TypeError: If prefix is not a string.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if not isinstance(prefix, str):
            raise TypeError(f"Need: `str`. Got: `{prefix!r}` type=`{type(prefix).__name__}`")
        return self._add(partial(_step_removeprefix, prefix=prefix))

    def removesuffix(self, suffix: str) -> Self:
        """Append a step that removes a suffix from the string; non-strings yield ``None``.

        Args:
            suffix: The string suffix to remove from the end of each string value.

        Raises:
            TypeError: If suffix is not a string.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if not isinstance(suffix, str):
            raise TypeError(f"Need: `str`. Got: `{suffix!r}` type=`{type(suffix).__name__}`")
        return self._add(partial(_step_removesuffix, suffix=suffix))

    def replace(self, old: str, new: str, count: int | None = None) -> Self:
        """Append a step that replaces substrings; non-strings yield ``None``.

        Args:
            old: The substring to replace.
            new: The replacement string.
            count: The maximum number of replacements to perform. If ``None``, replaces all occurrences.

        Raises:
            TypeError: If old or new are not strings, or count is not an
                integer or is a boolean.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if not isinstance(old, str):
            raise TypeError(f"replace() argument 1 must be `str`. Got: `{type(old).__name__}`")
        if not isinstance(new, str):
            raise TypeError(f"replace() argument 2 must be `str`. Got: `{type(new).__name__}`")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
            raise TypeError(f"replace() argument 3 must be `int`. Got: `{type(count).__name__}`")
        return self._add(partial(_step_replace, old=old, new=new, count=count))

    def rstrip(self, chars: str | None = None) -> Self:
        """Append a step that strips trailing characters from the string; non-strings yield ``None``.

        Args:
            chars: Characters to strip from the right. ``None`` strips Unicode
                whitespace recognized by ``str.rstrip``.

        Raises:
            TypeError: If chars is not a string or None.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if chars is not None and not isinstance(chars, str):
            raise TypeError(f"Need: `None` or `str`. Got: `{chars!r}` type=`{type(chars).__name__}`")
        if chars is None:
            return self._add(_step_rstrip)
        return self._add(partial(_step_rstrip_chars, chars=chars))

    def startswith(
        self,
        prefix: str | tuple[str, ...],
        start: int | None = None,
        end: int | None = None,
    ) -> Self:
        """Append a step that passes strings matching the given prefix; others yield ``None``.

        Args:
            prefix: Prefix string or tuple of prefix strings to test.
            start: Slice start position for the prefix check.
            end: Slice end position for the prefix check.

        Raises:
            TypeError: If prefix contains non-string elements, or if start or
                end are not integers or are booleans.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if isinstance(prefix, tuple):
            for p in prefix:
                if not isinstance(p, str):
                    raise TypeError(f"Need: `str`. Got: `{p!r}` type=`{type(p).__name__}`")
        elif not isinstance(prefix, str):
            raise TypeError(f"Need: `str` or `tuple[str, ...]`. Got: `{prefix!r}` type=`{type(prefix).__name__}`")
        if start is not None and (isinstance(start, bool) or not isinstance(start, int)):
            raise TypeError(f"Need: `int`. Got: `{start!r}` type=`{type(start).__name__}`")
        if end is not None and (isinstance(end, bool) or not isinstance(end, int)):
            raise TypeError(f"Need: `int`. Got: `{end!r}` type=`{type(end).__name__}`")
        return self._add(partial(_step_startswith, prefix=prefix, start=start, end=end))

    def strip(self, chars: str | None = None) -> Self:
        """Append a step that strips leading and trailing characters; non-strings yield ``None``.

        Args:
            chars: Characters to strip from both ends. ``None`` strips Unicode
                whitespace recognized by ``str.strip``.

        Raises:
            TypeError: If chars is not a string or None.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """
        if chars is not None and not isinstance(chars, str):
            raise TypeError(f"Need: `None` or `str`. Got: `{chars!r}` type=`{type(chars).__name__}`")
        if chars is None:
            return self._add(_step_strip)
        return self._add(partial(_step_strip_chars, chars=chars))

    def upper(self) -> Self:
        """Append a step that uppercases the string; non-strings yield ``None``.

        Side Effects:
            Mutates this normalizer by appending a pipeline step.
        """

        return self._add(_step_upper)
