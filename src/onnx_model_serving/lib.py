"""Library functions for legacy scaffold tests."""


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def greet() -> str:
    """Return a default greeting message."""
    return "Hello, world!"


def is_even(n: int) -> bool:
    """Return whether an integer is even."""
    return n % 2 == 0


def clamp(n: int, minimum: int, maximum: int) -> int:
    """Clamp an integer between a minimum and maximum value."""
    if n < minimum:
        return minimum
    if n > maximum:
        return maximum
    return n


def repeat_word(word: str, times: int) -> str:
    """Repeat a word ``times`` separated by spaces."""
    return " ".join([word] * times)


def sum_list(nums: list[int]) -> int:
    """Return the sum of an integer list."""
    return sum(nums)


def first_non_empty(values: list[str]) -> str | None:
    """Return the first non-empty string in a list."""
    for value in values:
        if value:
            return value
    return None


def toggle(flag: bool) -> bool:
    """Invert a boolean flag."""
    return not flag
