"""Unit tests for legacy onnx_model_serving helpers."""

from __future__ import annotations

from pytest import mark as pytest_mark

from onnx_model_serving import __all__ as onnx_model_serving___all__
from onnx_model_serving import (
    add,
    clamp,
    first_non_empty,
    greet,
    is_even,
    repeat_word,
    sum_list,
    toggle,
)

pytestmark = pytest_mark.unit


def test_add_and_greet() -> None:
    """Validate add and greet helpers."""
    assert add(2, 3) == 5
    assert greet() == "Hello, world!"


def test_is_even_and_toggle() -> None:
    """Validate parity and boolean toggling helpers."""
    assert is_even(4) is True
    assert is_even(5) is False
    assert toggle(True) is False
    assert toggle(False) is True


def test_clamp_boundaries() -> None:
    """Validate clamp lower/upper/in-range behavior."""
    assert clamp(-10, 0, 5) == 0
    assert clamp(3, 0, 5) == 3
    assert clamp(10, 0, 5) == 5


def test_repeat_sum_and_first_non_empty() -> None:
    """Validate repeat/sum/first-non-empty helper behavior."""
    assert repeat_word("x", 0) == ""
    assert repeat_word("x", 3) == "x x x"
    assert sum_list([1, 2, 3]) == 6
    assert sum_list([]) == 0
    assert first_non_empty(["", "a", "b"]) == "a"
    assert first_non_empty(["", ""]) is None


def test_package_exports_expected_symbols() -> None:
    """Validate package __all__ exports helper function names."""
    exported = set(onnx_model_serving___all__)
    assert exported == {
        "add",
        "clamp",
        "first_non_empty",
        "greet",
        "is_even",
        "repeat_word",
        "sum_list",
        "toggle",
    }
