# ruff: noqa: D101, D102
"""Direct unit tests for ParsedInput and batch_size extraction."""

from __future__ import annotations

import numpy as np
import pytest

from parsed_types import ParsedInput, batch_size

pytestmark = pytest.mark.unit


class TestBatchSize:
    def test_batch_size_from_X_matrix(self) -> None:
        parsed = ParsedInput(X=np.zeros((5, 3), dtype=np.float32))
        assert batch_size(parsed) == 5

    def test_batch_size_single_row(self) -> None:
        parsed = ParsedInput(X=np.zeros((1, 3), dtype=np.float32))
        assert batch_size(parsed) == 1

    def test_batch_size_from_first_tensor(self) -> None:
        parsed = ParsedInput(
            tensors={
                "a": np.zeros((4, 2), dtype=np.float32),
                "b": np.zeros((4, 7), dtype=np.float32),
            }
        )
        assert batch_size(parsed) == 4

    def test_zero_d_tensor_returns_one(self) -> None:
        # A scalar tensor (0-d) means a single example.
        parsed = ParsedInput(tensors={"scalar": np.array(3.14, dtype=np.float32)})
        assert batch_size(parsed) == 1

    def test_one_d_tensor_uses_shape_zero(self) -> None:
        parsed = ParsedInput(tensors={"vec": np.zeros(7, dtype=np.float32)})
        assert batch_size(parsed) == 7

    def test_x_takes_priority_over_tensors(self) -> None:
        # When both are set, X wins.
        parsed = ParsedInput(
            X=np.zeros((10, 3), dtype=np.float32),
            tensors={"a": np.zeros((99, 4), dtype=np.float32)},
        )
        assert batch_size(parsed) == 10

    def test_empty_parsed_raises(self) -> None:
        with pytest.raises(ValueError, match="no features/tensors"):
            batch_size(ParsedInput())

    def test_empty_tensors_dict_raises(self) -> None:
        # tensors set to empty dict is falsy, so raises.
        with pytest.raises(ValueError, match="no features/tensors"):
            batch_size(ParsedInput(tensors={}))

    @pytest.mark.parametrize("batch", [1, 2, 8, 64, 1024])
    def test_parametrized_batch_sizes(self, batch: int) -> None:
        parsed = ParsedInput(X=np.zeros((batch, 5), dtype=np.float32))
        assert batch_size(parsed) == batch
