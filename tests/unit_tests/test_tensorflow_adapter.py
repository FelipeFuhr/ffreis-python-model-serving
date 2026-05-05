"""Unit tests for tensorflow adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as types_SimpleNamespace
from typing import Any, Self, cast

from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import ndarray as np_ndarray
from numpy import sum as np_sum
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from config import Settings
from parsed_types import ParsedInput
from tensorflow_adapter import TensorflowAdapter

pytestmark = pytest_mark.unit


class _FakeTensorflowModel:
    """Fake keras model for adapter tests."""

    def __call__(
        self: Self, features: np_ndarray, training: bool = False
    ) -> np_ndarray:
        _ = training
        return np_sum(features, axis=1, keepdims=True).astype(np_float32)


class _NumpyScalarWrapper:
    """Numpy-compatible scalar wrapper for conversion branch coverage."""

    def __init__(self: Self, value: float) -> None:
        self._value = value

    def numpy(self: Self) -> float:
        return self._value


def _fake_tensorflow_module() -> object:
    """Build fake tensorflow module with keras load_model."""
    keras_models = types_SimpleNamespace(
        load_model=lambda _path: _FakeTensorflowModel()
    )
    keras = types_SimpleNamespace(models=keras_models)
    return types_SimpleNamespace(keras=keras)


class TestTensorflowAdapter:
    """Test suite for tensorflow adapter behavior."""

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.X with fake tensorflow module."""
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "tensorflow":
                return _fake_tensorflow_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        adapter = TensorflowAdapter(Settings())

        result = adapter.predict(
            ParsedInput(X=np_asarray([[1, 2, 3], [4, 5, 6]], dtype=np_float32))
        )
        assert result == [[6.0], [15.0]]

    def test_predict_raises_for_empty_payload(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise validation error when neither X nor tensors are provided."""
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "tensorflow":
                return _fake_tensorflow_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        adapter = TensorflowAdapter(Settings())

        with pytest_raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())

    def test_predict_rejects_invalid_parsed_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise type error when parsed input object is invalid."""
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "tensorflow":
                return _fake_tensorflow_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        adapter = TensorflowAdapter(Settings())

        with pytest_raises(TypeError, match="TensorFlow adapter expects ParsedInput"):
            adapter.predict(cast(Any, object()))

    def test_predict_uses_predict_branch_and_nested_output_conversion(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Use non-callable model and cover nested conversion branches."""
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"fake")

        class _PredictOnlyModel:
            def predict(self: Self, features: np_ndarray, verbose: int = 0) -> object:
                _ = verbose
                summed = np_sum(features, axis=1, keepdims=True).astype(np_float32)
                return {
                    "predictions": summed,
                    "details": (np_float32(3.0), _NumpyScalarWrapper(9.0)),
                }

        tensorflow_module = types_SimpleNamespace(
            keras=types_SimpleNamespace(
                models=types_SimpleNamespace(
                    load_model=lambda _path: _PredictOnlyModel()
                )
            )
        )

        def _fake_import_module(name: str) -> object:
            if name == "tensorflow":
                return tensorflow_module
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        adapter = TensorflowAdapter(Settings())
        adapter.model = cast(Any, _PredictOnlyModel())
        result = adapter.predict(
            ParsedInput(tensors={"x": np_asarray([1.0, 2.0, 3.0], dtype=np_float32)})
        )
        assert result == {"details": [3.0, 9.0], "predictions": [[6.0]]}

    def test_raises_when_model_file_is_missing(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise file-not-found when configured artifact is absent."""
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        monkeypatch.setenv("MODEL_FILENAME", "missing.keras")
        with pytest_raises(FileNotFoundError, match="TensorFlow model not found"):
            TensorflowAdapter(Settings())
