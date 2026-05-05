"""Unit tests for sklearn adapter."""

from __future__ import annotations

from pathlib import Path
from pickle import dump as pickle_dump
from types import SimpleNamespace as types_SimpleNamespace
from typing import Any, Self, cast

from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray
from numpy import sum as np_sum
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from config import Settings
from parsed_types import ParsedInput
from sklearn_adapter import SklearnAdapter

pytestmark = pytest_mark.unit


class _FakeModel:
    """Simple fake estimator exposing ``predict``."""

    def predict(self: Self, features: np_ndarray) -> np_ndarray:
        return np_sum(features, axis=1).astype(np_float64)


class TestSklearnAdapter:
    """Test suite for sklearn adapter behavior."""

    def test_loads_joblib_model(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Load model through joblib when dependency is available."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types_SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")

        adapter = SklearnAdapter(Settings())
        assert adapter.is_ready() is True

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.X with deterministic fake model."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types_SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())

        result = adapter.predict(
            ParsedInput(X=np_asarray([[1, 2, 3], [4, 5, 6]], dtype=np_float32))
        )
        assert result == [6.0, 15.0]

    def test_predict_raises_for_empty_payload(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise validation error when neither X nor tensors are provided."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types_SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())

        with pytest_raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())

    def test_predict_rejects_invalid_parsed_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise type error when parsed input object is invalid."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types_SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())

        with pytest_raises(TypeError, match="Sklearn adapter expects ParsedInput"):
            adapter.predict(cast(Any, object()))

    def test_loads_pickle_when_joblib_is_missing(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Load fallback pickle artifact when joblib is unavailable."""
        model_path = tmp_path / "model.joblib"
        with model_path.open("wb") as handle:
            pickle_dump(_FakeModel(), handle)

        def _fake_import_module(_name: str) -> object:
            raise ModuleNotFoundError("joblib")

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())
        assert adapter.is_ready() is True

    def test_predicts_from_tensor_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.tensors and cover 1D reshape path."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types_SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())
        result = adapter.predict(
            ParsedInput(tensors={"x": np_asarray([1.0, 2.0, 3.0], dtype=np_float32)})
        )
        assert result == [6.0]

    def test_raises_when_model_file_is_missing(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise file-not-found when configured artifact is absent."""
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        monkeypatch.setenv("MODEL_FILENAME", "missing.joblib")
        with pytest_raises(FileNotFoundError, match="scikit-learn model not found"):
            SklearnAdapter(Settings())
