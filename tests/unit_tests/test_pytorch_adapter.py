"""Unit tests for pytorch adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as types_SimpleNamespace
from typing import Any, Self, cast

from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import floating as np_floating
from numpy import issubdtype as np_issubdtype
from numpy import ndarray as np_ndarray
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from config import Settings
from parsed_types import ParsedInput
from pytorch_adapter import PytorchAdapter

pytestmark = pytest_mark.unit


class _NoGrad:
    """Minimal context manager used by fake torch module."""

    def __enter__(self: Self) -> _NoGrad:
        return self

    def __exit__(
        self: Self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> bool:
        return False


class _FakeTensor:
    """Small tensor-like wrapper for adapter conversion tests."""

    def __init__(self: Self, array: np_ndarray) -> None:
        self._array = array
        self.dtype = types_SimpleNamespace(
            is_floating_point=bool(np_issubdtype(array.dtype, np_floating))
        )

    def to(self: Self, dtype: object) -> _FakeTensor:
        _ = dtype
        return _FakeTensor(self._array.astype(np_float32, copy=False))

    def detach(self: Self) -> _FakeTensor:
        return self

    def cpu(self: Self) -> _FakeTensor:
        return self

    def numpy(self: Self) -> np_ndarray:
        return self._array


class _FakeModel:
    """Callable model returning sum over features."""

    def eval(self: Self) -> None:
        return None

    def __call__(self: Self, tensor: _FakeTensor) -> _FakeTensor:
        arr = tensor.numpy().astype(np_float32)
        summed = arr.sum(axis=1, keepdims=True)
        return _FakeTensor(summed)


class _FakeNestedModel:
    """Model returning nested outputs to cover output conversion branches."""

    def eval(self: Self) -> None:
        return None

    def __call__(self: Self, tensor: _FakeTensor) -> object:
        summed = tensor.numpy().sum(axis=1, keepdims=True).astype(np_float32)
        return {
            "predictions": _FakeTensor(summed),
            "details": (_FakeTensor(summed), np_float32(7.0)),
        }


def _fake_torch_module() -> object:
    """Build a fake torch module object."""
    return types_SimpleNamespace(
        Tensor=_FakeTensor,
        float32="float32",
        jit=types_SimpleNamespace(load=lambda _path, map_location=None: _FakeModel()),
        load=lambda _path, map_location=None: _FakeModel(),
        as_tensor=lambda array: _FakeTensor(np_asarray(array)),
        no_grad=lambda: _NoGrad(),
    )


class TestPytorchAdapter:
    """Test suite for pytorch adapter behavior."""

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.X with fake torch module."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _fake_torch_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())

        result = adapter.predict(
            ParsedInput(X=np_asarray([[1, 2, 3], [4, 5, 6]], dtype=np_float32))
        )
        assert result == [[6.0], [15.0]]

    def test_predict_raises_for_empty_payload(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise validation error when neither X nor tensors are provided."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _fake_torch_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())

        with pytest_raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())

    def test_predict_rejects_invalid_parsed_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise type error when parsed input object is invalid."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _fake_torch_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())

        with pytest_raises(TypeError, match="PyTorch adapter expects ParsedInput"):
            adapter.predict(cast(Any, object()))

    def test_load_falls_back_to_torch_load(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Fallback to torch.load when jit.load raises."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        class _FallbackTorch:
            Tensor = _FakeTensor
            float32 = "float32"

            def __init__(self: Self) -> None:
                self.jit = types_SimpleNamespace(load=self._jit_load)

            def _jit_load(self: Self, _path: str, map_location: str = "cpu") -> object:
                _ = map_location
                raise RuntimeError("jit load failed")

            def load(self: Self, _path: str, map_location: str = "cpu") -> object:
                _ = map_location
                return _FakeModel()

            def as_tensor(self: Self, array: np_ndarray) -> _FakeTensor:
                return _FakeTensor(np_asarray(array))

            def no_grad(self: Self) -> _NoGrad:
                return _NoGrad()

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _FallbackTorch()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())
        result = adapter.predict(
            ParsedInput(tensors={"x": np_asarray([1.0, 2.0, 3.0], dtype=np_float32)})
        )
        assert result == [[6.0]]

    def test_converts_nested_outputs_and_supports_missing_no_grad(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cover recursive output conversion and nullcontext fallback."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        class _TorchNoGradMissing:
            Tensor = _FakeTensor
            float32 = "float32"
            jit = types_SimpleNamespace(
                load=lambda _path, map_location=None: _FakeNestedModel()
            )
            load = staticmethod(lambda _path, map_location=None: _FakeNestedModel())
            as_tensor = staticmethod(lambda array: _FakeTensor(np_asarray(array)))

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _TorchNoGradMissing()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())
        result = adapter.predict(
            ParsedInput(X=np_asarray([[1, 2, 3]], dtype=np_float32))
        )
        assert result == {"details": [[[6.0]], 7.0], "predictions": [[6.0]]}
