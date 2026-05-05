"""Test module."""

from os import path as os_path
from pathlib import Path
from typing import Self

from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from base_adapter import BaseAdapter, load_adapter
from config import Settings

pytestmark = pytest_mark.unit


class _RaisesAdapter(BaseAdapter):
    def is_ready(self: Self) -> bool:
        """Run is ready.

        Returns
        -------
        bool
            Return value produced by helper logic in this test module.
        """
        return BaseAdapter.is_ready(self)

    def predict(self: Self, parsed_input: object) -> object:
        """Run predict.

        Parameters
        ----------
        parsed_input : object
            Parsed input payload object passed to adapter methods.

        Returns
        -------
        object
            Return value produced by helper logic in this test module.
        """
        return BaseAdapter.predict(self, parsed_input)


class TestBaseAdapter:
    """Test suite."""

    def test_base_methods_raise_not_implemented(self: Self) -> None:
        """Validate base methods raise not implemented.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        adapter = _RaisesAdapter()
        with pytest_raises(NotImplementedError):
            adapter.is_ready()
        with pytest_raises(NotImplementedError):
            adapter.predict(None)

    def test_load_adapter_uses_onnx_when_model_type_is_onnx(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate load adapter uses onnx when model type is onnx.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """

        class FakeOnnx:
            """Test suite."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import onnx_adapter as onnx_mod

        monkeypatch.setattr(onnx_mod, "OnnxAdapter", FakeOnnx)
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeOnnx)

    def test_load_adapter_uses_onnx_when_default_model_exists(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate load adapter uses onnx when default model exists.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : object
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"x")

        class FakeOnnx:
            """Test suite."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import onnx_adapter as onnx_mod

        monkeypatch.setattr(onnx_mod, "OnnxAdapter", FakeOnnx)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeOnnx)
        assert os_path.exists(model_path)

    def test_load_adapter_rejects_unknown_model_types(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate load adapter rejects unsupported model types.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MODEL_TYPE", "xgboost")
        settings = Settings()
        with pytest_raises(RuntimeError, match="not implemented"):
            load_adapter(settings)

    def test_load_adapter_uses_sklearn_when_model_type_is_sklearn(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate load adapter uses sklearn when model type is sklearn."""

        class FakeSklearn:
            """Lightweight fake sklearn adapter for dispatch checks."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import sklearn_adapter as sklearn_mod

        monkeypatch.setattr(sklearn_mod, "SklearnAdapter", FakeSklearn)
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeSklearn)

    def test_load_adapter_uses_pytorch_when_model_type_is_pytorch(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate load adapter uses pytorch when model type is pytorch."""

        class FakePytorch:
            """Lightweight fake pytorch adapter for dispatch checks."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import pytorch_adapter as pytorch_mod

        monkeypatch.setattr(pytorch_mod, "PytorchAdapter", FakePytorch)
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakePytorch)

    def test_load_adapter_uses_tensorflow_when_model_type_is_tensorflow(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate load adapter uses tensorflow when model type is tensorflow."""

        class FakeTensorflow:
            """Lightweight fake tensorflow adapter for dispatch checks."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import tensorflow_adapter as tensorflow_mod

        monkeypatch.setattr(tensorflow_mod, "TensorflowAdapter", FakeTensorflow)
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeTensorflow)

    def test_load_adapter_uses_sklearn_when_default_model_exists(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate load adapter auto-detects sklearn artifact."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"x")

        class FakeSklearn:
            """Lightweight fake sklearn adapter for dispatch checks."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import sklearn_adapter as sklearn_mod

        monkeypatch.setattr(sklearn_mod, "SklearnAdapter", FakeSklearn)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeSklearn)
        assert os_path.exists(model_path)

    def test_load_adapter_requires_model_type_or_file(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate load adapter requires model type or file.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : object
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        with pytest_raises(
            RuntimeError,
            match="Set MODEL_TYPE=onnx\\|sklearn\\|pytorch\\|tensorflow",
        ):
            load_adapter(settings)

    def test_load_adapter_detects_pytorch_by_filename(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Auto-detect pytorch adapter from MODEL_FILENAME extension."""
        model_path = tmp_path / "custom_model.pt"
        model_path.write_bytes(b"x")

        class FakePytorch:
            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import pytorch_adapter as pytorch_mod

        monkeypatch.setattr(pytorch_mod, "PytorchAdapter", FakePytorch)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("MODEL_FILENAME", "custom_model.pt")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        out = load_adapter(Settings())
        assert isinstance(out, FakePytorch)
        assert os_path.exists(model_path)

    def test_load_adapter_detects_tensorflow_by_filename(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Auto-detect tensorflow adapter from MODEL_FILENAME extension."""
        model_path = tmp_path / "custom_model.keras"
        model_path.write_bytes(b"x")

        class FakeTensorflow:
            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import tensorflow_adapter as tensorflow_mod

        monkeypatch.setattr(tensorflow_mod, "TensorflowAdapter", FakeTensorflow)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("MODEL_FILENAME", "custom_model.keras")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        out = load_adapter(Settings())
        assert isinstance(out, FakeTensorflow)
        assert os_path.exists(model_path)

    def test_load_adapter_detects_tensorflow_saved_model_dir(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Auto-detect tensorflow adapter from SavedModel folder."""
        saved_model_dir = tmp_path / "saved_model"
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        (saved_model_dir / "saved_model.pb").write_bytes(b"x")

        class FakeTensorflow:
            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import tensorflow_adapter as tensorflow_mod

        monkeypatch.setattr(tensorflow_mod, "TensorflowAdapter", FakeTensorflow)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("MODEL_FILENAME", "saved_model")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        out = load_adapter(Settings())
        assert isinstance(out, FakeTensorflow)

    def test_load_adapter_uses_default_pytorch_model(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Auto-detect default ``model.pt`` when MODEL_FILENAME is empty."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"x")

        class FakePytorch:
            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import pytorch_adapter as pytorch_mod

        monkeypatch.setattr(pytorch_mod, "PytorchAdapter", FakePytorch)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("MODEL_FILENAME", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        out = load_adapter(Settings())
        assert isinstance(out, FakePytorch)
        assert os_path.exists(model_path)
