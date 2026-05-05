"""Base adapter contracts and adapter factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from os import path as os_path

from config import Settings
from parsed_types import ParsedInput
from value_types import PredictionValue


class BaseAdapter(ABC):
    """Abstract contract for inference adapters."""

    @abstractmethod
    def is_ready(self: BaseAdapter) -> bool:
        """Return whether the adapter is ready to serve predictions.

        Returns
        -------
        bool
            ``True`` when the adapter is fully initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self: BaseAdapter, parsed_input: ParsedInput) -> PredictionValue:
        """Run inference for a parsed input payload.

        Parameters
        ----------
        parsed_input : ParsedInput
            Parsed model input.

        Returns
        -------
        PredictionValue
            Model prediction output.
        """
        raise NotImplementedError


_SUPPORTED_MODEL_TYPES = frozenset({"onnx", "sklearn", "pytorch", "tensorflow"})


def _missing_extra_runtime_error(extra: str, module_name: str) -> RuntimeError:
    """Build actionable error for missing optional runtime dependencies."""
    return RuntimeError(
        f"Missing optional dependency '{module_name}' for MODEL_TYPE={extra}. "
        f"Install with: uv sync --extra {extra}"
    )


def _build_onnx_adapter(settings: Settings) -> BaseAdapter:
    from onnx_adapter import OnnxAdapter

    try:
        return OnnxAdapter(settings)
    except ModuleNotFoundError as exc:
        if exc.name == "onnxruntime":
            raise _missing_extra_runtime_error("onnx", "onnxruntime") from exc
        raise


def _build_sklearn_adapter(settings: Settings) -> BaseAdapter:
    from sklearn_adapter import SklearnAdapter

    try:
        return SklearnAdapter(settings)
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            raise _missing_extra_runtime_error("sklearn", "scikit-learn") from exc
        raise


def _build_pytorch_adapter(settings: Settings) -> BaseAdapter:
    from pytorch_adapter import PytorchAdapter

    try:
        return PytorchAdapter(settings)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise _missing_extra_runtime_error("torch", "torch") from exc
        raise


def _build_tensorflow_adapter(settings: Settings) -> BaseAdapter:
    from tensorflow_adapter import TensorflowAdapter

    try:
        return TensorflowAdapter(settings)
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise _missing_extra_runtime_error("tensorflow", "tensorflow") from exc
        raise


_ADAPTER_BUILDERS: dict[str, Callable[[Settings], BaseAdapter]] = {
    "onnx": _build_onnx_adapter,
    "sklearn": _build_sklearn_adapter,
    "pytorch": _build_pytorch_adapter,
    "tensorflow": _build_tensorflow_adapter,
}


def _build_adapter_for(model_type: str, settings: Settings) -> BaseAdapter:
    return _ADAPTER_BUILDERS[model_type](settings)


def _validated_model_type(settings: Settings) -> str | None:
    model_type = settings.model_type
    if not model_type:
        return None
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise RuntimeError(
            f"MODEL_TYPE={model_type} is not implemented in this package"
        )
    return model_type


def _infer_model_type_from_filename(settings: Settings) -> str | None:
    model_filename = settings.model_filename.strip()
    if not model_filename:
        return None

    model_path = os_path.join(settings.model_dir, model_filename)
    lowered = model_filename.lower()

    if lowered.endswith(".onnx") and os_path.exists(model_path):
        return "onnx"
    if lowered.endswith((".joblib", ".pkl")) and os_path.exists(model_path):
        return "sklearn"
    if lowered.endswith((".pt", ".pth", ".jit", ".torchscript")) and os_path.exists(
        model_path
    ):
        return "pytorch"
    if lowered.endswith((".keras", ".h5", ".hdf5")) and os_path.exists(model_path):
        return "tensorflow"
    if os_path.isdir(model_path) and os_path.exists(
        os_path.join(model_path, "saved_model.pb")
    ):
        return "tensorflow"
    return None


def _infer_model_type_from_defaults(settings: Settings) -> str | None:
    onnx_default = os_path.join(settings.model_dir, "model.onnx")
    sklearn_default = os_path.join(settings.model_dir, "model.joblib")
    pickle_default = os_path.join(settings.model_dir, "model.pkl")
    pytorch_default = os_path.join(settings.model_dir, "model.pt")
    tensorflow_default = os_path.join(settings.model_dir, "model.keras")
    tensorflow_saved_model_default = os_path.join(settings.model_dir, "saved_model")

    if os_path.exists(onnx_default):
        return "onnx"
    if os_path.exists(sklearn_default) or os_path.exists(pickle_default):
        return "sklearn"
    if os_path.exists(pytorch_default):
        return "pytorch"
    if os_path.exists(tensorflow_default) or os_path.exists(
        tensorflow_saved_model_default
    ):
        return "tensorflow"
    return None


def load_adapter(settings: Settings) -> BaseAdapter:
    """Instantiate the appropriate adapter for current settings.

    Parameters
    ----------
    settings : Settings
        Runtime configuration.

    Returns
    -------
    BaseAdapter
        Instantiated inference adapter.
    """
    explicit_model_type = _validated_model_type(settings)
    if explicit_model_type is not None:
        return _build_adapter_for(explicit_model_type, settings)

    inferred_model_type = _infer_model_type_from_filename(settings)
    if inferred_model_type is None:
        inferred_model_type = _infer_model_type_from_defaults(settings)
    if inferred_model_type is not None:
        return _build_adapter_for(inferred_model_type, settings)

    raise RuntimeError(
        "Set MODEL_TYPE=onnx|sklearn|pytorch|tensorflow or place a known model "
        "artifact under SM_MODEL_DIR"
    )
