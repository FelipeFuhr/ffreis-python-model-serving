"""PyTorch adapter implementation."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager, nullcontext
from os import path as os_path
from typing import Protocol, Self, cast

from numpy import asarray as np_asarray
from numpy import generic as np_generic
from numpy import ndarray as np_ndarray

from base_adapter import BaseAdapter
from config import Settings
from parsed_types import ParsedInput
from value_types import JsonDict, PredictionValue


class _TensorDType(Protocol):
    is_floating_point: bool


class _TorchDTypeLike(Protocol):
    """Marker protocol for torch dtype values."""


class _TorchTensor(Protocol):
    dtype: _TensorDType

    def to(self: Self, *, dtype: _TorchDTypeLike) -> _TorchTensor:
        """Cast tensor dtype."""

    def detach(self: Self) -> _TorchTensor:
        """Detach from graph."""

    def cpu(self: Self) -> _TorchTensor:
        """Move tensor to CPU."""

    def numpy(self: Self) -> np_ndarray:
        """Convert tensor to NumPy array."""


class _TorchModel(Protocol):
    def eval(self: Self) -> None:
        """Switch model to eval mode."""

    def __call__(self: Self, tensor: _TorchTensor) -> PredictionValue:
        """Run model forward pass."""


class _TorchJitModule(Protocol):
    def load(self: Self, model_path: str, map_location: str = "cpu") -> _TorchModel:
        """Load TorchScript model."""


class _TorchModule(Protocol):
    Tensor: type
    float32: _TorchDTypeLike
    jit: _TorchJitModule

    def load(self: Self, model_path: str, map_location: str = "cpu") -> _TorchModel:
        """Load eager model."""

    def as_tensor(self: Self, array: np_ndarray) -> _TorchTensor:
        """Create tensor from ndarray."""

    def no_grad(self: Self) -> AbstractContextManager[None]:
        """Return no-grad context manager."""


class PytorchAdapter(BaseAdapter):
    """Inference adapter backed by a serialized PyTorch model."""

    def __init__(self: Self, settings: Settings) -> None:
        """Load torch model from disk."""
        self.settings = settings
        model_filename = settings.model_filename.strip() or "model.pt"
        model_path = os_path.join(settings.model_dir, model_filename)
        if not os_path.exists(model_path):
            raise FileNotFoundError(f"PyTorch model not found at: {model_path}")

        self._torch = cast(_TorchModule, importlib.import_module("torch"))
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self: Self, model_path: str) -> _TorchModel:
        """Load scripted or eager PyTorch model artifact."""
        try:
            return self._torch.jit.load(model_path, map_location="cpu")
        except Exception:
            return self._torch.load(model_path, map_location="cpu")

    def is_ready(self: Self) -> bool:
        """Report whether model is loaded."""
        return self.model is not None

    def _extract_features(self: Self, parsed_input: ParsedInput) -> np_ndarray:
        """Extract model input matrix from parsed input."""
        if not isinstance(parsed_input, ParsedInput):
            raise TypeError("PyTorch adapter expects ParsedInput")
        if parsed_input.X is not None:
            features = parsed_input.X
        elif parsed_input.tensors:
            features = next(iter(parsed_input.tensors.values()))
        else:
            raise ValueError(
                "PyTorch adapter requires ParsedInput.X or ParsedInput.tensors"
            )
        array = np_asarray(features)
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array

    def _to_python(self: Self, value: PredictionValue) -> PredictionValue:
        """Recursively convert framework outputs to JSON-serializable shapes."""
        if hasattr(self._torch, "Tensor") and isinstance(value, self._torch.Tensor):
            tensor_value = cast(_TorchTensor, value)
            return cast(PredictionValue, tensor_value.detach().cpu().numpy().tolist())
        if isinstance(value, np_ndarray):
            return cast(PredictionValue, value.tolist())
        if isinstance(value, list):
            return [self._to_python(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_python(item) for item in value]
        if isinstance(value, dict):
            mapped: JsonDict = {str(k): self._to_python(v) for k, v in value.items()}
            return mapped
        if isinstance(value, np_generic):
            return cast(PredictionValue, value.item())
        return value

    def predict(self: Self, parsed_input: ParsedInput) -> PredictionValue:
        """Run prediction with PyTorch model."""
        features = self._extract_features(parsed_input)
        tensor = self._torch.as_tensor(features)
        if tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype=self._torch.float32)

        no_grad = getattr(self._torch, "no_grad", None)
        context = (
            cast(AbstractContextManager[None], no_grad())
            if callable(no_grad)
            else nullcontext()
        )
        with context:
            output = self.model(tensor)
        return self._to_python(output)
