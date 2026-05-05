"""Scikit-learn adapter implementation."""

from __future__ import annotations

import importlib
from os import path as os_path
from pickle import load as pickle_load
from typing import Protocol, Self, cast

from numpy import asarray as np_asarray
from numpy import ndarray as np_ndarray

from base_adapter import BaseAdapter
from config import Settings
from parsed_types import ParsedInput
from value_types import PredictionValue


class _SklearnModel(Protocol):
    def predict(self: Self, features: np_ndarray) -> PredictionValue:
        """Predict output for an input matrix."""


class _JoblibModule(Protocol):
    def load(self: Self, path: str) -> _SklearnModel:
        """Load model from file."""


class SklearnAdapter(BaseAdapter):
    """Inference adapter backed by a serialized scikit-learn model."""

    def __init__(self: Self, settings: Settings) -> None:
        """Load scikit-learn model from disk."""
        self.settings = settings
        model_filename = settings.model_filename.strip() or "model.joblib"
        model_path = os_path.join(settings.model_dir, model_filename)
        if not os_path.exists(model_path):
            raise FileNotFoundError(f"scikit-learn model not found at: {model_path}")
        self.model = self._load_model(model_path)

    def _load_model(self: Self, model_path: str) -> _SklearnModel:
        """Load serialized model using joblib when available."""
        try:
            joblib_module = cast(_JoblibModule, importlib.import_module("joblib"))
            return joblib_module.load(model_path)
        except ModuleNotFoundError:
            with open(model_path, "rb") as handle:
                loaded_model = pickle_load(handle)
            return cast(_SklearnModel, loaded_model)

    def is_ready(self: Self) -> bool:
        """Report whether model is loaded."""
        return self.model is not None

    def predict(self: Self, parsed_input: ParsedInput) -> PredictionValue:
        """Run prediction with sklearn model."""
        if not isinstance(parsed_input, ParsedInput):
            raise TypeError("Sklearn adapter expects ParsedInput")
        if parsed_input.X is not None:
            features = parsed_input.X
        elif parsed_input.tensors:
            first_tensor = next(iter(parsed_input.tensors.values()))
            features = first_tensor
        else:
            raise ValueError(
                "Sklearn adapter requires ParsedInput.X or ParsedInput.tensors"
            )

        array = np_asarray(features)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        predictions = self.model.predict(array)
        prediction_array = np_asarray(predictions)
        return cast(PredictionValue, prediction_array.tolist())
