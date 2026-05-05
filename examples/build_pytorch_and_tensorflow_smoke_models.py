"""Build paired PyTorch and TensorFlow smoke models with matching linear behavior."""

from __future__ import annotations

from pathlib import Path
from sys import argv as sys_argv
from typing import Self

from numpy import float32 as np_float32
from numpy import ones as np_ones

try:
    from tensorflow import keras as tf_keras
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "TensorFlow is required for this example. Install with: "
        "uv sync --extra tensorflow"
    ) from exc

try:
    from torch import Tensor as torch_Tensor
    from torch import jit as torch_jit
    from torch import nn as torch_nn
    from torch import no_grad as torch_no_grad
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "PyTorch is required for this example. Install with: uv sync --extra torch"
    ) from exc


def _build_pytorch_model(path: Path) -> None:
    """Write TorchScript model computing y = x @ 1."""

    class SumModel(torch_nn.Module):
        def __init__(self: Self) -> None:
            super().__init__()
            self.linear = torch_nn.Linear(3, 1, bias=False)
            with torch_no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self: Self, x: torch_Tensor) -> torch_Tensor:
            return self.linear(x.float())

    model = SumModel().eval()
    scripted = torch_jit.script(model)
    scripted.save(str(path))


def _build_tensorflow_model(path: Path) -> None:
    """Write Keras model computing y = x @ 1."""
    model = tf_keras.Sequential(
        [
            tf_keras.Input(shape=(3,)),
            tf_keras.layers.Dense(1, use_bias=False),
        ]
    )
    model.layers[0].set_weights([np_ones((3, 1), dtype=np_float32)])
    model.save(path)


def main() -> None:
    """Write PyTorch and TensorFlow smoke model artifacts."""
    output_dir = (
        Path(sys_argv[1]) if len(sys_argv) > 1 else Path("/tmp/onnx-runner-comparison")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pytorch_path = output_dir / "model.pt"
    tensorflow_path = output_dir / "model.keras"

    _build_pytorch_model(pytorch_path)
    _build_tensorflow_model(tensorflow_path)
    print(f"wrote PyTorch model to {pytorch_path}")
    print(f"wrote TensorFlow model to {tensorflow_path}")


if __name__ == "__main__":
    main()
