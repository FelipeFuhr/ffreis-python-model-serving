"""Build paired sklearn and ONNX smoke models from the same training data."""

from __future__ import annotations

from pathlib import Path
from sys import argv as sys_argv

from joblib import dump as joblib_dump
from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression


def _train_model() -> LinearRegression:
    x_train = np_asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np_float32,
    )
    y_train = np_asarray([0, 1, 1, 2, 1, 2, 2, 3], dtype=np_float32)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def main() -> None:
    """Write paired sklearn and ONNX artifacts to target directory."""
    output_dir = (
        Path(sys_argv[1]) if len(sys_argv) > 1 else Path("/tmp/onnx-runner-comparison")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    sklearn_path = output_dir / "model.joblib"

    model = _train_model()
    joblib_dump(model, sklearn_path)

    onnx_model = convert_sklearn(
        model,
        initial_types=[("x", FloatTensorType([None, 3]))],
        target_opset=14,
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())
    print(f"wrote sklearn model to {sklearn_path}")
    print(f"wrote ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
