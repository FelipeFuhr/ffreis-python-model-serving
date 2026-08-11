"""Build paired sklearn and ONNX smoke models from the same training data."""

from __future__ import annotations

from pathlib import Path
from sys import argv as sys_argv

from joblib import dump as joblib_dump
from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from onnx import TensorProto, helper, numpy_helper
from sklearn.linear_model import LinearRegression

# Matches the fixed 2-row payload used by every consumer of this smoke model
# (onnx-runner-comparison's scenarios/sepal-sum/payload.csv and
# config/scenarios.yaml both hardcode this exact 2-row CSV) — this script
# regenerates the model fresh before each run, so pinning the batch dim to
# what the harness actually sends is safe, not a general-serving constraint.
BATCH_SIZE = 2


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


def _build_onnx_graph(model: LinearRegression) -> bytes:
    """Hand-build a MatMul+Add ONNX graph equivalent to the fitted model.

    skl2onnx's own LinearRegression converter emits the ai.onnx.ml
    LinearRegressor operator for float32 inputs (float64 inputs get
    decomposed to MatMul+Add instead — see skl2onnx's dtype-branched
    converter). tract, the ONNX engine behind the Rust serving consumer of
    this artifact, has no typed-translation support for LinearRegressor
    (fails "Unimplemented(LinearRegressor)" — confirmed empirically against
    tract 0.22.x), so building the equivalent MatMul+Add graph by hand
    keeps this smoke model on float32 (matching every consumer's hardcoded
    float32 tensor handling) while staying entirely in the ai.onnx domain,
    which both onnxruntime and tract fully support.
    """
    coef = model.coef_.astype(np_float32).reshape(-1, 1)
    intercept = np_asarray([model.intercept_], dtype=np_float32)

    coef_init = numpy_helper.from_array(coef, name="coef")
    intercept_init = numpy_helper.from_array(intercept, name="intercept")
    matmul_node = helper.make_node(
        "MatMul", ["x", "coef"], ["matmul_out"], name="matmul"
    )
    add_node = helper.make_node(
        "Add", ["matmul_out", "intercept"], ["variable"], name="add"
    )

    n_features = coef.shape[0]
    x_input = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [BATCH_SIZE, n_features]
    )
    y_output = helper.make_tensor_value_info(
        "variable", TensorProto.FLOAT, [BATCH_SIZE, 1]
    )

    graph = helper.make_graph(
        [matmul_node, add_node],
        "linear_regression_matmul_add",
        [x_input],
        [y_output],
        initializer=[coef_init, intercept_init],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_model.ir_version = 8
    return onnx_model.SerializeToString()


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
    onnx_path.write_bytes(_build_onnx_graph(model))
    print(f"wrote sklearn model to {sklearn_path}")
    print(f"wrote ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
