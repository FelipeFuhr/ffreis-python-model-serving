"""Build a minimal ONNX model used by compose smoke tests."""

from __future__ import annotations

from pathlib import Path
from sys import argv as sys_argv

from onnx import TensorProto, helper
from onnx import save as onnx_save


def write_tiny_sum_model(path: Path) -> None:
    """Write a minimal ONNX model that sums 3 features into one output."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "tiny_sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx_save(model, str(path))


def main() -> None:
    """Write the smoke ONNX model to provided path or default location."""
    output = Path(sys_argv[1]) if len(sys_argv) > 1 else Path("/models/model.onnx")
    write_tiny_sum_model(output)
    print(f"wrote smoke ONNX model to {output}")


if __name__ == "__main__":
    main()
