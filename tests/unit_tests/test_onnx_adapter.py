"""Tests for onnx adapter."""

from pathlib import Path
from typing import Self

from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import importorskip as pytest_importorskip
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from config import Settings
from onnx_adapter import OnnxAdapter
from parsed_types import ParsedInput

onnx = pytest_importorskip("onnx")
pytest_importorskip("onnxruntime")

TensorProto = onnx.TensorProto
helper = onnx.helper

pytestmark = pytest_mark.unit


def _write_sum_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _write_two_output_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, ["N", 2])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, ["N", 2])
    add = helper.make_node("Add", inputs=["x", "x"], outputs=["y1"])
    ident = helper.make_node("Identity", inputs=["x"], outputs=["y2"])
    graph = helper.make_graph([add, ident], "two_out_graph", [x], [y1, y2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


class TestOnnxAdapter:
    """Test suite for TestOnnxAdapter."""

    def test_raises_when_model_is_missing(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate raises when model is missing.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : Path
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        with pytest_raises(FileNotFoundError, match="ONNX model not found"):
            OnnxAdapter(Settings())

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate predicts from tabular input.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : Path
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        _write_sum_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(X=np_asarray([[1, 2, 3], [4, 5, 6]], dtype=np_float32))
        preds = adapter.predict(inp)
        assert preds == [[6.0], [15.0]]

    def test_predicts_from_tensor_inputs_with_output_map(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate predicts from tensor inputs with output map.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : Path
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        _write_two_output_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        monkeypatch.setenv("ONNX_OUTPUT_MAP_JSON", '{"double":"y1","raw":"y2"}')
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(
            tensors={"x": np_asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np_float32)}
        )
        preds = adapter.predict(inp)
        assert preds == {
            "double": [[2.0, 4.0], [6.0, 8.0]],
            "raw": [[1.0, 2.0], [3.0, 4.0]],
        }

    def test_selects_named_output(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify selects named output.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : Path
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        _write_two_output_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        monkeypatch.setenv("ONNX_OUTPUT_NAME", "y2")
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(X=np_asarray([[2.0, 4.0]], dtype=np_float32))
        preds = adapter.predict(inp)
        assert preds == [[2.0, 4.0]]

    def test_selects_output_by_index(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate selects output by index.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : Path
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        _write_two_output_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        monkeypatch.setenv("ONNX_OUTPUT_INDEX", "1")
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(X=np_asarray([[2.0, 4.0]], dtype=np_float32))
        preds = adapter.predict(inp)
        assert preds == [[2.0, 4.0]]

    def test_rejects_empty_input(
        self: Self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify rejects empty input.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : Path
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        _write_sum_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        adapter = OnnxAdapter(Settings())
        with pytest_raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())
