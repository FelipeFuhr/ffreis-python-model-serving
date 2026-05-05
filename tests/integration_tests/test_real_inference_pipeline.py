"""Tests for real inference pipeline."""

from pathlib import Path

from httpx import ASGITransport as httpx_ASGITransport
from httpx import AsyncClient as httpx_AsyncClient
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import importorskip as pytest_importorskip
from pytest import mark as pytest_mark

from application import create_application
from config import Settings

onnx = pytest_importorskip("onnx")
pytest_importorskip("onnxruntime")

TensorProto = onnx.TensorProto
helper = onnx.helper

pytestmark = pytest_mark.integration


def _write_tiny_sum_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "tiny_sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _write_tiny_two_input_add_model(path: Path) -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["N", 1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, ["N", 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    add = helper.make_node("Add", inputs=["a", "b"], outputs=["y"])
    graph = helper.make_graph([add], "tiny_two_input_add_graph", [a, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


@pytest_mark.asyncio
async def test_real_model_pipeline_integration(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate real model pipeline integration.

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
    _write_tiny_sum_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        ping_response = await client.get("/ping")
        assert ping_response.status_code == 200

        invoke_response = await client.post(
            "/invocations",
            content=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == [[6.0], [15.0]]


@pytest_mark.asyncio
async def test_real_model_pipeline_json_and_metrics_fallback(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate JSON payloads and fallback metrics exposure with real model."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")

    import application as application_module

    monkeypatch.setattr(application_module, "Instrumentator", None)

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        live_response = await client.get("/live")
        ready_response = await client.get("/ready")
        metrics_response = await client.get("/metrics")
        assert live_response.status_code == 200
        assert ready_response.status_code == 200
        assert metrics_response.status_code == 200
        assert "byoc_up 1" in metrics_response.text

        invoke_response = await client.post(
            "/invocations",
            content=b'{"instances": [[1,2,3], [4,5,6]]}',
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == [[6.0], [15.0]]


@pytest_mark.asyncio
async def test_real_model_pipeline_jsonl_multi_input_mapping(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate JSONL multi-input payload parsing with ONNX input mapping."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_two_input_add_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("ONNX_INPUT_MAP_JSON", '{"feature_a":"a","feature_b":"b"}')
    monkeypatch.setenv("ONNX_INPUT_DTYPE_MAP_JSON", '{"a":"float32","b":"float32"}')
    monkeypatch.setenv("ONNX_DYNAMIC_BATCH", "true")

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        invoke_response = await client.post(
            "/invocations",
            content=(
                b'{"feature_a":[1.0],"feature_b":[2.0]}\n'
                b'{"feature_a":[3.0],"feature_b":[4.0]}\n'
            ),
            headers={
                "Content-Type": "application/x-ndjson",
                "Accept": "application/json",
            },
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == [[3.0], [7.0]]


@pytest_mark.asyncio
async def test_real_model_pipeline_csv_header_auto_and_wrapped_json(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate CSV header auto-detection and wrapped JSON output mode."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "auto")
    monkeypatch.setenv("RETURN_PREDICTIONS_ONLY", "false")
    monkeypatch.setenv("JSON_OUTPUT_KEY", "predictions")

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        invoke_response = await client.post(
            "/invocations",
            content=b"f1,f2,f3\n1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == {"predictions": [[6.0], [15.0]]}


@pytest_mark.asyncio
async def test_real_model_pipeline_id_and_feature_column_split(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate tabular identifier/feature column splitting with real model."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("TABULAR_ID_COLUMNS", "0")
    monkeypatch.setenv("TABULAR_FEATURE_COLUMNS", "1:4")
    monkeypatch.setenv("TABULAR_NUM_FEATURES", "3")

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        invoke_response = await client.post(
            "/invocations",
            content=b"99,1,2,3\n100,4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == [[6.0], [15.0]]


@pytest_mark.asyncio
async def test_real_model_pipeline_csv_response_and_body_limit_error(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate CSV output formatting and max-body guardrail with real model."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("MAX_BODY_BYTES", "10")

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        csv_response = await client.post(
            "/invocations",
            content=b"1,2,3\n",
            headers={"Content-Type": "text/csv", "Accept": "text/csv"},
        )
        assert csv_response.status_code == 200
        assert csv_response.text.strip() == "6.0"

        too_large_response = await client.post(
            "/invocations",
            content=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv"},
        )
        assert too_large_response.status_code == 413
        assert too_large_response.json()["error"] == "payload_too_large"


@pytest_mark.asyncio
async def test_readiness_reports_500_when_model_cannot_be_loaded(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Validate readiness failure path when model directory is empty."""
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        ready_response = await client.get("/ready")
        ping_response = await client.get("/ping")
        assert ready_response.status_code == 500
        assert ping_response.status_code == 500
