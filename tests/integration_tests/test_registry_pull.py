"""Integration test: registry-backed startup (B4).

Creates an in-memory SQLite registry with a registered production-stage ONNX
model, starts the serving process with the registry env vars set, and asserts
the HTTP ``/predict`` endpoint returns correct predictions.

The test covers:
- ``MODEL_REGISTRY_BACKEND=sqlite`` + ``MODEL_REGISTRY_URI=<db path>``
- Automatic ``onnx_uri`` artifact materialisation into a temp dir.
- The ``/invocations`` predict path end-to-end.
- ``pull_model_from_registry`` short-circuit when env vars are absent.
- Error paths: missing backend, missing uri, missing model name, no production
  version, missing onnx_uri, unsupported scheme.
"""

from __future__ import annotations

# pylint: disable=wrong-import-position
# Imports placed after pytest_importorskip() calls intentionally — this is the
# standard pytest skip-when-dep-absent pattern; moving imports above the skip
# guard would import the package unconditionally and raise ImportError.
from pathlib import Path

import pytest
from httpx import ASGITransport as httpx_ASGITransport
from httpx import AsyncClient as httpx_AsyncClient
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import importorskip as pytest_importorskip

onnx = pytest_importorskip("onnx")
pytest_importorskip("onnxruntime")
pytest_importorskip("ml_registry")

from ml_registry.adapters.sqlite_registry import SqliteModelRegistry  # noqa: E402
from ml_registry.core.entities import ModelVersion, Stage  # noqa: E402

from application import create_application  # noqa: E402
from config import Settings  # noqa: E402
from registry_pull import (  # noqa: E402
    _materialize_artifact,
    _open_registry,
    _resolve_onnx_uri,
    pull_model_from_registry,
)

TensorProto = onnx.TensorProto
helper = onnx.helper

pytestmark = pytest.mark.integration


# ── helpers ──────────────────────────────────────────────────────────────────


def _write_tiny_sum_model(path: Path) -> None:
    """Write a 3-input sum ONNX model (W=[1,1,1]) to ``path``."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "tiny_sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _register_production_model(
    registry: SqliteModelRegistry,
    name: str,
    onnx_uri: str,
) -> ModelVersion:
    """Register and promote an ONNX model to production in ``registry``."""
    version = registry.register(
        ModelVersion(
            id=f"{name}-v1",
            name=name,
            model_id=f"{name}-model",
            version="1",
            onnx_uri=onnx_uri,
        )
    )
    return registry.promote(name=name, version=version.version, stage=Stage.PRODUCTION)


# ── happy-path integration test ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_pull_serves_correct_predictions(
    monkeypatch: pytest_MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Serving started via registry env vars returns correct predictions.

    Sets up a SQLite registry with a production-stage ONNX model, configures
    the serving process via env vars, and asserts the ``/invocations`` endpoint
    returns the expected sum values.
    """
    # 1. Write ONNX model artifact to disk. Named "model.onnx" so the ONNX
    # adapter's default filename detection finds it in the materialised temp dir.
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    onnx_path = model_dir / "model.onnx"
    _write_tiny_sum_model(onnx_path)

    # 2. Create a SQLite registry and register the model at production stage.
    db_path = str(tmp_path / "registry.db")
    registry = SqliteModelRegistry(path=db_path)
    _register_production_model(registry, name="tiny-sum", onnx_uri=str(onnx_path))
    registry.close()

    # 3. Configure env vars for registry-backed startup.
    monkeypatch.setenv("MODEL_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setenv("MODEL_REGISTRY_URI", db_path)
    monkeypatch.setenv("MODEL_NAME", "tiny-sum")
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    # SM_MODEL_DIR intentionally points at an empty dir to prove the registry
    # path is used instead.
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()

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


@pytest.mark.asyncio
async def test_registry_pull_with_file_uri_scheme(
    monkeypatch: pytest_MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry pull works when ``onnx_uri`` uses a ``file://`` URI scheme."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    onnx_path = model_dir / "model.onnx"
    _write_tiny_sum_model(onnx_path)

    db_path = str(tmp_path / "registry.db")
    registry = SqliteModelRegistry(path=db_path)
    _register_production_model(
        registry,
        name="tiny-sum-file",
        onnx_uri=f"file://{onnx_path}",
    )
    registry.close()

    monkeypatch.setenv("MODEL_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setenv("MODEL_REGISTRY_URI", db_path)
    monkeypatch.setenv("MODEL_NAME", "tiny-sum-file")
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir(exist_ok=True)

    application = create_application(Settings())
    transport = httpx_ASGITransport(app=application)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        invoke_response = await client.post(
            "/invocations",
            content=b"1,2,3\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == [[6.0]]


# ── no-op path: env vars absent ───────────────────────────────────────────────


def test_pull_model_from_registry_returns_none_when_not_configured() -> None:
    """pull_model_from_registry returns None when registry env vars are absent."""
    settings = Settings.model_construct(
        model_registry_backend="",
        model_registry_uri="",
        model_name="",
        model_dir="/opt/ml/model",
        model_type="",
        model_filename="",
        port=8080,
        log_level="INFO",
        service_name="test",
        service_version="dev",
        deployment_env="local",
        input_mode="tabular",
        output_mode="predictions",
        default_content_type="application/json",
        default_accept="application/json",
        tabular_dtype="float32",
        csv_delimiter=",",
        csv_has_header="auto",
        csv_skip_blank_lines=True,
        json_key_instances="instances",
        jsonl_features_key="features",
        tabular_id_columns="",
        tabular_feature_columns="",
        predictions_only=True,
        json_output_key="predictions",
        max_body_bytes=6 * 1024 * 1024,
        max_records=5000,
        max_inflight=16,
        acquire_timeout_s=0.25,
        gunicorn_workers=1,
        gunicorn_threads=4,
        gunicorn_timeout=60,
        gunicorn_graceful_timeout=30,
        gunicorn_keepalive=5,
        prometheus_enabled=False,
        prometheus_path="/metrics",
        swagger_enabled=False,
        otel_enabled=False,
        otel_endpoint="",
        otel_headers="",
        otel_timeout_s=10.0,
        onnx_providers="CPUExecutionProvider",
        onnx_intra_op_threads=0,
        onnx_inter_op_threads=0,
        onnx_graph_opt_level="all",
        onnx_input_name="",
        onnx_output_name="",
        onnx_output_index=0,
        tabular_num_features=0,
        onnx_input_map_json="",
        onnx_output_map_json="",
        onnx_input_dtype_map_json="",
        onnx_dynamic_batch=True,
    )
    result = pull_model_from_registry(settings)
    assert result is None


# ── error-path unit tests ─────────────────────────────────────────────────────


def _make_settings(**overrides: object) -> Settings:
    """Build a minimal Settings instance for error-path tests."""
    defaults: dict[str, object] = {
        "model_registry_backend": "",
        "model_registry_uri": "",
        "model_name": "",
        "model_dir": "/opt/ml/model",
        "model_type": "",
        "model_filename": "",
        "port": 8080,
        "log_level": "INFO",
        "service_name": "test",
        "service_version": "dev",
        "deployment_env": "local",
        "input_mode": "tabular",
        "output_mode": "predictions",
        "default_content_type": "application/json",
        "default_accept": "application/json",
        "tabular_dtype": "float32",
        "csv_delimiter": ",",
        "csv_has_header": "auto",
        "csv_skip_blank_lines": True,
        "json_key_instances": "instances",
        "jsonl_features_key": "features",
        "tabular_id_columns": "",
        "tabular_feature_columns": "",
        "predictions_only": True,
        "json_output_key": "predictions",
        "max_body_bytes": 6 * 1024 * 1024,
        "max_records": 5000,
        "max_inflight": 16,
        "acquire_timeout_s": 0.25,
        "gunicorn_workers": 1,
        "gunicorn_threads": 4,
        "gunicorn_timeout": 60,
        "gunicorn_graceful_timeout": 30,
        "gunicorn_keepalive": 5,
        "prometheus_enabled": False,
        "prometheus_path": "/metrics",
        "swagger_enabled": False,
        "otel_enabled": False,
        "otel_endpoint": "",
        "otel_headers": "",
        "otel_timeout_s": 10.0,
        "onnx_providers": "CPUExecutionProvider",
        "onnx_intra_op_threads": 0,
        "onnx_inter_op_threads": 0,
        "onnx_graph_opt_level": "all",
        "onnx_input_name": "",
        "onnx_output_name": "",
        "onnx_output_index": 0,
        "tabular_num_features": 0,
        "onnx_input_map_json": "",
        "onnx_output_map_json": "",
        "onnx_input_dtype_map_json": "",
        "onnx_dynamic_batch": True,
    }
    defaults.update(overrides)
    return Settings.model_construct(**defaults)  # type: ignore[arg-type]


def test_pull_raises_when_only_backend_is_set() -> None:
    """RuntimeError when backend is set but uri is missing."""
    settings = _make_settings(model_registry_backend="sqlite", model_registry_uri="")
    with pytest.raises(RuntimeError, match="MODEL_REGISTRY_URI"):
        pull_model_from_registry(settings)


def test_pull_raises_when_only_uri_is_set() -> None:
    """RuntimeError when uri is set but backend is missing."""
    settings = _make_settings(
        model_registry_backend="", model_registry_uri="/db/reg.db"
    )
    with pytest.raises(RuntimeError, match="MODEL_REGISTRY_BACKEND"):
        pull_model_from_registry(settings)


def test_pull_raises_when_model_name_is_missing() -> None:
    """RuntimeError when both backend+uri are set but MODEL_NAME is absent."""
    settings = _make_settings(
        model_registry_backend="sqlite",
        model_registry_uri="/db/reg.db",
        model_name="",
    )
    with pytest.raises(RuntimeError, match="MODEL_NAME"):
        pull_model_from_registry(settings)


def test_open_registry_raises_on_unsupported_backend() -> None:
    """_open_registry raises RuntimeError for unknown backend names."""
    with pytest.raises(RuntimeError, match="not supported"):
        _open_registry("redis", "redis://localhost")


def test_resolve_onnx_uri_raises_when_no_production_version() -> None:
    """_resolve_onnx_uri raises RuntimeError when no production version exists."""
    registry = SqliteModelRegistry(path=":memory:")
    registry.register(
        ModelVersion(
            id="m-v1",
            name="mymodel",
            model_id="m",
            version="1",
            onnx_uri="/tmp/model.onnx",
        )
    )
    # Not promoted to any stage, so resolving "production" finds no version.
    with pytest.raises(RuntimeError, match="No version at stage="):
        _resolve_onnx_uri(registry, "mymodel", "production")
    registry.close()


def test_resolve_onnx_uri_raises_when_onnx_uri_is_missing() -> None:
    """_resolve_onnx_uri raises RuntimeError when production version has no onnx_uri."""
    registry = SqliteModelRegistry(path=":memory:")
    version = registry.register(
        ModelVersion(
            id="m-v1",
            name="mymodel",
            model_id="m",
            version="1",
            onnx_uri=None,  # deliberately absent
        )
    )
    registry.promote(name="mymodel", version=version.version, stage=Stage.PRODUCTION)
    with pytest.raises(RuntimeError, match="no onnx_uri"):
        _resolve_onnx_uri(registry, "mymodel", "production")
    registry.close()


def test_materialize_artifact_raises_on_missing_local_path(tmp_path: Path) -> None:
    """_materialize_artifact raises RuntimeError when local path does not exist."""
    with pytest.raises(RuntimeError, match="does not exist"):
        _materialize_artifact("/nonexistent/path/model.onnx", tmp_path)


def test_materialize_artifact_raises_on_s3_uri(tmp_path: Path) -> None:
    """_materialize_artifact raises NotImplementedError for s3:// URIs."""
    with pytest.raises(NotImplementedError, match="s3://"):
        _materialize_artifact("s3://my-bucket/models/model.onnx", tmp_path)


def test_materialize_artifact_raises_on_unsupported_scheme(tmp_path: Path) -> None:
    """_materialize_artifact raises RuntimeError for unknown URI schemes."""
    with pytest.raises(RuntimeError, match="Unsupported onnx_uri scheme"):
        _materialize_artifact("gs://my-bucket/model.onnx", tmp_path)
