"""Parity checks between HTTP and gRPC serving contracts and behavior."""

from __future__ import annotations

from asyncio import run as asyncio_run
from dataclasses import dataclass
from json import dumps as json_dumps
from json import loads as json_loads
from os import getenv as os_getenv
from typing import Any, cast

from httpx import ASGITransport as httpx_ASGITransport
from httpx import AsyncClient as httpx_AsyncClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import skip as pytest_skip

from application import create_application
from config import Settings
from parsed_types import ParsedInput

try:
    # Keep this guarded block stable because these imports are optional in CI images.
    from grpc import StatusCode as grpc_StatusCode  # noqa: I001

    from onnx_model_serving.grpc.server import InferenceGrpcService
    from onnx_serving_grpc.inference_pb2 import (
        DESCRIPTOR as inference_pb2_DESCRIPTOR,
    )
    from onnx_serving_grpc.inference_pb2 import (
        LiveRequest as inference_pb2_LiveRequest,
    )
    from onnx_serving_grpc.inference_pb2 import (
        PredictReply as inference_pb2_PredictReply,
    )
    from onnx_serving_grpc.inference_pb2 import (
        PredictRequest as inference_pb2_PredictRequest,
    )
    from onnx_serving_grpc.inference_pb2 import (
        ReadyRequest as inference_pb2_ReadyRequest,
    )
    from onnx_serving_grpc.inference_pb2 import (
        StatusReply as inference_pb2_StatusReply,
    )
except (ImportError, ModuleNotFoundError) as exc:
    pytest_skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)

pytestmark = pytest_mark.integration
_HYPOTHESIS_MAX_EXAMPLES = int(os_getenv("HYPOTHESIS_MAX_EXAMPLES", "30"))

_HTTP_TO_GRPC_SURFACE_MAP: dict[str, str] = {
    "/live": "Live",
    "/healthz": "Live",
    "/ready": "Ready",
    "/readyz": "Ready",
    "/ping": "Ready",
    "/invocations": "Predict",
}

_UNMAPPED_HTTP_PATHS: set[str] = set()
_UNMAPPED_GRPC_METHODS: set[str] = set()


class _DummyAdapter:
    """Deterministic adapter used for parity checks."""

    def is_ready(self: _DummyAdapter) -> bool:
        """Return adapter readiness."""
        return True

    def predict(self: _DummyAdapter, parsed_input: ParsedInput) -> object:
        """Return deterministic predictions based on batch size."""
        if parsed_input.X is not None:
            size = int(parsed_input.X.shape[0])
        elif parsed_input.tensors:
            first = next(iter(parsed_input.tensors.values()))
            size = int(first.shape[0]) if first.ndim > 0 else 1
        else:
            size = 1
        return [0] * size


@dataclass
class _RecordingContext:
    """Minimal context mock to capture gRPC error status."""

    code: object | None = None
    details: str | None = None

    def set_code(self: _RecordingContext, code: object) -> None:
        """Capture status code."""
        self.code = code

    def set_details(self: _RecordingContext, details: str) -> None:
        """Capture status details."""
        self.details = details


def _set_base_env(monkeypatch: pytest_MonkeyPatch) -> None:
    """Set baseline environment variables for parity tests."""
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("INPUT_MODE", "tabular")
    monkeypatch.setenv("DEFAULT_CONTENT_TYPE", "application/json")
    monkeypatch.setenv("DEFAULT_ACCEPT", "application/json")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("TABULAR_DTYPE", "float32")


def _patch_adapters(monkeypatch: pytest_MonkeyPatch) -> None:
    """Patch both HTTP and gRPC loaders to use deterministic adapter."""
    import application as application_module
    import onnx_model_serving.grpc.server as grpc_server_module

    monkeypatch.setattr(
        application_module,
        "load_adapter",
        lambda settings: _DummyAdapter(),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "load_adapter",
        lambda settings: _DummyAdapter(),
    )


def _grpc_method_names() -> set[str]:
    """Return gRPC method names from protobuf service descriptor."""
    service = inference_pb2_DESCRIPTOR.services_by_name["InferenceService"]
    return {method.name for method in service.methods}


def _http_route_paths(application: object) -> set[str]:
    """Return documented HTTP paths from OpenAPI schema."""
    app_obj = cast(Any, application)
    return set(app_obj.openapi().get("paths", {}).keys())


def test_surface_parity_http_to_grpc_map_is_exhaustive(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Fail when HTTP/gRPC surfaces diverge without explicit mapping."""
    _set_base_env(monkeypatch)
    _patch_adapters(monkeypatch)
    app = create_application(Settings())

    discovered_http = _http_route_paths(app)
    discovered_grpc = _grpc_method_names()

    mapped_http = set(_HTTP_TO_GRPC_SURFACE_MAP.keys())
    mapped_grpc = set(_HTTP_TO_GRPC_SURFACE_MAP.values())

    assert discovered_http == mapped_http | _UNMAPPED_HTTP_PATHS
    assert discovered_grpc == mapped_grpc | _UNMAPPED_GRPC_METHODS


def test_schema_mapping_predict_fields_are_explicit() -> None:
    """Keep explicit translation table between HTTP and gRPC Predict fields."""
    grpc_predict_request_fields = set(
        inference_pb2_PredictRequest.DESCRIPTOR.fields_by_name.keys()
    )
    grpc_predict_reply_fields = set(
        inference_pb2_PredictReply.DESCRIPTOR.fields_by_name.keys()
    )

    expected_request_mapping = {
        "payload": "HTTP request body bytes",
        "content_type": "HTTP Content-Type header",
        "accept": "HTTP Accept header",
    }
    expected_reply_mapping = {
        "body": "HTTP response body bytes",
        "content_type": "HTTP response Content-Type header",
        "metadata": "HTTP response metadata extension",
    }

    assert grpc_predict_request_fields == set(expected_request_mapping.keys())
    assert grpc_predict_reply_fields == set(expected_reply_mapping.keys())


def test_grpc_contract_for_predict_messages() -> None:
    """Validate gRPC I/O contract fields for predict RPC."""
    request_fields = set(inference_pb2_PredictRequest.DESCRIPTOR.fields_by_name.keys())
    reply_fields = set(inference_pb2_PredictReply.DESCRIPTOR.fields_by_name.keys())
    status_fields = set(inference_pb2_StatusReply.DESCRIPTOR.fields_by_name.keys())

    assert request_fields == {"payload", "content_type", "accept"}
    assert reply_fields == {"body", "content_type", "metadata"}
    assert status_fields == {"ok", "status"}


@pytest_mark.asyncio
@pytest_mark.parametrize(
    "instances",
    [
        [[1.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]],
    ],
)
async def test_http_and_grpc_predict_parity_across_batches(
    monkeypatch: pytest_MonkeyPatch,
    instances: list[list[float]],
) -> None:
    """Compare HTTP and gRPC outputs for multiple valid payload shapes."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)
    payload_obj = {"instances": instances}
    payload_bytes = json_dumps(payload_obj).encode("utf-8")

    transport = httpx_ASGITransport(app=app)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        live_response = await client.get("/live")
        healthz_response = await client.get("/healthz")
        ready_response = await client.get("/ready")
        readyz_response = await client.get("/readyz")
        http_response = await client.post(
            "/invocations",
            content=payload_bytes,
            headers={"content-type": "application/json", "accept": "application/json"},
        )

    grpc_live = await grpc_service.Live(inference_pb2_LiveRequest(), None)
    grpc_ready = await grpc_service.Ready(inference_pb2_ReadyRequest(), None)
    grpc_predict = await grpc_service.Predict(
        inference_pb2_PredictRequest(
            payload=payload_bytes,
            content_type="application/json",
            accept="application/json",
        ),
        None,
    )

    assert live_response.status_code == 200
    assert healthz_response.status_code == 200
    assert ready_response.status_code == 200
    assert readyz_response.status_code == 200
    assert http_response.status_code == 200
    assert grpc_live.ok is True
    assert grpc_ready.ok is True
    assert grpc_predict.content_type == "application/json"

    http_json = json_loads(http_response.content.decode("utf-8"))
    grpc_json = json_loads(grpc_predict.body.decode("utf-8"))
    assert http_json == grpc_json


@pytest_mark.asyncio
async def test_error_parity_for_invalid_json_input(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate invalid JSON maps to equivalent HTTP/gRPC error semantics."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)
    bad_payload = b"{not-json"

    transport = httpx_ASGITransport(app=app)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/invocations",
            content=bad_payload,
            headers={"content-type": "application/json", "accept": "application/json"},
        )

    context = _RecordingContext()
    grpc_reply = await grpc_service.Predict(
        inference_pb2_PredictRequest(
            payload=bad_payload,
            content_type="application/json",
            accept="application/json",
        ),
        context,
    )

    assert http_response.status_code == 400
    assert context.code == grpc_StatusCode.INVALID_ARGUMENT
    assert context.details is not None
    assert grpc_reply.body == b""


@pytest_mark.asyncio
async def test_error_parity_for_record_limit_exceeded(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate batch-limit violations map to equivalent error category."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)
    payload_bytes = json_dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8")

    transport = httpx_ASGITransport(app=app)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/invocations",
            content=payload_bytes,
            headers={"content-type": "application/json", "accept": "application/json"},
        )

    context = _RecordingContext()
    await grpc_service.Predict(
        inference_pb2_PredictRequest(
            payload=payload_bytes,
            content_type="application/json",
            accept="application/json",
        ),
        context,
    )

    assert http_response.status_code == 400
    assert "too_many_records" in http_response.json()["error"]
    assert context.code == grpc_StatusCode.INVALID_ARGUMENT
    assert context.details is not None
    assert "too_many_records" in context.details


@pytest_mark.asyncio
@pytest_mark.parametrize(
    ("payload", "content_type", "accept"),
    [
        (b"1,2\n3,4\n", "text/csv", "application/json"),
        (
            json_dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8"),
            "application/json",
            "application/json",
        ),
    ],
)
async def test_http_and_grpc_predict_parity_for_multiple_content_types(
    monkeypatch: pytest_MonkeyPatch,
    payload: bytes,
    content_type: str,
    accept: str,
) -> None:
    """Compare HTTP/gRPC outputs for CSV and JSON payload variants."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)

    transport = httpx_ASGITransport(app=app)
    async with httpx_AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/invocations",
            content=payload,
            headers={"content-type": content_type, "accept": accept},
        )

    grpc_predict = await grpc_service.Predict(
        inference_pb2_PredictRequest(
            payload=payload,
            content_type=content_type,
            accept=accept,
        ),
        None,
    )

    assert http_response.status_code == 200
    assert grpc_predict.content_type == "application/json"
    assert json_loads(http_response.content.decode("utf-8")) == json_loads(
        grpc_predict.body.decode("utf-8")
    )


@st.composite
def _rectangular_instances(
    draw: st.DrawFn,
) -> list[list[float]]:
    row_count = draw(st.integers(min_value=1, max_value=12))
    col_count = draw(st.integers(min_value=1, max_value=8))
    return draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-1_000.0,
                    max_value=1_000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=col_count,
                max_size=col_count,
            ),
            min_size=row_count,
            max_size=row_count,
        )
    )


@pytest_mark.property
@settings(
    max_examples=_HYPOTHESIS_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    instances=_rectangular_instances(),
)
def test_http_and_grpc_predict_property_parity_for_json_instances(
    monkeypatch: pytest_MonkeyPatch,
    instances: list[list[float]],
) -> None:
    """Property check: valid JSON batches produce equivalent HTTP/gRPC predictions."""

    async def _run() -> None:
        _set_base_env(monkeypatch)
        monkeypatch.setenv("MAX_RECORDS", "1000")
        _patch_adapters(monkeypatch)

        settings_obj = Settings()
        app = create_application(settings_obj)
        grpc_service = InferenceGrpcService(settings_obj)
        payload_bytes = json_dumps({"instances": instances}).encode("utf-8")

        transport = httpx_ASGITransport(app=app)
        async with httpx_AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            http_response = await client.post(
                "/invocations",
                content=payload_bytes,
                headers={
                    "content-type": "application/json",
                    "accept": "application/json",
                },
            )

        grpc_predict = await grpc_service.Predict(
            inference_pb2_PredictRequest(
                payload=payload_bytes,
                content_type="application/json",
                accept="application/json",
            ),
            None,
        )

        assert http_response.status_code == 200
        assert grpc_predict.content_type == "application/json"
        assert json_loads(http_response.content.decode("utf-8")) == json_loads(
            grpc_predict.body.decode("utf-8")
        )

    asyncio_run(_run())


@pytest_mark.asyncio
async def test_grpc_predict_uses_defaults_when_content_type_and_accept_missing(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate gRPC predict falls back to default content-type and accept."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    grpc_service = InferenceGrpcService(settings)
    payload_bytes = json_dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8")

    grpc_predict = await grpc_service.Predict(
        inference_pb2_PredictRequest(
            payload=payload_bytes,
            content_type="",
            accept="",
        ),
        None,
    )

    assert grpc_predict.content_type == "application/json"
    assert json_loads(grpc_predict.body.decode("utf-8")) == [0, 0]


@pytest_mark.asyncio
async def test_swagger_routes_respect_toggle_flag(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Ensure /docs and /openapi.yaml are exposed only when enabled."""
    _set_base_env(monkeypatch)
    _patch_adapters(monkeypatch)

    monkeypatch.setenv("SWAGGER_ENABLED", "false")
    app_disabled = create_application(Settings())
    transport_disabled = httpx_ASGITransport(app=app_disabled)
    async with httpx_AsyncClient(
        transport=transport_disabled,
        base_url="http://test",
    ) as client:
        docs = await client.get("/docs")
        spec = await client.get("/openapi.yaml")
        assert docs.status_code == 404
        assert spec.status_code == 404

    monkeypatch.setenv("SWAGGER_ENABLED", "true")
    app_enabled = create_application(Settings())
    transport_enabled = httpx_ASGITransport(app=app_enabled)
    async with httpx_AsyncClient(
        transport=transport_enabled,
        base_url="http://test",
    ) as client:
        docs = await client.get("/docs")
        spec = await client.get("/openapi.yaml")
        assert docs.status_code == 200
        assert "SwaggerUIBundle" in docs.text
        assert spec.status_code == 200
        assert "openapi" in spec.text.lower()
