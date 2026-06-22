"""Unit tests for gRPC server helpers and error mapping."""

from __future__ import annotations

from asyncio import run as asyncio_run
from types import SimpleNamespace as types_SimpleNamespace

from numpy import array as np_array
from numpy import float32 as np_float32
from numpy import zeros as np_zeros
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import raises as pytest_raises
from pytest import skip as pytest_skip

from config import Settings
from parsed_types import ParsedInput

try:
    from grpc import StatusCode as grpc_StatusCode
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest_skip(f"grpc dependencies unavailable: {exc}", allow_module_level=True)


class _AdapterStub:
    """Minimal adapter for service tests."""

    def __init__(
        self: _AdapterStub, *, ready: bool = True, value: object = None
    ) -> None:
        self._ready = ready
        self._value = value if value is not None else [1]

    def is_ready(self: _AdapterStub) -> bool:
        return self._ready

    def predict(self: _AdapterStub, parsed_input: ParsedInput) -> object:
        _ = parsed_input
        return self._value


class _ContextRecorder:
    """Capture gRPC status code/details written by handlers."""

    def __init__(self: _ContextRecorder) -> None:
        self.code: object | None = None
        self.details: str | None = None

    def set_code(self: _ContextRecorder, code: object) -> None:
        self.code = code

    def set_details(self: _ContextRecorder, details: str) -> None:
        self.details = details


def test_set_grpc_error_handles_none_context() -> None:
    """No-op when context is None."""
    import onnx_model_serving.grpc.server as module

    module._set_grpc_error(None, grpc_StatusCode.INTERNAL, "boom")


def test_set_grpc_error_sets_code_and_details() -> None:
    """Record code/details when context is present."""
    import onnx_model_serving.grpc.server as module

    context = _ContextRecorder()
    module._set_grpc_error(context, grpc_StatusCode.INVALID_ARGUMENT, "bad")
    assert context.code == grpc_StatusCode.INVALID_ARGUMENT
    assert context.details == "bad"


def test_batch_size_from_tabular_tensor_and_empty_payload() -> None:
    """Compute batch size for supported shapes and reject empty payloads."""
    import onnx_model_serving.grpc.server as module

    assert module._batch_size(ParsedInput(X=np_zeros((3, 2), dtype=np_float32))) == 3
    assert (
        module._batch_size(
            ParsedInput(tensors={"x": np_zeros((4, 1), dtype=np_float32)})
        )
        == 4
    )
    assert (
        module._batch_size(ParsedInput(tensors={"x": np_array(1.0, dtype=np_float32)}))
        == 1
    )
    with pytest_raises(ValueError, match="no features/tensors"):
        module._batch_size(ParsedInput())


def test_live_and_ready_statuses(monkeypatch: pytest_MonkeyPatch) -> None:
    """Expose expected liveness/readiness payloads."""
    import onnx_model_serving.grpc.server as module

    monkeypatch.setattr(
        module,
        "load_adapter",
        lambda settings: _AdapterStub(ready=True),
    )
    service = module.InferenceGrpcService(Settings())

    live = asyncio_run(service.Live(object(), object()))
    ready = asyncio_run(service.Ready(object(), object()))

    assert live.ok is True
    assert live.status == "live"
    assert ready.ok is True
    assert ready.status == "ready"


def test_ready_reports_not_ready(monkeypatch: pytest_MonkeyPatch) -> None:
    """Report not_ready when adapter is unavailable."""
    import onnx_model_serving.grpc.server as module

    monkeypatch.setattr(
        module,
        "load_adapter",
        lambda settings: _AdapterStub(ready=False),
    )
    service = module.InferenceGrpcService(Settings())
    ready = asyncio_run(service.Ready(object(), object()))
    assert ready.ok is False
    assert ready.status == "not_ready"


def test_predict_success_uses_defaults(monkeypatch: pytest_MonkeyPatch) -> None:
    """Use default content-type/accept and return metadata batch size."""
    import onnx_model_serving.grpc.server as module

    monkeypatch.setattr(
        module,
        "load_adapter",
        lambda settings: _AdapterStub(value=[42]),
    )

    def _parse_payload(
        payload: bytes, *, content_type: str, settings: Settings
    ) -> ParsedInput:
        _ = (payload, content_type, settings)
        return ParsedInput(X=np_zeros((2, 2), dtype=np_float32))

    def _format_output(
        predictions: object, *, accept: str, settings: Settings
    ) -> tuple[bytes, str]:
        _ = (predictions, accept, settings)
        return (b'{"predictions":[42]}', "application/json")

    monkeypatch.setattr(
        module,
        "parse_payload",
        _parse_payload,
    )
    monkeypatch.setattr(module, "format_output", _format_output)

    service = module.InferenceGrpcService(Settings())
    request = types_SimpleNamespace(payload=b"{}", content_type="", accept="")
    reply = asyncio_run(service.Predict(request, None))

    assert reply.body == b'{"predictions":[42]}'
    assert reply.content_type == "application/json"
    assert reply.metadata["batch_size"] == "2"


def test_predict_maps_value_error_to_invalid_argument(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Map validation errors to INVALID_ARGUMENT and empty reply payload."""
    import onnx_model_serving.grpc.server as module

    monkeypatch.setattr(module, "load_adapter", lambda settings: _AdapterStub())

    def _boom(*_args: object, **_kwargs: object) -> ParsedInput:
        raise ValueError("invalid input")

    monkeypatch.setattr(module, "parse_payload", _boom)
    service = module.InferenceGrpcService(Settings())
    context = _ContextRecorder()
    request = types_SimpleNamespace(
        payload=b"{bad}",
        content_type="application/json",
        accept="application/json",
    )
    reply = asyncio_run(service.Predict(request, context))

    assert context.code == grpc_StatusCode.INVALID_ARGUMENT
    assert context.details == "invalid input"
    assert reply.body == b""


def test_predict_maps_unexpected_error_to_internal(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Map unexpected adapter/runtime errors to INTERNAL."""
    import onnx_model_serving.grpc.server as module

    monkeypatch.setattr(module, "load_adapter", lambda settings: _AdapterStub())

    def _parse_payload(
        payload: bytes, *, content_type: str, settings: Settings
    ) -> ParsedInput:
        _ = (payload, content_type, settings)
        return ParsedInput(X=np_zeros((1, 1), dtype=np_float32))

    monkeypatch.setattr(
        module,
        "parse_payload",
        _parse_payload,
    )

    def _explode(*_args: object, **_kwargs: object) -> tuple[bytes, str]:
        raise RuntimeError("explode")

    monkeypatch.setattr(module, "format_output", _explode)
    service = module.InferenceGrpcService(Settings())
    context = _ContextRecorder()
    request = types_SimpleNamespace(
        payload=b"{}",
        content_type="application/json",
        accept="application/json",
    )
    reply = asyncio_run(service.Predict(request, context))

    assert context.code == grpc_StatusCode.INTERNAL
    assert context.details == "internal_server_error"
    assert reply.body == b""


def test_create_server_registers_service(monkeypatch: pytest_MonkeyPatch) -> None:
    """Register service and bind host/port."""
    import onnx_model_serving.grpc.server as module

    monkeypatch.setattr(module, "load_adapter", lambda settings: _AdapterStub())
    calls: dict[str, object] = {}

    class _FakeServer:
        def add_insecure_port(self: _FakeServer, address: str) -> None:
            calls["address"] = address

    fake_server = _FakeServer()

    def _fake_grpc_server(**kwargs: object) -> _FakeServer:
        calls["kwargs"] = kwargs
        return fake_server

    def _fake_register(servicer: object, server: object) -> None:
        calls["servicer"] = servicer
        calls["server"] = server

    monkeypatch.setattr(module.grpc.aio, "server", _fake_grpc_server)
    monkeypatch.setattr(
        module,
        "_require_grpc_stubs_module",
        lambda: types_SimpleNamespace(
            add_InferenceServiceServicer_to_server=_fake_register
        ),
    )
    created = module.create_server(
        Settings(),
        host="127.0.0.1",
        port=50052,
        max_concurrent_rpcs=2,
    )
    assert created is fake_server
    assert calls["address"] == "127.0.0.1:50052"
    assert calls["server"] is fake_server


def test_main_starts_server(monkeypatch: pytest_MonkeyPatch) -> None:
    """Parse args and run grpc server lifecycle."""
    import onnx_model_serving.grpc.server as module

    started = {"start": False, "wait": False}

    class _Server:
        async def start(self: _Server) -> None:
            started["start"] = True

        async def wait_for_termination(self: _Server) -> None:
            started["wait"] = True

    monkeypatch.setattr(module, "create_server", lambda *args, **kwargs: _Server())
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: module.argparse.Namespace(
            host="0.0.0.0",
            port=50052,
            max_concurrent_rpcs=8,
        ),
    )
    module.main()
    assert started["start"] is True
    assert started["wait"] is True
