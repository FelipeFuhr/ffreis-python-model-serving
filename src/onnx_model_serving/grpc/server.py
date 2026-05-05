"""gRPC server exposing ONNX inference endpoints."""

from __future__ import annotations

import argparse
import importlib
from asyncio import run as asyncio_run
from asyncio import sleep as asyncio_sleep
from asyncio import to_thread as asyncio_to_thread
from dataclasses import dataclass, field
from functools import partial
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger
from types import ModuleType
from typing import Protocol, cast

import grpc
from grpc import aio as grpc_aio

from base_adapter import BaseAdapter, load_adapter
from config import Settings
from input_output import format_output, parse_payload
from parsed_types import batch_size as _batch_size
from value_types import PredictionValue

log = logging_getLogger("byoc.grpc")


@dataclass
class _FallbackStatusReply:
    """Fallback status reply used when generated stubs are unavailable."""

    ok: bool = False
    status: str = ""


@dataclass
class _FallbackPredictReply:
    """Fallback predict reply used when generated stubs are unavailable."""

    body: bytes = b""
    content_type: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class _StatusReplyLike(Protocol):
    """Protocol for status reply payload."""

    ok: bool
    status: str


class _PredictReplyLike(Protocol):
    """Protocol for predict reply payload."""

    body: bytes
    content_type: str
    metadata: dict[str, str]


def _load_grpc_stub_module(module_name: str) -> ModuleType | None:
    """Load grpc generated module by name when present."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


_INFERENCE_PB2 = _load_grpc_stub_module("onnx_serving_grpc.inference_pb2")
_INFERENCE_PB2_GRPC = _load_grpc_stub_module("onnx_serving_grpc.inference_pb2_grpc")


def _status_reply(*, ok: bool, status: str) -> _StatusReplyLike:
    """Create status reply from generated stubs or fallback type."""
    if _INFERENCE_PB2 is None:
        return _FallbackStatusReply(ok=ok, status=status)
    return cast(_StatusReplyLike, _INFERENCE_PB2.StatusReply(ok=ok, status=status))


def _predict_reply(
    *,
    body: bytes = b"",
    content_type: str = "",
    metadata: dict[str, str] | None = None,
) -> _PredictReplyLike:
    """Create predict reply from generated stubs or fallback type."""
    if _INFERENCE_PB2 is None:
        return _FallbackPredictReply(
            body=body,
            content_type=content_type,
            metadata=metadata or {},
        )
    return cast(
        _PredictReplyLike,
        _INFERENCE_PB2.PredictReply(
            body=body,
            content_type=content_type,
            metadata=metadata or {},
        ),
    )


def _require_grpc_stubs_module() -> ModuleType:
    """Return generated grpc service module or raise actionable error."""
    if _INFERENCE_PB2_GRPC is None:
        raise RuntimeError(
            "gRPC stubs are missing. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return _INFERENCE_PB2_GRPC


class _PredictRequestLike(Protocol):
    """Protocol for predict request object shape."""

    payload: bytes
    content_type: str
    accept: str


class _HealthRequestLike(Protocol):
    """Protocol for empty health request messages."""


def _set_grpc_error(
    context: grpc.ServicerContext | grpc_aio.ServicerContext | None,
    code: grpc.StatusCode,
    details: str,
) -> None:
    """Safely set error code/details when context is available."""
    if context is None:
        return
    context.set_code(code)
    context.set_details(details)


class InferenceGrpcService:
    """gRPC inference service implementation."""

    _RPC_METHOD_ALIASES: dict[str, str] = {
        "Live": "live",
        "Ready": "ready",
        "Predict": "predict",
    }

    def __init__(self: InferenceGrpcService, settings: Settings) -> None:
        """Initialize service and eagerly load model adapter.

        Parameters
        ----------
        settings : Settings
            Runtime configuration.
        """
        self.settings = settings
        self.adapter: BaseAdapter = load_adapter(settings)

    async def live(
        self: InferenceGrpcService,
        request: _HealthRequestLike,
        context: grpc.ServicerContext | grpc_aio.ServicerContext,
    ) -> _StatusReplyLike:
        """Report process liveness."""
        _ = (request, context)
        await asyncio_sleep(0)
        return _status_reply(ok=True, status="live")

    async def ready(
        self: InferenceGrpcService,
        request: _HealthRequestLike,
        context: grpc.ServicerContext | grpc_aio.ServicerContext,
    ) -> _StatusReplyLike:
        """Report model readiness."""
        _ = (request, context)
        is_ready = bool(await asyncio_to_thread(self.adapter.is_ready))
        return _status_reply(
            ok=is_ready,
            status="ready" if is_ready else "not_ready",
        )

    async def predict(
        self: InferenceGrpcService,
        request: _PredictRequestLike,
        context: grpc.ServicerContext | grpc_aio.ServicerContext | None,
    ) -> _PredictReplyLike:
        """Run model prediction for a payload."""
        content_type = request.content_type or self.settings.default_content_type
        accept = request.accept or self.settings.default_accept
        payload = bytes(request.payload)

        try:
            parsed_input = await asyncio_to_thread(
                partial(
                    parse_payload,
                    payload,
                    content_type=content_type,
                    settings=self.settings,
                )
            )
            batch_size = _batch_size(parsed_input)
            if batch_size > self.settings.max_records:
                raise ValueError(
                    f"too_many_records: {batch_size} > {self.settings.max_records}"
                )
            predictions: PredictionValue = await asyncio_to_thread(
                self.adapter.predict, parsed_input
            )
            body, output_content_type = await asyncio_to_thread(
                partial(
                    format_output,
                    predictions,
                    accept=accept,
                    settings=self.settings,
                )
            )
            body_bytes = body if isinstance(body, bytes) else body.encode("utf-8")
            return _predict_reply(
                body=body_bytes,
                content_type=output_content_type,
                metadata={"batch_size": str(batch_size)},
            )
        except ValueError as exc:
            _set_grpc_error(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return _predict_reply()
        except Exception:  # pragma: no cover - unexpected adapter/runtime errors
            log.exception("Predict RPC failed")
            _set_grpc_error(context, grpc.StatusCode.INTERNAL, "internal_server_error")
            return _predict_reply()

    def __getattr__(self: InferenceGrpcService, name: str) -> object:
        """Expose gRPC PascalCase RPC handlers via lowercase implementations."""
        target = self._RPC_METHOD_ALIASES.get(name)
        if target is None:
            raise AttributeError(name)
        return getattr(self, target)


def create_server(
    settings: Settings,
    *,
    host: str,
    port: int,
    max_concurrent_rpcs: int = 16,
) -> grpc_aio.Server:
    """Create configured gRPC server instance.

    Parameters
    ----------
    settings : Settings
        Runtime settings.
    host : str
        Bind host.
    port : int
        Bind TCP port.
    max_concurrent_rpcs : int, default=16
        Maximum number of concurrent RPCs the server will accept.

    Returns
    -------
    grpc.aio.Server
        Configured server.
    """
    server = grpc_aio.server(maximum_concurrent_rpcs=max_concurrent_rpcs)
    grpc_stubs = _require_grpc_stubs_module()
    grpc_stubs.add_InferenceServiceServicer_to_server(
        InferenceGrpcService(settings),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server


async def _serve(
    *,
    settings: Settings,
    host: str,
    port: int,
    max_concurrent_rpcs: int,
) -> None:
    """Start async gRPC server and block until termination."""
    server = create_server(
        settings,
        host=host,
        port=port,
        max_concurrent_rpcs=max_concurrent_rpcs,
    )
    await server.start()
    log.info("gRPC inference server listening on %s:%s", host, port)
    await server.wait_for_termination()


def main() -> None:
    """Run gRPC inference server entrypoint."""
    settings = Settings()
    logging_basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="ONNX model serving gRPC endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--max-concurrent-rpcs", type=int, default=16)
    args = parser.parse_args()

    asyncio_run(
        _serve(
            settings=settings,
            host=cast(str, args.host),
            port=cast(int, args.port),
            max_concurrent_rpcs=cast(int, args.max_concurrent_rpcs),
        )
    )


if __name__ == "__main__":
    main()
