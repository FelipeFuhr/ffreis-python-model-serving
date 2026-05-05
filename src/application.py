"""HTTP application factory and request handling flow."""

from __future__ import annotations

from asyncio import Semaphore as asyncio_Semaphore
from asyncio import wait_for as asyncio_wait_for
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from importlib import import_module as importlib_import_module
from logging import getLogger as logging_getLogger
from pathlib import Path
from time import time as time_time
from types import TracebackType
from typing import Literal, Protocol, cast

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from base_adapter import BaseAdapter, load_adapter
from config import Settings
from input_output import format_output, parse_payload
from parsed_types import ParsedInput
from parsed_types import batch_size as _batch_size
from telemetry import (
    get_current_trace_identifiers,
    get_tracer,
    instrument_fastapi_application,
    setup_telemetry,
)
from value_types import PredictionValue, SpanAttributeValue

log = logging_getLogger("byoc")
_OPENAPI_CONTRACT_FILE = Path(__file__).resolve().parents[1] / "docs" / "openapi.yaml"
_SWAGGER_UI_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.yaml",
      dom_id: "#swagger-ui",
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
    });
  </script>
</body>
</html>
"""

try:
    Instrumentator = importlib_import_module(
        "prometheus_fastapi_instrumentator"
    ).Instrumentator
except Exception:  # pragma: no cover - optional dependency
    Instrumentator = None


class _NoopSpan:
    """No-op tracing span used when OpenTelemetry is not available."""

    def __enter__(self: _NoopSpan) -> _NoopSpan:
        return self

    def __exit__(
        self: _NoopSpan,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False

    def set_attribute(self: _NoopSpan, _k: str, _v: SpanAttributeValue) -> None:
        return None


@contextmanager
def _noop_span() -> Iterator[_NoopSpan]:
    """Return a no-op span context manager."""
    yield _NoopSpan()


class _SpanLike(Protocol):
    """Span protocol required by this module."""

    def set_attribute(self: _SpanLike, key: str, value: SpanAttributeValue) -> None:
        """Set span attribute."""


class _TracerLike(Protocol):
    """Tracer protocol required by this module."""

    def start_as_current_span(
        self: _TracerLike, name: str
    ) -> AbstractContextManager[_SpanLike]:
        """Start a span context manager."""


class _NoopTracer:
    """No-op tracer used when OpenTelemetry is unavailable."""

    def start_as_current_span(
        self: _NoopTracer, _name: str
    ) -> AbstractContextManager[_NoopSpan]:
        """Return no-op span context manager."""
        return _noop_span()


tracer: _TracerLike = cast(_TracerLike, get_tracer("byoc") or _NoopTracer())


class InferenceApplicationBuilder:
    """Compose and configure FastAPI inference application."""

    def __init__(self: InferenceApplicationBuilder, settings: Settings) -> None:
        """Initialize builder with runtime settings.

        Parameters
        ----------
        settings : Settings
            Runtime settings.
        """
        self.settings = settings
        self.application = FastAPI(
            title=settings.service_name,
            version=settings.service_version,
            openapi_url=None,
            docs_url=None,
            redoc_url=None,
        )
        self._adapter: BaseAdapter | None = None
        self._semaphore = asyncio_Semaphore(settings.max_inflight)

    def build(self: InferenceApplicationBuilder) -> FastAPI:
        """Build and return configured FastAPI application."""
        self._configure_telemetry()
        self._configure_metrics()
        self._register_body_limit_middleware()
        self._register_routes()
        return self.application

    def _configure_telemetry(self: InferenceApplicationBuilder) -> None:
        """Enable application telemetry when configured."""
        if setup_telemetry(self.settings):
            instrument_fastapi_application(self.settings, self.application)

    def _configure_metrics(self: InferenceApplicationBuilder) -> None:
        """Expose metrics endpoint when Prometheus is enabled."""
        if not self.settings.prometheus_enabled:
            return
        if Instrumentator is not None:
            Instrumentator().instrument(self.application).expose(
                self.application,
                endpoint=self.settings.prometheus_path,
                include_in_schema=False,
            )
            log.info("Prometheus enabled at %s", self.settings.prometheus_path)
            return

        @self.application.get(self.settings.prometheus_path, include_in_schema=False)
        def _fallback_metrics() -> PlainTextResponse:
            return PlainTextResponse(
                "# HELP byoc_up Service readiness\n# TYPE byoc_up gauge\nbyoc_up 1\n",
                status_code=200,
            )

    def _register_body_limit_middleware(self: InferenceApplicationBuilder) -> None:
        """Register middleware enforcing maximum payload size."""

        @self.application.middleware("http")
        async def limit_body(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            body = await request.body()
            if len(body) > self.settings.max_body_bytes:
                return JSONResponse(
                    {
                        "error": "payload_too_large",
                        "max_bytes": self.settings.max_body_bytes,
                    },
                    status_code=413,
                )
            request.state.cached_body = body
            return await call_next(request)

    def _register_routes(self: InferenceApplicationBuilder) -> None:
        """Register health and inference routes."""

        @self.application.get("/live")
        def live() -> PlainTextResponse:
            """Return process liveness status.

            Returns
            -------
            fastapi.responses.PlainTextResponse
                HTTP 200 response when process is up.
            """
            return self._build_liveness_response()

        @self.application.get("/healthz")
        def healthz() -> PlainTextResponse:
            """Return process liveness status (Kubernetes-style endpoint)."""
            return self._build_liveness_response()

        @self.application.get("/ready")
        def ready() -> PlainTextResponse:
            """Return model readiness status.

            Returns
            -------
            fastapi.responses.PlainTextResponse
                HTTP 200 when model is ready, otherwise HTTP 500.
            """
            return self._build_readiness_response()

        @self.application.get("/readyz")
        def readyz() -> PlainTextResponse:
            """Return model readiness status (Kubernetes-style endpoint)."""
            return self._build_readiness_response()

        @self.application.get("/ping")
        def ping() -> PlainTextResponse:
            """Return readiness status for SageMaker compatibility.

            Returns
            -------
            fastapi.responses.PlainTextResponse
                HTTP 200 when model is ready, otherwise HTTP 500.
            """
            return self._build_readiness_response()

        @self.application.post("/invocations")
        async def invocations(request: Request) -> Response:
            return await self._handle_invocation(request)

        if self.settings.swagger_enabled:

            @self.application.get("/openapi.yaml", include_in_schema=False)
            def openapi_contract() -> Response:
                if not _OPENAPI_CONTRACT_FILE.exists():
                    return JSONResponse(
                        {"error": "openapi_contract_not_found"}, status_code=404
                    )
                return PlainTextResponse(
                    _OPENAPI_CONTRACT_FILE.read_text(encoding="utf-8"),
                    media_type="application/yaml",
                    status_code=200,
                )

            @self.application.get("/docs", include_in_schema=False)
            def swagger_ui() -> HTMLResponse:
                return HTMLResponse(_SWAGGER_UI_HTML, status_code=200)

    def _build_liveness_response(
        self: InferenceApplicationBuilder,
    ) -> PlainTextResponse:
        """Build liveness response.

        Returns
        -------
        fastapi.responses.PlainTextResponse
            HTTP 200 response indicating process is alive.
        """
        return PlainTextResponse("\n", status_code=200)

    def _build_readiness_response(
        self: InferenceApplicationBuilder,
    ) -> PlainTextResponse:
        """Build readiness response from adapter availability.

        Returns
        -------
        fastapi.responses.PlainTextResponse
            HTTP 200 when adapter reports ready, otherwise HTTP 500.
        """
        try:
            self._ensure_adapter_loaded()
            adapter = self._adapter
            if adapter is None:
                return PlainTextResponse("\n", status_code=500)
            return PlainTextResponse(
                "\n",
                status_code=200 if adapter.is_ready() else 500,
            )
        except Exception:
            log.exception("Readiness check failed")
            return PlainTextResponse("\n", status_code=500)

    def _ensure_adapter_loaded(self: InferenceApplicationBuilder) -> None:
        """Load model adapter on first use."""
        if self._adapter is None:
            self._adapter = load_adapter(self.settings)
            log.info(
                "Loaded adapter=%s model_dir=%s",
                type(self._adapter).__name__,
                self.settings.model_dir,
            )

    async def _handle_invocation(
        self: InferenceApplicationBuilder, request: Request
    ) -> Response:
        """Handle invocation request with concurrency guard.

        Parameters
        ----------
        request : Request
            Incoming HTTP request.

        Returns
        -------
        Response
            Inference response.
        """
        acquired = await self._try_acquire_request_slot()
        if not acquired:
            return JSONResponse({"error": "too_many_requests"}, status_code=429)

        start_time = time_time()
        try:
            return await self._run_inference(request=request, start_time=start_time)
        except ValueError as error:
            log.warning("Bad request: %s", error)
            return JSONResponse({"error": str(error)}, status_code=400)
        except Exception:
            log.exception("Inference failed")
            return JSONResponse({"error": "internal_server_error"}, status_code=500)
        finally:
            if acquired:
                self._semaphore.release()

    async def _try_acquire_request_slot(self: InferenceApplicationBuilder) -> bool:
        """Try to acquire one concurrency slot within timeout."""
        try:
            await asyncio_wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.acquire_timeout_s,
            )
            return True
        except TimeoutError:
            return False

    async def _run_inference(
        self: InferenceApplicationBuilder, request: Request, start_time: float
    ) -> Response:
        """Run parsing, prediction, and response formatting.

        Parameters
        ----------
        request : Request
            Incoming HTTP request.
        start_time : float
            Start timestamp used for latency calculation.

        Returns
        -------
        Response
            Successful inference response.
        """
        self._ensure_adapter_loaded()
        adapter = self._adapter
        if adapter is None:
            raise RuntimeError("Adapter failed to load")
        content_type, accept = self._resolve_content_preferences(request)
        payload = getattr(request.state, "cached_body", None) or await request.body()

        with tracer.start_as_current_span("invocations") as span:
            span.set_attribute("request.content_type", content_type)
            span.set_attribute("request.bytes", len(payload))

            parsed_input = parse_payload(
                payload,
                content_type=content_type,
                settings=self.settings,
            )
            batch_size = _batch_size(parsed_input)
            span.set_attribute("batch.size", batch_size)
            self._validate_batch_limit(batch_size)

            predictions = self._predict(adapter, parsed_input)
            body, output_content_type = format_output(
                predictions,
                accept=accept,
                settings=self.settings,
            )
            span.set_attribute("response.content_type", output_content_type)
            span.set_attribute("latency_ms", (time_time() - start_time) * 1000.0)
            response = Response(
                content=body, media_type=output_content_type, status_code=200
            )
            self._attach_trace_correlation_headers(response)
            return response

    def _resolve_content_preferences(
        self: InferenceApplicationBuilder, request: Request
    ) -> tuple[str, str]:
        """Resolve request and response content preferences from headers."""
        content_type = (
            request.headers.get("content-type")
            or request.headers.get("x-amzn-sagemaker-content-type")
            or self.settings.default_content_type
        )
        accept = (
            request.headers.get("accept")
            or request.headers.get("x-amzn-sagemaker-accept")
            or self.settings.default_accept
        )
        return content_type, accept

    def _validate_batch_limit(
        self: InferenceApplicationBuilder, batch_size: int
    ) -> None:
        """Validate that batch size is within configured limits."""
        if batch_size > self.settings.max_records:
            raise ValueError(
                f"too_many_records: {batch_size} > {self.settings.max_records}"
            )

    def _predict(
        self: InferenceApplicationBuilder,
        adapter: BaseAdapter,
        parsed_input: ParsedInput,
    ) -> PredictionValue:
        """Run adapter prediction under model tracing span."""
        with tracer.start_as_current_span("model.predict"):
            return adapter.predict(parsed_input)

    def _attach_trace_correlation_headers(
        self: InferenceApplicationBuilder, response: Response
    ) -> None:
        """Attach trace identifiers to HTTP response headers.

        Parameters
        ----------
        response : Response
            Outgoing response object to annotate.
        """
        trace_identifiers = get_current_trace_identifiers()
        trace_id = trace_identifiers.get("trace_id")
        span_id = trace_identifiers.get("span_id")
        if trace_id:
            response.headers["x-trace-id"] = trace_id
        if span_id:
            response.headers["x-span-id"] = span_id


def create_application(settings: Settings) -> FastAPI:
    """Create and configure an inference FastAPI application.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    fastapi.FastAPI
        Configured application instance.
    """
    return InferenceApplicationBuilder(settings).build()
