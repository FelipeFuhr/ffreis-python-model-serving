"""Telemetry setup and instrumentation helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib import import_module as importlib_import_module
from logging import getLogger as logging_getLogger
from typing import Protocol, cast

from fastapi import FastAPI

from config import Settings

log = logging_getLogger("otel")


class SpanContextProtocol(Protocol):
    """Span context fields used for correlation."""

    trace_id: int
    span_id: int
    is_valid: bool


class SpanProtocol(Protocol):
    """Current-span protocol used in correlation helpers."""

    def get_span_context(self: SpanProtocol) -> SpanContextProtocol:
        """Return span context."""


class TracerProtocol(Protocol):
    """OpenTelemetry tracer protocol."""

    def start_as_current_span(
        self: TracerProtocol, name: str
    ) -> AbstractContextManager[SpanProtocol]:
        """Start span context manager."""


class ResourceInstanceProtocol(Protocol):
    """Marker protocol for resource instances."""


class ExporterProtocol(Protocol):
    """Marker protocol for OTLP exporters."""


class BatchProcessorProtocol(Protocol):
    """Marker protocol for batch span processors."""


class TraceModuleProtocol(Protocol):
    """Subset of `opentelemetry.trace` module used by this service."""

    def get_tracer(self: TraceModuleProtocol, name: str) -> TracerProtocol:
        """Return tracer by name."""

    def set_tracer_provider(
        self: TraceModuleProtocol, provider: TracerProviderProtocol
    ) -> None:
        """Set tracer provider."""

    def get_current_span(self: TraceModuleProtocol) -> SpanProtocol:
        """Return current span."""


class ResourceProtocol(Protocol):
    """OpenTelemetry resource factory protocol."""

    @staticmethod
    def create(payload: dict[str, str]) -> ResourceInstanceProtocol:
        """Create resource instance."""


class TracerProviderProtocol(Protocol):
    """Trace provider methods used by setup."""

    def add_span_processor(
        self: TracerProviderProtocol, processor: BatchProcessorProtocol
    ) -> None:
        """Attach span processor."""


class TracerProviderFactoryProtocol(Protocol):
    """Trace provider constructor protocol."""

    def __call__(
        self: TracerProviderFactoryProtocol, *, resource: ResourceInstanceProtocol
    ) -> TracerProviderProtocol:
        """Create provider instance."""


class ExporterFactoryProtocol(Protocol):
    """Exporter constructor protocol."""

    def __call__(
        self: ExporterFactoryProtocol,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
    ) -> ExporterProtocol:
        """Create exporter."""


class BatchProcessorFactoryProtocol(Protocol):
    """Batch processor constructor protocol."""

    def __call__(
        self: BatchProcessorFactoryProtocol, exporter: ExporterProtocol
    ) -> BatchProcessorProtocol:
        """Create batch span processor."""


class InstrumentorProtocol(Protocol):
    """Protocol for request/logging instrumentors."""

    def instrument(
        self: InstrumentorProtocol, set_logging_format: bool = False
    ) -> None:
        """Enable instrumentation."""


class InstrumentorFactoryProtocol(Protocol):
    """Instrumentation factory protocol."""

    def __call__(self: InstrumentorFactoryProtocol) -> InstrumentorProtocol:
        """Create instrumentor instance."""


class PropagatorModuleProtocol(Protocol):
    """OpenTelemetry propagator module protocol."""

    def inject(self: PropagatorModuleProtocol, carrier: dict[str, str]) -> None:
        """Inject active context into a carrier mapping."""


class FastApiInstrumentorProtocol(Protocol):
    """FastAPI instrumentor protocol."""

    @staticmethod
    def instrument_app(application: FastAPI) -> None:
        """Instrument FastAPI application."""


trace: TraceModuleProtocol | None = None
propagate: PropagatorModuleProtocol | None = None
OTLPSpanExporter: ExporterFactoryProtocol | None = None
FastAPIInstrumentor: FastApiInstrumentorProtocol | None = None
LoggingInstrumentor: InstrumentorFactoryProtocol | None = None
RequestsInstrumentor: InstrumentorFactoryProtocol | None = None
HTTPXClientInstrumentor: InstrumentorFactoryProtocol | None = None
Resource: ResourceProtocol | None = None
TracerProvider: TracerProviderFactoryProtocol | None = None
BatchSpanProcessor: BatchProcessorFactoryProtocol | None = None


def _load_optional_telemetry_components() -> None:
    """Load optional telemetry dependencies if available."""
    global trace
    global propagate
    global OTLPSpanExporter
    global FastAPIInstrumentor
    global LoggingInstrumentor
    global RequestsInstrumentor
    global HTTPXClientInstrumentor
    global Resource
    global TracerProvider
    global BatchSpanProcessor

    try:
        trace = cast(
            TraceModuleProtocol, importlib_import_module("opentelemetry").trace
        )
        propagate = cast(
            PropagatorModuleProtocol,
            importlib_import_module("opentelemetry.propagate"),
        )
        OTLPSpanExporter = cast(
            ExporterFactoryProtocol,
            importlib_import_module(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter"
            ).OTLPSpanExporter,
        )
        FastAPIInstrumentor = cast(
            FastApiInstrumentorProtocol,
            importlib_import_module(
                "opentelemetry.instrumentation.fastapi"
            ).FastAPIInstrumentor,
        )
        LoggingInstrumentor = cast(
            InstrumentorFactoryProtocol,
            importlib_import_module(
                "opentelemetry.instrumentation.logging"
            ).LoggingInstrumentor,
        )
        RequestsInstrumentor = cast(
            InstrumentorFactoryProtocol,
            importlib_import_module(
                "opentelemetry.instrumentation.requests"
            ).RequestsInstrumentor,
        )
        HTTPXClientInstrumentor = cast(
            InstrumentorFactoryProtocol,
            importlib_import_module(
                "opentelemetry.instrumentation.httpx"
            ).HTTPXClientInstrumentor,
        )
        Resource = cast(
            ResourceProtocol,
            importlib_import_module("opentelemetry.sdk.resources").Resource,
        )
        TracerProvider = cast(
            TracerProviderFactoryProtocol,
            importlib_import_module("opentelemetry.sdk.trace").TracerProvider,
        )
        BatchSpanProcessor = cast(
            BatchProcessorFactoryProtocol,
            importlib_import_module(
                "opentelemetry.sdk.trace.export"
            ).BatchSpanProcessor,
        )
    except Exception:  # pragma: no cover - optional dependency
        trace = None
        propagate = None
        OTLPSpanExporter = None
        FastAPIInstrumentor = None
        LoggingInstrumentor = None
        RequestsInstrumentor = None
        HTTPXClientInstrumentor = None
        Resource = None
        TracerProvider = None
        BatchSpanProcessor = None


_load_optional_telemetry_components()


def get_tracer(name: str) -> TracerProtocol | None:
    """Return configured tracer when telemetry dependency is present."""
    if trace is None:
        return None
    return trace.get_tracer(name)


def _parse_headers(s: str) -> dict[str, str]:
    """Parse OTLP headers from comma-separated key-value string."""
    parsed_headers = {}
    for part in [p.strip() for p in (s or "").split(",") if p.strip()]:
        if "=" in part:
            key, value = part.split("=", 1)
            parsed_headers[key.strip()] = value.strip()
    return parsed_headers


def setup_telemetry(settings: Settings) -> bool:
    """Configure OpenTelemetry exporters and instrumentors."""
    if not settings.otel_enabled:
        log.info("OTEL disabled")
        return False

    if (
        trace is None
        or Resource is None
        or TracerProvider is None
        or OTLPSpanExporter is None
        or BatchSpanProcessor is None
        or RequestsInstrumentor is None
        or LoggingInstrumentor is None
    ):
        log.info("OTEL dependencies unavailable; exporter disabled.")
        return False

    if not settings.otel_endpoint:
        log.info("OTEL enabled but no endpoint configured; exporter disabled.")
        return False

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.service_version,
            "deployment.environment": settings.deployment_env,
            "cloud.provider": "aws",
            "cloud.platform": "aws_sagemaker",
        }
    )
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(
        endpoint=settings.otel_endpoint,
        headers=_parse_headers(settings.otel_headers),
        timeout=settings.otel_timeout_s,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    RequestsInstrumentor().instrument()
    if HTTPXClientInstrumentor is not None:
        HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=True)

    log.info("OTEL exporter configured: %s", settings.otel_endpoint)
    return True


def instrument_fastapi_application(settings: Settings, application: FastAPI) -> None:
    """Instrument a FastAPI application when telemetry is enabled."""
    if settings.otel_enabled and FastAPIInstrumentor is not None:
        FastAPIInstrumentor.instrument_app(application)


def get_current_trace_identifiers() -> dict[str, str]:
    """Return current trace/span identifiers when available."""
    if trace is None:
        return {}
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return {}
    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }


def inject_current_trace_context(
    carrier: dict[str, str] | None = None,
) -> dict[str, str]:
    """Inject active trace context headers into a carrier."""
    target_carrier = carrier or {}
    if propagate is None:
        return target_carrier
    propagate.inject(target_carrier)
    return target_carrier
