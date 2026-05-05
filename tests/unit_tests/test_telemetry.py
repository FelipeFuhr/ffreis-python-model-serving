"""Test module."""

from types import SimpleNamespace as types_SimpleNamespace
from typing import Self

from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark

from config import Settings
from telemetry import (
    _parse_headers,
    get_current_trace_identifiers,
    inject_current_trace_context,
    instrument_fastapi_application,
    setup_telemetry,
)

pytestmark = pytest_mark.unit


class _Recorder:
    def __init__(self: Self) -> None:
        self.calls = []

    def add(self: Self, *args: object, **kwargs: object) -> object:
        """Run add.

        Parameters
        ----------
        args : object
            Parameter used by this test scenario.
        kwargs : object
            Parameter used by this test scenario.

        Returns
        -------
        object
            Return value produced by helper logic in this test module.
        """
        self.calls.append((args, kwargs))


def test_parse_headers_handles_empty_and_pairs() -> None:
    """Validate parse headers handles empty and pairs.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    assert _parse_headers("") == {}
    assert _parse_headers("a=b, c=d ,bad, x=1=2") == {"a": "b", "c": "d", "x": "1=2"}


def test_setup_telemetry_returns_false_when_disabled(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate setup telemetry returns false when disabled.

    Parameters
    ----------
    monkeypatch : object
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("OTEL_ENABLED", "false")
    assert setup_telemetry(Settings()) is False


def test_setup_telemetry_returns_false_when_dependencies_missing(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate setup telemetry returns false when dependencies missing.

    Parameters
    ----------
    monkeypatch : object
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")
    import telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "trace", None)
    assert setup_telemetry(Settings()) is False


def test_setup_telemetry_returns_false_without_endpoint(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate setup telemetry returns false without endpoint.

    Parameters
    ----------
    monkeypatch : object
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    import telemetry as telemetry_module

    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(set_tracer_provider=lambda p: None),
    )
    assert setup_telemetry(Settings()) is False


def test_setup_telemetry_happy_path(monkeypatch: pytest_MonkeyPatch) -> None:
    """Validate setup telemetry happy path.

    Parameters
    ----------
    monkeypatch : object
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "a=b")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "3")

    import telemetry as telemetry_module

    trace_rec = _Recorder()
    req_rec = _Recorder()
    httpx_rec = _Recorder()
    log_rec = _Recorder()
    resource_rec = _Recorder()
    provider_rec = _Recorder()

    class FakeProvider:
        """Test suite."""

        def __init__(self: Self, resource: object) -> None:
            provider_rec.add(resource=resource)

        def add_span_processor(self: Self, processor: object) -> object:
            """Run add span processor.

            Parameters
            ----------
            processor : object
                Parameter used by this test scenario.

            Returns
            -------
            object
                Return value produced by helper logic in this test module.
            """
            provider_rec.add(processor=processor)

    class FakeResource:
        """Test suite."""

        @staticmethod
        def create(payload: object) -> object:
            """Run create.

            Parameters
            ----------
            payload : object
                Raw payload bytes used to exercise parsing or invocation behavior.

            Returns
            -------
            object
                Return value produced by helper logic in this test module.
            """
            resource_rec.add(payload=payload)
            return {"resource": payload}

    class FakeExporter:
        """Test suite."""

        def __init__(
            self: Self, endpoint: object, headers: object, timeout: object
        ) -> None:
            self.endpoint = endpoint
            self.headers = headers
            self.timeout = timeout

    class FakeBatch:
        """Test suite."""

        def __init__(self: Self, exporter: object) -> None:
            self.exporter = exporter

    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(
            set_tracer_provider=lambda provider: trace_rec.add(provider=provider)
        ),
    )
    monkeypatch.setattr(telemetry_module, "Resource", FakeResource)
    monkeypatch.setattr(telemetry_module, "TracerProvider", FakeProvider)
    monkeypatch.setattr(telemetry_module, "OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr(telemetry_module, "BatchSpanProcessor", FakeBatch)
    monkeypatch.setattr(
        telemetry_module,
        "RequestsInstrumentor",
        lambda: types_SimpleNamespace(instrument=lambda: req_rec.add()),
    )
    monkeypatch.setattr(
        telemetry_module,
        "HTTPXClientInstrumentor",
        lambda: types_SimpleNamespace(instrument=lambda: httpx_rec.add()),
    )
    monkeypatch.setattr(
        telemetry_module,
        "LoggingInstrumentor",
        lambda: types_SimpleNamespace(
            instrument=lambda set_logging_format: log_rec.add(
                set_logging_format=set_logging_format
            )
        ),
    )

    out = setup_telemetry(Settings())
    assert out is True
    assert trace_rec.calls
    assert resource_rec.calls
    assert provider_rec.calls
    assert req_rec.calls
    assert httpx_rec.calls
    assert log_rec.calls


def test_instrument_fastapi_when_enabled(monkeypatch: pytest_MonkeyPatch) -> None:
    """Validate instrument fastapi when enabled.

    Parameters
    ----------
    monkeypatch : object
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("OTEL_ENABLED", "true")
    import telemetry as telemetry_module

    rec = _Recorder()
    monkeypatch.setattr(
        telemetry_module,
        "FastAPIInstrumentor",
        types_SimpleNamespace(
            instrument_app=lambda application: rec.add(application=application)
        ),
    )
    application = object()
    instrument_fastapi_application(Settings(), application)
    assert rec.calls


def test_instrument_fastapi_noop_when_disabled(monkeypatch: pytest_MonkeyPatch) -> None:
    """Validate instrument fastapi noop when disabled.

    Parameters
    ----------
    monkeypatch : object
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("OTEL_ENABLED", "false")
    import telemetry as telemetry_module

    rec = _Recorder()
    monkeypatch.setattr(
        telemetry_module,
        "FastAPIInstrumentor",
        types_SimpleNamespace(
            instrument_app=lambda application: rec.add(application=application)
        ),
    )
    instrument_fastapi_application(Settings(), object())
    assert rec.calls == []


def test_get_current_trace_identifiers_returns_empty_without_trace(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate current trace identifiers are empty without trace module."""
    import telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "trace", None)
    assert get_current_trace_identifiers() == {}


def test_get_current_trace_identifiers_returns_ids(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate current trace identifiers are rendered as lowercase hex."""
    import telemetry as telemetry_module

    span_context = types_SimpleNamespace(
        trace_id=0x1234,
        span_id=0xABCD,
        is_valid=True,
    )
    fake_span = types_SimpleNamespace(get_span_context=lambda: span_context)
    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(get_current_span=lambda: fake_span),
    )

    assert get_current_trace_identifiers() == {
        "trace_id": "00000000000000000000000000001234",
        "span_id": "000000000000abcd",
    }


def test_inject_current_trace_context_uses_propagator(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate active trace context injection populates provided carrier."""
    import telemetry as telemetry_module

    def _inject(carrier: dict[str, str]) -> None:
        carrier["traceparent"] = "00-abc-def-01"

    monkeypatch.setattr(
        telemetry_module,
        "propagate",
        types_SimpleNamespace(inject=_inject),
    )

    carrier: dict[str, str] = {}
    assert inject_current_trace_context(carrier) == {"traceparent": "00-abc-def-01"}


def test_setup_telemetry_happy_path_without_httpx_instrumentor(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate setup succeeds when HTTPX instrumentation is unavailable."""
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")
    import telemetry as telemetry_module

    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(set_tracer_provider=lambda provider: None),
    )
    monkeypatch.setattr(
        telemetry_module,
        "Resource",
        types_SimpleNamespace(create=lambda payload: {"resource": payload}),
    )
    monkeypatch.setattr(
        telemetry_module,
        "TracerProvider",
        lambda *, resource: types_SimpleNamespace(add_span_processor=lambda p: None),
    )
    monkeypatch.setattr(
        telemetry_module,
        "OTLPSpanExporter",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        telemetry_module,
        "BatchSpanProcessor",
        lambda exporter: object(),
    )
    monkeypatch.setattr(
        telemetry_module,
        "RequestsInstrumentor",
        lambda: types_SimpleNamespace(instrument=lambda: None),
    )
    monkeypatch.setattr(telemetry_module, "HTTPXClientInstrumentor", None)
    monkeypatch.setattr(
        telemetry_module,
        "LoggingInstrumentor",
        lambda: types_SimpleNamespace(instrument=lambda set_logging_format: None),
    )

    assert setup_telemetry(Settings()) is True


def test_get_current_trace_identifiers_returns_empty_for_invalid_context(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate invalid span contexts produce empty identifier mapping."""
    import telemetry as telemetry_module

    span_context = types_SimpleNamespace(trace_id=1, span_id=2, is_valid=False)
    fake_span = types_SimpleNamespace(get_span_context=lambda: span_context)
    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(get_current_span=lambda: fake_span),
    )
    assert get_current_trace_identifiers() == {}


def test_inject_current_trace_context_returns_input_when_propagator_missing(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate context injection is a no-op when propagate is unavailable."""
    import telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "propagate", None)
    carrier: dict[str, str] = {"x-existing": "1"}
    assert inject_current_trace_context(carrier) == {"x-existing": "1"}


def test_get_tracer_returns_value_when_trace_module_present(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate get_tracer delegates to trace.get_tracer when available."""
    import telemetry as telemetry_module

    sentinel = object()
    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(get_tracer=lambda name: sentinel),
    )
    assert telemetry_module.get_tracer("any") is sentinel


def test_setup_telemetry_returns_false_without_endpoint_when_deps_present(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Validate explicit no-endpoint branch when dependencies are present."""
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    import telemetry as telemetry_module

    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types_SimpleNamespace(set_tracer_provider=lambda provider: None),
    )
    monkeypatch.setattr(telemetry_module, "Resource", object())
    monkeypatch.setattr(telemetry_module, "TracerProvider", object())
    monkeypatch.setattr(telemetry_module, "OTLPSpanExporter", object())
    monkeypatch.setattr(telemetry_module, "BatchSpanProcessor", object())
    monkeypatch.setattr(telemetry_module, "RequestsInstrumentor", object())
    monkeypatch.setattr(telemetry_module, "LoggingInstrumentor", object())
    assert setup_telemetry(Settings()) is False
