"""Tests for conftest."""

from collections.abc import AsyncIterator, Iterator
from typing import Self

from fastapi import FastAPI
from httpx import ASGITransport as httpx_ASGITransport
from httpx import AsyncClient as httpx_AsyncClient
from pytest import MonkeyPatch, fixture
from pytest_asyncio import fixture as asyncio_fixture

from application import create_application
from config import Settings
from parsed_types import ParsedInput


class DummyAdapter:
    """Test suite for DummyAdapter."""

    def __init__(self: Self, mode: str = "list") -> None:
        """Initialize adapter mode for tests.

        Parameters
        ----------
        mode : str
            Parameter used by this test scenario.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        self.mode = mode

    def is_ready(self: Self) -> bool:
        """Report adapter readiness for tests.

        Returns
        -------
        bool
            Return value produced by helper logic in this test module.
        """
        return True

    def predict(self: Self, parsed_input: ParsedInput) -> object:
        """Return deterministic outputs for fixture-backed tests.

        Parameters
        ----------
        parsed_input : ParsedInput
            Parsed input payload object passed to adapter methods.

        Returns
        -------
        object
            Return value produced by helper logic in this test module.
        """
        if self.mode == "list":
            if parsed_input.X is not None:
                n = parsed_input.X.shape[0]
            elif parsed_input.tensors:
                n = next(iter(parsed_input.tensors.values())).shape[0]
            else:
                n = 1
            return [0] * int(n)
        if self.mode == "dict":
            return {"logits": [[1.0, 2.0]], "proba": [[0.1, 0.9]]}
        raise ValueError("Unknown mode")


@fixture
def base_env(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Set baseline environment variables for application tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    Iterator[None]
        Return value produced by helper logic in this test module.
    """
    monkeypatch.setenv("INPUT_MODE", "tabular")
    monkeypatch.setenv("DEFAULT_CONTENT_TYPE", "application/json")
    monkeypatch.setenv("DEFAULT_ACCEPT", "application/json")
    monkeypatch.setenv("CSV_DELIMITER", ",")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("TABULAR_DTYPE", "float32")
    monkeypatch.setenv("TABULAR_NUM_FEATURES", "0")
    monkeypatch.setenv("MAX_BODY_BYTES", "1000000")
    monkeypatch.setenv("MAX_RECORDS", "1000")
    monkeypatch.setenv("MAX_INFLIGHT", "4")
    monkeypatch.setenv("ACQUIRE_TIMEOUT_S", "0.2")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")
    monkeypatch.setenv("PROMETHEUS_PATH", "/metrics")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("SM_MODEL_DIR", "/opt/ml/model")
    yield


def _make_client(app: FastAPI) -> httpx_AsyncClient:
    """Create an ASGI test client for the given FastAPI application.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance under test.

    Returns
    -------
    httpx.AsyncClient
        Return value produced by helper logic in this test module.
    """
    transport = httpx_ASGITransport(app=app)
    return httpx_AsyncClient(transport=transport, base_url="http://test")


@fixture
def app_list_adapter(monkeypatch: MonkeyPatch, base_env: None) -> FastAPI:
    """Build app fixture backed by list-style dummy predictions.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to configure environment and runtime hooks.
    base_env : None
        Fixture that applies baseline environment variables used by application tests.

    Returns
    -------
    FastAPI
        Return value produced by helper logic in this test module.
    """
    import application as application_module

    monkeypatch.setattr(
        application_module, "load_adapter", lambda settings: DummyAdapter(mode="list")
    )
    return create_application(Settings())


@fixture
def app_dict_adapter(monkeypatch: MonkeyPatch, base_env: None) -> FastAPI:
    """Build app fixture backed by dict-style dummy predictions.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to configure environment and runtime hooks.
    base_env : None
        Fixture that applies baseline environment variables used by application tests.

    Returns
    -------
    FastAPI
        Return value produced by helper logic in this test module.
    """
    import application as application_module

    monkeypatch.setattr(
        application_module, "load_adapter", lambda settings: DummyAdapter(mode="dict")
    )
    return create_application(Settings())


@asyncio_fixture
async def client_list(app_list_adapter: FastAPI) -> AsyncIterator[httpx_AsyncClient]:
    """Provide async client fixture for list-adapter application.

    Parameters
    ----------
    app_list_adapter : FastAPI
        FastAPI application fixture configured with list-style adapter outputs.

    Returns
    -------
    AsyncIterator[httpx.AsyncClient]
        Return value produced by helper logic in this test module.
    """
    async with _make_client(app_list_adapter) as client:
        yield client


@asyncio_fixture
async def client_dict(app_dict_adapter: FastAPI) -> AsyncIterator[httpx_AsyncClient]:
    """Provide async client fixture for dict-adapter application.

    Parameters
    ----------
    app_dict_adapter : FastAPI
        FastAPI application fixture configured with dictionary-style adapter outputs.

    Returns
    -------
    AsyncIterator[httpx.AsyncClient]
        Return value produced by helper logic in this test module.
    """
    async with _make_client(app_dict_adapter) as client:
        yield client
