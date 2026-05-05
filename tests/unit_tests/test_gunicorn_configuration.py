"""Tests for gunicorn configuration."""

from importlib import reload as importlib_reload

from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark

pytestmark = pytest_mark.unit


def test_gunicorn_configuration_uses_settings_from_env(
    monkeypatch: pytest_MonkeyPatch,
) -> None:
    """Verify gunicorn configuration uses settings from env.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("GUNICORN_WORKERS", "3")
    monkeypatch.setenv("GUNICORN_THREADS", "9")
    monkeypatch.setenv("GUNICORN_TIMEOUT", "88")
    monkeypatch.setenv("GUNICORN_GRACEFUL_TIMEOUT", "44")
    monkeypatch.setenv("GUNICORN_KEEPALIVE", "7")

    import gunicorn_configuration as gunicorn_configuration

    gunicorn_settings_module = importlib_reload(gunicorn_configuration)
    assert gunicorn_settings_module.bind == "0.0.0.0:9999"
    assert gunicorn_settings_module.workers == 3
    assert gunicorn_settings_module.threads == 9
    assert gunicorn_settings_module.timeout == 88
    assert gunicorn_settings_module.graceful_timeout == 44
    assert gunicorn_settings_module.keepalive == 7
    assert gunicorn_settings_module.worker_class == "uvicorn.workers.UvicornWorker"
    assert gunicorn_settings_module.accesslog == "-"
    assert gunicorn_settings_module.errorlog == "-"
