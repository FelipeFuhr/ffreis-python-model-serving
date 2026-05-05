"""Tests for serving."""

from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

pytestmark = pytest_mark.unit


def test_main_execs_gunicorn(monkeypatch: pytest_MonkeyPatch) -> None:
    """Verify main execs gunicorn.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    from serving import main as serving_module_main
    from serving import os as serving_module_os

    seen = {}

    def fake_execvp(cmd: object, argv: object) -> object:
        """Run fake execvp.

        Parameters
        ----------
        cmd : object
            Parameter used by this test scenario.
        argv : object
            Parameter used by this test scenario.

        Returns
        -------
        object
            Return value produced by helper logic in this test module.
        """
        seen["cmd"] = cmd
        seen["argv"] = argv
        raise RuntimeError("stop")

    monkeypatch.setattr(serving_module_os, "execvp", fake_execvp)
    with pytest_raises(RuntimeError, match="stop"):
        serving_module_main()

    assert seen["cmd"] == "gunicorn"
    assert seen["argv"] == [
        "gunicorn",
        "-c",
        "python:gunicorn_configuration",
        "serving:application",
    ]


def test_module_exposes_asgi_app() -> None:
    """Verify module exposes asgi app.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    from serving import application as serving_module_application

    assert serving_module_application is not None
