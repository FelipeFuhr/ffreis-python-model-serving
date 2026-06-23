# ruff: noqa: D101, D102
"""Unit tests for registry-pull startup-mode selection logic (B4).

These tests cover the *decision* logic in ``pull_model_from_registry`` only —
no real registry backend or ONNX file is touched. The registry client and
the ``_pull_model_inner`` helper are mocked so the suite runs without the
``ml_registry`` optional extra installed.

Scenarios covered:
- All three registry env vars set + package installed → enters registry path.
- Registry vars set + ``ml_registry`` NOT installed → raises clear RuntimeError.
- Only ``MODEL_PATH`` / ``SM_MODEL_DIR`` set (no registry vars) → returns None
  (legacy path, caller uses the local dir unchanged).
- Only backend set without URI → raises RuntimeError.
- Only URI set without backend → raises RuntimeError.
- All registry vars set but ``MODEL_NAME`` missing → raises RuntimeError.
- ``MODEL_STAGE`` defaults to ``"production"`` when not explicitly set.
- ``MODEL_REGISTRY_TIMEOUT`` defaults to 30 when not explicitly set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from config import Settings

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────


def _settings(**overrides: Any) -> Settings:  # noqa: ANN401
    """Build a Settings instance with registry vars cleared by default.

    Any keyword arg directly sets the corresponding Settings field.
    """
    defaults: dict[str, Any] = {
        "model_registry_backend": "",
        "model_registry_uri": "",
        "model_name": "",
        "model_stage": "production",
        "model_registry_timeout": 30,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ── startup-mode selection ────────────────────────────────────────────────────


class TestStartupModeSelection:
    """``pull_model_from_registry`` routes to the correct startup mode."""

    def test_returns_none_when_no_registry_vars(self) -> None:
        """Legacy path: no registry env vars → returns None immediately."""
        from registry_pull import pull_model_from_registry

        settings = _settings()
        result = pull_model_from_registry(settings)
        assert result is None

    def test_registry_mode_entered_when_all_vars_set(self, tmp_path: Path) -> None:
        """Registry vars + package installed → enters registry pull path."""
        from registry_pull import pull_model_from_registry

        fake_temp = str(tmp_path / "pulled-model")

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="/db/registry.db",
            model_name="my-model",
            model_stage="production",
            model_registry_timeout=30,
        )

        with (
            patch("registry_pull._require_ml_registry"),
            patch(
                "registry_pull._pull_model_inner", return_value=fake_temp
            ) as mock_inner,
        ):
            result = pull_model_from_registry(settings)

        assert result == fake_temp
        mock_inner.assert_called_once_with(
            "sqlite", "/db/registry.db", "my-model", "production"
        )

    def test_clear_error_when_registry_package_not_installed(self) -> None:
        """Registry vars set but ``ml_registry`` absent → RuntimeError with hint."""
        from registry_pull import pull_model_from_registry

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="/db/registry.db",
            model_name="my-model",
        )

        def _raise_import(*_: Any, **__: Any) -> None:  # noqa: ANN401
            raise RuntimeError(
                "MODEL_REGISTRY_BACKEND is set but the 'ml_registry' package is not "
                "installed. Install with: uv sync --extra registry"
            )

        with patch("registry_pull._require_ml_registry", side_effect=_raise_import):
            with pytest.raises(RuntimeError, match="ml_registry.*not installed"):
                pull_model_from_registry(settings)

    def test_uses_model_path_when_only_local_dir_set(self) -> None:
        """``SM_MODEL_DIR`` only (no registry vars) → returns None (legacy path)."""
        from registry_pull import pull_model_from_registry

        settings = _settings(model_dir="/opt/ml/model")
        result = pull_model_from_registry(settings)
        assert result is None

    def test_raises_when_backend_set_without_uri(self) -> None:
        """Partial registry config (backend but no URI) → RuntimeError."""
        from registry_pull import pull_model_from_registry

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="",
            model_name="my-model",
        )
        with pytest.raises(
            RuntimeError, match="Both MODEL_REGISTRY_BACKEND and MODEL_REGISTRY_URI"
        ):
            pull_model_from_registry(settings)

    def test_raises_when_uri_set_without_backend(self) -> None:
        """Partial registry config (URI but no backend) → RuntimeError."""
        from registry_pull import pull_model_from_registry

        settings = _settings(
            model_registry_backend="",
            model_registry_uri="sqlite:///db/registry.db",
            model_name="my-model",
        )
        with pytest.raises(
            RuntimeError, match="Both MODEL_REGISTRY_BACKEND and MODEL_REGISTRY_URI"
        ):
            pull_model_from_registry(settings)

    def test_raises_when_model_name_missing(self) -> None:
        """Registry vars set but MODEL_NAME empty → RuntimeError."""
        from registry_pull import pull_model_from_registry

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="/db/registry.db",
            model_name="",
        )
        with patch("registry_pull._require_ml_registry"):
            with pytest.raises(RuntimeError, match="MODEL_NAME must be set"):
                pull_model_from_registry(settings)

    def test_model_stage_defaults_to_production(self, tmp_path: Path) -> None:
        """``MODEL_STAGE`` defaults to ``"production"`` when not overridden."""
        from registry_pull import pull_model_from_registry

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="/db/registry.db",
            model_name="my-model",
            # model_stage not set → uses default "production"
        )
        assert settings.model_stage == "production"

        with (
            patch("registry_pull._require_ml_registry"),
            patch(
                "registry_pull._pull_model_inner", return_value=str(tmp_path)
            ) as mock_inner,
        ):
            pull_model_from_registry(settings)

        _, _, _, stage_arg = mock_inner.call_args.args
        assert stage_arg == "production"

    def test_model_stage_custom_value_forwarded(self, tmp_path: Path) -> None:
        """Custom ``MODEL_STAGE`` value is forwarded to ``_pull_model_inner``."""
        from registry_pull import pull_model_from_registry

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="/db/registry.db",
            model_name="my-model",
            model_stage="staging",
        )

        with (
            patch("registry_pull._require_ml_registry"),
            patch(
                "registry_pull._pull_model_inner", return_value=str(tmp_path)
            ) as mock_inner,
        ):
            pull_model_from_registry(settings)

        _, _, _, stage_arg = mock_inner.call_args.args
        assert stage_arg == "staging"

    def test_model_registry_timeout_defaults_to_30(self) -> None:
        """``MODEL_REGISTRY_TIMEOUT`` defaults to 30 seconds."""
        settings = _settings()
        assert settings.model_registry_timeout == 30

    def test_model_registry_timeout_applied_to_future(self, tmp_path: Path) -> None:
        """Custom timeout is forwarded to ``future.result(timeout=...)``.

        We patch ``ThreadPoolExecutor`` so we can inspect the ``timeout``
        kwarg passed to ``future.result()``.
        """
        from concurrent.futures import Future

        from registry_pull import pull_model_from_registry

        mock_future: MagicMock = MagicMock(spec=Future)
        mock_future.result.return_value = str(tmp_path)
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future

        settings = _settings(
            model_registry_backend="sqlite",
            model_registry_uri="/db/registry.db",
            model_name="my-model",
            model_registry_timeout=15,
        )

        with (
            patch("registry_pull._require_ml_registry"),
            patch("registry_pull.ThreadPoolExecutor", return_value=mock_pool),
        ):
            pull_model_from_registry(settings)

        mock_future.result.assert_called_once_with(timeout=15)
