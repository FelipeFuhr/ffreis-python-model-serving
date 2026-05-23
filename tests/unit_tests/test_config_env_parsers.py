# ruff: noqa: D101, D102
"""Direct tests for config env parsers and Settings construction.

The audit called out that env parsing has no malformed-input coverage despite
running in production via Gunicorn. These tests pin the error-handling
contract explicitly.
"""

from __future__ import annotations

import pytest

from config import Settings, _env_bool, _env_float, _env_int, _env_str

pytestmark = pytest.mark.unit


class TestEnvBool:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("y", True),
            ("on", True),
            ("  on  ", True),
            ("0", False),
            ("false", False),
            ("FALSE", False),
            ("no", False),
            ("n", False),
            ("off", False),
            ("", False),
            ("nope", False),
            ("garbage", False),
            ("2", False),
        ],
    )
    def test_truthy_set(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        monkeypatch.setenv("X", raw)
        assert _env_bool("X", default=not expected) is expected

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_BOOL", raising=False)
        assert _env_bool("UNSET_BOOL", default=True) is True
        assert _env_bool("UNSET_BOOL", default=False) is False


class TestEnvInt:
    def test_valid_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "42")
        assert _env_int("X", default=0) == 42

    def test_negative_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "-7")
        assert _env_int("X", default=0) == -7

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_INT", raising=False)
        assert _env_int("UNSET_INT", default=99) == 99

    @pytest.mark.parametrize("raw", ["abc", "1.5", "", "   ", "1,000", "0x10"])
    def test_malformed_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("X", raw)
        with pytest.raises(ValueError):
            _env_int("X", default=0)


class TestEnvFloat:
    def test_valid_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "3.14")
        assert _env_float("X", default=0.0) == pytest.approx(3.14)

    def test_integer_string_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "42")
        assert _env_float("X", default=0.0) == pytest.approx(42.0)

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_FLOAT", raising=False)
        assert _env_float("UNSET_FLOAT", default=1.5) == pytest.approx(1.5)

    @pytest.mark.parametrize("raw", ["abc", "", " ", "1.5.0", "--3"])
    def test_malformed_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("X", raw)
        with pytest.raises(ValueError):
            _env_float("X", default=0.0)


class TestEnvStr:
    def test_returns_string_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "hello")
        assert _env_str("X", default="x") == "hello"

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_STR", raising=False)
        assert _env_str("UNSET_STR", default="fallback") == "fallback"

    def test_empty_string_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # _env_str preserves explicit empty strings; only None falls through.
        monkeypatch.setenv("X", "")
        assert _env_str("X", default="fallback") == ""


class TestSettingsRoundTrip:
    """Settings is frozen, so each fixture builds a fresh instance."""

    def test_defaults_when_environment_clean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clear common vars that might be set in CI.
        for name in (
            "PORT",
            "LOG_LEVEL",
            "INPUT_MODE",
            "OUTPUT_MODE",
            "MAX_BODY_BYTES",
            "PROMETHEUS_ENABLED",
            "OTEL_ENABLED",
        ):
            monkeypatch.delenv(name, raising=False)
        s = Settings()
        assert s.port == 8080
        assert s.log_level == "INFO"
        assert s.input_mode == "tabular"
        assert s.output_mode == "predictions"
        assert s.max_body_bytes == 6 * 1024 * 1024
        assert s.prometheus_enabled is True
        assert s.otel_enabled is True

    def test_settings_picks_up_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        monkeypatch.setenv("MAX_INFLIGHT", "32")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
        s = Settings()
        assert s.port == 9000
        assert s.log_level == "DEBUG"  # uppercased per default_factory
        assert s.max_inflight == 32
        assert s.prometheus_enabled is False

    def test_malformed_max_body_bytes_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MAX_BODY_BYTES", "not-a-number")
        with pytest.raises(ValueError):
            Settings()

    def test_malformed_otel_timeout_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "ten")
        with pytest.raises(ValueError):
            Settings()

    def test_lower_case_normalized_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INPUT_MODE", "  JSON  ")
        monkeypatch.setenv("MODEL_TYPE", "ONNX")
        s = Settings()
        assert s.input_mode == "json"
        assert s.model_type == "onnx"
