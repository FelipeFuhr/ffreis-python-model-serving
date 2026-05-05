"""Unit tests for OpenAPI contract loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as types_SimpleNamespace
from typing import Self

from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark

from openapi_contract import load_openapi_contract

pytestmark = pytest_mark.unit


class TestOpenApiContract:
    """Test suite for ``load_openapi_contract``."""

    def teardown_method(self: Self) -> None:
        """Clear cache between tests."""
        load_openapi_contract.cache_clear()

    def test_returns_none_when_yaml_dependency_is_unavailable(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Return ``None`` when PyYAML import is unavailable."""
        monkeypatch.setattr("openapi_contract.yaml", None)
        assert load_openapi_contract() is None

    def test_returns_none_when_openapi_file_is_missing(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Return ``None`` when docs/openapi.yaml does not exist."""
        fake_yaml = types_SimpleNamespace(safe_load=lambda _raw: {"openapi": "3.1.0"})
        monkeypatch.setattr("openapi_contract.yaml", fake_yaml)
        monkeypatch.setattr(Path, "exists", lambda _self: False)
        assert load_openapi_contract() is None

    def test_returns_none_when_yaml_is_not_object(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Return ``None`` when loaded contract is not a mapping."""
        fake_yaml = types_SimpleNamespace(
            safe_load=lambda _raw: ["not", "an", "object"]
        )
        monkeypatch.setattr("openapi_contract.yaml", fake_yaml)
        monkeypatch.setattr(Path, "exists", lambda _self: True)
        monkeypatch.setattr(Path, "read_text", lambda _self, encoding="utf-8": "x")
        assert load_openapi_contract() is None

    def test_returns_mapping_when_yaml_loads_object(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Return parsed object when contract exists and is valid YAML mapping."""
        fake_yaml = types_SimpleNamespace(safe_load=lambda _raw: {"openapi": "3.1.0"})
        monkeypatch.setattr("openapi_contract.yaml", fake_yaml)
        monkeypatch.setattr(Path, "exists", lambda _self: True)
        monkeypatch.setattr(Path, "read_text", lambda _self, encoding="utf-8": "x")
        loaded = load_openapi_contract()
        assert loaded == {"openapi": "3.1.0"}
