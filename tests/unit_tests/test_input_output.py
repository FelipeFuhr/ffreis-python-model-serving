"""Tests for input output."""

from json import loads as json_loads
from typing import Self

from numpy import int64 as np_int64
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from config import Settings
from input_output import format_output, parse_payload

pytestmark = pytest_mark.unit


class TestParsePayload:
    """Test suite for TestParsePayload."""

    def test_csv_parse_basic(self: Self, monkeypatch: pytest_MonkeyPatch) -> None:
        """Verify csv parse basic.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("CSV_HAS_HEADER", "false")
        settings = Settings()
        parsed = parse_payload(b"1,2,3\n4,5,6\n", "text/csv", settings)
        assert parsed.X.shape == (2, 3)
        assert parsed.tensors is None

    def test_json_parse_instances(self: Self) -> None:
        """Verify json parse instances.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        settings = Settings()
        parsed = parse_payload(
            b'{"instances":[[1,2],[3,4]]}', "application/json", settings
        )
        assert parsed.X.shape == (2, 2)

    def test_jsonl_parse_features_key(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Verify jsonl parse features key.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("JSONL_FEATURES_KEY", "features")
        settings = Settings()
        parsed = parse_payload(
            b'{"features":[1,2,3]}\n{"features":[4,5,6]}\n',
            "application/jsonl",
            settings,
        )
        assert parsed.X.shape == (2, 3)

    def test_strict_feature_count(self: Self, monkeypatch: pytest_MonkeyPatch) -> None:
        """Verify strict feature count.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("TABULAR_NUM_FEATURES", "3")
        settings = Settings()
        with pytest_raises(ValueError, match="Feature count mismatch"):
            parse_payload(b"1,2\n3,4\n", "text/csv", settings)

    def test_split_id_and_feature_columns(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Verify split id and feature columns.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("TABULAR_ID_COLUMNS", "0")
        monkeypatch.setenv("TABULAR_FEATURE_COLUMNS", "1:3")
        settings = Settings()
        parsed = parse_payload(b"9,1,2\n8,3,4\n", "text/csv", settings)
        assert parsed.X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert parsed.meta == {"ids": [[9.0], [8.0]], "id_columns": [0]}

    def test_rejects_unsupported_content_type(self: Self) -> None:
        """Verify rejects unsupported content type.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        settings = Settings()
        with pytest_raises(ValueError, match="Unsupported Content-Type"):
            parse_payload(b"<x/>", "application/xml", settings)

    def test_rejects_non_tabular_mode(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Verify rejects non tabular mode.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("INPUT_MODE", "image")
        settings = Settings()
        with pytest_raises(ValueError, match="not implemented"):
            parse_payload(b"1,2", "text/csv", settings)

    def test_onnx_multi_input_builds_tensors(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Verify onnx multi input builds tensors.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv(
            "ONNX_INPUT_MAP_JSON",
            '{"input_ids":"input_ids","attention_mask":"attention_mask"}',
        )
        monkeypatch.setenv(
            "ONNX_INPUT_DTYPE_MAP_JSON",
            '{"input_ids":"int64","attention_mask":"int64"}',
        )
        monkeypatch.setenv("ONNX_DYNAMIC_BATCH", "true")
        settings = Settings()

        payload = (
            b'{"instances":[{"input_ids":[101,102],"attention_mask":[1,1]},'
            b'{"input_ids":[101,0],"attention_mask":[1,0]}]}'
        )
        parsed = parse_payload(payload, "application/json", settings)

        assert parsed.X is None
        assert parsed.tensors is not None
        assert parsed.tensors["input_ids"].dtype == np_int64
        assert parsed.tensors["attention_mask"].shape[0] == 2

    def test_onnx_multi_input_dtype_by_onnx_name(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Verify onnx multi input dtype by onnx name.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("ONNX_INPUT_MAP_JSON", '{"ids":"input_ids"}')
        monkeypatch.setenv("ONNX_INPUT_DTYPE_MAP_JSON", '{"input_ids":"int64"}')
        settings = Settings()

        payload = b'{"instances":[{"ids":[1,2,3]},{"ids":[4,5,6]}]}'
        parsed = parse_payload(payload, "application/json", settings)
        assert parsed.tensors["input_ids"].dtype == np_int64

    def test_onnx_multi_input_requires_record_keys(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Verify onnx multi input requires record keys.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv(
            "ONNX_INPUT_MAP_JSON", '{"ids":"input_ids","mask":"attention_mask"}'
        )
        settings = Settings()
        payload = b'{"instances":[{"ids":[1,2,3]}]}'
        with pytest_raises(ValueError, match="Missing key 'mask'"):
            parse_payload(payload, "application/json", settings)

    def test_onnx_multi_input_requires_object_records(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate onnx multi input requires object records.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("ONNX_INPUT_MAP_JSON", '{"ids":"input_ids"}')
        settings = Settings()
        payload = b'{"instances":[[1,2,3],[4,5,6]]}'
        with pytest_raises(ValueError, match="expects each record to be a JSON object"):
            parse_payload(payload, "application/json", settings)


class TestFormatOutput:
    """Test suite for TestFormatOutput."""

    def test_dict_forces_json(self: Self) -> None:
        """Verify dict forces json.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        settings = Settings()
        body, content_type = format_output(
            {"a": [1]}, accept="text/csv", settings=settings
        )
        assert content_type == "application/json"
        assert json_loads(body) == {"a": [1]}

    def test_csv_from_vector(self: Self) -> None:
        """Verify csv from vector.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        settings = Settings()
        body, content_type = format_output(
            [1, 2, 3], accept="text/csv", settings=settings
        )
        assert content_type == "text/csv"
        assert body == "1\n2\n3"

    def test_csv_from_matrix(self: Self) -> None:
        """Verify csv from matrix.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        settings = Settings()
        body, content_type = format_output(
            [[1, 2], [3, 4]], accept="text/csv", settings=settings
        )
        assert content_type == "text/csv"
        assert body == "1,2\n3,4"

    def test_wrapped_json_when_predictions_only_false(
        self: Self, monkeypatch: pytest_MonkeyPatch
    ) -> None:
        """Validate wrapped json when predictions only false.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("RETURN_PREDICTIONS_ONLY", "false")
        monkeypatch.setenv("JSON_OUTPUT_KEY", "outputs")
        settings = Settings()
        body, content_type = format_output(
            [7, 8], accept="application/json", settings=settings
        )
        assert content_type == "application/json"
        assert json_loads(body) == {"outputs": [7, 8]}
