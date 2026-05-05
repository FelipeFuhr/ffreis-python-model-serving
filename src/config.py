"""Application configuration objects."""

from __future__ import annotations

from os import getenv as os_getenv

from pydantic import BaseModel, ConfigDict, Field


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from environment variables.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool
        Fallback value when the variable is not set.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    value = os_getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variables.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Fallback value when the variable is not set.

    Returns
    -------
    int
        Parsed integer value.
    """
    value = os_getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    """Read a float from environment variables.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Fallback value when the variable is not set.

    Returns
    -------
    float
        Parsed float value.
    """
    value = os_getenv(name)
    return default if value is None else float(value)


def _env_str(name: str, default: str) -> str:
    """Read a string from environment variables.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Fallback value when the variable is not set.

    Returns
    -------
    str
        Parsed string value.
    """
    value = os_getenv(name)
    return default if value is None else value


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    port: int = Field(default_factory=lambda: _env_int("PORT", 8080))
    log_level: str = Field(
        default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper()
    )
    service_name: str = Field(
        default_factory=lambda: _env_str("SERVICE_NAME", "model-serving-universal")
    )
    service_version: str = Field(
        default_factory=lambda: _env_str("SERVICE_VERSION", "dev")
    )
    deployment_env: str = Field(
        default_factory=lambda: _env_str("DEPLOYMENT_ENV", "local")
    )

    model_dir: str = Field(
        default_factory=lambda: _env_str("SM_MODEL_DIR", "/opt/ml/model")
    )
    model_type: str = Field(
        default_factory=lambda: _env_str("MODEL_TYPE", "").strip().lower()
    )
    model_filename: str = Field(
        default_factory=lambda: _env_str("MODEL_FILENAME", "").strip()
    )

    input_mode: str = Field(
        default_factory=lambda: _env_str("INPUT_MODE", "tabular").strip().lower()
    )
    output_mode: str = Field(
        default_factory=lambda: _env_str("OUTPUT_MODE", "predictions").strip().lower()
    )

    default_content_type: str = Field(
        default_factory=lambda: _env_str("DEFAULT_CONTENT_TYPE", "application/json")
    )
    default_accept: str = Field(
        default_factory=lambda: _env_str("DEFAULT_ACCEPT", "application/json")
    )

    tabular_dtype: str = Field(
        default_factory=lambda: _env_str("TABULAR_DTYPE", "float32").strip().lower()
    )
    csv_delimiter: str = Field(default_factory=lambda: _env_str("CSV_DELIMITER", ","))
    csv_has_header: str = Field(
        default_factory=lambda: _env_str("CSV_HAS_HEADER", "auto").strip().lower()
    )
    csv_skip_blank_lines: bool = Field(
        default_factory=lambda: _env_bool("CSV_SKIP_BLANK_LINES", True)
    )

    json_key_instances: str = Field(
        default_factory=lambda: _env_str("JSON_KEY_INSTANCES", "instances")
    )
    jsonl_features_key: str = Field(
        default_factory=lambda: _env_str("JSONL_FEATURES_KEY", "features")
    )

    tabular_id_columns: str = Field(
        default_factory=lambda: _env_str("TABULAR_ID_COLUMNS", "").strip()
    )
    tabular_feature_columns: str = Field(
        default_factory=lambda: _env_str("TABULAR_FEATURE_COLUMNS", "").strip()
    )

    predictions_only: bool = Field(
        default_factory=lambda: _env_bool("RETURN_PREDICTIONS_ONLY", True)
    )
    json_output_key: str = Field(
        default_factory=lambda: _env_str("JSON_OUTPUT_KEY", "predictions")
    )

    max_body_bytes: int = Field(
        default_factory=lambda: _env_int("MAX_BODY_BYTES", 6 * 1024 * 1024)
    )
    max_records: int = Field(default_factory=lambda: _env_int("MAX_RECORDS", 5000))
    max_inflight: int = Field(default_factory=lambda: _env_int("MAX_INFLIGHT", 16))
    acquire_timeout_s: float = Field(
        default_factory=lambda: _env_float("ACQUIRE_TIMEOUT_S", 0.25)
    )

    gunicorn_workers: int = Field(
        default_factory=lambda: _env_int("GUNICORN_WORKERS", 1)
    )
    gunicorn_threads: int = Field(
        default_factory=lambda: _env_int("GUNICORN_THREADS", 4)
    )
    gunicorn_timeout: int = Field(
        default_factory=lambda: _env_int("GUNICORN_TIMEOUT", 60)
    )
    gunicorn_graceful_timeout: int = Field(
        default_factory=lambda: _env_int("GUNICORN_GRACEFUL_TIMEOUT", 30)
    )
    gunicorn_keepalive: int = Field(
        default_factory=lambda: _env_int("GUNICORN_KEEPALIVE", 5)
    )

    prometheus_enabled: bool = Field(
        default_factory=lambda: _env_bool("PROMETHEUS_ENABLED", True)
    )
    prometheus_path: str = Field(
        default_factory=lambda: _env_str("PROMETHEUS_PATH", "/metrics")
    )
    swagger_enabled: bool = Field(
        default_factory=lambda: _env_bool("SWAGGER_ENABLED", False)
    )

    otel_enabled: bool = Field(default_factory=lambda: _env_bool("OTEL_ENABLED", True))
    otel_endpoint: str = Field(
        default_factory=lambda: _env_str("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )
    otel_headers: str = Field(
        default_factory=lambda: _env_str("OTEL_EXPORTER_OTLP_HEADERS", "")
    )
    otel_timeout_s: float = Field(
        default_factory=lambda: _env_float("OTEL_EXPORTER_OTLP_TIMEOUT", 10.0)
    )

    onnx_providers: str = Field(
        default_factory=lambda: _env_str("ONNX_PROVIDERS", "CPUExecutionProvider")
    )
    onnx_intra_op_threads: int = Field(
        default_factory=lambda: _env_int("ONNX_INTRA_OP_THREADS", 0)
    )
    onnx_inter_op_threads: int = Field(
        default_factory=lambda: _env_int("ONNX_INTER_OP_THREADS", 0)
    )
    onnx_graph_opt_level: str = Field(
        default_factory=lambda: _env_str("ONNX_GRAPH_OPT_LEVEL", "all").strip().lower()
    )
    onnx_input_name: str = Field(
        default_factory=lambda: _env_str("ONNX_INPUT_NAME", "").strip()
    )
    onnx_output_name: str = Field(
        default_factory=lambda: _env_str("ONNX_OUTPUT_NAME", "").strip()
    )
    onnx_output_index: int = Field(
        default_factory=lambda: _env_int("ONNX_OUTPUT_INDEX", 0)
    )

    tabular_num_features: int = Field(
        default_factory=lambda: _env_int("TABULAR_NUM_FEATURES", 0)
    )
    onnx_input_map_json: str = Field(
        default_factory=lambda: _env_str("ONNX_INPUT_MAP_JSON", "").strip()
    )
    onnx_output_map_json: str = Field(
        default_factory=lambda: _env_str("ONNX_OUTPUT_MAP_JSON", "").strip()
    )
    onnx_input_dtype_map_json: str = Field(
        default_factory=lambda: _env_str("ONNX_INPUT_DTYPE_MAP_JSON", "").strip()
    )
    onnx_dynamic_batch: bool = Field(
        default_factory=lambda: _env_bool("ONNX_DYNAMIC_BATCH", True)
    )
