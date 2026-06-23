"""Registry-backed model resolution at serving startup.

When ``MODEL_REGISTRY_BACKEND`` and ``MODEL_REGISTRY_URI`` are both set the
serving process resolves ``MODEL_NAME`` at ``stage=production`` via the
appropriate ml-registry backend, downloads (or copies) the ``onnx_uri``
artifact to a local temp directory, and returns the path so the caller can pass
it to the existing adapter auto-detection — exactly as if the artifact had been
placed under ``SM_MODEL_DIR``.

The registry integration is an **optional extra** (``[registry]``). When
``ffreis-ml-registry`` is not installed and the env vars are not set, the
function is a lightweight no-op — existing ``SM_MODEL_DIR`` behaviour is
completely unchanged.

Supported backends: ``sqlite`` (zero extra deps — stdlib ``sqlite3``),
``dynamodb`` (requires ``ml_registry[aws]``), ``postgres`` (requires
``ml_registry[postgres]``).

URI semantics per backend
--------------------------
``sqlite``     path to the SQLite file, e.g. ``/models/.registry/registry.db``
``dynamodb``   DynamoDB table name, e.g. ``ml-model-registry``
``postgres``   DSN string,            e.g. ``postgresql://user:pw@host/db``

``onnx_uri`` artifact semantics
---------------------------------
The registry stores the artifact address registered at ``ModelVersion.onnx_uri``.
This implementation supports:
  - Local file paths (``/opt/ml/models/iris_v1.onnx``) — copied into tmp.
  - ``file://`` URIs — stripped and treated as a local path.
  - ``s3://`` URIs — reserved for future extension; raises ``NotImplementedError``.
"""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as futures_TimeoutError
from logging import getLogger as logging_getLogger
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from config import Settings

log = logging_getLogger("byoc")

_SUPPORTED_BACKENDS = frozenset({"sqlite", "local", "dynamodb", "postgres"})


# Protocols have minimal public surface: they capture only the methods
# this module calls, not the full adapter contract.
class _ResolvedModel(Protocol):  # pylint: disable=too-few-public-methods
    """Minimal structural type for a resolved model version.

    Satisfies the ml-registry ``ResolvedModel`` shape we actually need.
    Avoids importing the real class at module level (optional dep).
    """

    onnx_uri: str | None


class _RegistryLike(Protocol):  # pylint: disable=too-few-public-methods
    """Minimal structural type satisfied by every ml-registry adapter.

    We do not import the real ``ModelRegistry`` Protocol at module level because
    ``ml_registry`` is an optional dependency — importing it unconditionally
    would break startup when the package is absent.
    """

    def resolve(
        self,
        name: str,
        *,
        version: str | None = None,
        alias: str | None = None,
        stage: object = None,
    ) -> _ResolvedModel:
        """Resolve a model version by name and optional discriminator."""
        ...  # pylint: disable=unnecessary-ellipsis

    def close(self) -> None:
        """Release backend resources."""
        ...  # pylint: disable=unnecessary-ellipsis


def _require_ml_registry() -> None:
    """Raise a clear error when the optional ``ml_registry`` package is absent."""
    try:
        # Guard import: the import side-effect confirms the package is installed;
        # the symbol is intentionally discarded.
        import ml_registry  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        raise RuntimeError(
            "MODEL_REGISTRY_BACKEND is set but the 'ml_registry' package is not "
            "installed. Install with: uv sync --extra registry"
        ) from exc


def _open_registry(backend: str, uri: str) -> _RegistryLike:
    """Instantiate the appropriate ModelRegistry adapter.

    Parameters
    ----------
    backend : str
        Registry backend key (``sqlite`` / ``dynamodb`` / ``postgres``).
    uri : str
        Connection string or path appropriate for the chosen backend.

    Returns
    -------
    _RegistryLike
        A ``ModelRegistry``-shaped adapter instance (``resolve`` method).
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"MODEL_REGISTRY_BACKEND={backend!r} is not supported. "
            f"Supported values: {sorted(_SUPPORTED_BACKENDS)}"
        )

    if backend in ("sqlite", "local"):
        # pylint: disable-next=import-outside-toplevel
        from ml_registry.adapters.sqlite_registry import (  # noqa: I001
            SqliteModelRegistry,
        )

        return cast(_RegistryLike, SqliteModelRegistry(path=uri))

    if backend == "dynamodb":
        try:
            # pylint: disable-next=import-outside-toplevel
            from ml_registry.adapters.dynamodb_registry import (  # noqa: I001
                DynamoDbModelRegistry,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MODEL_REGISTRY_BACKEND=dynamodb requires the 'ml_registry[aws]' "
                "extra. Install with: uv sync --extra registry"
            ) from exc
        return cast(_RegistryLike, DynamoDbModelRegistry(table_name=uri))

    if backend == "postgres":
        try:
            # pylint: disable-next=import-outside-toplevel
            from ml_registry.adapters.postgres_registry import (  # noqa: I001
                PostgresModelRegistry,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MODEL_REGISTRY_BACKEND=postgres requires the 'ml_registry[postgres]' "
                "extra. Install with: uv sync --extra registry"
            ) from exc
        return cast(_RegistryLike, PostgresModelRegistry(dsn=uri))

    raise RuntimeError(f"Unhandled backend: {backend!r}")  # pragma: no cover


def _resolve_onnx_uri(registry: _RegistryLike, model_name: str, stage: str) -> str:
    """Resolve the ``onnx_uri`` for the requested stage of ``model_name``.

    Parameters
    ----------
    registry : _RegistryLike
        A ``ModelRegistry``-shaped adapter.
    model_name : str
        The logical model name to resolve.
    stage : str
        Stage name to resolve (e.g. ``"production"``, ``"staging"``).

    Returns
    -------
    str
        The ``onnx_uri`` stored in the target ``ModelVersion``.

    Raises
    ------
    RuntimeError
        When no version at the requested stage exists, or the version has no
        ``onnx_uri``, or the stage value is not recognised by the registry.
    """
    # Lazy import: guards the optional ml_registry dep.
    # pylint: disable-next=import-outside-toplevel
    from ml_registry.core.entities import Stage  # noqa: I001

    # pylint: disable-next=import-outside-toplevel
    from ml_registry.ports.model_registry import ModelVersionNotFoundError  # noqa: I001

    try:
        resolved_stage = Stage(stage)
    except ValueError as exc:
        valid = [s.value for s in Stage]
        raise RuntimeError(
            f"MODEL_STAGE={stage!r} is not a valid stage. Valid values: {valid}"
        ) from exc

    try:
        resolved = registry.resolve(model_name, stage=resolved_stage)
    except ModelVersionNotFoundError as exc:
        raise RuntimeError(
            f"No version at stage={stage!r} found for MODEL_NAME={model_name!r} in "
            "registry. Promote a version to the target stage before starting the "
            "serving process."
        ) from exc

    onnx_uri: str | None = resolved.onnx_uri
    if not onnx_uri:
        raise RuntimeError(
            f"Version of {model_name!r} at stage={stage!r} has no onnx_uri. "
            "Register the ONNX artifact path before promoting to that stage."
        )
    return onnx_uri


def _materialize_artifact(onnx_uri: str, dest_dir: Path) -> Path:
    """Copy or download the artifact at ``onnx_uri`` into ``dest_dir``.

    Parameters
    ----------
    onnx_uri : str
        URI or path to the ONNX model artifact.
    dest_dir : Path
        Destination directory (already exists).

    Returns
    -------
    Path
        Absolute path to the local artifact copy inside ``dest_dir``.

    Raises
    ------
    NotImplementedError
        When the URI scheme is not yet supported (e.g. ``s3://``).
    RuntimeError
        When the resolved local path does not exist.
    """
    parsed = urlparse(onnx_uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        # Local path — strip the file:// prefix if present.
        local_path = Path(parsed.path if scheme == "file" else onnx_uri)
        if not local_path.exists():
            raise RuntimeError(
                f"Registry onnx_uri {onnx_uri!r} points to a path that does not exist: "
                f"{local_path}"
            )
        dest_file = dest_dir / local_path.name
        shutil.copy2(local_path, dest_file)
        return dest_file

    if scheme == "s3":
        raise NotImplementedError(
            "s3:// artifact URIs are not yet supported by the registry pull adapter. "
            "Copy the artifact to a local path and re-register with a file:// URI."
        )

    raise RuntimeError(
        f"Unsupported onnx_uri scheme {scheme!r} in {onnx_uri!r}. "
        "Supported schemes: '' (bare path), 'file://'."
    )


def _pull_model_inner(backend: str, uri: str, model_name: str, stage: str) -> str:
    """Resolve and materialise the model artifact; called inside the timeout wrapper.

    Parameters
    ----------
    backend : str
        Registry backend key.
    uri : str
        Connection string or path for the backend.
    model_name : str
        Logical model name.
    stage : str
        Stage name to resolve.

    Returns
    -------
    str
        Temp directory path containing the pulled artifact.
    """
    registry = _open_registry(backend, uri)
    try:
        onnx_uri = _resolve_onnx_uri(registry, model_name, stage)
        temp_dir = Path(tempfile.mkdtemp(prefix="byoc-registry-"))
        artifact_path = _materialize_artifact(onnx_uri, temp_dir)
        log.info("registry-pull: artifact materialised at %s", artifact_path)
        return str(temp_dir)
    finally:
        registry.close()


def pull_model_from_registry(settings: Settings) -> str | None:
    """Resolve and materialise a registry-backed model artifact at startup.

    Called once at adapter load time. When ``MODEL_REGISTRY_BACKEND`` and
    ``MODEL_REGISTRY_URI`` are both set the function:

    1. Validates that ``MODEL_NAME`` is also set.
    2. Opens the configured backend.
    3. Resolves the ``ModelVersion`` for ``MODEL_NAME`` at ``MODEL_STAGE``.
    4. Copies the ``onnx_uri`` artifact into a fresh ``tempfile.mkdtemp`` dir.
    5. Returns the temp dir path so the caller can set it as the effective
       ``model_dir`` passed into the adapter factory.

    The entire pull operation is bounded by ``MODEL_REGISTRY_TIMEOUT`` seconds
    (default 30). A timeout causes a fast-fail ``RuntimeError`` so the process
    exits non-zero rather than hanging indefinitely.

    When the registry env vars are absent the function returns ``None`` and the
    caller continues with the legacy ``SM_MODEL_DIR`` path unchanged.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    str | None
        Local directory containing the pulled ONNX artifact, or ``None`` when
        registry mode is not configured.

    Raises
    ------
    RuntimeError
        On any configuration, resolution, materialisation, or timeout failure.
    """
    backend = settings.model_registry_backend
    uri = settings.model_registry_uri

    if not backend and not uri:
        return None

    if not backend or not uri:
        raise RuntimeError(
            "Both MODEL_REGISTRY_BACKEND and MODEL_REGISTRY_URI must be set together. "
            f"Got backend={backend!r}, uri={uri!r}."
        )

    model_name = settings.model_name
    if not model_name:
        raise RuntimeError(
            "MODEL_NAME must be set when MODEL_REGISTRY_BACKEND and "
            "MODEL_REGISTRY_URI are configured."
        )

    _require_ml_registry()

    stage = settings.model_stage
    timeout = settings.model_registry_timeout

    log.info(
        "registry-pull: resolving backend=%s uri=%s model_name=%s stage=%s timeout=%ds",
        backend,
        uri,
        model_name,
        stage,
        timeout,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_pull_model_inner, backend, uri, model_name, stage)
        try:
            return future.result(timeout=timeout)
        except futures_TimeoutError as exc:
            raise RuntimeError(
                f"Registry pull timed out after {timeout}s "
                f"(MODEL_REGISTRY_TIMEOUT={timeout}). "
                "Check that the registry backend is reachable."
            ) from exc
