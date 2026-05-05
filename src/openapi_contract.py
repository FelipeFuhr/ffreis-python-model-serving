"""Load optional checked-in OpenAPI contract."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from value_types import JsonDict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency for contract loading
    yaml = None


@lru_cache(maxsize=1)
def load_openapi_contract() -> JsonDict | None:
    """Load docs/openapi.yaml when available.

    Returns ``None`` when the file does not exist or PyYAML is unavailable.
    """
    if yaml is None:
        return None
    contract_path = Path(__file__).resolve().parents[1] / "docs" / "openapi.yaml"
    if not contract_path.exists():
        return None
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None
