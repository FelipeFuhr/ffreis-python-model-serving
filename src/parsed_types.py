"""Shared parsed data structures."""

from __future__ import annotations

from numpy import ndarray as np_ndarray
from pydantic import BaseModel, ConfigDict

from value_types import MetadataValue


class ParsedInput(BaseModel):
    """Normalized request input consumed by model adapters.

    Parameters
    ----------
    X : numpy.ndarray | None, default=None
        Tabular features represented as a two-dimensional array.
    tensors : dict[str, numpy.ndarray] | None, default=None
        Named tensors for multi-input models.
    meta : dict[str, MetadataValue] | None, default=None
        Auxiliary metadata generated during parsing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: np_ndarray | None = None
    tensors: dict[str, np_ndarray] | None = None
    meta: dict[str, MetadataValue] | None = None


def batch_size(parsed: ParsedInput) -> int:
    """Extract batch size from parsed input payload.

    Parameters
    ----------
    parsed : ParsedInput
        Parsed input object.

    Returns
    -------
    int
        Inferred batch size.
    """
    if parsed.X is not None:
        return int(parsed.X.shape[0])
    if parsed.tensors:
        first = next(iter(parsed.tensors.values()))
        return int(first.shape[0]) if getattr(first, "ndim", 0) > 0 else 1
    raise ValueError("Parsed input contained no features/tensors")
