"""Shared type aliases for JSON-like payloads and model predictions."""

from __future__ import annotations

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
type JsonDict = dict[str, JsonValue]
type JsonArray = list[JsonValue]

type PredictionValue = JsonValue
type MetadataValue = JsonValue

type SpanAttributeValue = (
    str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
)
