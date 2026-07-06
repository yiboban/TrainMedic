"""Shared diagnostic data structures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def _to_json_compatible(value: Any) -> JsonValue:
    """Convert arbitrary values into JSON-compatible values.

    Values that cannot be represented safely as JSON are converted with ``str(value)``.
    Non-finite floats are also converted to strings to avoid emitting non-standard JSON.
    """
    if isinstance(value, Enum):
        enum_value = value.value
        return enum_value if isinstance(enum_value, str) else str(enum_value)

    if value is None or isinstance(value, str | int | bool):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, list | tuple):
        return [_to_json_compatible(item) for item in value]

    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}

    return str(value)


class Severity(str, Enum):
    """Severity level for a diagnostic finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Evidence:
    """A single observed fact that supports a diagnostic."""

    name: str
    value: Any
    description: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible dictionary."""
        return {
            "name": self.name,
            "value": _to_json_compatible(self.value),
            "description": self.description,
        }


@dataclass(frozen=True)
class Diagnostic:
    """Structured diagnosis emitted by TrainMedic rules."""

    code: str
    severity: Severity
    title: str
    message: str
    object_name: str | None = None
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    possible_causes: tuple[str, ...] = field(default_factory=tuple)
    suggestions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible dictionary."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "object_name": self.object_name,
            "evidence": [item.to_dict() for item in self.evidence],
            "possible_causes": list(self.possible_causes),
            "suggestions": list(self.suggestions),
        }
