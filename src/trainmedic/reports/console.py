"""Plain-text diagnostic reporting."""

from __future__ import annotations

import json
from collections.abc import Sequence

from trainmedic.types import Diagnostic, JsonValue

_NO_ISSUES_MESSAGE = "TrainMedic found no diagnostics."


def format_diagnostics(diagnostics: Sequence[Diagnostic]) -> str:
    """Format diagnostics as deterministic plain text without printing."""
    if not diagnostics:
        return _NO_ISSUES_MESSAGE

    sections = []
    for index, diagnostic in enumerate(diagnostics, start=1):
        lines = [
            f"[{index}] {diagnostic.code} {diagnostic.severity.value.upper()} - {diagnostic.title}",
        ]

        if diagnostic.object_name is not None:
            lines.append(f"Object: {diagnostic.object_name}")

        lines.append(f"Message: {diagnostic.message}")

        if diagnostic.evidence:
            lines.append("Evidence:")
            for item in diagnostic.evidence:
                lines.append(f"  - {item.name}: {_format_value(item.to_dict()['value'])}")

        if diagnostic.possible_causes:
            lines.append("Possible causes:")
            for cause in diagnostic.possible_causes:
                lines.append(f"  - {cause}")

        if diagnostic.suggestions:
            lines.append("Suggestions:")
            for suggestion in diagnostic.suggestions:
                lines.append(f"  - {suggestion}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _format_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
