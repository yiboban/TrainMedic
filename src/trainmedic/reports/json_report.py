"""JSON diagnostic reporting."""

from __future__ import annotations

import json
from collections.abc import Sequence

from trainmedic.types import Diagnostic


def diagnostics_to_json(
    diagnostics: Sequence[Diagnostic],
    *,
    indent: int | None = 2,
) -> str:
    """Serialize diagnostics to strict standard JSON."""
    return json.dumps(
        [diagnostic.to_dict() for diagnostic in diagnostics],
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )
