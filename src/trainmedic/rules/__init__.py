"""Diagnostic rules."""

from trainmedic.rules.numerical_rules import (
    FORWARD_OUTPUT_CONTAINS_INF,
    FORWARD_OUTPUT_CONTAINS_NAN,
)
from trainmedic.rules.optimizer_rules import inspect_optimizer

__all__ = [
    "FORWARD_OUTPUT_CONTAINS_INF",
    "FORWARD_OUTPUT_CONTAINS_NAN",
    "inspect_optimizer",
]
