"""Public API for TrainMedic."""

from trainmedic.rules.optimizer_rules import inspect_optimizer
from trainmedic.types import Diagnostic, Evidence, Severity

__version__ = "0.1.0.dev0"

__all__ = [
    "Diagnostic",
    "Evidence",
    "Severity",
    "__version__",
    "inspect_optimizer",
]
