"""Public API for TrainMedic."""

from trainmedic.monitors.forward import watch_forward
from trainmedic.monitors.gradients import watch_gradients
from trainmedic.monitors.updates import watch_updates
from trainmedic.rules.optimizer_rules import inspect_optimizer
from trainmedic.types import Diagnostic, Evidence, Severity

__version__ = "0.1.0.dev0"

__all__ = [
    "Diagnostic",
    "Evidence",
    "Severity",
    "__version__",
    "inspect_optimizer",
    "watch_forward",
    "watch_gradients",
    "watch_updates",
]
