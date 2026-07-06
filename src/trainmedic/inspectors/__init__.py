"""Static model and optimizer inspection helpers."""

from trainmedic.inspectors.model import ModelParameterRecord, collect_model_parameters
from trainmedic.inspectors.optimizer import (
    OptimizerParameterRecord,
    collect_optimizer_parameters,
)

__all__ = [
    "ModelParameterRecord",
    "OptimizerParameterRecord",
    "collect_model_parameters",
    "collect_optimizer_parameters",
]
