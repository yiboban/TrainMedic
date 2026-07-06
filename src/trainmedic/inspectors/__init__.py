"""Static model and optimizer inspection helpers."""

from trainmedic.inspectors.model import ModelParameterRecord, collect_model_parameters
from trainmedic.inspectors.modules import ModelModuleRecord, collect_model_modules
from trainmedic.inspectors.optimizer import (
    OptimizerParameterRecord,
    collect_optimizer_parameters,
)

__all__ = [
    "ModelParameterRecord",
    "ModelModuleRecord",
    "OptimizerParameterRecord",
    "collect_model_parameters",
    "collect_model_modules",
    "collect_optimizer_parameters",
]
