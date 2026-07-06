"""Numerical diagnostics for runtime tensor observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainmedic.types import Diagnostic, Evidence, Severity

if TYPE_CHECKING:
    from trainmedic.monitors.forward import ForwardTensorObservation

FORWARD_OUTPUT_CONTAINS_NAN = "TM3001"
FORWARD_OUTPUT_CONTAINS_INF = "TM3002"


def diagnose_forward_observation(
    observation: ForwardTensorObservation,
    *,
    nan_already_reported: bool,
    inf_already_reported: bool,
) -> tuple[Diagnostic, ...]:
    """Create diagnostics for the first observed NaN and Inf forward outputs."""
    diagnostics: list[Diagnostic] = []

    if observation.nan_count > 0 and not nan_already_reported:
        diagnostics.append(
            Diagnostic(
                code=FORWARD_OUTPUT_CONTAINS_NAN,
                severity=Severity.ERROR,
                title="Forward output contains NaN",
                message="This is the first observed module output containing NaN.",
                object_name=observation.module_name,
                evidence=_observation_evidence(observation),
                possible_causes=(
                    "A log or square root operation may have received negative values.",
                    "A division or normalization operation may have become invalid.",
                    "A numerical overflow may have occurred upstream.",
                    "The module input may already contain non-finite values.",
                    "Low-precision computation may be unstable for this operation.",
                ),
                suggestions=(
                    "Check the input range for this module.",
                    "Inspect nearby division, log, sqrt, exp, and softmax operations.",
                    "Add input assertions around the suspected operation.",
                    "Try FP32 or BF16 for the unstable region.",
                    "Check the learning rate and outputs from previous layers.",
                ),
            )
        )

    if observation.inf_count > 0 and not inf_already_reported:
        diagnostics.append(
            Diagnostic(
                code=FORWARD_OUTPUT_CONTAINS_INF,
                severity=Severity.ERROR,
                title="Forward output contains Inf",
                message="This is the first observed module output containing Inf.",
                object_name=observation.module_name,
                evidence=_observation_evidence(observation),
                possible_causes=(
                    "A division by zero may have occurred.",
                    "An exp operation may have overflowed.",
                    "An activation value may have grown too large.",
                    "The selected precision may not represent this value range.",
                    "The module input may already contain Inf.",
                ),
                suggestions=(
                    "Check denominators and normalization constants.",
                    "Inspect exp, softmax, and scaling operations near this module.",
                    "Check activation magnitudes in previous layers.",
                    "Try FP32 or BF16 for the unstable region.",
                    "Check the learning rate and initialization scale.",
                ),
            )
        )

    return tuple(diagnostics)


def _observation_evidence(
    observation: ForwardTensorObservation,
) -> tuple[Evidence, ...]:
    return (
        Evidence("module_name", observation.module_name),
        Evidence("module_aliases", observation.module_aliases),
        Evidence("module_type", observation.module_type),
        Evidence("module_call_index", observation.module_call_index),
        Evidence("observation_sequence_index", observation.sequence_index),
        Evidence("tensor_path", observation.tensor_path),
        Evidence("shape", observation.shape),
        Evidence("dtype", observation.dtype),
        Evidence("device", observation.device),
        Evidence("numel", observation.numel),
        Evidence("nan_count", observation.nan_count),
        Evidence("inf_count", observation.inf_count),
    )
