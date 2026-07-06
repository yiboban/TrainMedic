"""Gradient diagnostics for accumulated parameter gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainmedic.types import Diagnostic, Evidence, Severity

if TYPE_CHECKING:
    from trainmedic.monitors.gradients import GradientObservation

NO_PARAMETERS_SELECTED_FOR_GRADIENT_MONITORING = "TM2000"
PARAMETER_GRADIENT_IS_NONE = "TM2001"
GRADIENT_CONTAINS_NAN = "TM2002"
GRADIENT_CONTAINS_INF = "TM2003"
GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD = "TM2004"
GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD = "TM2005"


def diagnose_gradient_observation(
    observation: GradientObservation,
    *,
    nan_already_reported: bool,
    inf_already_reported: bool,
) -> tuple[Diagnostic, ...]:
    """Create diagnostics for the first observed NaN and Inf parameter gradients."""
    diagnostics: list[Diagnostic] = []

    if observation.nan_count > 0 and not nan_already_reported:
        diagnostics.append(
            Diagnostic(
                code=GRADIENT_CONTAINS_NAN,
                severity=Severity.ERROR,
                title="Parameter gradient contains NaN",
                message=(
                    "This is the first parameter gradient observed to contain NaN "
                    "during this monitor session."
                ),
                object_name=observation.parameter_name,
                evidence=_observation_evidence(observation),
                possible_causes=(
                    "Forward activations or the loss may already contain non-finite values.",
                    "A division, log, sqrt, exp, or normalization operation may be unstable.",
                    "The learning rate or loss scale may be too large.",
                    "Low-precision computation may have overflowed.",
                    "A custom autograd backward implementation may be invalid.",
                ),
                suggestions=(
                    "Use watch_forward() at the same site to check forward outputs.",
                    "Check whether the loss value is finite before backward.",
                    "Try FP32 or BF16 for the unstable region.",
                    "Review AMP GradScaler usage if mixed precision is enabled.",
                    "Check the learning rate and numerical stability of the loss.",
                    "Use gradcheck for custom autograd Functions.",
                ),
            )
        )

    if observation.inf_count > 0 and not inf_already_reported:
        diagnostics.append(
            Diagnostic(
                code=GRADIENT_CONTAINS_INF,
                severity=Severity.ERROR,
                title="Parameter gradient contains Inf",
                message=(
                    "This is the first parameter gradient observed to contain Inf "
                    "during this monitor session."
                ),
                object_name=observation.parameter_name,
                evidence=_observation_evidence(observation),
                possible_causes=(
                    "Forward activations or the loss may already contain Inf.",
                    "A division by zero or exp overflow may have occurred.",
                    "The learning rate or loss scale may be too large.",
                    "Low-precision computation may not represent the gradient range.",
                    "A custom autograd backward implementation may be invalid.",
                ),
                suggestions=(
                    "Use watch_forward() at the same site to check forward outputs.",
                    "Check whether the loss value is finite before backward.",
                    "Review denominators, exponentials, and scaling operations.",
                    "Try FP32 or BF16 for the unstable region.",
                    "Review AMP GradScaler usage if mixed precision is enabled.",
                ),
            )
        )

    return tuple(diagnostics)


def no_parameters_selected_diagnostic(
    *,
    optimizer_provided: bool,
    model_parameter_count: int,
    trainable_parameter_count: int,
    optimizer_model_parameter_count: int,
    selected_parameter_count: int,
) -> Diagnostic:
    """Create a diagnostic for an empty gradient monitoring selection."""
    return Diagnostic(
        code=NO_PARAMETERS_SELECTED_FOR_GRADIENT_MONITORING,
        severity=Severity.INFO,
        title="No parameters selected for gradient monitoring",
        message="No model parameters matched the gradient monitoring selection.",
        evidence=(
            Evidence("optimizer_provided", optimizer_provided),
            Evidence("model_parameter_count", model_parameter_count),
            Evidence("trainable_parameter_count", trainable_parameter_count),
            Evidence("optimizer_model_parameter_count", optimizer_model_parameter_count),
            Evidence("selected_parameter_count", selected_parameter_count),
        ),
        possible_causes=(
            "The model may have no parameters.",
            "All model parameters may be frozen.",
            "The optimizer may not manage trainable parameters from this model.",
            "The provided model and optimizer may not match.",
        ),
        suggestions=(
            "Use inspect_optimizer() to check the model/optimizer relationship.",
            "Check requires_grad flags for the intended trainable parameters.",
            "If this is an inference-only model, this diagnostic can be ignored.",
        ),
    )


def none_gradients_diagnostic(
    *,
    checked_parameter_count: int,
    none_gradient_count: int,
    non_none_gradient_count: int,
    hook_observation_count: int,
    any_backward_observed: bool,
    none_parameter_names_preview: tuple[str, ...],
    omitted_name_count: int,
    selection_scope: str,
) -> Diagnostic:
    """Create a diagnostic for selected parameters with grad=None."""
    return Diagnostic(
        code=PARAMETER_GRADIENT_IS_NONE,
        severity=Severity.WARNING,
        title="Selected parameters have grad=None",
        message="At the explicit gradient check, one or more selected parameters had grad=None.",
        evidence=(
            Evidence("checked_parameter_count", checked_parameter_count),
            Evidence("none_gradient_count", none_gradient_count),
            Evidence("non_none_gradient_count", non_none_gradient_count),
            Evidence("hook_observation_count", hook_observation_count),
            Evidence("any_backward_observed", any_backward_observed),
            Evidence("none_parameter_names_preview", none_parameter_names_preview),
            Evidence("omitted_name_count", omitted_name_count),
            Evidence("selection_scope", selection_scope),
        ),
        possible_causes=(
            "The check may have happened before backward.",
            "The parameters may not participate in the current forward path.",
            "The computation graph may have been detached.",
            "The loss may not depend on these parameters.",
            "A conditional branch may not have used these parameters for this batch.",
            "zero_grad(set_to_none=True) may have been called before the check.",
            "The parameters may be intentionally unused in this batch.",
        ),
        suggestions=(
            "Call check_gradients() after backward and before zero_grad().",
            "Confirm the loss depends on the relevant module outputs.",
            "Look for detach(), torch.no_grad(), item(), or other graph breaks.",
            "Check conditional branches and intentionally unused parameters.",
            "Use inspect_optimizer() to verify optimizer configuration.",
        ),
    )


def global_norm_exceeds_threshold_diagnostic(
    *,
    actual_global_norm: float,
    configured_threshold: float,
    contributing_parameter_count: int,
    none_gradient_count: int,
    selection_scope: str,
) -> Diagnostic:
    """Create a diagnostic for a global gradient norm above the user threshold."""
    return Diagnostic(
        code=GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD,
        severity=Severity.WARNING,
        title="Global gradient norm exceeds threshold",
        message="The global gradient norm is above the user-configured threshold.",
        evidence=(
            Evidence("actual_global_norm", actual_global_norm),
            Evidence("configured_threshold", configured_threshold),
            Evidence("contributing_parameter_count", contributing_parameter_count),
            Evidence("none_gradient_count", none_gradient_count),
            Evidence("selection_scope", selection_scope),
        ),
        possible_causes=(
            "The loss scale or learning rate may be too large.",
            "Gradients may be accumulating across multiple backward calls.",
            "The configured threshold may be too low for this model and loss.",
        ),
        suggestions=(
            "Inspect the loss scale and learning rate.",
            "Check whether gradient accumulation is intentional.",
            "If needed, apply gradient clipping explicitly in user code.",
        ),
    )


def global_norm_below_threshold_diagnostic(
    *,
    actual_global_norm: float,
    configured_threshold: float,
    contributing_parameter_count: int,
    none_gradient_count: int,
    selection_scope: str,
) -> Diagnostic:
    """Create a diagnostic for a global gradient norm below the user threshold."""
    return Diagnostic(
        code=GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD,
        severity=Severity.WARNING,
        title="Global gradient norm is below threshold",
        message="The global gradient norm is below the user-configured threshold.",
        evidence=(
            Evidence("actual_global_norm", actual_global_norm),
            Evidence("configured_threshold", configured_threshold),
            Evidence("contributing_parameter_count", contributing_parameter_count),
            Evidence("none_gradient_count", none_gradient_count),
            Evidence("selection_scope", selection_scope),
        ),
        possible_causes=(
            "The selected parameters may receive very small gradients.",
            "The loss scale may be small.",
            "The configured threshold may be too high for this model and loss.",
        ),
        suggestions=(
            "Confirm the threshold is appropriate for the model and task.",
            "Check whether the loss depends on the monitored parameters.",
            "Review activation scales and initialization.",
        ),
    )


def _observation_evidence(
    observation: GradientObservation,
) -> tuple[Evidence, ...]:
    return (
        Evidence("sequence_index", observation.sequence_index),
        Evidence("parameter_name", observation.parameter_name),
        Evidence("parameter_aliases", observation.parameter_aliases),
        Evidence("parameter_shape", observation.parameter_shape),
        Evidence("parameter_dtype", observation.parameter_dtype),
        Evidence("parameter_device", observation.parameter_device),
        Evidence("parameter_numel", observation.parameter_numel),
        Evidence("hook_call_index", observation.hook_call_index),
        Evidence("gradient_shape", observation.gradient_shape),
        Evidence("gradient_dtype", observation.gradient_dtype),
        Evidence("gradient_device", observation.gradient_device),
        Evidence("gradient_layout", observation.gradient_layout),
        Evidence("gradient_numel", observation.gradient_numel),
        Evidence("gradient_nnz", observation.gradient_nnz),
        Evidence("nan_count", observation.nan_count),
        Evidence("inf_count", observation.inf_count),
    )
