"""Diagnostics for parameter update monitoring."""

from __future__ import annotations

from trainmedic.types import Diagnostic, Evidence, JsonValue, Severity

NO_PARAMETERS_SELECTED_FOR_UPDATE_MONITORING = "TM4000"
OPTIMIZER_STEP_NOT_OBSERVED = "TM4001"
ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES = "TM4002"
PARAMETER_UPDATE_NOT_DETECTED = "TM4003"
OPTIMIZER_STEP_DID_NOT_COMPLETE = "TM4004"
UPDATE_CHECK_SKIPPED_FOR_UNKNOWN_LEARNING_RATE = "TM4005"


def no_parameters_selected_diagnostic(
    *,
    model_parameter_count: int,
    trainable_parameter_count: int,
    optimizer_model_parameter_count: int,
    selected_parameter_count: int,
    optimizer_group_count: int,
) -> Diagnostic:
    """Create a diagnostic for an empty update monitoring selection."""
    return Diagnostic(
        code=NO_PARAMETERS_SELECTED_FOR_UPDATE_MONITORING,
        severity=Severity.INFO,
        title="No parameters selected for update monitoring",
        message="No model parameters matched the parameter update monitoring selection.",
        evidence=(
            Evidence("model_parameter_count", model_parameter_count),
            Evidence("trainable_parameter_count", trainable_parameter_count),
            Evidence("optimizer_model_parameter_count", optimizer_model_parameter_count),
            Evidence("selected_parameter_count", selected_parameter_count),
            Evidence("optimizer_group_count", optimizer_group_count),
        ),
        possible_causes=(
            "The model may have no parameters.",
            "All model parameters may be frozen.",
            "The optimizer may not manage parameters from this model.",
            "The provided model and optimizer may not match.",
        ),
        suggestions=(
            "Use inspect_optimizer() to check the model/optimizer relationship.",
            "Check requires_grad flags for the intended trainable parameters.",
            "Confirm that the optimizer was created from this model instance.",
        ),
    )


def step_not_observed_diagnostic(
    *,
    selected_parameter_count: int,
    optimizer_group_count: int,
) -> Diagnostic:
    """Create a diagnostic for a session that did not observe optimizer.step()."""
    return Diagnostic(
        code=OPTIMIZER_STEP_NOT_OBSERVED,
        severity=Severity.WARNING,
        title="Optimizer step was not observed",
        message="No optimizer.step() call was observed during this monitor session.",
        evidence=(
            Evidence("selected_parameter_count", selected_parameter_count),
            Evidence("successful_step_count", 0),
            Evidence("optimizer_group_count", optimizer_group_count),
        ),
        possible_causes=(
            "optimizer.step() may have been skipped.",
            "The monitor context may not cover the optimizer.step() call.",
            "A different optimizer may have been stepped.",
            "Conditional training logic may have skipped this update.",
            "optimizer.step() may have been called before the monitor started or after it closed.",
        ),
        suggestions=(
            "Wrap both backward and optimizer.step() in the monitor context.",
            "Confirm that the stepped optimizer is the same object passed to watch_updates().",
            "Check gradient accumulation and skipped-step control flow.",
        ),
    )


def zero_learning_rate_diagnostic(
    *,
    step_index: int,
    affected_parameter_count: int,
    affected_parameter_names_preview: tuple[str, ...],
    omitted_parameter_count: int,
    optimizer_group_indices: tuple[int, ...],
    learning_rate_values: tuple[float | str, ...],
    selection_count: int,
) -> Diagnostic:
    """Create a diagnostic for finite nonzero gradients in zero-lr groups."""
    return Diagnostic(
        code=ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES,
        severity=Severity.WARNING,
        title="Finite nonzero gradients are in zero learning-rate groups",
        message=(
            f"At optimizer step {step_index}, one or more parameters had finite nonzero "
            "gradients but belonged only to optimizer groups with learning rate 0."
        ),
        evidence=(
            Evidence("step_index", step_index),
            Evidence("affected_parameter_count", affected_parameter_count),
            Evidence("affected_parameter_names_preview", affected_parameter_names_preview),
            Evidence("omitted_parameter_count", omitted_parameter_count),
            Evidence("optimizer_group_indices", optimizer_group_indices),
            Evidence("learning_rate_values", learning_rate_values),
            Evidence("selection_count", selection_count),
        ),
        possible_causes=(
            "A scheduler may have reduced the learning rate to 0.",
            "The optimizer param group may have been initialized with lr=0.",
            "Warmup or schedule configuration may be wrong.",
            "The parameter group may be intentionally paused.",
        ),
        suggestions=(
            "Check the learning rate for each optimizer param group.",
            "Review scheduler and warmup configuration.",
            "If the group is intentionally paused, this warning can be ignored.",
        ),
    )


def parameter_update_not_detected_diagnostic(
    *,
    step_index: int,
    candidate_parameter_count: int,
    changed_candidate_count: int,
    unchanged_candidate_count: int,
    exact_unchanged_count: int,
    sampled_unchanged_count: int,
    unsupported_or_skipped_count: int,
    unchanged_parameter_names_preview: tuple[str, ...],
    omitted_parameter_count: int,
    per_parameter_preview: tuple[dict[str, JsonValue], ...],
    configured_sample_size: int,
    configured_max_snapshot_elements: int,
) -> Diagnostic:
    """Create a diagnostic for finite nonzero-gradient parameters that did not update."""
    sampled_clause = (
        " Some results are based on sampled elements."
        if sampled_unchanged_count > 0
        else ""
    )
    return Diagnostic(
        code=PARAMETER_UPDATE_NOT_DETECTED,
        severity=Severity.WARNING,
        title="Parameter update was not detected",
        message=(
            f"At optimizer step {step_index}, no parameter change was detected for one or "
            f"more parameters with finite nonzero gradients.{sampled_clause}"
        ),
        evidence=(
            Evidence("step_index", step_index),
            Evidence("candidate_parameter_count", candidate_parameter_count),
            Evidence("changed_candidate_count", changed_candidate_count),
            Evidence("unchanged_candidate_count", unchanged_candidate_count),
            Evidence("exact_unchanged_count", exact_unchanged_count),
            Evidence("sampled_unchanged_count", sampled_unchanged_count),
            Evidence("unsupported_or_skipped_count", unsupported_or_skipped_count),
            Evidence("unchanged_parameter_names_preview", unchanged_parameter_names_preview),
            Evidence("omitted_parameter_count", omitted_parameter_count),
            Evidence("per_parameter_preview", per_parameter_preview),
            Evidence("configured_sample_size", configured_sample_size),
            Evidence("configured_max_snapshot_elements", configured_max_snapshot_elements),
        ),
        possible_causes=(
            "The optimizer implementation may have skipped the parameter.",
            "The learning rate may be extremely small.",
            "The update may be below the representable precision of the parameter dtype.",
            "A custom optimizer may be a no-op.",
            "Optimizer state or conditional logic may have prevented the update.",
            "Mixed precision or quantization may have affected the update.",
            "Sampled mode may have missed changes that occurred only at unsampled elements.",
        ),
        suggestions=(
            "Check the learning rate for each param group.",
            "Run watch_gradients() at the same site.",
            "Increase sample_size and max_snapshot_elements for this diagnostic step.",
            "Inspect optimizer state and skipped-step logic.",
            "Reproduce the suspicious step in FP32.",
            "Check custom optimizer.step implementations.",
        ),
    )


def step_did_not_complete_diagnostic(
    *,
    attempted_step_count: int,
    successful_step_count: int,
    incomplete_step_count: int,
    last_attempted_step_index: int,
    pending_snapshot_present: bool,
) -> Diagnostic:
    """Create a diagnostic for attempted optimizer steps that did not complete."""
    return Diagnostic(
        code=OPTIMIZER_STEP_DID_NOT_COMPLETE,
        severity=Severity.WARNING,
        title="Optimizer step did not complete",
        message=(
            "One or more optimizer.step() calls entered the pre-hook but did not reach "
            "the post-hook during this monitor session."
        ),
        evidence=(
            Evidence("attempted_step_count", attempted_step_count),
            Evidence("successful_step_count", successful_step_count),
            Evidence("incomplete_step_count", incomplete_step_count),
            Evidence("last_attempted_step_index", last_attempted_step_index),
            Evidence("pending_snapshot_present", pending_snapshot_present),
        ),
        possible_causes=(
            "optimizer.step() may have raised an exception that user code caught.",
            "A custom optimizer may have interrupted execution inside step().",
            "Nested or recursive optimizer.step() calls may be unsupported.",
            "A closure may have raised an exception.",
            "The optimizer implementation may not have completed normally.",
        ),
        suggestions=(
            "Inspect any optimizer exception that was caught by user code.",
            "Close and restart the monitor after a failed optimizer step.",
            "Check custom optimizer or closure implementations.",
            "Avoid reusing one monitor session after a failed step.",
        ),
    )


def unknown_learning_rate_diagnostic(
    *,
    step_index: int,
    affected_parameter_count: int,
    affected_parameter_names_preview: tuple[str, ...],
    omitted_parameter_count: int,
    optimizer_group_indices: tuple[int, ...],
    learning_rate_representations: tuple[str, ...],
    selected_parameter_count: int,
) -> Diagnostic:
    """Create a diagnostic for finite nonzero gradients with unknown learning rates."""
    return Diagnostic(
        code=UPDATE_CHECK_SKIPPED_FOR_UNKNOWN_LEARNING_RATE,
        severity=Severity.INFO,
        title="Update check skipped for unknown learning rate",
        message=(
            "Update comparison was skipped for one or more parameters because their "
            "learning rate could not be interpreted as a finite scalar."
        ),
        evidence=(
            Evidence("step_index", step_index),
            Evidence("affected_parameter_count", affected_parameter_count),
            Evidence("affected_parameter_names_preview", affected_parameter_names_preview),
            Evidence("omitted_parameter_count", omitted_parameter_count),
            Evidence("optimizer_group_indices", optimizer_group_indices),
            Evidence("learning_rate_representations", learning_rate_representations),
            Evidence("selected_parameter_count", selected_parameter_count),
        ),
        possible_causes=(
            "The learning rate may be a non-scalar Tensor.",
            "The learning rate may be NaN or Inf.",
            "The learning rate may be a bool or custom object.",
            "A scheduler or custom optimizer may store learning rate in an unsupported form.",
        ),
        suggestions=(
            "Use a finite Python float or int learning rate.",
            "Use a scalar Tensor learning rate only if the optimizer supports it.",
            "Inspect custom optimizer param group configuration.",
        ),
    )
