"""Diagnostics for train/eval mode and gradient context monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainmedic.types import Diagnostic, Evidence, Severity

if TYPE_CHECKING:
    from trainmedic.monitors.modes import ModeObservation

FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING = "TM5000"
MODEL_IN_EVAL_DURING_TRAINING = "TM5001"
MODEL_IN_TRAIN_DURING_EVALUATION = "TM5002"
CALLED_SUBMODULE_MODE_MISMATCH = "TM5003"
DROPOUT_MODE_MISMATCH = "TM5004"
BATCHNORM_MODE_MISMATCH = "TM5005"
GRADIENT_TRACKING_ENABLED_DURING_EVALUATION = "TM5006"
GRADIENT_TRACKING_DISABLED_DURING_TRAINING = "TM5007"


def forward_not_observed_diagnostic(*, expected_mode: str) -> Diagnostic:
    """Create a diagnostic for a mode-monitoring session with no forward calls."""
    return Diagnostic(
        code=FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING,
        severity=Severity.INFO,
        title="Forward was not observed during mode monitoring",
        message="No forward call was observed during this mode monitoring session.",
        evidence=(Evidence("expected_mode", expected_mode),),
        possible_causes=(
            "The monitor context may not cover a model forward call.",
            "A conditional branch may have skipped this batch.",
            "The model may have been called before the monitor started or after it closed.",
        ),
        suggestions=(
            "Wrap the forward call in the watch_modes() context.",
            "Check whether the current branch intentionally skipped execution.",
        ),
    )


def model_in_eval_during_training_diagnostic(observation: ModeObservation) -> Diagnostic:
    """Create a diagnostic for root model eval mode during expected training."""
    return Diagnostic(
        code=MODEL_IN_EVAL_DURING_TRAINING,
        severity=Severity.WARNING,
        title="Root model is in eval mode during training",
        message=(
            "The root model was first observed in eval mode during an expected training "
            "forward."
        ),
        object_name=observation.module_name,
        evidence=_observation_evidence(observation),
        possible_causes=(
            "model.train() may not have been called before training.",
            "Training may have resumed after validation without restoring train mode.",
            "Checkpoint or wrapper code may have changed the mode.",
            "The monitor context may cover the wrong phase.",
        ),
        suggestions=(
            "Call model.train() before the training forward.",
            "Restore train mode after validation.",
            "Check wrapper and checkpoint loading code that changes module modes.",
        ),
    )


def model_in_train_during_evaluation_diagnostic(observation: ModeObservation) -> Diagnostic:
    """Create a diagnostic for root model train mode during expected evaluation."""
    return Diagnostic(
        code=MODEL_IN_TRAIN_DURING_EVALUATION,
        severity=Severity.WARNING,
        title="Root model is in train mode during evaluation",
        message=(
            "The root model was first observed in train mode during an expected evaluation "
            "forward."
        ),
        object_name=observation.module_name,
        evidence=_observation_evidence(observation),
        possible_causes=(
            "model.eval() may not have been called before evaluation.",
            "Evaluation code may have accidentally called train().",
            "The wrong model instance may have been used.",
            "Another component may have changed mode before forward.",
        ),
        suggestions=(
            "Call model.eval() before validation or inference.",
            "Check evaluation helpers for accidental train() calls.",
            "Confirm that the monitored model is the instance being evaluated.",
        ),
    )


def called_submodule_mode_mismatch_diagnostic(observation: ModeObservation) -> Diagnostic:
    """Create a neutral diagnostic for a called non-sensitive submodule mode mismatch."""
    return Diagnostic(
        code=CALLED_SUBMODULE_MODE_MISMATCH,
        severity=Severity.INFO,
        title="Called submodule mode differs from expected phase",
        message=(
            "A called submodule was first observed in a mode that differs from the "
            "expected phase."
        ),
        object_name=observation.module_name,
        evidence=_observation_evidence(observation),
        possible_causes=(
            "The submodule may have been intentionally fixed in train or eval mode.",
            "A wrapper may have changed only part of the model.",
            "The current batch may have reached a branch with a different mode.",
        ),
        suggestions=(
            "Confirm whether this submodule is intentionally fixed to this mode.",
            "If not intentional, check code that calls train() or eval() on submodules.",
        ),
    )


def dropout_mode_mismatch_diagnostic(
    observation: ModeObservation,
    *,
    p: float | None,
    inplace: bool | None,
) -> Diagnostic:
    """Create a diagnostic for Dropout mode mismatch."""
    expected_stochastic = observation.expected_mode == "train"
    actual_stochastic = observation.actual_mode == "train"
    return Diagnostic(
        code=DROPOUT_MODE_MISMATCH,
        severity=Severity.WARNING,
        title="Dropout mode differs from expected phase",
        message=(
            "A called Dropout module was first observed in a mode that differs from the "
            "expected phase."
        ),
        object_name=observation.module_name,
        evidence=(
            *_observation_evidence(observation),
            Evidence("dropout_probability", p),
            Evidence("inplace", inplace),
            Evidence("expected_stochastic_behavior", expected_stochastic),
            Evidence("actual_stochastic_behavior", actual_stochastic),
        ),
        possible_causes=(
            "Dropout may have been disabled during training.",
            "Dropout may remain stochastic during evaluation.",
            "A submodule may have been put into a different mode than the root model.",
        ),
        suggestions=(
            "Check calls to train() and eval() on the root model and submodules.",
            "If this Dropout behavior is intentional, document the fixed mode.",
        ),
    )


def batchnorm_mode_mismatch_diagnostic(
    observation: ModeObservation,
    *,
    num_features: int | None,
    affine: bool | None,
    track_running_stats: bool | None,
    momentum: float | None,
) -> Diagnostic:
    """Create a diagnostic for BatchNorm mode mismatch."""
    if track_running_stats is False:
        message = (
            "A called BatchNorm module was first observed in a mode that differs from the "
            "expected phase. track_running_stats=False changes BatchNorm statistics "
            "semantics."
        )
    else:
        message = (
            "A called BatchNorm module was first observed in a mode that differs from the "
            "expected phase."
        )
    return Diagnostic(
        code=BATCHNORM_MODE_MISMATCH,
        severity=Severity.WARNING,
        title="BatchNorm mode differs from expected phase",
        message=message,
        object_name=observation.module_name,
        evidence=(
            *_observation_evidence(observation),
            Evidence("num_features", num_features),
            Evidence("affine", affine),
            Evidence("track_running_stats", track_running_stats),
            Evidence("momentum", momentum),
        ),
        possible_causes=(
            "BatchNorm may use batch statistics during evaluation.",
            "BatchNorm may use running statistics during training.",
            "A submodule may have been put into a different mode than the root model.",
            "track_running_stats=False may make the intended behavior different.",
        ),
        suggestions=(
            "Check calls to train() and eval() on the root model and BatchNorm modules.",
            "Confirm whether BatchNorm statistics behavior is intentional.",
        ),
    )


def eval_grad_enabled_diagnostic(observation: ModeObservation) -> Diagnostic:
    """Create a diagnostic for gradient tracking enabled during expected evaluation."""
    return Diagnostic(
        code=GRADIENT_TRACKING_ENABLED_DURING_EVALUATION,
        severity=Severity.INFO,
        title="Gradient tracking is enabled during evaluation",
        message="Gradient tracking was enabled during an expected evaluation forward.",
        object_name=observation.module_name,
        evidence=_grad_context_evidence(observation),
        possible_causes=(
            "The evaluation forward may not be wrapped in torch.no_grad().",
            "The code may intentionally need gradients for evaluation.",
            "The monitor may cover an analysis or adversarial phase.",
        ),
        suggestions=(
            "Use torch.no_grad() for ordinary validation.",
            "Use torch.inference_mode() for inference when appropriate.",
            "Ignore this diagnostic if the evaluation intentionally needs gradients.",
        ),
    )


def train_grad_disabled_diagnostic(observation: ModeObservation) -> Diagnostic:
    """Create a diagnostic for gradient tracking disabled during expected training."""
    return Diagnostic(
        code=GRADIENT_TRACKING_DISABLED_DURING_TRAINING,
        severity=Severity.WARNING,
        title="Gradient tracking is disabled during training",
        message="Gradient tracking was disabled during an expected training forward.",
        object_name=observation.module_name,
        evidence=_grad_context_evidence(observation),
        possible_causes=(
            "Training code may be wrapped in torch.no_grad().",
            "Training code may be wrapped in torch.inference_mode().",
            "A validation context may have been reused for training.",
            "Gradient tracking may have been disabled over too broad a scope.",
        ),
        suggestions=(
            "Remove torch.no_grad() or torch.inference_mode() around training forward.",
            "Narrow the no-grad scope to evaluation-only code.",
            "If only a frozen branch is intended, confirm the scope is correct.",
        ),
    )


def _observation_evidence(observation: ModeObservation) -> tuple[Evidence, ...]:
    return (
        Evidence("sequence_index", observation.sequence_index),
        Evidence("module_name", observation.module_name),
        Evidence("module_aliases", observation.module_aliases),
        Evidence("module_type", observation.module_type),
        Evidence("module_call_index", observation.module_call_index),
        Evidence("is_root", observation.is_root),
        Evidence("expected_mode", observation.expected_mode),
        Evidence("actual_mode", observation.actual_mode),
        Evidence("grad_enabled", observation.grad_enabled),
        Evidence("inference_mode_enabled", observation.inference_mode_enabled),
    )


def _grad_context_evidence(observation: ModeObservation) -> tuple[Evidence, ...]:
    return (
        Evidence("sequence_index", observation.sequence_index),
        Evidence("expected_mode", observation.expected_mode),
        Evidence("actual_mode", observation.actual_mode),
        Evidence("root_call_index", observation.module_call_index),
        Evidence("grad_enabled", observation.grad_enabled),
        Evidence("inference_mode_enabled", observation.inference_mode_enabled),
    )
