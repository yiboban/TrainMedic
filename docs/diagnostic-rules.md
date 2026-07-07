# Diagnostic Rules

TrainMedic diagnostics are deterministic and evidence-first. Current rules do not use
language-model judgment.

## Optimizer Rules

### TM1001 PARAMETER_NOT_IN_OPTIMIZER

- Severity: `error`
- Trigger: a model parameter has `requires_grad=True` and its object identity is absent
  from all optimizer parameter groups.
- Boundary: parameters intentionally excluded from training should be marked
  `requires_grad=False`.

### TM1002 OPTIMIZER_PARAMETER_NOT_IN_MODEL

- Severity: `warning`
- Trigger: an optimizer parameter object identity is not present in the current model.
- Boundary: external trainable parameters may be intentional, so this is not an error.

### TM1003 FROZEN_PARAMETER_IN_OPTIMIZER

- Severity: `info`
- Trigger: a model parameter is present in the optimizer and has `requires_grad=False`.
- Boundary: keeping frozen parameters in the optimizer can be intentional for later
  unfreezing and does not by itself prove a training failure.

### TM1004 MODEL_HAS_FROZEN_PARAMETERS

- Severity: `info`
- Trigger: a model has both trainable and frozen parameters.
- Boundary: this is a neutral aggregate notice, not one diagnostic per frozen parameter.

### TM1005 ALL_MODEL_PARAMETERS_FROZEN

- Severity: `warning`
- Trigger: a model has at least one parameter and all parameters have
  `requires_grad=False`.
- Boundary: inference-only models may ignore this.

### TM1006 MODEL_HAS_NO_PARAMETERS

- Severity: `info`
- Trigger: `model.parameters()` is empty.
- Boundary: stateless modules are valid PyTorch modules.

### TM1007 DUPLICATE_PARAMETER_IN_OPTIMIZER

- Severity: `warning`
- Trigger: the same Parameter object identity appears more than once in optimizer
  parameter groups.
- Boundary: tied model parameter aliases are not duplicates unless the optimizer contains
  the same object more than once.

## Forward Numerical Rules

### TM3001 FORWARD_OUTPUT_CONTAINS_NAN

- Severity: `error`
- Trigger: the first observed floating-point or complex module output tensor has
  `nan_count > 0`.
- Boundary: this is the first observed module output containing NaN, not proof that the
  module is the root cause. Inputs or unmonitored upstream operations may already be
  abnormal.

### TM3002 FORWARD_OUTPUT_CONTAINS_INF

- Severity: `error`
- Trigger: the first observed floating-point or complex module output tensor has
  `inf_count > 0`.
- Boundary: this is the first observed module output containing Inf, not proof that the
  module is the root cause.

## Gradient Rules

### TM2000 NO_PARAMETERS_SELECTED_FOR_GRADIENT_MONITORING

- Severity: `info`
- Trigger: `check_gradients()` is called and the selected parameter set is empty.
- Evidence: optimizer presence, total model parameter count, trainable parameter count,
  optimizer-managed model parameter count, and selected parameter count.
- Boundary: emitted only at explicit check time; construction and `start()` do not emit
  diagnostics.

### TM2001 PARAMETER_GRADIENT_IS_NONE

- Severity: `warning`
- Trigger: at explicit gradient check, one or more selected parameters have `grad is None`.
- Evidence: checked count, None count, non-None count, hook observation count, whether
  backward was observed, preview of names, omitted name count, and selection scope.
- Boundary: this does not prove parameters are broken. Checks before backward,
  `zero_grad(set_to_none=True)`, unused branches, graph detaches, and intentionally
  unused parameters can all produce `grad=None`.

### TM2002 GRADIENT_CONTAINS_NAN

- Severity: `error`
- Trigger: first observed accumulated parameter gradient has `nan_count > 0`.
- Evidence: parameter name and aliases, parameter summary, hook call index, gradient
  summary, layout, nnz for sparse COO gradients, NaN count, and Inf count.
- Boundary: first observed gradient containing NaN is not necessarily the root cause.
  Forward outputs, loss computation, or custom backward code may have produced it.

### TM2003 GRADIENT_CONTAINS_INF

- Severity: `error`
- Trigger: first observed accumulated parameter gradient has `inf_count > 0`.
- Evidence: same shape as TM2002.
- Boundary: first observed gradient containing Inf is not necessarily the root cause.

### TM2004 GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD

- Severity: `warning`
- Trigger: user sets `max_global_norm` and the finite global L2 norm is greater than the
  threshold.
- Evidence: actual norm, threshold, contributing parameter count, None gradient count,
  and selection scope.
- Boundary: TrainMedic does not clip gradients and does not set a default threshold.

### TM2005 GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD

- Severity: `warning`
- Trigger: user sets `min_global_norm`, at least one gradient contributes, all
  contributing gradients are finite, and the global L2 norm is below the threshold.
- Evidence: same shape as TM2004.
- Boundary: absolute gradient size is model-, task-, and loss-scale-dependent, so this
  rule only runs with an explicit user threshold.

## Parameter Update Rules

### TM4000 NO_PARAMETERS_SELECTED_FOR_UPDATE_MONITORING

- Severity: `info`
- Trigger: the update monitor session finalizes normally and no trainable model
  parameter is also present in the optimizer.
- Evidence: model parameter count, trainable parameter count, unique optimizer-managed
  model parameter count, selected parameter count, and optimizer group count.
- Boundary: emitted instead of TM4001 when the selection is empty. Missing optimizer
  parameters remain TM1001, and optimizer parameters outside the model remain TM1002.

### TM4001 OPTIMIZER_STEP_NOT_OBSERVED

- Severity: `warning`
- Trigger: the session finalizes normally, selected parameters are non-empty, and no
  optimizer step reached the post-hook.
- Evidence: selected parameter count, successful step count, and optimizer group count.
- Boundary: scoped only to the current monitor session. If a user exception is
  propagating, the monitor cleans up hooks and snapshots without adding TM4001.

### TM4002 ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES

- Severity: `warning`
- Trigger: at optimizer step pre-hook time, one or more selected parameters have finite
  nonzero gradients and belong only to optimizer groups whose readable learning rate is
  exactly 0.
- Evidence: first step index, affected parameter count, parameter preview, omitted
  count, optimizer group indices, learning-rate values, and selection count.
- Boundary: emitted once per session. Parameters reported here are excluded from TM4003
  for the same evidence. Unknown, NaN, or Inf learning rates are not treated as ordinary
  nonzero learning rates.

### TM4003 PARAMETER_UPDATE_NOT_DETECTED

- Severity: `warning`
- Trigger: after a successful optimizer step post-hook, a selected parameter had a
  finite nonzero gradient, a known nonzero learning rate, successful before/after
  snapshots, and identical sampled values.
- Evidence: first problem step index, candidate counts, changed and unchanged counts,
  exact and sampled unchanged counts, unsupported/skipped count, unchanged parameter
  preview, bounded per-parameter preview, configured sample size, and configured global
  snapshot budget.
- Boundary: not emitted for `grad=None`, zero gradients, NaN/Inf gradients, unsupported
  gradients, unsupported snapshots, explicit zero learning-rate groups, failed optimizer
  steps, or parameters whose sampled values changed. `grad=None` is TM2001, NaN/Inf
  gradients are TM2002/TM2003, and zero learning rate is TM4002.
- Sampled limitation: sampled coverage only proves that TrainMedic did not detect a
  change in the sampled elements. It can miss updates that occur only outside sampled
  positions.

### TM4004 OPTIMIZER_STEP_DID_NOT_COMPLETE

- Severity: `warning`
- Trigger: the session finalizes normally, at least one optimizer step entered the
  pre-hook, and at least one attempted step did not reach the post-hook.
- Evidence: attempted step count, successful step count, incomplete step count, last
  attempted step index, and whether a pending snapshot existed at finalization.
- Boundary: if the optimizer exception propagates out of the monitor context, TrainMedic
  only cleans resources and does not add TM4004.
- Suggestions: inspect caught optimizer exceptions, close and restart after failed
  steps, and check custom optimizer or closure code.

### TM4005 UPDATE_CHECK_SKIPPED_FOR_UNKNOWN_LEARNING_RATE

- Severity: `info`
- Trigger: a selected parameter has a finite nonzero gradient, but its current
  learning rate cannot be interpreted as a finite scalar.
- Evidence: first step index, affected parameter count, preview names, group indices,
  stable learning-rate representations, and selected parameter count.
- Boundary: unknown learning rates are neither zero nor nonzero for update comparison.
  These parameters are excluded from TM4002 and TM4003.
- Suggestions: use finite scalar Python or Tensor learning rates, and inspect custom
  optimizer param group configuration.

## Mode Rules

### TM5000 FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING

- Severity: `info`
- Trigger: a mode-monitoring session finalizes normally without any forward pre-hook
  observation.
- Evidence: expected mode.
- Boundary: not emitted when a user exception exits the context.

### TM5001 MODEL_IN_EVAL_DURING_TRAINING

- Severity: `warning`
- Trigger: `expected_mode="train"` and the root model is actually called while
  `.training is False`.
- Evidence: expected and actual mode, sequence index, root call index, module type,
  grad enabled, and inference-mode state.
- Boundary: reports the first observed root call only; it does not scan uncalled modules.
- Suggestions: call `model.train()` before training and restore train mode after
  validation.

### TM5002 MODEL_IN_TRAIN_DURING_EVALUATION

- Severity: `warning`
- Trigger: `expected_mode="eval"` and the root model is actually called while
  `.training is True`.
- Evidence: same shape as TM5001.
- Boundary: this is scoped to the expected evaluation session.
- Suggestions: call `model.eval()` before validation or inference.

### TM5003 CALLED_SUBMODULE_MODE_MISMATCH

- Severity: `info`
- Trigger: a called non-root, non-Dropout, non-BatchNorm submodule has a mode that
  differs from the expected phase.
- Evidence: module name, aliases, type, expected and actual mode, sequence index, and
  call index.
- Boundary: INFO because fixed-mode submodules can be intentional.
- Suggestions: confirm whether the fixed submodule mode is intended.

### TM5004 DROPOUT_MODE_MISMATCH

- Severity: `warning`
- Trigger: a called Dropout module has a mode that differs from the expected phase.
- Evidence: common mode observation evidence plus dropout probability, inplace flag,
  expected stochastic behavior, and actual stochastic behavior.
- Boundary: reports the first called Dropout mismatch only and does not modify Dropout.
- Suggestions: check root and submodule `train()` / `eval()` calls.

### TM5005 BATCHNORM_MODE_MISMATCH

- Severity: `warning`
- Trigger: a called BatchNorm module has a mode that differs from the expected phase.
- Evidence: common mode observation evidence plus num_features, affine,
  track_running_stats, and momentum.
- Boundary: when `track_running_stats=False`, BatchNorm statistics semantics differ; the
  diagnostic does not unconditionally claim running statistics are used or updated.
- Suggestions: confirm intended BatchNorm statistics behavior.

### TM5006 GRADIENT_TRACKING_ENABLED_DURING_EVALUATION

- Severity: `info`
- Trigger: `expected_mode="eval"`, grad-context checks are enabled, the root model is
  called, and `torch.is_grad_enabled()` is true.
- Evidence: expected mode, actual mode, root call index, grad enabled, and
  inference-mode state.
- Boundary: INFO because gradient-based evaluation, attribution, or adversarial methods
  may intentionally need gradients.
- Suggestions: use `torch.no_grad()` for ordinary validation or `torch.inference_mode()`
  for inference.

### TM5007 GRADIENT_TRACKING_DISABLED_DURING_TRAINING

- Severity: `warning`
- Trigger: `expected_mode="train"`, grad-context checks are enabled, the root model is
  called, and `torch.is_grad_enabled()` is false.
- Evidence: same shape as TM5006.
- Boundary: does not claim backward must fail; some branches may be intentionally frozen.
- Suggestions: remove overly broad `torch.no_grad()` or `torch.inference_mode()` scopes
  around training.

## Current Limitations

- Forward and gradient monitoring report first observations, not every propagated
  occurrence.
- Sparse, meta, quantized, and backend-specific tensors may be skipped and counted as
  unsupported.
- Parameter update monitoring does not formally support LBFGS or complex closure-based
  optimizer semantics.
- Module backward `grad_input` / `grad_output`, distributed training,
  `torch.compile`, and TorchScript are not formally supported yet.
