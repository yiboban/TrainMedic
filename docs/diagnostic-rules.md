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

## Current Limitations

- Forward monitoring reports the first observed NaN and first observed Inf, not every
  propagated occurrence.
- Sparse, meta, quantized, and backend-specific tensors may be skipped and counted as
  unsupported.
- Module backward `grad_input` / `grad_output`, parameter updates, train/eval mode
  diagnostics, distributed training, `torch.compile`, and TorchScript are not formally
  supported yet.

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
