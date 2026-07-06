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
- Backward gradients, parameter updates, train/eval mode diagnostics, distributed
  training, `torch.compile`, and TorchScript are not formally supported yet.
