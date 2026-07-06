# TrainMedic

TrainMedic is an evidence-first diagnostics toolkit for PyTorch training failures.

TrainMedic is currently under active development. Phase 3 provides static inspection of
model/optimizer parameter relationships, forward output monitoring for the first observed
NaN and Inf, and accumulated parameter gradient monitoring. It does not yet inspect
parameter updates, train/eval mode, or full training loops.

## Goal

The project aims to help users answer:

- Why did training fail or stop making progress?
- Where did the first suspicious signal appear?
- What evidence supports the diagnosis?
- What can the user try next?

## MVP Scope

The first MVP will focus on five classes of issues:

- Trainable parameters missing from the optimizer.
- Gradients that are `None`, NaN, Inf, exploding, or vanishing.
- Forward or backward tensors that first contain NaN or Inf.
- Parameters that do not update after optimizer steps.
- Model train/eval mode mistakes.

## Development Install

```bash
python -m pip install -e ".[dev]"
```

Install the package in editable mode before running examples from the repository.

## Minimal Data Structure Example

```python
import json

import trainmedic

diagnostic = trainmedic.Diagnostic(
    code="TM0001",
    severity=trainmedic.Severity.INFO,
    title="TrainMedic initialized",
    message="The diagnostic system is available.",
    evidence=(
        trainmedic.Evidence(name="version", value=trainmedic.__version__),
    ),
)

print(json.dumps(diagnostic.to_dict(), indent=2))
```

Example output:

```json
{
  "code": "TM0001",
  "severity": "info",
  "title": "TrainMedic initialized",
  "message": "The diagnostic system is available.",
  "object_name": null,
  "evidence": [
    {
      "name": "version",
      "value": "0.1.0.dev0",
      "description": null
    }
  ],
  "possible_causes": [],
  "suggestions": []
}
```

TrainMedic evidence values are JSON-compatible. Standard TrainMedic evidence uses stable
primitive values such as strings, numbers, booleans, lists, and dictionaries. Arbitrary
third-party Python objects are converted with `str(value)` for JSON compatibility, but
their string form is not guaranteed to be stable across processes.

## Static Optimizer Inspection

Currently implemented:

- Static model and optimizer parameter inspection.
- Forward output NaN/Inf monitoring.
- Accumulated parameter gradient monitoring.

Example:

```python
import torch
from torch import nn

from trainmedic import inspect_optimizer
from trainmedic.reports.console import format_diagnostics


class MissingParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 3, bias=False)
        self.decoder = nn.Linear(3, 1, bias=False)


model = MissingParameterModel()
optimizer = torch.optim.SGD(model.encoder.parameters(), lr=0.1)

diagnostics = inspect_optimizer(model, optimizer)
print(format_diagnostics(diagnostics))
```

Output:

```text
[1] TM1001 ERROR - Trainable parameter is not managed by the optimizer
Object: decoder.weight
Message: Parameter decoder.weight is trainable but is not managed by the current optimizer.
Evidence:
  - parameter_name: decoder.weight
  - aliases: ["decoder.weight"]
  - shape: [1, 3]
  - dtype: torch.float32
  - device: cpu
  - requires_grad: true
  - optimizer_group_count: 1
Possible causes:
  - The module may have been omitted when constructing the optimizer.
  - The optimizer may have been created before the model structure was finalized.
Suggestions:
  - Check whether optimizer construction includes this module's parameters.
  - Create the optimizer after the full model structure is built.
  - If this parameter is intentionally frozen, set requires_grad=False explicitly.
```

Supported Phase 1 diagnostic codes:

- `TM1001 PARAMETER_NOT_IN_OPTIMIZER`
- `TM1002 OPTIMIZER_PARAMETER_NOT_IN_MODEL`
- `TM1003 FROZEN_PARAMETER_IN_OPTIMIZER`
- `TM1004 MODEL_HAS_FROZEN_PARAMETERS`
- `TM1005 ALL_MODEL_PARAMETERS_FROZEN`
- `TM1006 MODEL_HAS_NO_PARAMETERS`
- `TM1007 DUPLICATE_PARAMETER_IN_OPTIMIZER`

## Forward Monitoring

TrainMedic can monitor module forward outputs without modifying the returned tensors or
the computation graph:

```python
import torch
from torch import nn

from trainmedic import watch_forward
from trainmedic.reports.console import format_diagnostics


class InvalidLog(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(inputs)


class NaNForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_log = InvalidLog()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.invalid_log(inputs)


model = NaNForwardModel()

with watch_forward(model) as monitor:
    model(torch.tensor([-1.0, 1.0]))

print(format_diagnostics(monitor.diagnostics))
```

NaN output:

```text
[1] TM3001 ERROR - Forward output contains NaN
Object: invalid_log
Message: This is the first observed module output containing NaN.
Evidence:
  - module_name: invalid_log
  - module_aliases: ["invalid_log"]
  - module_type: __main__.InvalidLog
  - module_call_index: 1
  - observation_sequence_index: 1
  - tensor_path: output
  - shape: [2]
  - dtype: torch.float32
  - device: cpu
  - numel: 2
  - nan_count: 1
  - inf_count: 0
Possible causes:
  - A log or square root operation may have received negative values.
  - A division or normalization operation may have become invalid.
  - A numerical overflow may have occurred upstream.
  - The module input may already contain non-finite values.
  - Low-precision computation may be unstable for this operation.
Suggestions:
  - Check the input range for this module.
  - Inspect nearby division, log, sqrt, exp, and softmax operations.
  - Add input assertions around the suspected operation.
  - Try FP32 or BF16 for the unstable region.
  - Check the learning rate and outputs from previous layers.
```

Inf output example:

```text
[1] TM3002 ERROR - Forward output contains Inf
Object: divide_by_zero
Message: This is the first observed module output containing Inf.
Evidence:
  - module_name: divide_by_zero
  - module_aliases: ["divide_by_zero"]
  - module_type: __main__.DivideByZero
  - module_call_index: 1
  - observation_sequence_index: 1
  - tensor_path: output
  - shape: [2]
  - dtype: torch.float32
  - device: cpu
  - numel: 2
  - nan_count: 0
  - inf_count: 2
Possible causes:
  - A division by zero may have occurred.
  - An exp operation may have overflowed.
  - An activation value may have grown too large.
  - The selected precision may not represent this value range.
  - The module input may already contain Inf.
Suggestions:
  - Check denominators and normalization constants.
  - Inspect exp, softmax, and scaling operations near this module.
  - Check activation magnitudes in previous layers.
  - Try FP32 or BF16 for the unstable region.
  - Check the learning rate and initialization scale.
```

TrainMedic reports the first observed NaN and Inf outputs, not every propagated
occurrence. If one tensor contains both, `TM3001` is emitted before `TM3002`.

`watch_forward()` defaults to `module_scope="all"` so it can catch functional operations
inside non-leaf modules. `module_scope="leaf"` monitors leaf modules and always also
monitors the root model. The context manager removes hooks on normal and exceptional
exit, and it does not suppress model exceptions.

Forward monitoring currently checks floating-point and complex strided tensors. Sparse,
meta, quantized, and backend-specific tensors that do not support `torch.isnan` or
`torch.isinf` may be skipped and counted by `monitor.unsupported_tensor_count`.

On CUDA, the `.item()` calls used to count NaN/Inf values can synchronize the device.
TrainMedic is a diagnostic tool and should not be left permanently enabled during
performance benchmarking. Sampling and lower-overhead modes are planned for later phases.

`torch.compile`, TorchScript, distributed training, DeepSpeed, FSDP, and Lightning are
not formally supported yet.

Supported Phase 2 diagnostic codes:

- `TM3001 FORWARD_OUTPUT_CONTAINS_NAN`
- `TM3002 FORWARD_OUTPUT_CONTAINS_INF`

## Gradient Monitoring

Gradient monitoring checks accumulated `Parameter.grad` values. Call
`check_gradients()` after `backward()` and before `optimizer.step()` or `zero_grad()`:

```python
import torch
from torch import nn

from trainmedic import watch_gradients
from trainmedic.reports.console import format_diagnostics


class BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.used(inputs)


model = BranchModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with watch_gradients(model, optimizer) as monitor:
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    monitor.check_gradients()

print(format_diagnostics(monitor.diagnostics))
```

`TM2001` output:

```text
[1] TM2001 WARNING - Selected parameters have grad=None
Message: At the explicit gradient check, one or more selected parameters had grad=None.
Evidence:
  - checked_parameter_count: 2
  - none_gradient_count: 1
  - non_none_gradient_count: 1
  - hook_observation_count: 1
  - any_backward_observed: true
  - none_parameter_names_preview: ["unused.weight"]
  - omitted_name_count: 0
  - selection_scope: optimizer
```

`TM2002` output:

```text
[1] TM2002 ERROR - Parameter gradient contains NaN
Object: weight
Message: This is the first parameter gradient observed to contain NaN during this monitor session.
Evidence:
  - sequence_index: 1
  - parameter_name: weight
  - parameter_aliases: ["weight"]
  - parameter_shape: [2]
  - parameter_dtype: torch.float32
  - parameter_device: cpu
  - parameter_numel: 2
  - hook_call_index: 1
  - gradient_shape: [2]
  - gradient_dtype: torch.float32
  - gradient_device: cpu
  - gradient_layout: torch.strided
  - gradient_numel: 2
  - gradient_nnz: null
  - nan_count: 2
  - inf_count: 0
```

`TM2003` output:

```text
[1] TM2003 ERROR - Parameter gradient contains Inf
Object: weight
Message: This is the first parameter gradient observed to contain Inf during this monitor session.
Evidence:
  - sequence_index: 1
  - parameter_name: weight
  - parameter_aliases: ["weight"]
  - parameter_shape: [2]
  - parameter_dtype: torch.float32
  - parameter_device: cpu
  - parameter_numel: 2
  - hook_call_index: 1
  - gradient_shape: [2]
  - gradient_dtype: torch.float32
  - gradient_device: cpu
  - gradient_layout: torch.strided
  - gradient_numel: 2
  - gradient_nnz: null
  - nan_count: 0
  - inf_count: 2
```

When an optimizer is provided, `watch_gradients()` monitors trainable model parameters
that are also managed by that optimizer. Without an optimizer, it monitors all trainable
model parameters. Static relationship issues such as missing optimizer parameters remain
the responsibility of `inspect_optimizer()`.

GradientMonitor observes the accumulated `.grad` value available when the
post-accumulate hook runs. If you call backward multiple times without `zero_grad()`,
gradients accumulate and hook call indexes increase. If `zero_grad(set_to_none=True)` is
called before `check_gradients()`, selected gradients will be reported as `grad=None`.
If gradient clipping runs before the check, global norm diagnostics reflect clipped
gradients.

Global gradient norm checks are off by default. They only run when `max_global_norm` or
`min_global_norm` is explicitly provided, and TrainMedic never clips gradients.

Gradient NaN and Inf diagnostics report the first runtime observation of each issue, not
every propagated occurrence. Sparse COO gradients are checked through `coalesce().values()`
without densifying. Other sparse layouts and special tensor backends may be skipped and
counted by `monitor.unsupported_gradient_count`.

Gradient hooks and `.item()` calls add overhead and can synchronize CUDA. Use gradient
monitoring for a small number of diagnostic steps rather than permanent benchmarking.

AMP GradScaler internals, `torch.compile`, TorchScript, distributed training, DeepSpeed,
FSDP, and Lightning are not formally supported yet.

Supported Phase 3 diagnostic codes:

- `TM2000 NO_PARAMETERS_SELECTED_FOR_GRADIENT_MONITORING`
- `TM2001 PARAMETER_GRADIENT_IS_NONE`
- `TM2002 GRADIENT_CONTAINS_NAN`
- `TM2003 GRADIENT_CONTAINS_INF`
- `TM2004 GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD`
- `TM2005 GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD`

## Run Checks

```bash
pytest --cov=trainmedic --cov-report=term-missing
ruff check .
mypy src/trainmedic
```

## Roadmap

- Phase 0: project skeleton, tooling, CI, and diagnostic data structures.
- Phase 1: static model and optimizer inspection.
- Phase 2: forward numerical monitoring.
- Phase 3: accumulated parameter gradient monitoring.
- Phase 4: parameter update monitoring.
- Phase 5: train/eval mode checks.

## Contributing

Keep changes small and evidence-based. New diagnostic behavior should include focused
tests that show the issue is detected and that the fixed version no longer reports it.
