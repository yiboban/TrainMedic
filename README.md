# TrainMedic

TrainMedic is an evidence-first diagnostics toolkit for PyTorch training failures.

TrainMedic is currently under active development. Phase 1 provides static inspection of
model parameters and optimizer parameter groups. It does not yet inspect gradients,
activations, parameter updates, train/eval mode, or training loops.

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
- Phase 3: backward and gradient monitoring.
- Phase 4: parameter update monitoring.
- Phase 5: train/eval mode checks.

## Contributing

Keep changes small and evidence-based. New diagnostic behavior should include focused
tests that show the issue is detected and that the fixed version no longer reports it.
