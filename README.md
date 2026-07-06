# TrainMedic

TrainMedic is an evidence-first diagnostics toolkit for PyTorch training failures.

TrainMedic is currently under active development. Phase 0 provides the installable
package skeleton, project tooling, and stable diagnostic data structures. It does not
yet inspect models, optimizers, gradients, activations, or training loops.

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
