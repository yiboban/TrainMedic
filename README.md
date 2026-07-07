# TrainMedic

English | [简体中文](README.zh-CN.md)

[![CI](https://github.com/yiboban/TrainMedic/actions/workflows/ci.yml/badge.svg)](https://github.com/yiboban/TrainMedic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch training diagnostics toolkit that helps you find why a model is not
training correctly.

TrainMedic checks model parameters, forward outputs, gradients, optimizer
updates, and train/eval modes without automatically changing your training
behavior.

## Why Use TrainMedic?

PyTorch training failures are often difficult to diagnose:

- loss suddenly becomes NaN;
- some parameters always have `grad=None`;
- the optimizer was created without part of the model;
- `optimizer.step()` is called, but parameters do not change;
- validation accidentally runs in train mode;
- Dropout or BatchNorm uses the wrong behavior.

TrainMedic observes these signals and produces structured diagnostics with
evidence, possible causes, and suggested next steps.

TrainMedic can help you answer:

- Which trainable parameters were forgotten when creating the optimizer?
- Where was the first NaN or Inf observed?
- Which parameters have `grad=None`?
- Did `optimizer.step()` actually update the parameters?
- Is the model using the wrong train/eval mode?

## 30 Second Quick Start

```python
import torch
from torch import nn

from trainmedic import watch_forward
from trainmedic.reports.console import format_diagnostics


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)


model = Model()

with watch_forward(model) as monitor:
    model(torch.tensor([-1.0, 1.0]))

print(format_diagnostics(monitor.diagnostics))
```

Trimmed output:

```text
[1] TM3001 ERROR - Forward output contains NaN
Object: <root>
Message: This is the first observed module output containing NaN.
Evidence:
  - tensor_path: output
  - shape: [2]
  - nan_count: 1
  - inf_count: 0
```

TrainMedic reports the first observed abnormal output. It does not
automatically claim that the module is the root cause.

## Installation

TrainMedic has not been published to PyPI yet. Install it from the repository:

```bash
git clone https://github.com/yiboban/TrainMedic.git
cd TrainMedic
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

Requirements:

- Python >= 3.10
- PyTorch >= 2.1

## What You Can Check

### Optimizer Parameter Mistakes

Useful when part of the model never learns even though the training loop runs
normally.

```python
from trainmedic import inspect_optimizer

diagnostics = inspect_optimizer(model, optimizer)
```

This can detect:

- trainable model parameters missing from the optimizer;
- optimizer parameters that do not belong to the model;
- frozen parameters still managed by the optimizer;
- models where every parameter is frozen;
- duplicate parameter registration in optimizer groups.

### Forward NaN / Inf

Useful when the loss becomes non-finite and you need the first suspicious module
output.

```python
from trainmedic import watch_forward

with watch_forward(model) as monitor:
    output = model(inputs)
```

Forward monitoring:

- reports the first observed NaN and Inf output;
- supports nested `list`, `tuple`, and `dict` outputs;
- does not save complete activations;
- does not claim that the first observed module is always the original root
  cause.

### Parameter Gradients

Useful when a branch is unused, the graph is detached, or gradients become
non-finite.

```python
from trainmedic import watch_gradients

with watch_gradients(model, optimizer) as monitor:
    loss.backward()
    monitor.check_gradients()
```

Call `check_gradients()` after `backward()` and before `optimizer.step()` or
`zero_grad()`.

This can detect:

- `grad=None`;
- gradient NaN;
- gradient Inf;
- user-configured global gradient norm thresholds.

### Parameter Updates

Useful when gradients exist but training still does not move.

```python
from trainmedic import watch_updates

with watch_updates(model, optimizer) as monitor:
    loss.backward()
    optimizer.step()
```

This can detect:

- no `optimizer.step()` observed in the current monitor session;
- finite nonzero gradients in zero learning-rate groups;
- an optimizer step that entered the pre-hook but did not complete;
- finite nonzero gradients where no parameter update was detected;
- skipped update checks when the learning rate cannot be interpreted as a
  finite scalar.

Large parameters are checked using deterministic sampled elements. A sampled
result means no change was detected in the sampled positions, not that the
entire parameter was proven unchanged.

### Train / Eval Mode

Useful when validation accidentally runs with training behavior, or training
runs with evaluation behavior.

```python
from trainmedic import watch_modes

model.eval()

with watch_modes(model, expected_mode="eval") as monitor:
    with torch.no_grad():
        output = model(inputs)
```

This can detect:

- root model in eval mode during expected training;
- root model in train mode during expected evaluation;
- Dropout mode mismatch;
- BatchNorm mode mismatch;
- gradient tracking enabled during evaluation;
- gradient tracking disabled during training.

TrainMedic never calls `model.train()` or `model.eval()` for you. The expected
mode must be explicit.

## Understanding a Diagnostic

A diagnostic may contain:

- `code`: a stable issue identifier, such as `TM3001`;
- `severity`: `INFO`, `WARNING`, `ERROR`, or `CRITICAL`;
- `object`: the related module or parameter, when available;
- `message`: what was observed;
- `evidence`: concrete data that supports the diagnostic;
- `possible_causes`: hypotheses to investigate;
- `suggestions`: practical next steps.

Possible causes are hypotheses based on observed evidence. They are not
guaranteed root-cause conclusions.

## Support Matrix

| Area                                 | Available                |
| ------------------------------------ | ------------------------ |
| Model/optimizer parameter inspection | Yes                      |
| Forward NaN/Inf monitoring           | Yes                      |
| Parameter gradient diagnostics       | Yes                      |
| Parameter update monitoring          | Yes                      |
| Train/eval mode monitoring           | Yes                      |
| CPU                                  | Yes                      |
| CUDA                                 | Basic support            |
| `torch.compile`                      | Not officially supported |
| DDP/FSDP/DeepSpeed                   | Not officially supported |
| Lightning/Transformers Trainer       | Not officially supported |
| Automatic fixes                      | No                       |

CUDA tensors are supported by the diagnostic code paths, but operations such as
`.item()` may synchronize the device. TrainMedic has not yet been validated
across every CUDA device, distributed stack, or training framework.

## Project Status

TrainMedic is currently an alpha developer preview.

The core diagnostics are usable, tested, and covered by CPU CI on Python 3.10,
3.11, and 3.12. However, APIs may still change, and the project has not yet
been validated across every PyTorch training stack.

## Limitations

- There is no unified `watch()` high-level entry point yet.
- TrainMedic has not been published to PyPI yet.
- `torch.compile` and TorchScript are not officially supported.
- DDP, FSDP, DeepSpeed, Lightning, and Transformers Trainer are not officially
  supported.
- AMP GradScaler internals are not fully diagnosed.
- Complex closure optimizers such as LBFGS are not officially supported.
- Sampled update checks can miss changes outside sampled positions.
- Hooks and tensor statistics add overhead and may synchronize CUDA.
- TrainMedic is best used for a small number of diagnostic steps, not permanent
  benchmarking.

For more detail, see [Architecture](docs/architecture.md) and
[Diagnostic Rules](docs/diagnostic-rules.md).

## Examples

Run examples after installing TrainMedic in editable mode:

```bash
python examples/nan_forward.py
```

Useful examples:

- [examples/missing_optimizer_parameter.py](examples/missing_optimizer_parameter.py):
  an optimizer was created without part of the model.
- [examples/nan_forward.py](examples/nan_forward.py): a forward output first
  contains NaN.
- [examples/none_gradient.py](examples/none_gradient.py): a selected parameter
  has `grad=None`.
- [examples/nan_gradient.py](examples/nan_gradient.py): a parameter gradient
  contains NaN.
- [examples/missing_optimizer_step.py](examples/missing_optimizer_step.py): no
  `optimizer.step()` was observed.
- [examples/zero_learning_rate.py](examples/zero_learning_rate.py): gradients
  are finite and nonzero, but the learning rate is zero.
- [examples/model_eval_during_training.py](examples/model_eval_during_training.py):
  the root model is in eval mode during expected training.
- [examples/dropout_active_during_evaluation.py](examples/dropout_active_during_evaluation.py):
  Dropout remains active during expected evaluation.

See the full [examples guide](examples/README.md).

## Development Checks

```bash
pytest --cov=trainmedic --cov-report=term-missing
ruff check .
mypy src/trainmedic
```

The project is tested on Python 3.10, 3.11, and 3.12 with CPU PyTorch in GitHub
Actions.

## Contributing

The most valuable contributions are:

- minimal scripts that reproduce real training failures;
- false-positive reports;
- models or optimizers that TrainMedic does not handle correctly;
- tests for edge cases;
- clearer diagnostic messages and documentation.

When opening an issue, please include:

- Python version;
- PyTorch version;
- device;
- minimal reproduction;
- TrainMedic output;
- expected behavior.

Contributions are welcome, especially when they are small, reproducible, and
backed by tests.

## Roadmap

- Unified `watch()` session
- First GitHub alpha release
- PyPI packaging
- Better CUDA validation
- AMP and framework integrations
- Real-world diagnostic case library
