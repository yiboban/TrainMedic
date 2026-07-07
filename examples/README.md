# TrainMedic Examples

Run examples from the repository root after installing TrainMedic in editable
mode:

```bash
python -m pip install -e ".[dev]"
```

Each example is CPU-friendly and intended to show one diagnostic situation.

## Optimizer Setup

- `healthy_optimizer.py`: a correctly configured optimizer produces no
  diagnostics.
- `missing_optimizer_parameter.py`: a trainable model parameter was not added to
  the optimizer.

## Forward NaN / Inf

- `healthy_forward.py`: a normal forward pass produces no diagnostics.
- `nan_forward.py`: a module output first contains NaN.
- `inf_forward.py`: a module output first contains Inf.

## Gradients

- `healthy_gradients.py`: a normal backward pass produces no diagnostics.
- `none_gradient.py`: one selected parameter has `grad=None`.
- `nan_gradient.py`: a parameter gradient contains NaN.
- `inf_gradient.py`: a parameter gradient contains Inf.

## Optimizer Updates

- `healthy_updates.py`: gradients and optimizer updates look healthy.
- `missing_optimizer_step.py`: no `optimizer.step()` was observed in the monitor
  session.
- `zero_learning_rate.py`: finite nonzero gradients are in a zero learning-rate
  group.
- `parameter_not_updated.py`: a parameter has finite nonzero gradients, but no
  update was detected.

## Train / Eval Mode

- `healthy_modes.py`: healthy training and evaluation mode usage.
- `model_eval_during_training.py`: the root model is in eval mode during
  expected training.
- `dropout_active_during_evaluation.py`: Dropout remains active during expected
  evaluation.
- `eval_without_no_grad.py`: evaluation runs with gradient tracking enabled.
- `batchnorm_train_during_evaluation.py`: BatchNorm remains in train mode during
  expected evaluation.

## Suggested First Runs

```bash
python examples/nan_forward.py
python examples/none_gradient.py
python examples/missing_optimizer_step.py
python examples/model_eval_during_training.py
```
