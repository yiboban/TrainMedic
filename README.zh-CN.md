# TrainMedic

[English](README.md) | [简体中文](README.zh-CN.md)

TrainMedic 是一个面向 PyTorch 训练失败场景、以证据优先的诊断工具包。

TrainMedic is currently under active development. 当前 Phase 5 已提供模型与
optimizer 参数关系的静态检查、forward 输出首次 NaN/Inf 监控、累积参数梯度监控，
围绕 `optimizer.step()` 的有界参数更新监控，以及显式 train/eval 模式和梯度上下文监控。
它尚未提供统一 watch 入口或完整训练循环监控。

## 项目目标

TrainMedic 的目标是帮助用户回答：

- 训练为什么失败，或者为什么不再取得进展？
- 第一个可疑信号出现在哪里？
- 哪些证据支持这个诊断？
- 用户下一步可以尝试什么？

## MVP 范围

第一个 MVP 将聚焦五类问题：

- 可训练参数没有加入 optimizer。
- 梯度为 `None`、NaN、Inf、爆炸或消失。
- Forward 或 backward tensor 首次出现 NaN 或 Inf。
- optimizer step 后参数没有更新。
- 模型 train/eval 模式使用错误。

## 开发安装

```bash
python -m pip install -e ".[dev]"
```

从仓库运行 examples 前，请先用 editable 模式安装本包。

## 最小数据结构示例

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

示例输出：

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

TrainMedic 的 evidence value 会保持 JSON 兼容。标准 evidence 使用字符串、数字、
布尔值、列表、字典等稳定基础类型。任意第三方 Python 对象会通过 `str(value)`
转换以保证 JSON 兼容，但该字符串形式不保证跨进程稳定。

## 静态 Optimizer 检查

当前已经实现：

- 静态模型与 optimizer 参数检查。
- Forward 输出 NaN/Inf 监控。
- 累积参数梯度监控。
- 围绕 `optimizer.step()` 的有界参数更新监控。
- 显式 train/eval 模式与梯度上下文监控。

示例：

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

输出：

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

Phase 1 支持的诊断代码：

- `TM1001 PARAMETER_NOT_IN_OPTIMIZER`
- `TM1002 OPTIMIZER_PARAMETER_NOT_IN_MODEL`
- `TM1003 FROZEN_PARAMETER_IN_OPTIMIZER`
- `TM1004 MODEL_HAS_FROZEN_PARAMETERS`
- `TM1005 ALL_MODEL_PARAMETERS_FROZEN`
- `TM1006 MODEL_HAS_NO_PARAMETERS`
- `TM1007 DUPLICATE_PARAMETER_IN_OPTIMIZER`

## Forward 监控

TrainMedic 可以监控 module forward 输出，不修改返回 tensor，也不修改计算图：

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

NaN 输出：

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
```

Inf 输出示例：

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
```

TrainMedic 报告首次观测到的 NaN 和 Inf 输出，而不是每一次传播后的重复出现。
如果同一个 tensor 同时包含二者，先报告 `TM3001`，再报告 `TM3002`。

`watch_forward()` 默认使用 `module_scope="all"`，因此可以捕捉非叶子 module 内部的
functional 操作。`module_scope="leaf"` 监控叶子 module，并始终额外监控 root model。
context manager 会在正常退出或异常退出时移除 hooks，也不会吞掉模型异常。

Forward 监控当前检查 floating-point 和 complex strided tensors。Sparse、meta、
quantized 以及特定后端 tensor 如果不支持 `torch.isnan` 或 `torch.isinf`，可能会被跳过，
并计入 `monitor.unsupported_tensor_count`。

在 CUDA 上，用于统计 NaN/Inf 的 `.item()` 调用可能导致设备同步。TrainMedic 是诊断工具，
不应长期保留在性能 benchmark 中。采样和更低开销模式会在后续阶段考虑。

`torch.compile`、TorchScript、distributed training、DeepSpeed、FSDP 和 Lightning
尚未正式支持。

Phase 2 支持的诊断代码：

- `TM3001 FORWARD_OUTPUT_CONTAINS_NAN`
- `TM3002 FORWARD_OUTPUT_CONTAINS_INF`

## 梯度监控

梯度监控检查累积后的 `Parameter.grad`。请在 `backward()` 之后、
`optimizer.step()` 或 `zero_grad()` 之前调用 `check_gradients()`：

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

`TM2001` 输出：

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

`TM2002` 输出：

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

`TM2003` 输出：

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

提供 optimizer 时，`watch_gradients()` 只监控属于该 model、可训练、且由该 optimizer
管理的参数。不提供 optimizer 时，它监控 model 中全部可训练参数。缺少 optimizer 参数等
静态关系问题仍由 `inspect_optimizer()` 负责。

GradientMonitor 观测 post-accumulate hook 运行时可用的累积 `.grad`。如果连续多次调用
backward 且没有 `zero_grad()`，梯度会累积，hook call index 也会增加。如果在
`check_gradients()` 前调用 `zero_grad(set_to_none=True)`，被选中的梯度会报告为
`grad=None`。如果梯度裁剪发生在检查前，全局范数诊断反映裁剪后的梯度。

全局梯度范数检查默认关闭。只有显式传入 `max_global_norm` 或 `min_global_norm` 时才会运行，
TrainMedic 永远不会裁剪梯度。

梯度 NaN 和 Inf 诊断只报告每类问题的首次运行时观测，而不是每一次传播后的重复出现。
Sparse COO 梯度会通过 `coalesce().values()` 检查，不会 densify。其他 sparse layout
和特殊 tensor 后端可能会被跳过，并计入 `monitor.unsupported_gradient_count`。

梯度 hooks 和 `.item()` 调用会增加开销，并可能在 CUDA 上同步设备。请将梯度监控用于少量
诊断 step，而不是长期 benchmark。

AMP GradScaler 内部、`torch.compile`、TorchScript、distributed training、DeepSpeed、
FSDP 和 Lightning 尚未正式支持。

Phase 3 支持的诊断代码：

- `TM2000 NO_PARAMETERS_SELECTED_FOR_GRADIENT_MONITORING`
- `TM2001 PARAMETER_GRADIENT_IS_NONE`
- `TM2002 GRADIENT_CONTAINS_NAN`
- `TM2003 GRADIENT_CONTAINS_INF`
- `TM2004 GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD`
- `TM2005 GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD`

## 参数更新监控

`watch_updates()` 使用 PyTorch optimizer step pre-hook 和 post-hook 观察真实发生的
`optimizer.step()`。它不会 monkey patch `optimizer.step()`，不会替换 optimizer，
不会修改 closure、step 参数或返回值，也不会主动调用 backward、step、`zero_grad()`，
或执行梯度裁剪。

monitor 会在 session start 和每次 step pre-hook 时刷新 model/optimizer 参数关系和学习率。
`step_count` 表示成功到达 post-hook 的 completed step；`attempted_step_count` 包含进入
pre-hook 但未完成的 step。

```python
import torch
from torch import nn

from trainmedic import watch_updates
from trainmedic.reports.console import format_diagnostics


model = nn.Linear(2, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with watch_updates(model, optimizer) as monitor:
    optimizer.zero_grad()
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

print(format_diagnostics(monitor.diagnostics))
```

健康输出：

```text
TrainMedic found no diagnostics.
```

如果 context 正常退出但没有观察到成功完成的 `optimizer.step()`，诊断只限定当前
monitor session：

```text
[1] TM4001 WARNING - Optimizer step was not observed
Message: No optimizer.step() call was observed during this monitor session.
Evidence:
  - selected_parameter_count: 1
  - successful_step_count: 0
  - optimizer_group_count: 1
```

如果参数有有限非零梯度，但只属于零学习率组，会报告 `TM4002`，且不会对同一参数再报告
`TM4003`：

```text
[1] TM4002 WARNING - Finite nonzero gradients are in zero learning-rate groups
Message: At optimizer step 1, one or more parameters had finite nonzero gradients but belonged only to optimizer groups with learning rate 0.
Evidence:
  - step_index: 1
  - affected_parameter_count: 1
  - affected_parameter_names_preview: ["weight"]
  - omitted_parameter_count: 0
  - optimizer_group_indices: [0]
  - learning_rate_values: [0.0]
  - selection_count: 1
```

如果参数有有限非零梯度、已知非零学习率，并且 before/after 未检测到变化，会报告
`TM4003`：

```text
[1] TM4003 WARNING - Parameter update was not detected
Message: At optimizer step 1, no parameter change was detected for one or more parameters with finite nonzero gradients.
Evidence:
  - step_index: 1
  - candidate_parameter_count: 1
  - changed_candidate_count: 0
  - unchanged_candidate_count: 1
  - exact_unchanged_count: 1
  - sampled_unchanged_count: 0
  - unsupported_or_skipped_count: 0
  - unchanged_parameter_names_preview: ["weight"]
  - omitted_parameter_count: 0
  - per_parameter_preview: [{"name": "weight", "aliases": ["weight"], "group_index": 0, "learning_rate": 0.1, "coverage": "exact", "parameter_numel": 2, "sampled_element_count": 2, "gradient_norm": 1.4142135381698608}]
  - configured_sample_size: 64
  - configured_max_snapshot_elements: 100000
```

只监控同时属于 model 与 optimizer、且 `requires_grad=True` 的参数。optimizer 中的
模型外参数仍由 `TM1002` 负责；optimizer 遗漏的可训练模型参数仍由 `TM1001` 负责。

只有 `finite_nonzero` 梯度会成为更新候选。`grad=None`、全零梯度、NaN/Inf 梯度、
unsupported 梯度或 snapshot、以及明确零学习率组不会产生 `TM4003`。

参数更新监控使用有界 snapshot。默认值是 `sample_size=64` 和
`max_snapshot_elements=100_000`。小参数且预算足够时 coverage 为 `exact`，会比较全部元素。
大参数使用确定性 sampled flat indices。sampled 模式可能遗漏只发生在非采样位置的更新，
因此 TrainMedic 只表述“采样元素未检测到变化”，不会把 sampled 结果写成绝对结论。

LBFGS 和复杂 closure-based optimizer 语义尚未正式支持，因为梯度可能在
`optimizer.step(closure)` 内部才计算。TrainMedic 不会修改 closure 或其返回值。

AMP GradScaler 内部、`torch.compile`、TorchScript、distributed training、DeepSpeed、
FSDP 和 Lightning 尚未正式支持。

Phase 4 支持的诊断代码：

- `TM4000 NO_PARAMETERS_SELECTED_FOR_UPDATE_MONITORING`
- `TM4001 OPTIMIZER_STEP_NOT_OBSERVED`
- `TM4002 ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES`
- `TM4003 PARAMETER_UPDATE_NOT_DETECTED`
- `TM4004 OPTIMIZER_STEP_DID_NOT_COMPLETE`
- `TM4005 UPDATE_CHECK_SKIPPED_FOR_UNKNOWN_LEARNING_RATE`

## 模式监控

`watch_modes()` 会在实际 forward 调用期间检查 runtime train/eval 模式和梯度上下文。
用户必须显式提供 expected phase；TrainMedic 不会根据 optimizer、backward 或
`model.training` 猜测阶段，也不会自动调用 `train()` 或 `eval()`。

```python
import torch
from torch import nn

from trainmedic import watch_modes
from trainmedic.reports.console import format_diagnostics


model = nn.Linear(2, 1)
model.eval()

with watch_modes(model, expected_mode="train") as monitor:
    model(torch.ones(1, 2))

print(format_diagnostics(monitor.diagnostics))
```

`TM5001` 输出：

```text
[1] TM5001 WARNING - Root model is in eval mode during training
Object: <root>
Message: The root model was first observed in eval mode during an expected training forward.
Evidence:
  - sequence_index: 1
  - module_name: <root>
  - module_aliases: ["<root>"]
  - module_type: torch.nn.modules.linear.Linear
  - module_call_index: 1
  - is_root: true
  - expected_mode: train
  - actual_mode: eval
  - grad_enabled: true
  - inference_mode_enabled: false
```

Dropout 和 BatchNorm 有专门诊断，因为 train/eval 会直接影响随机性和统计量：

```text
[1] TM5004 WARNING - Dropout mode differs from expected phase
Object: dropout
Message: A called Dropout module was first observed in a mode that differs from the expected phase.
```

```text
[1] TM5005 WARNING - BatchNorm mode differs from expected phase
Object: batch_norm
Message: A called BatchNorm module was first observed in a mode that differs from the expected phase.
```

评估阶段仍启用梯度跟踪是 INFO，因为普通验证通常使用 `torch.no_grad()`，但梯度型评估、
可解释性和对抗攻击可能有意需要梯度：

```text
[1] TM5006 INFO - Gradient tracking is enabled during evaluation
Object: <root>
Message: Gradient tracking was enabled during an expected evaluation forward.
```

模式监控使用唯一 Module 对象上的局部 forward pre-hook。它只诊断当前 session 中实际调用的
模块，保留共享模块 aliases，不修改 `.training`，不替换输入，也不保存 input/output。
Forward pre-hook 有诊断开销，建议只在聚焦排查时启用。

Phase 5 支持的诊断代码：

- `TM5000 FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING`
- `TM5001 MODEL_IN_EVAL_DURING_TRAINING`
- `TM5002 MODEL_IN_TRAIN_DURING_EVALUATION`
- `TM5003 CALLED_SUBMODULE_MODE_MISMATCH`
- `TM5004 DROPOUT_MODE_MISMATCH`
- `TM5005 BATCHNORM_MODE_MISMATCH`
- `TM5006 GRADIENT_TRACKING_ENABLED_DURING_EVALUATION`
- `TM5007 GRADIENT_TRACKING_DISABLED_DURING_TRAINING`

## 运行检查

```bash
pytest --cov=trainmedic --cov-report=term-missing
ruff check .
mypy src/trainmedic
```

## 路线图

- Phase 0：项目骨架、工具链、CI 和诊断数据结构。
- Phase 1：静态模型与 optimizer 检查。
- Phase 2：forward 数值异常监控。
- Phase 3：累积参数梯度监控。
- Phase 4：围绕 optimizer step 的有界参数更新监控。
- Phase 5：train/eval 模式与梯度上下文检查。

## 贡献说明

请保持改动小而基于证据。新增诊断行为应包含聚焦测试，证明问题可以被检测到，
并且修复后的场景不再报告该诊断。
