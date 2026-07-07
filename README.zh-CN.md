# TrainMedic

[English](README.md) | 简体中文

[![CI](https://github.com/yiboban/TrainMedic/actions/workflows/ci.yml/badge.svg)](https://github.com/yiboban/TrainMedic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个帮助你定位 PyTorch 模型为什么没有正常训练的诊断工具。

TrainMedic 会检查模型参数、forward 输出、梯度、optimizer 更新以及 train/eval
模式，但不会自动改变你的训练行为。

## 为什么需要 TrainMedic？

PyTorch 训练问题经常不容易定位：

- loss 突然变成 NaN；
- 某些参数一直是 `grad=None`；
- 创建 optimizer 时漏掉了一部分模型；
- 明明调用了 `optimizer.step()`，参数却没有变化；
- 验证阶段不小心跑在 train mode；
- Dropout 或 BatchNorm 使用了错误的行为。

TrainMedic 会观察这些信号，并给出结构化诊断：包含证据、可能原因和下一步建议。

它可以帮你回答：

- 哪些可训练参数在创建 optimizer 时被漏掉了？
- 第一个 NaN 或 Inf 是在哪里被观察到的？
- 哪些参数是 `grad=None`？
- `optimizer.step()` 是否真的更新了参数？
- 模型是否使用了错误的 train/eval 模式？

## 30 秒快速开始

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

精简后的输出：

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

TrainMedic 报告的是“首次观察到异常输出”的位置。它不会自动断言该模块一定就是根因。

## 安装

TrainMedic 目前还没有发布到 PyPI。请从仓库安装：

```bash
git clone https://github.com/yiboban/TrainMedic.git
cd TrainMedic
python -m pip install -e .
```

如果要运行测试和开发工具：

```bash
python -m pip install -e ".[dev]"
```

要求：

- Python >= 3.10
- PyTorch >= 2.1

## 可以检查什么？

### Optimizer 参数遗漏

适合排查“训练循环正常运行，但模型某一部分一直学不动”的情况。

```python
from trainmedic import inspect_optimizer

diagnostics = inspect_optimizer(model, optimizer)
```

它可以发现：

- 可训练模型参数没有加入 optimizer；
- optimizer 中包含不属于当前模型的参数；
- 冻结参数仍然被 optimizer 管理；
- 整个模型的参数都被冻结；
- optimizer 参数组中重复注册了同一个参数。

### Forward NaN / Inf

适合在 loss 变成非有限值时，寻找第一个可疑的模块输出。

```python
from trainmedic import watch_forward

with watch_forward(model) as monitor:
    output = model(inputs)
```

Forward 监控会：

- 报告首次观察到 NaN 或 Inf 的输出；
- 支持嵌套的 `list`、`tuple` 和 `dict` 输出；
- 不保存完整 activation；
- 不声称首次观察到异常的模块一定是最早根因。

### 参数梯度

适合排查分支未使用、计算图断开、梯度变成 NaN/Inf 等问题。

```python
from trainmedic import watch_gradients

with watch_gradients(model, optimizer) as monitor:
    loss.backward()
    monitor.check_gradients()
```

请在 `backward()` 之后、`optimizer.step()` 或 `zero_grad()` 之前调用
`check_gradients()`。

它可以发现：

- `grad=None`；
- gradient NaN；
- gradient Inf；
- 用户显式配置的全局梯度范数阈值异常。

### 参数是否真的更新

适合排查“梯度存在，但训练仍然没有动”的情况。

```python
from trainmedic import watch_updates

with watch_updates(model, optimizer) as monitor:
    loss.backward()
    optimizer.step()
```

它可以发现：

- 当前 monitor session 没有观察到 `optimizer.step()`；
- finite nonzero gradient 位于零学习率参数组；
- optimizer step 进入了 pre-hook 但没有完成；
- finite nonzero gradient 存在，但没有检测到参数更新；
- learning rate 无法解释为有限标量，因此跳过更新检查并给出证据。

大参数会使用确定性采样元素进行检查。采样结果表示“采样位置没有检测到变化”，
不表示已经证明整个参数完全没有变化。

### Train / Eval 模式

适合排查验证阶段误用 train 行为，或者训练阶段误用 eval 行为。

```python
from trainmedic import watch_modes

model.eval()

with watch_modes(model, expected_mode="eval") as monitor:
    with torch.no_grad():
        output = model(inputs)
```

它可以发现：

- 期望训练时 root model 处于 eval mode；
- 期望评估时 root model 处于 train mode；
- Dropout 模式不符合预期；
- BatchNorm 模式不符合预期；
- eval 阶段仍开启 gradient tracking；
- train 阶段关闭了 gradient tracking。

TrainMedic 永远不会替你调用 `model.train()` 或 `model.eval()`。你必须显式提供
期望模式。

## 如何阅读诊断结果

一条诊断可能包含：

- `code`：稳定的问题编号，例如 `TM3001`；
- `severity`：`INFO`、`WARNING`、`ERROR` 或 `CRITICAL`；
- `object`：相关模块或参数；
- `message`：观察到了什么；
- `evidence`：支持诊断的具体观测数据；
- `possible_causes`：可以优先排查的可能原因；
- `suggestions`：下一步建议。

Possible causes 是基于证据给出的假设，不是保证正确的根因结论。

## 支持矩阵

| 领域                                 | 当前状态                 |
| ------------------------------------ | ------------------------ |
| 模型/optimizer 参数关系检查          | Yes                      |
| Forward NaN/Inf 监控                 | Yes                      |
| 参数梯度诊断                         | Yes                      |
| 参数更新监控                         | Yes                      |
| Train/eval 模式监控                  | Yes                      |
| CPU                                  | Yes                      |
| CUDA                                 | Basic support            |
| `torch.compile`                      | Not officially supported |
| DDP/FSDP/DeepSpeed                   | Not officially supported |
| Lightning/Transformers Trainer       | Not officially supported |
| 自动修复                             | No                       |

诊断代码路径支持 CUDA Tensor，但 `.item()` 等统计操作可能触发设备同步。
TrainMedic 还没有在所有 CUDA 设备、分布式训练栈和训练框架中完成验证。

## 项目状态

TrainMedic 当前处于 Alpha 测试阶段。

核心诊断功能已经可以使用，并通过 Python 3.10、3.11、3.12 的 CPU CI 测试。
但是 API 仍可能调整，也尚未覆盖所有 PyTorch 训练框架和运行环境。

## 当前限制

- 尚无统一的 `watch()` 高层入口。
- 尚未发布到 PyPI。
- 不正式支持 `torch.compile` 和 TorchScript。
- 不正式支持 DDP、FSDP、DeepSpeed、Lightning 和 Transformers Trainer。
- 不完整诊断 AMP GradScaler 内部语义。
- 不正式支持 LBFGS 等复杂 closure optimizer。
- sampled 参数更新检查可能遗漏未采样位置的变化。
- hook 和 tensor 统计会带来额外开销，并可能同步 CUDA。
- 建议只在少量诊断 step 中启用，不要长期用于性能 benchmark。

更多细节见 [架构说明](docs/architecture.md) 和
[诊断规则](docs/diagnostic-rules.md)。

## 示例

安装 editable 版本后，可以运行示例：

```bash
python examples/nan_forward.py
```

常用示例：

- [examples/missing_optimizer_parameter.py](examples/missing_optimizer_parameter.py)：
  optimizer 创建时漏掉了一部分模型。
- [examples/nan_forward.py](examples/nan_forward.py)：forward 输出首次出现 NaN。
- [examples/none_gradient.py](examples/none_gradient.py)：某个被选中的参数是
  `grad=None`。
- [examples/nan_gradient.py](examples/nan_gradient.py)：参数梯度出现 NaN。
- [examples/missing_optimizer_step.py](examples/missing_optimizer_step.py)：没有观察到
  `optimizer.step()`。
- [examples/zero_learning_rate.py](examples/zero_learning_rate.py)：梯度有限且非零，
  但学习率为零。
- [examples/model_eval_during_training.py](examples/model_eval_during_training.py)：
  期望训练时 root model 处于 eval mode。
- [examples/dropout_active_during_evaluation.py](examples/dropout_active_during_evaluation.py)：
  期望评估时 Dropout 仍然处于激活状态。

完整列表见 [examples guide](examples/README.md)。

## 开发检查

```bash
pytest --cov=trainmedic --cov-report=term-missing
ruff check .
mypy src/trainmedic
```

项目在 GitHub Actions 中使用 CPU PyTorch 测试 Python 3.10、3.11 和 3.12。

## 贡献

最有价值的贡献包括：

- 能复现真实训练问题的最小脚本；
- false positive 报告；
- TrainMedic 尚不能正确处理的模型或 optimizer；
- 边界情况测试；
- 更清晰的诊断信息和文档。

提交 issue 时，建议提供：

- Python 版本；
- PyTorch 版本；
- device；
- 最小复现代码；
- TrainMedic 输出；
- 你期望的行为。

欢迎贡献，尤其是小而可复现、并带有测试的改动。

## 路线图

- 统一的 `watch()` session
- 首个 GitHub alpha release
- PyPI packaging
- 更完整的 CUDA 验证
- AMP 和训练框架集成
- 真实诊断案例库
