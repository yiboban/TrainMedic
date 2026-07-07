# TrainMedic Xiaohongshu Copywriting

## 3 Alternate Titles

1. 我把 PyTorch 训练里的 NaN 和梯度 Bug 做成了开源工具
2. PyTorch loss 变 NaN、梯度异常后，我写了个开源工具定位训练问题
3. 开源｜不用一层层 print，自动检查 PyTorch NaN、梯度和训练问题

## Full Post Body

最近复现论文和调 PyTorch 模型时，我又被几类问题折磨了一遍：

loss 突然变成 NaN。

有些参数一直是 `grad=None`。

训练循环看起来正常跑完了，但参数根本没更新。

optimizer 创建时不小心漏掉了一层。
验证时忘了 `model.eval()`，Dropout 和 BatchNorm 行为都不对。

这些问题最麻烦的地方是：代码不一定报错，但模型就是学不会。

以前我通常只能一层层 print，手动查梯度，反复看 optimizer 参数组，然后猜是哪一步出了问题。

所以我做了一个开源小工具：TrainMedic。

它现在可以帮你检查 5 类 PyTorch 训练问题：

1. optimizer 有没有漏掉可训练参数；
2. Forward 哪一层首次观察到 NaN / Inf；
3. 哪些参数是 `grad=None`，或者梯度出现 NaN / Inf；
4. `optimizer.step()` 后参数有没有真的变化；
5. train / eval mode、Dropout、BatchNorm、评估时梯度开关是否符合预期。

一个简单例子：

```python
with watch_forward(model) as monitor:
    output = model(inputs)
```

如果 forward 输出里出现 NaN，它会给出类似这样的诊断：

```text
TM3001 ERROR
Forward output contains NaN
Object: invalid_log
tensor_path: output
nan_count: 1
```

这里要说清楚：TrainMedic 报告的是“首次观察到异常输出的位置”，不会武断地说这一定是根因。它更像一个训练诊断面板，把证据、可能原因和下一步建议整理出来，帮你更快缩小排查范围。

项目目前还是 Alpha 测试版。

核心诊断功能已经可以运行，也有 Python 3.10 / 3.11 / 3.12 的 CPU CI。它还没有发布到 PyPI，API 后续也可能调整；DDP / FSDP / DeepSpeed、Lightning、Transformers Trainer 这些训练栈也还没有正式支持。

所以它现在更适合：试用、排查小规模问题、提交真实复现和误报反馈。

GitHub 搜索：yiboban/TrainMedic

相比单纯获得 Star，我更希望收集真实训练故障案例和误报反馈。
如果你遇到过很难排查的 PyTorch 训练问题，欢迎提 Issue，最好带一个最小复现。

## Short Post Body

我把 PyTorch 里最折磨人的几类训练问题，做成了一个开源诊断工具：TrainMedic。

它现在可以检查：

- optimizer 有没有漏掉可训练参数；
- Forward 哪一层首次观察到 NaN / Inf；
- 哪些参数是 `grad=None`，或者梯度异常；
- `optimizer.step()` 后参数有没有真的变化；
- train / eval mode、Dropout、BatchNorm 是否符合预期。

比如 forward 里出现 NaN 时，可以这样用：

```python
with watch_forward(model) as monitor:
    output = model(inputs)
```

它会给出诊断编号、问题对象、观测证据、可能原因和排查建议。注意它不会武断地说“这就是根因”，而是报告“首次观察到异常输出的位置”，帮你更快缩小排查范围。

项目目前还是 Alpha 测试版，还没有发布到 PyPI，也没有正式支持 DDP / FSDP / DeepSpeed、Lightning、Transformers Trainer。适合现在试用、反馈真实训练问题和误报。

GitHub 搜索：yiboban/TrainMedic

相比单纯获得 Star，更希望收集真实训练故障案例和误报反馈。遇到问题可以直接提 Issue，最好附最小复现。

## Pinned Comment

项目完全开源，GitHub 搜索 `yiboban/TrainMedic` 即可找到。

目前还是 Alpha 版本，最希望收到真实训练故障案例和误报反馈。遇到问题可以直接提 Issue，最好带一个最小复现。

## Recommended Hashtags

#PyTorch
#NaN
#梯度
#深度学习
#机器学习
#开源项目
#人工智能
#程序员
#科研工具
#GitHub
