# TrainMedic Xiaohongshu Image Plan

All images use `1080 x 1440` vertical format, dark technical background, medical
diagnostic accents, terminal cards, and the same TrainMedic top label and page number.

## 01-cover

Text:

- PyTorch 训不动？
- 我做了个开源工具
- 自动帮你找问题
- NaN｜grad=None｜参数没更新
- 开源项目
- TrainMedic

Design:

- Large headline as the main visual focus.
- Terminal card shows a compact diagnostic output.
- Heartbeat line reinforces the "training medic" metaphor.
- Red is used for `ERROR`, green/blue for evidence and suggestions.

## 02-pain-points

Text:

- 这些问题你遇到过吗？
- loss 突然变成 NaN
- 有些参数一直 grad=None
- 训练正常运行，但参数根本没更新
- optimizer 创建时漏掉了某一层
- 验证时忘了 model.eval()
- 代码不一定报错，但模型就是学不会。

Design:

- Five pain points are shown as separate cards.
- Each card uses a small numbered icon so users can scan quickly on mobile.

## 03-why-trainmedic

Text:

- 以前只能这样排查
- 一层层 print
- 手动检查梯度
- 反复查看 optimizer
- 猜是哪一步出了问题
- 所以我做了 TrainMedic
- 让训练问题变成有证据的诊断结果。

Design:

- Left side presents the old manual workflow.
- Right side presents a structured diagnostic panel.
- The layout shows "from messy guessing to evidence".

## 04-nan-example

Text:

- 不用再一层层 print
- `with watch_forward(model) as monitor:`
- `output = model(inputs)`
- TM3001 ERROR
- Forward output contains NaN
- Object: invalid_log
- tensor_path: output
- nan_count: 1
- 它报告的是首次观察到异常的位置，不会武断地说这一定是根因。

Design:

- Top terminal card shows the real public API.
- Diagnostic card uses fields verified from `python examples/nan_forward.py`.
- Bottom note keeps the NaN claim precise and not exaggerated.

## 05-features

Text:

- 它现在能检查什么？
- 01 optimizer 有没有漏参数
- 02 Forward 哪一层首次出现 NaN / Inf
- 03 哪些参数 grad=None 或梯度异常
- 04 optimizer.step 后参数有没有变化
- 05 train / eval、Dropout、BatchNorm 是否正确
- 不是自动修复，而是把问题讲清楚。

Design:

- Five rows map to the five current user-facing diagnostic areas.
- No internal phase numbers are used.
- The last line prevents users from interpreting TrainMedic as an auto-fixer.

## 06-diagnostic-structure

Text:

- 不只是告诉你“出错了”
- 问题对象
- Object: decoder.weight
- 观测证据
- requires_grad: true
- 可能原因
- optimizer 创建时漏掉模块
- 排查建议
- 检查 optimizer 参数来源
- 每个结论都尽量附带可验证证据，而不是只给出模糊猜测。

Design:

- One central diagnostic card split into four clear parts.
- Evidence field names mimic real TrainMedic console output without overloading the
  image with terminal text.

## 07-project-status

Text:

- 目前是 Alpha 测试版
- 已具备：核心诊断功能已经可以运行；Python 3.10 / 3.11 / 3.12 CI；CPU 可用；
  基础 CUDA Tensor 支持；完全开源
- 当前限制：还没有发布到 PyPI；API 后续可能调整；尚未正式支持 DDP / FSDP /
  DeepSpeed；尚未正式支持 Lightning / Transformers Trainer
- 适合现在试用和反馈，暂时不建议直接作为生产依赖。

Design:

- Two cards separate available capabilities from current limitations.
- Uses honest wording and does not hide Alpha status.

## 08-call-to-action

Text:

- 比起 Star
- 我更需要真实 Bug
- 你遇到过哪些难排查的 PyTorch 问题？
- 它有没有误报？
- 哪些模型或 optimizer 还不能支持？
- GitHub 搜索：yiboban/TrainMedic
- 欢迎试用、提 Issue、提交最小复现

Design:

- Ends with a direct feedback request rather than a hard sell.
- Uses a simple self-drawn git-graph mark, not external logo artwork.
- No QR code is included.
