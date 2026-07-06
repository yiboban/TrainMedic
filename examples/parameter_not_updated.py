from typing import Any

import torch
from torch import nn

from trainmedic import watch_updates
from trainmedic.reports.console import format_diagnostics


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Any) -> None:
        super().__init__(params, {"lr": 0.1})

    def step(self, closure: Any = None) -> Any:
        return closure() if closure is not None else None


def main() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = NoOpOptimizer(model.parameters())

    with watch_updates(model, optimizer) as monitor:
        optimizer.zero_grad()
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()

    print(format_diagnostics(monitor.diagnostics))


if __name__ == "__main__":
    main()
