from typing import Any

import torch
from torch import nn

from trainmedic import watch_updates


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Any) -> None:
        super().__init__(params, {"lr": 0.1})

    def step(self, closure: Any = None) -> Any:
        return closure() if closure is not None else None


def test_parameter_not_updated_diagnostic_case() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = NoOpOptimizer(model.parameters())

    with watch_updates(model, optimizer) as monitor:
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM4003",)
