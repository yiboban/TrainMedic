from typing import Any

import torch
from torch import nn

from trainmedic import watch_gradients
from trainmedic.reports.console import format_diagnostics


class NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        return torch.full((2,), float("nan"))


class NaNGradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self) -> torch.Tensor:
        return NaNGradient.apply(self.weight).sum()


model = NaNGradientModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with watch_gradients(model, optimizer) as monitor:
    loss = model()
    loss.backward()
    monitor.check_gradients()

print(format_diagnostics(monitor.diagnostics))
