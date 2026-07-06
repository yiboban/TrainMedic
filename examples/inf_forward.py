import torch
from torch import nn

from trainmedic import watch_forward
from trainmedic.reports.console import format_diagnostics


class DivideByZero(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / torch.zeros_like(inputs)


class InfForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.divide_by_zero = DivideByZero()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.divide_by_zero(inputs)


model = InfForwardModel()
inputs = torch.tensor([1.0, 2.0])

with watch_forward(model) as monitor:
    model(inputs)

print(format_diagnostics(monitor.diagnostics))
