import torch
from torch import nn

from trainmedic import watch_forward
from trainmedic.reports.console import format_diagnostics


class HealthyForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


model = HealthyForwardModel()
inputs = torch.ones(1, 2)

with watch_forward(model) as monitor:
    model(inputs)

print(format_diagnostics(monitor.diagnostics))
