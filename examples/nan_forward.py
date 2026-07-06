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
inputs = torch.tensor([-1.0, 1.0])

with watch_forward(model) as monitor:
    model(inputs)

print(format_diagnostics(monitor.diagnostics))
