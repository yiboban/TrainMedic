import torch
from torch import nn

from trainmedic import watch_gradients
from trainmedic.reports.console import format_diagnostics


class HealthyGradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


model = HealthyGradientModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
inputs = torch.ones(2)

with watch_gradients(model, optimizer) as monitor:
    loss = model(inputs).sum()
    loss.backward()
    monitor.check_gradients()

print(format_diagnostics(monitor.diagnostics))
