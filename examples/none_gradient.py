import torch
from torch import nn

from trainmedic import watch_gradients
from trainmedic.reports.console import format_diagnostics


class BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.used(inputs)


model = BranchModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
inputs = torch.ones(1, 2)

with watch_gradients(model, optimizer) as monitor:
    loss = model(inputs).sum()
    loss.backward()
    monitor.check_gradients()

print(format_diagnostics(monitor.diagnostics))
