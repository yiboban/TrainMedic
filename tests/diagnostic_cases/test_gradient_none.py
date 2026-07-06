import torch
from torch import nn

from trainmedic import watch_gradients


class BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.used(inputs)


def test_gradient_none_case() -> None:
    model = BranchModel()

    with watch_gradients(model) as monitor:
        model(torch.ones(1, 2)).sum().backward()
        diagnostics = monitor.check_gradients()

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM2001",)
