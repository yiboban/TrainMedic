import torch
from torch import nn

from trainmedic import watch_forward


class DivideByZero(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / torch.zeros_like(inputs)


class InfForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.divide_by_zero = DivideByZero()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.divide_by_zero(inputs)


def test_forward_inf_case() -> None:
    model = InfForwardModel()

    with watch_forward(model) as monitor:
        model(torch.tensor([1.0, 2.0]))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM3002",)
    assert monitor.diagnostics[0].object_name == "divide_by_zero"
