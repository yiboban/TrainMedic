from typing import Any

import torch
from torch import nn

from trainmedic import watch_gradients


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


def test_gradient_nan_case() -> None:
    model = NaNGradientModel()

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM2002",)
