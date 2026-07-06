import torch
from torch import nn

from trainmedic import watch_forward


class InvalidLog(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(inputs)


class NaNForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_log = InvalidLog()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.invalid_log(inputs)


def test_forward_nan_case() -> None:
    model = NaNForwardModel()

    with watch_forward(model) as monitor:
        model(torch.tensor([-1.0, 1.0]))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM3001",)
    assert monitor.diagnostics[0].object_name == "invalid_log"
