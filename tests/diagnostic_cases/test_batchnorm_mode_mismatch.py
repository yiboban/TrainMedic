import torch
from torch import nn

from trainmedic import watch_modes


class BatchNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(inputs)


def test_batchnorm_mode_mismatch_diagnostic_case() -> None:
    model = BatchNormModel()
    model.eval()
    model.batch_norm.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(2, 2))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM5005",)
