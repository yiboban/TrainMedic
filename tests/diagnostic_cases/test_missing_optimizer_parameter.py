import torch
from torch import nn

from trainmedic import inspect_optimizer


class MissingParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 3, bias=False)
        self.decoder = nn.Linear(3, 1, bias=False)


def test_missing_optimizer_parameter_case() -> None:
    model = MissingParameterModel()
    optimizer = torch.optim.SGD(model.encoder.parameters(), lr=0.1)

    diagnostics = inspect_optimizer(model, optimizer)

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM1001",)
    assert diagnostics[0].object_name == "decoder.weight"
