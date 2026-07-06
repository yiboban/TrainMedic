import torch
from torch import nn

from trainmedic import inspect_optimizer


class HealthyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 2)


def test_healthy_optimizer_case() -> None:
    model = HealthyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert inspect_optimizer(model, optimizer) == ()
