import torch
from torch import nn

from trainmedic import watch_updates


def test_step_not_observed_diagnostic_case() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        model(torch.ones(1, 2)).sum().backward()

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM4001",)
