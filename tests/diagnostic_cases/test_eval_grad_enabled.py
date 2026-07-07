import torch
from torch import nn

from trainmedic import watch_modes


def test_eval_grad_enabled_diagnostic_case() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="eval") as monitor:
        model(torch.ones(1, 2))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM5006",)
