import torch
from torch import nn

from trainmedic import watch_modes


def test_model_train_during_evaluation_diagnostic_case() -> None:
    model = nn.Linear(2, 1)
    model.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(1, 2))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM5002",)
