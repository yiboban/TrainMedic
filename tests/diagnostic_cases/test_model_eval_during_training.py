import torch
from torch import nn

from trainmedic import watch_modes


def test_model_eval_during_training_diagnostic_case() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM5001",)
