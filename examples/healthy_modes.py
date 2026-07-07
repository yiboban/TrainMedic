import torch
from torch import nn

from trainmedic import watch_modes
from trainmedic.reports.console import format_diagnostics


def main() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())

    model.train()
    with watch_modes(model, expected_mode="train") as train_monitor:
        model(torch.ones(1, 2))

    model.eval()
    with watch_modes(model, expected_mode="eval") as eval_monitor, torch.no_grad():
        model(torch.ones(1, 2))

    print(format_diagnostics(train_monitor.diagnostics + eval_monitor.diagnostics))


if __name__ == "__main__":
    main()
