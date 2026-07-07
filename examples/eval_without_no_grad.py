import torch
from torch import nn

from trainmedic import watch_modes
from trainmedic.reports.console import format_diagnostics


def main() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="eval") as monitor:
        model(torch.ones(1, 2))

    print(format_diagnostics(monitor.diagnostics))


if __name__ == "__main__":
    main()
