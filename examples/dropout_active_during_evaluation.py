import torch
from torch import nn

from trainmedic import watch_modes
from trainmedic.reports.console import format_diagnostics


class DropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs)


def main() -> None:
    model = DropoutModel()
    model.eval()
    model.dropout.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(2, 2))

    print(format_diagnostics(monitor.diagnostics))


if __name__ == "__main__":
    main()
