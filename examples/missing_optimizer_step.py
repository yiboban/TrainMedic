import torch
from torch import nn

from trainmedic import watch_updates
from trainmedic.reports.console import format_diagnostics


def main() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        optimizer.zero_grad()
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()

    print(format_diagnostics(monitor.diagnostics))


if __name__ == "__main__":
    main()
