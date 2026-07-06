"""Optimizer parameter collection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.optim import Optimizer


@dataclass(frozen=True, eq=False)
class OptimizerParameterRecord:
    """A parameter reference found in an optimizer parameter group."""

    parameter: torch.Tensor
    group_index: int
    parameter_index: int

    @property
    def parameter_id(self) -> int:
        """Object identity for comparing parameters without Tensor equality."""
        return id(self.parameter)

    @property
    def position_name(self) -> str:
        """Stable optimizer position name for diagnostics."""
        return f"optimizer.param_groups[{self.group_index}].params[{self.parameter_index}]"

    @property
    def shape(self) -> tuple[int, ...]:
        """Parameter shape as plain integers."""
        return tuple(int(size) for size in self.parameter.shape)

    @property
    def dtype(self) -> str:
        """Parameter dtype as a stable string."""
        return str(self.parameter.dtype)

    @property
    def device(self) -> str:
        """Parameter device as a stable string."""
        return str(self.parameter.device)

    @property
    def requires_grad(self) -> bool:
        """Whether autograd should compute gradients for this optimizer parameter."""
        return bool(self.parameter.requires_grad)


def collect_optimizer_parameters(optimizer: Optimizer) -> tuple[OptimizerParameterRecord, ...]:
    """Collect optimizer parameters with stable group and parameter indices."""
    records: list[OptimizerParameterRecord] = []

    for group_index, group in enumerate(optimizer.param_groups):
        params = cast("list[Any]", group.get("params", []))
        for parameter_index, parameter in enumerate(params):
            tensor = cast(torch.Tensor, parameter)
            records.append(
                OptimizerParameterRecord(
                    parameter=tensor,
                    group_index=group_index,
                    parameter_index=parameter_index,
                )
            )

    return tuple(records)
