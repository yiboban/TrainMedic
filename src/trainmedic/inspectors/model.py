"""Model parameter collection utilities."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, eq=False)
class ModelParameterRecord:
    """A unique model parameter and all names that refer to it."""

    parameter: nn.Parameter
    primary_name: str
    aliases: tuple[str, ...]

    @property
    def parameter_id(self) -> int:
        """Object identity for comparing parameters without Tensor equality."""
        return id(self.parameter)

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
        """Whether autograd should compute gradients for this parameter."""
        return bool(self.parameter.requires_grad)


def collect_model_parameters(model: nn.Module) -> tuple[ModelParameterRecord, ...]:
    """Collect unique model parameters and all aliases for shared parameters.

    Parameters are grouped by Python object identity. This handles tied weights correctly
    and avoids Tensor equality comparisons.
    """
    parameters_by_id: dict[int, nn.Parameter] = {}
    aliases_by_id: dict[int, list[str]] = {}

    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_id = id(parameter)
        parameters_by_id.setdefault(parameter_id, parameter)
        aliases_by_id.setdefault(parameter_id, []).append(name)

    records = []
    for parameter_id, parameter in parameters_by_id.items():
        aliases = tuple(sorted(aliases_by_id[parameter_id]))
        records.append(
            ModelParameterRecord(
                parameter=parameter,
                primary_name=aliases[0],
                aliases=aliases,
            )
        )

    return tuple(sorted(records, key=lambda record: record.primary_name))
