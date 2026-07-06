"""Model module collection utilities."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

ROOT_MODULE_NAME = "<root>"


@dataclass(frozen=True, eq=False)
class ModelModuleRecord:
    """A unique model module and all names that refer to it."""

    module: nn.Module
    primary_name: str
    aliases: tuple[str, ...]
    is_root: bool
    is_leaf: bool

    @property
    def module_id(self) -> int:
        """Object identity for comparing modules without equality overloads."""
        return id(self.module)

    @property
    def module_type(self) -> str:
        """Stable module type label for diagnostics."""
        module_type = type(self.module)
        return f"{module_type.__module__}.{module_type.__qualname__}"


def collect_model_modules(model: nn.Module) -> tuple[ModelModuleRecord, ...]:
    """Collect unique model modules and aliases, preserving first-seen order."""
    modules_by_id: dict[int, nn.Module] = {}
    aliases_by_id: dict[int, list[str]] = {}
    order_by_id: dict[int, int] = {}
    root_id = id(model)

    for order, (name, module) in enumerate(model.named_modules(remove_duplicate=False)):
        module_id = id(module)
        alias = ROOT_MODULE_NAME if name == "" else name
        modules_by_id.setdefault(module_id, module)
        aliases_by_id.setdefault(module_id, []).append(alias)
        order_by_id.setdefault(module_id, order)

    records = []
    for module_id, module in modules_by_id.items():
        aliases = tuple(sorted(aliases_by_id[module_id], key=_module_name_sort_key))
        records.append(
            ModelModuleRecord(
                module=module,
                primary_name=aliases[0],
                aliases=aliases,
                is_root=module_id == root_id,
                is_leaf=not any(True for _ in module.children()),
            )
        )

    return tuple(sorted(records, key=lambda record: order_by_id[record.module_id]))


def _module_name_sort_key(name: str) -> tuple[int, str]:
    return (0, "") if name == ROOT_MODULE_NAME else (1, name)
