"""Forward output numerical monitoring."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, TypeAlias

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from trainmedic.inspectors.modules import ModelModuleRecord, collect_model_modules
from trainmedic.rules.numerical_rules import (
    FORWARD_OUTPUT_CONTAINS_INF,
    FORWARD_OUTPUT_CONTAINS_NAN,
    diagnose_forward_observation,
)
from trainmedic.types import Diagnostic

ModuleScope: TypeAlias = Literal["all", "leaf"]


@dataclass(frozen=True)
class ForwardTensorObservation:
    """Summary of a tensor observed in a module forward output."""

    sequence_index: int
    module_name: str
    module_aliases: tuple[str, ...]
    module_type: str
    module_call_index: int
    tensor_path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    nan_count: int
    inf_count: int


class ForwardMonitor:
    """Context manager that monitors module forward outputs for NaN and Inf."""

    def __init__(
        self,
        model: nn.Module,
        *,
        module_scope: ModuleScope = "all",
    ) -> None:
        if module_scope not in ("all", "leaf"):
            raise ValueError("module_scope must be one of: 'all', 'leaf'")

        self._model = model
        self._module_scope = module_scope
        self._handles: list[RemovableHandle] = []
        self._diagnostics: list[Diagnostic] = []
        self._module_call_counts: dict[int, int] = {}
        self._sequence_index = 0
        self._unsupported_tensor_count = 0
        self._active = False

    def __enter__(self) -> ForwardMonitor:
        """Start monitoring and return this monitor."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Clean up hooks without suppressing model exceptions."""
        self.close()
        return False

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Diagnostics collected during the active or completed monitor session."""
        return tuple(self._diagnostics)

    @property
    def unsupported_tensor_count(self) -> int:
        """Number of output tensors skipped because their layout/backend is unsupported."""
        return self._unsupported_tensor_count

    def start(self) -> None:
        """Register forward hooks on the selected model modules."""
        if self._active:
            raise RuntimeError("ForwardMonitor is already active; close it before starting again.")

        self._diagnostics.clear()
        self._module_call_counts.clear()
        self._sequence_index = 0
        self._unsupported_tensor_count = 0
        records = _select_module_records(
            collect_model_modules(self._model),
            self._module_scope,
        )
        new_handles: list[RemovableHandle] = []
        for record in records:
            self._module_call_counts[record.module_id] = 0
            try:
                new_handles.append(record.module.register_forward_hook(self._make_hook(record)))
            except BaseException:
                for handle in new_handles:
                    handle.remove()
                self._handles = []
                self._active = False
                raise

        self._handles = new_handles

        self._active = True

    def close(self) -> None:
        """Remove all TrainMedic hooks. Safe to call multiple times."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._active = False

    def _make_hook(self, record: ModelModuleRecord) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            self._handle_forward_output(record, output)
            return None

        return hook

    def _handle_forward_output(self, record: ModelModuleRecord, output: Any) -> None:
        if self._has_reported_nan() and self._has_reported_inf():
            return

        module_call_index = self._module_call_counts[record.module_id] + 1
        self._module_call_counts[record.module_id] = module_call_index

        for tensor_path, tensor in _iter_output_tensors(output):
            if self._has_reported_nan() and self._has_reported_inf():
                return

            observation = self._observe_tensor(record, module_call_index, tensor_path, tensor)
            if observation is None:
                continue

            new_diagnostics = diagnose_forward_observation(
                observation,
                nan_already_reported=self._has_reported_nan(),
                inf_already_reported=self._has_reported_inf(),
            )
            self._diagnostics.extend(new_diagnostics)

    def _observe_tensor(
        self,
        record: ModelModuleRecord,
        module_call_index: int,
        tensor_path: str,
        tensor: torch.Tensor,
    ) -> ForwardTensorObservation | None:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            return None

        if _is_unsupported_tensor(tensor):
            self._unsupported_tensor_count += 1
            return None

        try:
            with torch.no_grad():
                detached = tensor.detach()
                nan_count = int(torch.isnan(detached).sum().item())
                inf_count = int(torch.isinf(detached).sum().item())
        except (NotImplementedError, RuntimeError, TypeError):
            self._unsupported_tensor_count += 1
            return None

        self._sequence_index += 1
        return ForwardTensorObservation(
            sequence_index=self._sequence_index,
            module_name=record.primary_name,
            module_aliases=record.aliases,
            module_type=record.module_type,
            module_call_index=module_call_index,
            tensor_path=tensor_path,
            shape=tuple(int(size) for size in tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            numel=int(tensor.numel()),
            nan_count=nan_count,
            inf_count=inf_count,
        )

    def _has_reported_nan(self) -> bool:
        return any(
            diagnostic.code == FORWARD_OUTPUT_CONTAINS_NAN
            for diagnostic in self._diagnostics
        )

    def _has_reported_inf(self) -> bool:
        return any(
            diagnostic.code == FORWARD_OUTPUT_CONTAINS_INF
            for diagnostic in self._diagnostics
        )


def watch_forward(
    model: nn.Module,
    *,
    module_scope: ModuleScope = "all",
) -> ForwardMonitor:
    """Create a context manager that monitors module forward outputs."""
    return ForwardMonitor(model, module_scope=module_scope)


def _select_module_records(
    records: tuple[ModelModuleRecord, ...],
    module_scope: ModuleScope,
) -> tuple[ModelModuleRecord, ...]:
    if module_scope == "all":
        return records

    selected = [
        record
        for record in records
        if record.is_root or record.is_leaf
    ]
    return tuple(selected)


def _iter_output_tensors(output: Any) -> Iterator[tuple[str, torch.Tensor]]:
    visited_containers: set[int] = set()
    visited_tensors: set[int] = set()

    def walk(value: Any, path: str) -> Iterator[tuple[str, torch.Tensor]]:
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id in visited_tensors:
                return
            visited_tensors.add(tensor_id)
            yield path, value
            return

        if isinstance(value, Mapping):
            container_id = id(value)
            if container_id in visited_containers:
                return
            visited_containers.add(container_id)
            for index, (key, item) in enumerate(value.items()):
                yield from walk(item, f"{path}{_format_mapping_key(key, index)}")
            return

        if isinstance(value, list | tuple):
            container_id = id(value)
            if container_id in visited_containers:
                return
            visited_containers.add(container_id)
            for index, item in enumerate(value):
                yield from walk(item, f"{path}[{index}]")

    yield from walk(output, "output")


def _format_mapping_key(key: Any, index: int) -> str:
    if isinstance(key, str):
        return f"[{json.dumps(key, ensure_ascii=False)}]"
    if isinstance(key, bool):
        return f"[{str(key)}]"
    if isinstance(key, int):
        return f"[{key}]"
    if key is None:
        return "[None]"
    return f"[<key:{index}>]"


def _is_unsupported_tensor(tensor: torch.Tensor) -> bool:
    if tensor.is_meta or tensor.is_quantized:
        return True
    return tensor.layout is not torch.strided
