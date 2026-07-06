"""Accumulated parameter gradient monitoring."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from types import TracebackType
from typing import Literal, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from trainmedic.inspectors.model import ModelParameterRecord, collect_model_parameters
from trainmedic.inspectors.optimizer import collect_optimizer_parameters
from trainmedic.rules.gradient_rules import (
    GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD,
    GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD,
    GRADIENT_CONTAINS_INF,
    GRADIENT_CONTAINS_NAN,
    diagnose_gradient_observation,
    global_norm_below_threshold_diagnostic,
    global_norm_exceeds_threshold_diagnostic,
    no_parameters_selected_diagnostic,
    none_gradients_diagnostic,
)
from trainmedic.types import Diagnostic

_NONE_PREVIEW_LIMIT = 20


@dataclass(frozen=True)
class GradientObservation:
    """Summary of an accumulated parameter gradient."""

    sequence_index: int
    parameter_name: str
    parameter_aliases: tuple[str, ...]
    parameter_shape: tuple[int, ...]
    parameter_dtype: str
    parameter_device: str
    parameter_numel: int
    hook_call_index: int
    gradient_shape: tuple[int, ...]
    gradient_dtype: str
    gradient_device: str
    gradient_layout: str
    gradient_numel: int
    gradient_nnz: int | None
    nan_count: int
    inf_count: int


@dataclass(frozen=True)
class _GradientStats:
    gradient_shape: tuple[int, ...]
    gradient_dtype: str
    gradient_device: str
    gradient_layout: str
    gradient_numel: int
    gradient_nnz: int | None
    nan_count: int
    inf_count: int
    l2_norm: float
    is_finite_for_norm: bool


class GradientMonitor:
    """Context manager that monitors accumulated parameter gradients."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        *,
        max_global_norm: float | None = None,
        min_global_norm: float | None = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._max_global_norm = _validate_optional_threshold(
            max_global_norm,
            name="max_global_norm",
            allow_zero=False,
        )
        self._min_global_norm = _validate_optional_threshold(
            min_global_norm,
            name="min_global_norm",
            allow_zero=True,
        )
        if (
            self._max_global_norm is not None
            and self._min_global_norm is not None
            and self._min_global_norm > self._max_global_norm
        ):
            raise ValueError("min_global_norm must be less than or equal to max_global_norm")

        self._model_parameters = collect_model_parameters(model)
        self._selected_parameters = _select_parameter_records(
            self._model_parameters,
            optimizer,
        )
        self._optimizer_model_parameter_count = _optimizer_model_parameter_count(
            self._model_parameters,
            optimizer,
        )
        self._selection_scope = "optimizer" if optimizer is not None else "model_trainable"
        self._handles: list[RemovableHandle] = []
        self._diagnostics: list[Diagnostic] = []
        self._hook_call_counts: dict[int, int] = {}
        self._sequence_index = 0
        self._unsupported_gradient_count = 0
        self._observation_count = 0
        self._active = False
        self._started_once = False
        self._checked = False

    def __enter__(self) -> GradientMonitor:
        """Start monitoring and return this monitor."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Clean up hooks without suppressing user exceptions."""
        self.close()
        return False

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Diagnostics collected during this monitor session."""
        return tuple(self._diagnostics)

    @property
    def unsupported_gradient_count(self) -> int:
        """Number of gradients skipped because their layout/backend is unsupported."""
        return self._unsupported_gradient_count

    def start(self) -> None:
        """Register post-accumulate hooks on selected parameters."""
        if self._active:
            raise RuntimeError("GradientMonitor is already active; close it before starting again.")

        self._diagnostics.clear()
        self._hook_call_counts = {
            record.parameter_id: 0
            for record in self._selected_parameters
        }
        self._sequence_index = 0
        self._unsupported_gradient_count = 0
        self._observation_count = 0
        self._handles = []
        self._checked = False

        new_handles: list[RemovableHandle] = []
        try:
            for record in self._selected_parameters:
                register_hook = cast(
                    Callable[[Callable[[nn.Parameter], None]], RemovableHandle],
                    record.parameter.register_post_accumulate_grad_hook,
                )
                new_handles.append(register_hook(self._make_hook(record)))
        except BaseException:
            for handle in new_handles:
                handle.remove()
            self._handles = []
            self._active = False
            self._started_once = False
            raise

        self._handles = new_handles
        self._active = True
        self._started_once = True

    def close(self) -> None:
        """Remove all TrainMedic gradient hooks. Safe to call repeatedly."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._active = False

    def check_gradients(self) -> tuple[Diagnostic, ...]:
        """Inspect current accumulated gradients and append check-time diagnostics."""
        if not self._started_once:
            raise RuntimeError("GradientMonitor must be started before check_gradients().")
        if self._checked:
            raise RuntimeError("check_gradients() can only be called once per monitor session.")
        self._checked = True

        if not self._selected_parameters:
            self._diagnostics.append(
                no_parameters_selected_diagnostic(
                    optimizer_provided=self._optimizer is not None,
                    model_parameter_count=len(self._model_parameters),
                    trainable_parameter_count=sum(
                        1 for record in self._model_parameters if record.requires_grad
                    ),
                    optimizer_model_parameter_count=self._optimizer_model_parameter_count,
                    selected_parameter_count=0,
                )
            )
            return self.diagnostics

        none_records = []
        finite_norm_squares: list[float] = []
        contributing_parameter_count = 0
        finite_norms_only = True

        for record in self._selected_parameters:
            gradient = record.parameter.grad
            if gradient is None:
                none_records.append(record)
                continue

            stats = _summarize_gradient(gradient)
            if stats is None:
                self._unsupported_gradient_count += 1
                finite_norms_only = False
                continue

            contributing_parameter_count += 1
            if stats.is_finite_for_norm:
                finite_norm_squares.append(stats.l2_norm * stats.l2_norm)
            else:
                finite_norms_only = False

        if none_records:
            preview_names = tuple(
                record.primary_name
                for record in none_records[:_NONE_PREVIEW_LIMIT]
            )
            self._diagnostics.append(
                none_gradients_diagnostic(
                    checked_parameter_count=len(self._selected_parameters),
                    none_gradient_count=len(none_records),
                    non_none_gradient_count=len(self._selected_parameters) - len(none_records),
                    hook_observation_count=self._observation_count,
                    any_backward_observed=self._observation_count > 0,
                    none_parameter_names_preview=preview_names,
                    omitted_name_count=max(0, len(none_records) - len(preview_names)),
                    selection_scope=self._selection_scope,
                )
            )

        if finite_norms_only and contributing_parameter_count > 0:
            global_norm = math.sqrt(sum(finite_norm_squares))
            if (
                self._max_global_norm is not None
                and global_norm > self._max_global_norm
                and not self._has_diagnostic(GLOBAL_GRADIENT_NORM_EXCEEDS_THRESHOLD)
            ):
                self._diagnostics.append(
                    global_norm_exceeds_threshold_diagnostic(
                        actual_global_norm=global_norm,
                        configured_threshold=self._max_global_norm,
                        contributing_parameter_count=contributing_parameter_count,
                        none_gradient_count=len(none_records),
                        selection_scope=self._selection_scope,
                    )
                )
            if (
                self._min_global_norm is not None
                and global_norm < self._min_global_norm
                and not self._has_diagnostic(GLOBAL_GRADIENT_NORM_BELOW_THRESHOLD)
            ):
                self._diagnostics.append(
                    global_norm_below_threshold_diagnostic(
                        actual_global_norm=global_norm,
                        configured_threshold=self._min_global_norm,
                        contributing_parameter_count=contributing_parameter_count,
                        none_gradient_count=len(none_records),
                        selection_scope=self._selection_scope,
                    )
                )

        return self.diagnostics

    def _make_hook(self, record: ModelParameterRecord) -> Callable[[nn.Parameter], None]:
        def hook(parameter: nn.Parameter) -> None:
            self._handle_parameter_gradient(record, parameter)
            return None

        return hook

    def _handle_parameter_gradient(
        self,
        record: ModelParameterRecord,
        parameter: nn.Parameter,
    ) -> None:
        if self._has_diagnostic(GRADIENT_CONTAINS_NAN) and self._has_diagnostic(
            GRADIENT_CONTAINS_INF
        ):
            return

        self._hook_call_counts[record.parameter_id] += 1
        hook_call_index = self._hook_call_counts[record.parameter_id]
        gradient = parameter.grad
        if gradient is None:
            return

        stats = _summarize_gradient(gradient)
        if stats is None:
            self._unsupported_gradient_count += 1
            return

        self._observation_count += 1
        if stats.nan_count == 0 and stats.inf_count == 0:
            return

        self._sequence_index += 1
        observation = GradientObservation(
            sequence_index=self._sequence_index,
            parameter_name=record.primary_name,
            parameter_aliases=record.aliases,
            parameter_shape=record.shape,
            parameter_dtype=record.dtype,
            parameter_device=record.device,
            parameter_numel=int(record.parameter.numel()),
            hook_call_index=hook_call_index,
            gradient_shape=stats.gradient_shape,
            gradient_dtype=stats.gradient_dtype,
            gradient_device=stats.gradient_device,
            gradient_layout=stats.gradient_layout,
            gradient_numel=stats.gradient_numel,
            gradient_nnz=stats.gradient_nnz,
            nan_count=stats.nan_count,
            inf_count=stats.inf_count,
        )
        self._diagnostics.extend(
            diagnose_gradient_observation(
                observation,
                nan_already_reported=self._has_diagnostic(GRADIENT_CONTAINS_NAN),
                inf_already_reported=self._has_diagnostic(GRADIENT_CONTAINS_INF),
            )
        )

    def _has_diagnostic(self, code: str) -> bool:
        return any(diagnostic.code == code for diagnostic in self._diagnostics)


def watch_gradients(
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    max_global_norm: float | None = None,
    min_global_norm: float | None = None,
) -> GradientMonitor:
    """Create a monitor for accumulated parameter gradients."""
    return GradientMonitor(
        model,
        optimizer,
        max_global_norm=max_global_norm,
        min_global_norm=min_global_norm,
    )


def _select_parameter_records(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer: Optimizer | None,
) -> tuple[ModelParameterRecord, ...]:
    trainable_records = tuple(record for record in model_parameters if record.requires_grad)
    if optimizer is None:
        return trainable_records

    optimizer_parameter_ids = {
        record.parameter_id
        for record in collect_optimizer_parameters(optimizer)
    }
    return tuple(
        record
        for record in trainable_records
        if record.parameter_id in optimizer_parameter_ids
    )


def _optimizer_model_parameter_count(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer: Optimizer | None,
) -> int:
    if optimizer is None:
        return 0

    model_parameter_ids = {record.parameter_id for record in model_parameters}
    return sum(
        1
        for record in collect_optimizer_parameters(optimizer)
        if record.parameter_id in model_parameter_ids
    )


def _summarize_gradient(gradient: torch.Tensor) -> _GradientStats | None:
    if not (gradient.is_floating_point() or gradient.is_complex()):
        return None

    if gradient.is_meta or gradient.is_quantized:
        return None

    try:
        with torch.no_grad():
            detached = gradient.detach()
            values = _values_for_gradient_checks(detached)
            if values is None:
                return None
            nan_count = int(torch.isnan(values).sum().item())
            inf_count = int(torch.isinf(values).sum().item())
            l2_norm = float(torch.linalg.vector_norm(values).item())
    except (NotImplementedError, RuntimeError, TypeError):
        return None

    return _GradientStats(
        gradient_shape=tuple(int(size) for size in gradient.shape),
        gradient_dtype=str(gradient.dtype),
        gradient_device=str(gradient.device),
        gradient_layout=str(gradient.layout),
        gradient_numel=int(gradient.numel()),
        gradient_nnz=_gradient_nnz(gradient),
        nan_count=nan_count,
        inf_count=inf_count,
        l2_norm=l2_norm,
        is_finite_for_norm=math.isfinite(l2_norm),
    )


def _values_for_gradient_checks(gradient: torch.Tensor) -> torch.Tensor | None:
    if gradient.layout is torch.strided:
        return gradient
    if gradient.layout is torch.sparse_coo:
        return gradient.coalesce().values()
    return None


def _gradient_nnz(gradient: torch.Tensor) -> int | None:
    if gradient.layout is torch.sparse_coo:
        return int(gradient.coalesce()._nnz())
    return None


def _validate_optional_threshold(
    value: float | None,
    *,
    name: str,
    allow_zero: bool,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")

    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError(f"{name} must be finite")
    if allow_zero:
        if threshold < 0:
            raise ValueError(f"{name} must be greater than or equal to 0")
    elif threshold <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return threshold
