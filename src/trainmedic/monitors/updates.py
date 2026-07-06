"""Parameter update monitoring around optimizer.step()."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from types import TracebackType
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from trainmedic.inspectors.model import ModelParameterRecord, collect_model_parameters
from trainmedic.inspectors.optimizer import collect_optimizer_parameters
from trainmedic.rules.update_rules import (
    PARAMETER_UPDATE_NOT_DETECTED,
    ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES,
    no_parameters_selected_diagnostic,
    parameter_update_not_detected_diagnostic,
    step_not_observed_diagnostic,
    zero_learning_rate_diagnostic,
)
from trainmedic.types import Diagnostic, JsonValue

_PREVIEW_LIMIT = 20

LearningRateValue = float | Literal["unknown"]
GradientState = Literal["none", "unsupported", "non_finite", "finite_zero", "finite_nonzero"]
SnapshotCoverage = Literal["exact", "sampled", "unsupported"]
LearningRateState = Literal["zero", "nonzero", "unknown"]

OptimizerPreHook = Callable[[Optimizer, tuple[Any, ...], dict[str, Any]], None]
OptimizerPostHook = Callable[[Optimizer, tuple[Any, ...], dict[str, Any]], None]


@dataclass(frozen=True)
class _SelectedParameter:
    record: ModelParameterRecord
    group_indices: tuple[int, ...]
    learning_rates: tuple[LearningRateValue, ...]


@dataclass(frozen=True)
class _GradientSummary:
    state: GradientState
    norm: float | None


@dataclass
class _SnapshotRecord:
    selected: _SelectedParameter
    gradient_state: GradientState
    gradient_norm: float | None
    learning_rate_state: LearningRateState
    coverage: SnapshotCoverage
    sample_indices: tuple[int, ...]
    before_values: torch.Tensor | None
    sampled_element_count: int
    unsupported_reason: str | None = None

    @property
    def is_update_candidate(self) -> bool:
        return self.gradient_state == "finite_nonzero" and self.learning_rate_state == "nonzero"


@dataclass
class _PendingStepSnapshot:
    step_index: int
    records: tuple[_SnapshotRecord, ...]
    stored_element_count: int
    skipped_parameter_count: int


@dataclass(frozen=True)
class _ComparisonResult:
    snapshot: _SnapshotRecord
    changed: bool
    changed_element_count: int
    max_abs_delta: float


class ParameterUpdateMonitor:
    """Context manager that monitors parameter changes around optimizer.step()."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        sample_size: int = 64,
        max_snapshot_elements: int = 100_000,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._sample_size = _validate_positive_int(sample_size, name="sample_size")
        self._max_snapshot_elements = _validate_positive_int(
            max_snapshot_elements,
            name="max_snapshot_elements",
        )

        self._model_parameters = collect_model_parameters(model)
        self._selected_parameters = _select_parameters(self._model_parameters, optimizer)
        self._optimizer_model_parameter_count = _optimizer_model_parameter_count(
            self._model_parameters,
            optimizer,
        )

        self._handles: list[RemovableHandle] = []
        self._diagnostics: list[Diagnostic] = []
        self._pending_snapshot: _PendingStepSnapshot | None = None
        self._step_count = 0
        self._next_step_index = 0
        self._unsupported_parameter_count = 0
        self._active = False
        self._started_once = False
        self._finalized = False

    def __enter__(self) -> ParameterUpdateMonitor:
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
        del exc_value, traceback
        self.close(finalize=exc_type is None)
        return False

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Diagnostics collected during this monitor session."""
        return tuple(self._diagnostics)

    @property
    def step_count(self) -> int:
        """Number of optimizer steps that reached the post-hook."""
        return self._step_count

    @property
    def unsupported_parameter_count(self) -> int:
        """Number of selected parameters skipped because update comparison was unsupported."""
        return self._unsupported_parameter_count

    def start(self) -> None:
        """Register optimizer step hooks for one monitoring session."""
        if self._active:
            raise RuntimeError(
                "ParameterUpdateMonitor is already active; close it before starting again."
            )

        self._diagnostics.clear()
        self._pending_snapshot = None
        self._step_count = 0
        self._next_step_index = 0
        self._unsupported_parameter_count = 0
        self._handles = []
        self._finalized = False

        new_handles: list[RemovableHandle] = []
        try:
            register_pre_hook = cast(
                Callable[[OptimizerPreHook], RemovableHandle],
                self._optimizer.register_step_pre_hook,
            )
            register_post_hook = cast(
                Callable[[OptimizerPostHook], RemovableHandle],
                self._optimizer.register_step_post_hook,
            )
            new_handles.append(register_pre_hook(self._step_pre_hook))
            new_handles.append(register_post_hook(self._step_post_hook))
        except BaseException:
            for handle in new_handles:
                handle.remove()
            self._handles = []
            self._pending_snapshot = None
            self._active = False
            self._started_once = False
            raise

        self._handles = new_handles
        self._active = True
        self._started_once = True

    def close(self, *, finalize: bool = True) -> None:
        """Remove TrainMedic hooks and clear any pending snapshot."""
        if finalize and self._started_once and not self._finalized:
            self._finalize_session()
            self._finalized = True

        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._pending_snapshot = None
        self._active = False

    def _step_pre_hook(
        self,
        optimizer: Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del optimizer, args, kwargs
        if self._pending_snapshot is not None:
            raise RuntimeError(
                "ParameterUpdateMonitor found a pending snapshot before a new optimizer step. "
                "Nested or re-entrant optimizer.step() calls are not supported."
            )

        self._next_step_index += 1
        self._pending_snapshot = self._build_pending_snapshot(self._next_step_index)
        self._maybe_report_zero_learning_rate(self._pending_snapshot)
        return None

    def _step_post_hook(
        self,
        optimizer: Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del optimizer, args, kwargs
        pending = self._pending_snapshot
        if pending is None:
            return None

        try:
            self._evaluate_completed_step(pending)
            self._step_count += 1
        finally:
            self._pending_snapshot = None
        return None

    def _build_pending_snapshot(self, step_index: int) -> _PendingStepSnapshot:
        records: list[_SnapshotRecord] = []
        remaining_budget = self._max_snapshot_elements
        skipped_parameter_count = 0

        for selected in self._selected_parameters:
            gradient = selected.record.parameter.grad
            gradient_summary = _summarize_gradient(gradient)
            lr_state = _learning_rate_state(selected.learning_rates)

            if gradient_summary.state == "finite_nonzero" and lr_state == "nonzero":
                snapshot = _snapshot_parameter(
                    selected=selected,
                    gradient_summary=gradient_summary,
                    learning_rate_state=lr_state,
                    sample_size=self._sample_size,
                    remaining_budget=remaining_budget,
                )
                remaining_budget -= snapshot.sampled_element_count
                if snapshot.before_values is None:
                    skipped_parameter_count += 1
                    self._unsupported_parameter_count += 1
            else:
                snapshot = _SnapshotRecord(
                    selected=selected,
                    gradient_state=gradient_summary.state,
                    gradient_norm=gradient_summary.norm,
                    learning_rate_state=lr_state,
                    coverage="unsupported",
                    sample_indices=(),
                    before_values=None,
                    sampled_element_count=0,
                    unsupported_reason="not_update_candidate",
                )
                if gradient_summary.state == "unsupported":
                    self._unsupported_parameter_count += 1

            records.append(snapshot)

        return _PendingStepSnapshot(
            step_index=step_index,
            records=tuple(records),
            stored_element_count=self._max_snapshot_elements - remaining_budget,
            skipped_parameter_count=skipped_parameter_count,
        )

    def _maybe_report_zero_learning_rate(self, pending: _PendingStepSnapshot) -> None:
        if self._has_diagnostic(ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES):
            return

        affected = tuple(
            snapshot
            for snapshot in pending.records
            if (
                snapshot.gradient_state == "finite_nonzero"
                and snapshot.learning_rate_state == "zero"
            )
        )
        if not affected:
            return

        preview = tuple(
            snapshot.selected.record.primary_name
            for snapshot in affected[:_PREVIEW_LIMIT]
        )
        group_indices = tuple(
            sorted(
                {
                    group_index
                    for snapshot in affected
                    for group_index in snapshot.selected.group_indices
                }
            )
        )
        learning_rates = tuple(
            sorted(
                {
                    learning_rate
                    for snapshot in affected
                    for learning_rate in snapshot.selected.learning_rates
                },
                key=str,
            )
        )
        self._diagnostics.append(
            zero_learning_rate_diagnostic(
                step_index=pending.step_index,
                affected_parameter_count=len(affected),
                affected_parameter_names_preview=preview,
                omitted_parameter_count=max(0, len(affected) - len(preview)),
                optimizer_group_indices=group_indices,
                learning_rate_values=learning_rates,
                selection_count=len(self._selected_parameters),
            )
        )

    def _evaluate_completed_step(self, pending: _PendingStepSnapshot) -> None:
        candidate_snapshots = tuple(
            snapshot
            for snapshot in pending.records
            if snapshot.gradient_state == "finite_nonzero"
            and snapshot.learning_rate_state == "nonzero"
        )
        if not candidate_snapshots or self._has_diagnostic(PARAMETER_UPDATE_NOT_DETECTED):
            return

        comparison_results: list[_ComparisonResult] = []
        unsupported_or_skipped_count = 0

        for snapshot in candidate_snapshots:
            result = _compare_snapshot(snapshot)
            if result is None:
                unsupported_or_skipped_count += 1
                self._unsupported_parameter_count += 1
                continue
            comparison_results.append(result)

        unchanged = tuple(result for result in comparison_results if not result.changed)
        if not unchanged:
            return

        changed_count = sum(1 for result in comparison_results if result.changed)
        exact_unchanged_count = sum(
            1 for result in unchanged if result.snapshot.coverage == "exact"
        )
        sampled_unchanged_count = sum(
            1 for result in unchanged if result.snapshot.coverage == "sampled"
        )
        preview_results = unchanged[:_PREVIEW_LIMIT]
        preview_names = tuple(
            result.snapshot.selected.record.primary_name
            for result in preview_results
        )
        per_parameter_preview = tuple(
            _comparison_preview(result)
            for result in preview_results
        )

        self._diagnostics.append(
            parameter_update_not_detected_diagnostic(
                step_index=pending.step_index,
                candidate_parameter_count=len(candidate_snapshots),
                changed_candidate_count=changed_count,
                unchanged_candidate_count=len(unchanged),
                exact_unchanged_count=exact_unchanged_count,
                sampled_unchanged_count=sampled_unchanged_count,
                unsupported_or_skipped_count=unsupported_or_skipped_count,
                unchanged_parameter_names_preview=preview_names,
                omitted_parameter_count=max(0, len(unchanged) - len(preview_names)),
                per_parameter_preview=per_parameter_preview,
                configured_sample_size=self._sample_size,
                configured_max_snapshot_elements=self._max_snapshot_elements,
            )
        )

    def _finalize_session(self) -> None:
        if not self._selected_parameters:
            self._diagnostics.append(
                no_parameters_selected_diagnostic(
                    model_parameter_count=len(self._model_parameters),
                    trainable_parameter_count=sum(
                        1 for record in self._model_parameters if record.requires_grad
                    ),
                    optimizer_model_parameter_count=self._optimizer_model_parameter_count,
                    selected_parameter_count=0,
                    optimizer_group_count=len(self._optimizer.param_groups),
                )
            )
            return

        if self._step_count == 0:
            self._diagnostics.append(
                step_not_observed_diagnostic(
                    selected_parameter_count=len(self._selected_parameters),
                    optimizer_group_count=len(self._optimizer.param_groups),
                )
            )

    def _has_diagnostic(self, code: str) -> bool:
        return any(diagnostic.code == code for diagnostic in self._diagnostics)


def watch_updates(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    sample_size: int = 64,
    max_snapshot_elements: int = 100_000,
) -> ParameterUpdateMonitor:
    """Monitor parameter changes around optimizer.step()."""
    return ParameterUpdateMonitor(
        model,
        optimizer,
        sample_size=sample_size,
        max_snapshot_elements=max_snapshot_elements,
    )


def _select_parameters(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer: Optimizer,
) -> tuple[_SelectedParameter, ...]:
    records_by_parameter_id: dict[int, list[int]] = defaultdict(list)
    for optimizer_record in collect_optimizer_parameters(optimizer):
        records_by_parameter_id[optimizer_record.parameter_id].append(
            optimizer_record.group_index
        )

    selected = []
    for model_record in model_parameters:
        group_indices = tuple(
            sorted(set(records_by_parameter_id.get(model_record.parameter_id, ())))
        )
        if not model_record.requires_grad or not group_indices:
            continue

        selected.append(
            _SelectedParameter(
                record=model_record,
                group_indices=group_indices,
                learning_rates=tuple(
                    _read_learning_rate(optimizer.param_groups[group_index].get("lr"))
                    for group_index in group_indices
                ),
            )
        )

    return tuple(selected)


def _optimizer_model_parameter_count(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer: Optimizer,
) -> int:
    model_parameter_ids = {record.parameter_id for record in model_parameters}
    return len(
        {
            record.parameter_id
            for record in collect_optimizer_parameters(optimizer)
            if record.parameter_id in model_parameter_ids
        }
    )


def _summarize_gradient(gradient: torch.Tensor | None) -> _GradientSummary:
    if gradient is None:
        return _GradientSummary(state="none", norm=None)

    if not (gradient.is_floating_point() or gradient.is_complex()):
        return _GradientSummary(state="unsupported", norm=None)

    if gradient.is_meta or gradient.is_quantized:
        return _GradientSummary(state="unsupported", norm=None)

    try:
        with torch.no_grad():
            detached = gradient.detach()
            values = _values_for_gradient_checks(detached)
            if values is None:
                return _GradientSummary(state="unsupported", norm=None)
            has_nan = bool(torch.isnan(values).any().item())
            has_inf = bool(torch.isinf(values).any().item())
            if has_nan or has_inf:
                return _GradientSummary(state="non_finite", norm=None)

            norm = float(torch.linalg.vector_norm(values).item())
    except (NotImplementedError, RuntimeError, TypeError):
        return _GradientSummary(state="unsupported", norm=None)

    if not math.isfinite(norm):
        return _GradientSummary(state="non_finite", norm=None)
    if norm == 0.0:
        return _GradientSummary(state="finite_zero", norm=0.0)
    return _GradientSummary(state="finite_nonzero", norm=norm)


def _values_for_gradient_checks(gradient: torch.Tensor) -> torch.Tensor | None:
    if gradient.layout is torch.strided:
        return gradient
    if gradient.layout is torch.sparse_coo:
        return gradient.coalesce().values()
    return None


def _read_learning_rate(value: object) -> LearningRateValue:
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, Real):
        learning_rate = float(value)
        return learning_rate if math.isfinite(learning_rate) else "unknown"
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.numel() != 1 or value.dtype is torch.bool:
            return "unknown"
        try:
            learning_rate = float(value.detach().cpu().item())
        except (RuntimeError, TypeError, ValueError):
            return "unknown"
        return learning_rate if math.isfinite(learning_rate) else "unknown"
    return "unknown"


def _learning_rate_state(learning_rates: tuple[LearningRateValue, ...]) -> LearningRateState:
    if not learning_rates or any(value == "unknown" for value in learning_rates):
        return "unknown"
    float_rates = cast(tuple[float, ...], learning_rates)
    if all(value == 0.0 for value in float_rates):
        return "zero"
    return "nonzero"


def _snapshot_parameter(
    *,
    selected: _SelectedParameter,
    gradient_summary: _GradientSummary,
    learning_rate_state: LearningRateState,
    sample_size: int,
    remaining_budget: int,
) -> _SnapshotRecord:
    parameter = selected.record.parameter
    numel = int(parameter.numel())
    if parameter.layout is not torch.strided or parameter.is_meta or parameter.is_quantized:
        return _unsupported_snapshot(
            selected,
            gradient_summary,
            learning_rate_state,
            "unsupported_parameter_layout",
        )

    if remaining_budget <= 0:
        return _unsupported_snapshot(
            selected,
            gradient_summary,
            learning_rate_state,
            "budget_skipped",
        )

    if numel <= sample_size:
        requested_count = numel
        coverage: SnapshotCoverage = "exact"
    else:
        requested_count = min(sample_size, numel)
        coverage = "sampled"

    sample_count = min(requested_count, remaining_budget)
    if sample_count <= 0:
        return _unsupported_snapshot(
            selected,
            gradient_summary,
            learning_rate_state,
            "budget_skipped",
        )
    if sample_count < numel:
        coverage = "sampled"

    indices = _deterministic_sample_indices(numel, sample_count)
    try:
        before_values = _take_parameter_values(parameter, indices)
    except (NotImplementedError, RuntimeError, TypeError):
        return _unsupported_snapshot(
            selected,
            gradient_summary,
            learning_rate_state,
            "snapshot_failed",
        )

    return _SnapshotRecord(
        selected=selected,
        gradient_state=gradient_summary.state,
        gradient_norm=gradient_summary.norm,
        learning_rate_state=learning_rate_state,
        coverage=coverage,
        sample_indices=indices,
        before_values=before_values,
        sampled_element_count=int(before_values.numel()),
    )


def _unsupported_snapshot(
    selected: _SelectedParameter,
    gradient_summary: _GradientSummary,
    learning_rate_state: LearningRateState,
    reason: str,
) -> _SnapshotRecord:
    return _SnapshotRecord(
        selected=selected,
        gradient_state=gradient_summary.state,
        gradient_norm=gradient_summary.norm,
        learning_rate_state=learning_rate_state,
        coverage="unsupported",
        sample_indices=(),
        before_values=None,
        sampled_element_count=0,
        unsupported_reason=reason,
    )


def _deterministic_sample_indices(numel: int, sample_count: int) -> tuple[int, ...]:
    if numel <= 0 or sample_count <= 0:
        return ()
    if sample_count >= numel:
        return tuple(range(numel))
    if sample_count == 1:
        return (0,)
    return tuple(
        int(index * (numel - 1) / (sample_count - 1))
        for index in range(sample_count)
    )


def _take_parameter_values(parameter: nn.Parameter, indices: tuple[int, ...]) -> torch.Tensor:
    if not indices:
        return torch.empty(0, dtype=parameter.dtype)

    with torch.no_grad():
        index_tensor = torch.tensor(indices, dtype=torch.long, device=parameter.device)
        values = torch.take(parameter.detach(), index_tensor)
        return values.detach().clone().cpu()


def _compare_snapshot(snapshot: _SnapshotRecord) -> _ComparisonResult | None:
    before_values = snapshot.before_values
    if before_values is None:
        return None

    parameter = snapshot.selected.record.parameter
    if tuple(int(size) for size in parameter.shape) != snapshot.selected.record.shape:
        return None
    if str(parameter.dtype) != snapshot.selected.record.dtype:
        return None
    if str(parameter.device) != snapshot.selected.record.device:
        return None

    try:
        after_values = _take_parameter_values(parameter, snapshot.sample_indices)
    except (NotImplementedError, RuntimeError, TypeError):
        return None

    changed = not torch.equal(before_values, after_values)
    if changed:
        delta = after_values - before_values
        changed_element_count = int((delta != 0).sum().item())
        max_abs_delta = float(torch.max(torch.abs(delta)).item()) if delta.numel() > 0 else 0.0
    else:
        changed_element_count = 0
        max_abs_delta = 0.0

    return _ComparisonResult(
        snapshot=snapshot,
        changed=changed,
        changed_element_count=changed_element_count,
        max_abs_delta=max_abs_delta,
    )


def _comparison_preview(result: _ComparisonResult) -> dict[str, JsonValue]:
    snapshot = result.snapshot
    selected = snapshot.selected
    group_index_value: int | list[int]
    if len(selected.group_indices) == 1:
        group_index_value = selected.group_indices[0]
    else:
        group_index_value = list(selected.group_indices)

    learning_rate_value: float | str | list[float | str]
    if len(selected.learning_rates) == 1:
        learning_rate_value = selected.learning_rates[0]
    else:
        learning_rate_value = list(selected.learning_rates)

    group_index_json: JsonValue
    if isinstance(group_index_value, list):
        group_index_json = [int(item) for item in group_index_value]
    else:
        group_index_json = group_index_value

    learning_rate_json: JsonValue
    if isinstance(learning_rate_value, list):
        learning_rate_json = [
            item if isinstance(item, float) else "unknown"
            for item in learning_rate_value
        ]
    else:
        learning_rate_json = learning_rate_value

    return {
        "name": selected.record.primary_name,
        "aliases": list(selected.record.aliases),
        "group_index": group_index_json,
        "learning_rate": learning_rate_json,
        "coverage": snapshot.coverage,
        "parameter_numel": int(selected.record.parameter.numel()),
        "sampled_element_count": snapshot.sampled_element_count,
        "gradient_norm": snapshot.gradient_norm,
    }


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result
