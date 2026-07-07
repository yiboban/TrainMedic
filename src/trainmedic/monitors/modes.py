"""Runtime train/eval mode and gradient context monitoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from trainmedic.inspectors.modules import ModelModuleRecord, collect_model_modules
from trainmedic.rules.mode_rules import (
    BATCHNORM_MODE_MISMATCH,
    CALLED_SUBMODULE_MODE_MISMATCH,
    DROPOUT_MODE_MISMATCH,
    FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING,
    GRADIENT_TRACKING_DISABLED_DURING_TRAINING,
    GRADIENT_TRACKING_ENABLED_DURING_EVALUATION,
    MODEL_IN_EVAL_DURING_TRAINING,
    MODEL_IN_TRAIN_DURING_EVALUATION,
    batchnorm_mode_mismatch_diagnostic,
    called_submodule_mode_mismatch_diagnostic,
    dropout_mode_mismatch_diagnostic,
    eval_grad_enabled_diagnostic,
    forward_not_observed_diagnostic,
    model_in_eval_during_training_diagnostic,
    model_in_train_during_evaluation_diagnostic,
    train_grad_disabled_diagnostic,
)
from trainmedic.types import Diagnostic

ExpectedMode = Literal["train", "eval"]

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)

_BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)

ForwardPreHook = Callable[[nn.Module, tuple[Any, ...]], None]


@dataclass(frozen=True)
class ModeObservation:
    """Summary of a real forward pre-hook observation."""

    sequence_index: int
    module_name: str
    module_aliases: tuple[str, ...]
    module_type: str
    module_call_index: int
    is_root: bool
    expected_mode: str
    actual_mode: str
    grad_enabled: bool
    inference_mode_enabled: bool


@dataclass(frozen=True)
class _ModeModuleRecord:
    module_id: int
    primary_name: str
    aliases: tuple[str, ...]
    module_type: str
    is_root: bool


class ModeMonitor:
    """Context manager that monitors train/eval mode during forward calls."""

    def __init__(
        self,
        model: nn.Module,
        *,
        expected_mode: ExpectedMode,
        check_grad_context: bool = True,
    ) -> None:
        self._model = model
        self._expected_mode = _validate_expected_mode(expected_mode)
        self._check_grad_context = _validate_check_grad_context(check_grad_context)
        self._module_records: tuple[_ModeModuleRecord, ...] = ()
        self._handles: list[RemovableHandle] = []
        self._diagnostics: list[Diagnostic] = []
        self._module_call_counts: dict[int, int] = {}
        self._forward_call_count = 0
        self._active = False
        self._started_once = False
        self._finalized = False

    def __enter__(self) -> ModeMonitor:
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
    def forward_call_count(self) -> int:
        """Number of module forward pre-hook observations in this session."""
        return self._forward_call_count

    def start(self) -> None:
        """Register local forward pre-hooks on unique modules."""
        if self._active:
            raise RuntimeError("ModeMonitor is already active; close it before starting again.")

        collected_records = collect_model_modules(self._model)
        self._module_records = tuple(
            _lightweight_record(record)
            for record in collected_records
        )
        self._diagnostics.clear()
        self._module_call_counts = {
            record.module_id: 0
            for record in self._module_records
        }
        self._forward_call_count = 0
        self._handles = []
        self._finalized = False

        new_handles: list[RemovableHandle] = []
        try:
            for source_record, monitored_record in zip(
                collected_records,
                self._module_records,
                strict=True,
            ):
                register_hook = cast(
                    Callable[[ForwardPreHook], RemovableHandle],
                    source_record.module.register_forward_pre_hook,
                )
                new_handles.append(register_hook(self._make_hook(monitored_record)))
        except BaseException:
            for handle in new_handles:
                handle.remove()
            self._module_records = ()
            self._module_call_counts = {}
            self._handles = []
            self._active = False
            self._started_once = False
            raise

        self._handles = new_handles
        self._active = True
        self._started_once = True

    def close(self, *, finalize: bool = True) -> None:
        """Remove TrainMedic hooks and optionally finalize no-forward diagnostics."""
        if finalize and self._started_once and not self._finalized:
            self._finalize_session()
            self._finalized = True

        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._module_records = ()
        self._module_call_counts = {}
        self._active = False

    def _make_hook(self, record: _ModeModuleRecord) -> ForwardPreHook:
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            del args
            self._handle_module_call(record, module)
            return None

        return hook

    def _handle_module_call(self, record: _ModeModuleRecord, module: nn.Module) -> None:
        self._forward_call_count += 1
        self._module_call_counts[record.module_id] += 1
        observation = ModeObservation(
            sequence_index=self._forward_call_count,
            module_name=record.primary_name,
            module_aliases=record.aliases,
            module_type=record.module_type,
            module_call_index=self._module_call_counts[record.module_id],
            is_root=record.is_root,
            expected_mode=self._expected_mode,
            actual_mode="train" if module.training else "eval",
            grad_enabled=torch.is_grad_enabled(),
            inference_mode_enabled=torch.is_inference_mode_enabled(),
        )
        self._diagnose_observation(observation, module)

    def _diagnose_observation(self, observation: ModeObservation, module: nn.Module) -> None:
        mode_mismatch = observation.actual_mode != observation.expected_mode

        if observation.is_root and mode_mismatch:
            if (
                observation.expected_mode == "train"
                and not self._has_diagnostic(MODEL_IN_EVAL_DURING_TRAINING)
            ):
                self._diagnostics.append(
                    model_in_eval_during_training_diagnostic(observation)
                )
            elif (
                observation.expected_mode == "eval"
                and not self._has_diagnostic(MODEL_IN_TRAIN_DURING_EVALUATION)
            ):
                self._diagnostics.append(
                    model_in_train_during_evaluation_diagnostic(observation)
                )

        if not observation.is_root and mode_mismatch:
            if isinstance(module, _DROPOUT_TYPES):
                if not self._has_diagnostic(DROPOUT_MODE_MISMATCH):
                    self._diagnostics.append(
                        dropout_mode_mismatch_diagnostic(
                            observation,
                            p=_optional_float(getattr(module, "p", None)),
                            inplace=_optional_bool(getattr(module, "inplace", None)),
                        )
                    )
            elif isinstance(module, _BATCHNORM_TYPES):
                if not self._has_diagnostic(BATCHNORM_MODE_MISMATCH):
                    self._diagnostics.append(
                        batchnorm_mode_mismatch_diagnostic(
                            observation,
                            num_features=_optional_int(getattr(module, "num_features", None)),
                            affine=_optional_bool(getattr(module, "affine", None)),
                            track_running_stats=_optional_bool(
                                getattr(module, "track_running_stats", None)
                            ),
                            momentum=_optional_float(getattr(module, "momentum", None)),
                        )
                    )
            elif not self._has_diagnostic(CALLED_SUBMODULE_MODE_MISMATCH):
                self._diagnostics.append(called_submodule_mode_mismatch_diagnostic(observation))

        if observation.is_root and self._check_grad_context:
            if (
                observation.expected_mode == "eval"
                and observation.grad_enabled
                and not self._has_diagnostic(GRADIENT_TRACKING_ENABLED_DURING_EVALUATION)
            ):
                self._diagnostics.append(eval_grad_enabled_diagnostic(observation))
            elif (
                observation.expected_mode == "train"
                and not observation.grad_enabled
                and not self._has_diagnostic(GRADIENT_TRACKING_DISABLED_DURING_TRAINING)
            ):
                self._diagnostics.append(train_grad_disabled_diagnostic(observation))

    def _finalize_session(self) -> None:
        if (
            self._forward_call_count == 0
            and not self._has_diagnostic(FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING)
        ):
            self._diagnostics.append(
                forward_not_observed_diagnostic(expected_mode=self._expected_mode)
            )

    def _has_diagnostic(self, code: str) -> bool:
        return any(diagnostic.code == code for diagnostic in self._diagnostics)


def watch_modes(
    model: nn.Module,
    *,
    expected_mode: ExpectedMode,
    check_grad_context: bool = True,
) -> ModeMonitor:
    """Monitor runtime train/eval mode and gradient context during forward calls."""
    return ModeMonitor(
        model,
        expected_mode=expected_mode,
        check_grad_context=check_grad_context,
    )


def _lightweight_record(record: ModelModuleRecord) -> _ModeModuleRecord:
    return _ModeModuleRecord(
        module_id=record.module_id,
        primary_name=record.primary_name,
        aliases=record.aliases,
        module_type=record.module_type,
        is_root=record.is_root,
    )


def _validate_expected_mode(value: str) -> ExpectedMode:
    if value == "train" or value == "eval":
        return cast(ExpectedMode, value)
    raise ValueError("expected_mode must be one of: 'train', 'eval'")


def _validate_check_grad_context(value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError("check_grad_context must be a bool")
    return value


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
