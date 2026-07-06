"""Static model and optimizer relationship diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from torch import nn
from torch.optim import Optimizer

from trainmedic.inspectors.model import ModelParameterRecord, collect_model_parameters
from trainmedic.inspectors.optimizer import (
    OptimizerParameterRecord,
    collect_optimizer_parameters,
)
from trainmedic.types import Diagnostic, Evidence, Severity

PARAMETER_NOT_IN_OPTIMIZER = "TM1001"
OPTIMIZER_PARAMETER_NOT_IN_MODEL = "TM1002"
FROZEN_PARAMETER_IN_OPTIMIZER = "TM1003"
MODEL_HAS_FROZEN_PARAMETERS = "TM1004"
ALL_MODEL_PARAMETERS_FROZEN = "TM1005"
MODEL_HAS_NO_PARAMETERS = "TM1006"
DUPLICATE_PARAMETER_IN_OPTIMIZER = "TM1007"

_PREVIEW_LIMIT = 10


def inspect_optimizer(
    model: nn.Module,
    optimizer: Optimizer,
) -> tuple[Diagnostic, ...]:
    """Inspect the relationship between model parameters and an optimizer.

    The inspection is static: it does not run forward or backward passes, does not call
    ``optimizer.step()``, and does not mutate the model or optimizer.
    """
    model_parameters = collect_model_parameters(model)
    optimizer_parameters = collect_optimizer_parameters(optimizer)

    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_diagnose_model_parameter_state(model_parameters))
    diagnostics.extend(
        _diagnose_missing_trainable_parameters(model_parameters, optimizer_parameters)
    )
    diagnostics.extend(
        _diagnose_optimizer_parameters_not_in_model(model_parameters, optimizer_parameters)
    )
    diagnostics.extend(
        _diagnose_frozen_parameters_in_optimizer(model_parameters, optimizer_parameters)
    )
    diagnostics.extend(
        _diagnose_duplicate_optimizer_parameters(model_parameters, optimizer_parameters)
    )
    return tuple(diagnostics)


def _diagnose_model_parameter_state(
    model_parameters: tuple[ModelParameterRecord, ...],
) -> tuple[Diagnostic, ...]:
    total_count = len(model_parameters)
    if total_count == 0:
        return (
            Diagnostic(
                code=MODEL_HAS_NO_PARAMETERS,
                severity=Severity.INFO,
                title="Model has no parameters",
                message="The model does not expose any parameters.",
                evidence=(Evidence("total_parameter_count", 0),),
                possible_causes=(
                    "The model may be a stateless module.",
                    "The model may use tensors that were not registered as nn.Parameter.",
                ),
                suggestions=(
                    "If this model should train weights, check that layers are assigned "
                    "as module attributes.",
                    "If the model is intentionally stateless, this diagnostic can be ignored.",
                ),
            ),
        )

    frozen_parameters = tuple(record for record in model_parameters if not record.requires_grad)
    trainable_count = total_count - len(frozen_parameters)

    if len(frozen_parameters) == total_count:
        preview, omitted_count = _preview_names(record.primary_name for record in frozen_parameters)
        return (
            Diagnostic(
                code=ALL_MODEL_PARAMETERS_FROZEN,
                severity=Severity.WARNING,
                title="All model parameters are frozen",
                message="All model parameters have requires_grad=False.",
                evidence=(
                    Evidence("total_parameter_count", total_count),
                    Evidence("frozen_parameter_count", len(frozen_parameters)),
                    Evidence("parameter_names_preview", preview),
                    Evidence("omitted_parameter_count", omitted_count),
                ),
                possible_causes=(
                    "The target layers may not have been unfrozen before training.",
                    "Checkpoint loading or fine-tuning setup may have frozen the full model.",
                ),
                suggestions=(
                    "Check whether the intended trainable layers were unfrozen.",
                    "Review checkpoint loading and freeze logic.",
                    "If the model is only used for inference, this diagnostic can be ignored.",
                ),
            ),
        )

    if frozen_parameters:
        preview, omitted_count = _preview_names(record.primary_name for record in frozen_parameters)
        return (
            Diagnostic(
                code=MODEL_HAS_FROZEN_PARAMETERS,
                severity=Severity.INFO,
                title="Model contains frozen parameters",
                message=(
                    "The model contains frozen parameters; confirm this matches the "
                    "training plan."
                ),
                evidence=(
                    Evidence("frozen_parameter_count", len(frozen_parameters)),
                    Evidence("trainable_parameter_count", trainable_count),
                    Evidence("total_parameter_count", total_count),
                    Evidence("frozen_parameter_names_preview", preview),
                    Evidence("omitted_frozen_parameter_count", omitted_count),
                ),
                possible_causes=(
                    "Some layers may have been intentionally frozen for fine-tuning.",
                    "A freeze configuration may have matched more parameters than intended.",
                ),
                suggestions=(
                    "Verify that each frozen parameter is intentional for this training run.",
                    "If a parameter should train, set requires_grad=True before creating "
                    "the optimizer.",
                ),
            ),
        )

    return ()


def _diagnose_missing_trainable_parameters(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer_parameters: tuple[OptimizerParameterRecord, ...],
) -> tuple[Diagnostic, ...]:
    optimizer_parameter_ids = {record.parameter_id for record in optimizer_parameters}
    optimizer_group_count = _optimizer_group_count(optimizer_parameters)

    diagnostics = []
    for record in model_parameters:
        if record.requires_grad and record.parameter_id not in optimizer_parameter_ids:
            diagnostics.append(
                Diagnostic(
                    code=PARAMETER_NOT_IN_OPTIMIZER,
                    severity=Severity.ERROR,
                    title="Trainable parameter is not managed by the optimizer",
                    message=(
                        f"Parameter {record.primary_name} is trainable but is not managed "
                        "by the current optimizer."
                    ),
                    object_name=record.primary_name,
                    evidence=(
                        Evidence("parameter_name", record.primary_name),
                        Evidence("aliases", record.aliases),
                        Evidence("shape", record.shape),
                        Evidence("dtype", record.dtype),
                        Evidence("device", record.device),
                        Evidence("requires_grad", record.requires_grad),
                        Evidence("optimizer_group_count", optimizer_group_count),
                    ),
                    possible_causes=(
                        "The module may have been omitted when constructing the optimizer.",
                        "The optimizer may have been created before the model structure "
                        "was finalized.",
                    ),
                    suggestions=(
                        "Check whether optimizer construction includes this module's parameters.",
                        "Create the optimizer after the full model structure is built.",
                        "If this parameter is intentionally frozen, set requires_grad=False "
                        "explicitly.",
                    ),
                )
            )

    return tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.object_name or ""))


def _diagnose_optimizer_parameters_not_in_model(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer_parameters: tuple[OptimizerParameterRecord, ...],
) -> tuple[Diagnostic, ...]:
    model_parameter_ids = {record.parameter_id for record in model_parameters}
    diagnostics = []

    for record in optimizer_parameters:
        if record.parameter_id not in model_parameter_ids:
            diagnostics.append(
                Diagnostic(
                    code=OPTIMIZER_PARAMETER_NOT_IN_MODEL,
                    severity=Severity.WARNING,
                    title="Optimizer parameter is not part of the model",
                    message=(
                        f"{record.position_name} is managed by the optimizer but does not "
                        "belong to the current model."
                    ),
                    object_name=record.position_name,
                    evidence=(
                        Evidence("optimizer_group_index", record.group_index),
                        Evidence("optimizer_parameter_index", record.parameter_index),
                        Evidence("shape", record.shape),
                        Evidence("dtype", record.dtype),
                        Evidence("device", record.device),
                        Evidence("requires_grad", record.requires_grad),
                    ),
                    possible_causes=(
                        "The optimizer may have been created for an older model instance.",
                        "Model parameters may have been replaced after optimizer construction.",
                        "The user may intentionally optimize an external Parameter.",
                    ),
                    suggestions=(
                        "Recreate the optimizer after replacing model parameters.",
                        "If this external parameter is intentional, this warning can be ignored.",
                    ),
                )
            )

    return tuple(diagnostics)


def _diagnose_frozen_parameters_in_optimizer(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer_parameters: tuple[OptimizerParameterRecord, ...],
) -> tuple[Diagnostic, ...]:
    optimizer_positions_by_id = _optimizer_positions_by_id(optimizer_parameters)
    diagnostics = []

    for record in model_parameters:
        positions = optimizer_positions_by_id.get(record.parameter_id, ())
        if positions and not record.requires_grad:
            diagnostics.append(
                Diagnostic(
                    code=FROZEN_PARAMETER_IN_OPTIMIZER,
                    severity=Severity.INFO,
                    title="Frozen parameter is still managed by the optimizer",
                    message=(
                        f"Parameter {record.primary_name} has requires_grad=False but is still "
                        "present in the optimizer."
                    ),
                    object_name=record.primary_name,
                    evidence=(
                        Evidence("parameter_name", record.primary_name),
                        Evidence("aliases", record.aliases),
                        Evidence("optimizer_positions", positions),
                        Evidence("shape", record.shape),
                        Evidence("dtype", record.dtype),
                        Evidence("device", record.device),
                        Evidence("requires_grad", record.requires_grad),
                    ),
                    possible_causes=(
                        "The parameter may have been frozen after optimizer construction.",
                        "The parameter may be intentionally kept for later unfreezing.",
                    ),
                    suggestions=(
                        "If the parameter will stay frozen, consider omitting it from "
                        "the optimizer.",
                        "If the parameter will be unfrozen later, keeping it in the "
                        "optimizer may be fine.",
                    ),
                )
            )

    return tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.object_name or ""))


def _diagnose_duplicate_optimizer_parameters(
    model_parameters: tuple[ModelParameterRecord, ...],
    optimizer_parameters: tuple[OptimizerParameterRecord, ...],
) -> tuple[Diagnostic, ...]:
    model_parameters_by_id = {record.parameter_id: record for record in model_parameters}
    optimizer_records_by_id: dict[int, list[OptimizerParameterRecord]] = defaultdict(list)
    for record in optimizer_parameters:
        optimizer_records_by_id[record.parameter_id].append(record)

    diagnostics = []
    for parameter_id in sorted(
        optimizer_records_by_id,
        key=lambda current_id: optimizer_records_by_id[current_id][0].position_name,
    ):
        records = optimizer_records_by_id[parameter_id]
        if len(records) <= 1:
            continue

        model_record = model_parameters_by_id.get(parameter_id)
        positions = tuple(record.position_name for record in records)
        object_name = model_record.primary_name if model_record is not None else positions[0]
        aliases = model_record.aliases if model_record is not None else ()

        diagnostics.append(
            Diagnostic(
                code=DUPLICATE_PARAMETER_IN_OPTIMIZER,
                severity=Severity.WARNING,
                title="Parameter is registered multiple times in the optimizer",
                message=(
                    f"{object_name} appears {len(records)} times in optimizer "
                    "parameter groups."
                ),
                object_name=object_name,
                evidence=(
                    Evidence("occurrence_count", len(records)),
                    Evidence("optimizer_positions", positions),
                    Evidence("parameter_name", model_record.primary_name if model_record else None),
                    Evidence("aliases", aliases),
                    Evidence("shape", records[0].shape),
                ),
                possible_causes=(
                    "The same parameter may have been passed to multiple optimizer groups.",
                    "A parameter list may have been concatenated without deduplicating "
                    "by identity.",
                ),
                suggestions=(
                    "Ensure each Parameter object is registered in the optimizer only once.",
                    "When using tied weights, pass the shared Parameter once and keep "
                    "aliases in the model.",
                ),
            )
        )

    return tuple(diagnostics)


def _preview_names(names: Iterable[str]) -> tuple[tuple[str, ...], int]:
    sorted_names = tuple(sorted(names))
    preview = sorted_names[:_PREVIEW_LIMIT]
    return preview, max(0, len(sorted_names) - len(preview))


def _optimizer_group_count(
    optimizer_parameters: tuple[OptimizerParameterRecord, ...],
) -> int:
    if not optimizer_parameters:
        return 0
    return max(record.group_index for record in optimizer_parameters) + 1


def _optimizer_positions_by_id(
    optimizer_parameters: tuple[OptimizerParameterRecord, ...],
) -> dict[int, tuple[str, ...]]:
    positions: dict[int, list[str]] = defaultdict(list)
    for record in optimizer_parameters:
        positions[record.parameter_id].append(record.position_name)
    return {parameter_id: tuple(items) for parameter_id, items in positions.items()}
