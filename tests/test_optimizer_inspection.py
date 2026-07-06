import json

import torch
from torch import nn

from trainmedic import Diagnostic, Severity, inspect_optimizer
from trainmedic.inspectors.model import collect_model_parameters
from trainmedic.reports.json_report import diagnostics_to_json


class TwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 1, bias=False)


class NoParameterModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.ones(3))
        self.left = shared
        self.right = shared


def _codes(diagnostics: tuple[Diagnostic, ...]) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in diagnostics)


def test_healthy_model_optimizer_returns_no_diagnostics() -> None:
    model = TwoLayerModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert inspect_optimizer(model, optimizer) == ()


def test_missing_trainable_parameter_reports_tm1001() -> None:
    model = TwoLayerModel()
    optimizer = torch.optim.SGD(model.first.parameters(), lr=0.1)

    diagnostics = inspect_optimizer(model, optimizer)

    assert _codes(diagnostics) == ("TM1001",)
    diagnostic = diagnostics[0]
    assert diagnostic.object_name == "second.weight"
    evidence = diagnostic.to_dict()["evidence"]
    assert {"name": "parameter_name", "value": "second.weight", "description": None} in evidence
    assert {"name": "requires_grad", "value": True, "description": None} in evidence
    assert {"name": "optimizer_group_count", "value": 1, "description": None} in evidence


def test_optimizer_parameter_not_in_model_reports_stable_position() -> None:
    model = TwoLayerModel()
    external = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {"params": list(model.parameters())},
            {"params": [external]},
        ],
        lr=0.1,
    )

    diagnostics = inspect_optimizer(model, optimizer)

    assert _codes(diagnostics) == ("TM1002",)
    assert diagnostics[0].object_name == "optimizer.param_groups[1].params[0]"
    assert "0x" not in diagnostics[0].object_name
    evidence = diagnostics[0].to_dict()["evidence"]
    assert {"name": "optimizer_group_index", "value": 1, "description": None} in evidence
    assert {"name": "optimizer_parameter_index", "value": 0, "description": None} in evidence


def test_frozen_parameter_in_optimizer_reports_info_without_tm1001() -> None:
    model = TwoLayerModel()
    model.second.weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    diagnostics = inspect_optimizer(model, optimizer)
    frozen_diagnostics = [diagnostic for diagnostic in diagnostics if diagnostic.code == "TM1003"]

    assert "TM1001" not in _codes(diagnostics)
    assert len(frozen_diagnostics) == 1
    assert frozen_diagnostics[0].severity is Severity.INFO
    assert frozen_diagnostics[0].object_name == "second.weight"


def test_partially_frozen_model_reports_one_aggregate_tm1004() -> None:
    model = TwoLayerModel()
    model.second.weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.first.parameters(), lr=0.1)

    diagnostics = inspect_optimizer(model, optimizer)

    assert _codes(diagnostics) == ("TM1004",)
    evidence = diagnostics[0].to_dict()["evidence"]
    assert {"name": "frozen_parameter_count", "value": 1, "description": None} in evidence
    assert {"name": "trainable_parameter_count", "value": 1, "description": None} in evidence
    assert {
        "name": "frozen_parameter_names_preview",
        "value": ["second.weight"],
        "description": None,
    } in evidence


def test_all_frozen_model_reports_tm1005_without_tm1004_or_tm1001() -> None:
    model = TwoLayerModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    diagnostics = inspect_optimizer(model, optimizer)
    codes = _codes(diagnostics)

    assert codes[0] == "TM1005"
    assert "TM1004" not in codes
    assert "TM1001" not in codes


def test_no_parameter_model_reports_tm1006_without_crashing() -> None:
    model = NoParameterModel()
    placeholder = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([placeholder], lr=0.1)
    optimizer.param_groups[0]["params"] = []

    diagnostics = inspect_optimizer(model, optimizer)

    assert _codes(diagnostics) == ("TM1006",)


def test_duplicate_optimizer_parameter_reports_tm1007() -> None:
    model = TwoLayerModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.param_groups[0]["params"].append(model.first.weight)

    diagnostics = inspect_optimizer(model, optimizer)

    assert _codes(diagnostics) == ("TM1007",)
    diagnostic = diagnostics[0]
    assert diagnostic.object_name == "first.weight"
    evidence = diagnostic.to_dict()["evidence"]
    assert {"name": "occurrence_count", "value": 2, "description": None} in evidence
    assert {
        "name": "optimizer_positions",
        "value": [
            "optimizer.param_groups[0].params[0]",
            "optimizer.param_groups[0].params[2]",
        ],
        "description": None,
    } in evidence


def test_shared_parameter_aliases_do_not_count_as_optimizer_duplicates() -> None:
    model = SharedParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert inspect_optimizer(model, optimizer) == ()

    records = collect_model_parameters(model)
    assert len(records) == 1
    assert records[0].primary_name == "left"
    assert records[0].aliases == ("left", "right")


def test_optimizer_inspection_order_is_stable() -> None:
    model = TwoLayerModel()
    external = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {"params": list(model.first.parameters())},
            {"params": [external]},
        ],
        lr=0.1,
    )

    first = inspect_optimizer(model, optimizer)
    second = inspect_optimizer(model, optimizer)

    assert first == second
    assert diagnostics_to_json(first) == diagnostics_to_json(second)
    json.loads(diagnostics_to_json(first))


def test_optimizer_inspection_has_no_side_effects() -> None:
    model = TwoLayerModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameter_values = {
        id(parameter): parameter.detach().clone()
        for parameter in model.parameters()
    }
    requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    param_group_ids = [
        [id(parameter) for parameter in group["params"]]
        for group in optimizer.param_groups
    ]
    optimizer_state_keys = tuple(optimizer.state.keys())
    model_training = model.training

    inspect_optimizer(model, optimizer)

    for parameter in model.parameters():
        assert torch.equal(parameter, parameter_values[id(parameter)])
        assert parameter.requires_grad is requires_grad[id(parameter)]

    assert [
        [id(parameter) for parameter in group["params"]]
        for group in optimizer.param_groups
    ] == param_group_ids
    assert tuple(optimizer.state.keys()) == optimizer_state_keys
    assert model.training is model_training
