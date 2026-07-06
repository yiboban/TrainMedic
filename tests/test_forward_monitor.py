import copy
import gc
import weakref
from typing import Any

import pytest
import torch
from torch import nn

from trainmedic import Severity, watch_forward


class FiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class NaNModule(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(-torch.ones_like(inputs))


class InfModule(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / torch.zeros_like(inputs)


class NaNSequentialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(2, 2, bias=False)
        self.bad = NaNModule()
        self.linear2 = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.bad(self.linear1(inputs)))


class InfModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bad = InfModule()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bad(inputs)


class BothNaNAndInfModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.new_tensor([float("nan"), float("inf")])


class NestedOutputModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        abnormal = torch.log(-torch.ones_like(inputs))
        return {
            "logits": inputs + 1,
            "cache": [
                inputs + 2,
                {"hidden": abnormal},
            ],
        }


class RootFunctionalNaN(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(inputs)


class FunctionalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.identity(inputs)
        return torch.log(-torch.ones_like(values))


class WrapperWithFunctionalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = FunctionalBlock()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SharedModuleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = NaNModule()
        self.left = shared
        self.right = shared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.left(inputs) + self.right(inputs)


class SecondCallInf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 2:
            return inputs / torch.zeros_like(inputs)
        return inputs + 1


class RepeatedModuleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.repeat = SecondCallInf()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.repeat(self.repeat(inputs))


class RaisingModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise ValueError("model failed")


class NewOutputModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 2


class MixedUnsupportedModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        del inputs
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        with torch.sparse.check_sparse_tensor_invariants(False):
            sparse = torch.sparse_coo_tensor(indices, values, (2, 2))
        return (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([True, False]),
            torch.empty(0),
            sparse,
        )


def _diagnostic_evidence(diagnostic_index: int, diagnostics: tuple[Any, ...]) -> dict[str, Any]:
    return {
        item["name"]: item["value"]
        for item in diagnostics[diagnostic_index].to_dict()["evidence"]
    }


def _hook_count(model: nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def test_healthy_forward_has_no_diagnostics() -> None:
    model = FiniteModel()

    with watch_forward(model) as monitor:
        model(torch.ones(1, 2))

    assert monitor.diagnostics == ()


def test_nan_module_is_reported_before_downstream_propagation() -> None:
    model = NaNSequentialModel()

    with watch_forward(model) as monitor:
        model(torch.ones(1, 2))

    diagnostics = monitor.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM3001",)
    assert diagnostics[0].severity is Severity.ERROR
    assert diagnostics[0].object_name == "bad"
    evidence = _diagnostic_evidence(0, diagnostics)
    assert evidence["nan_count"] == 2
    assert evidence["tensor_path"] == "output"


def test_inf_module_is_reported() -> None:
    model = InfModel()

    with watch_forward(model) as monitor:
        model(torch.ones(2))

    diagnostics = monitor.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM3002",)
    assert diagnostics[0].object_name == "bad"
    assert _diagnostic_evidence(0, diagnostics)["inf_count"] == 2


def test_same_tensor_with_nan_and_inf_reports_nan_then_inf() -> None:
    model = BothNaNAndInfModel()

    with watch_forward(model) as monitor:
        model(torch.ones(1))

    assert tuple(diagnostic.code for diagnostic in monitor.diagnostics) == ("TM3001", "TM3002")


def test_nested_output_path_points_to_abnormal_tensor() -> None:
    model = NestedOutputModel()

    with watch_forward(model) as monitor:
        model(torch.ones(1, 2))

    evidence = _diagnostic_evidence(0, monitor.diagnostics)
    assert evidence["tensor_path"] == 'output["cache"][1]["hidden"]'


def test_root_functional_nan_is_detected_in_leaf_scope() -> None:
    model = RootFunctionalNaN()

    with watch_forward(model, module_scope="leaf") as monitor:
        model(torch.tensor([-1.0]))

    assert monitor.diagnostics[0].object_name == "<root>"


def test_all_scope_finds_non_leaf_functional_nan_more_precisely_than_leaf_scope() -> None:
    all_model = WrapperWithFunctionalBlock()
    leaf_model = WrapperWithFunctionalBlock()

    with watch_forward(all_model) as all_monitor:
        all_model(torch.ones(1, 2))

    with watch_forward(leaf_model, module_scope="leaf") as leaf_monitor:
        leaf_model(torch.ones(1, 2))

    assert all_monitor.diagnostics[0].object_name == "block"
    assert leaf_monitor.diagnostics[0].object_name == "<root>"


def test_shared_module_registers_one_hook_and_reports_aliases() -> None:
    model = SharedModuleModel()
    shared = model.left
    hooks_before = len(shared._forward_hooks)

    with watch_forward(model) as monitor:
        assert len(shared._forward_hooks) == hooks_before + 1
        model(torch.ones(1, 2))

    assert len(shared._forward_hooks) == hooks_before
    evidence = _diagnostic_evidence(0, monitor.diagnostics)
    assert evidence["module_aliases"] == ["left", "right"]


def test_repeated_module_call_index_is_reported() -> None:
    model = RepeatedModuleModel()

    with watch_forward(model) as monitor:
        model(torch.ones(1))

    evidence = _diagnostic_evidence(0, monitor.diagnostics)
    assert evidence["module_name"] == "repeat"
    assert evidence["module_call_index"] == 2


def test_hooks_are_removed_after_normal_exit_and_later_forward_is_not_recorded() -> None:
    model = NaNSequentialModel()
    hooks_before = _hook_count(model)

    with watch_forward(model) as monitor:
        model(torch.ones(1, 2))

    assert _hook_count(model) == hooks_before
    before = monitor.diagnostics
    model(torch.ones(1, 2))
    assert monitor.diagnostics == before


def test_hooks_are_removed_after_model_exception_and_exception_is_preserved() -> None:
    model = RaisingModel()
    hooks_before = _hook_count(model)

    with pytest.raises(ValueError, match="model failed"), watch_forward(model):
        model(torch.ones(1))

    assert _hook_count(model) == hooks_before


def test_monitor_has_no_training_side_effects_and_backward_matches() -> None:
    torch.manual_seed(0)
    base_model = FiniteModel()
    monitored_model = copy.deepcopy(base_model)
    inputs = torch.randn(3, 2, requires_grad=True)
    monitored_inputs = inputs.detach().clone().requires_grad_(True)
    params_before = {
        name: parameter.detach().clone()
        for name, parameter in monitored_model.named_parameters()
    }
    requires_grad_before = {
        name: parameter.requires_grad
        for name, parameter in monitored_model.named_parameters()
    }
    training_before = monitored_model.training

    base_output = base_model(inputs)
    base_loss = base_output.sum()
    base_loss.backward()

    with watch_forward(monitored_model) as monitor:
        monitored_output = monitored_model(monitored_inputs)
        monitored_grad_fn_type = type(monitored_output.grad_fn)
        monitored_loss = monitored_output.sum()
        monitored_loss.backward()

    assert monitor.diagnostics == ()
    assert torch.allclose(base_output, monitored_output)
    assert monitored_grad_fn_type is type(base_output.grad_fn)
    assert monitored_model.training is training_before
    for name, parameter in monitored_model.named_parameters():
        assert torch.equal(parameter, params_before[name])
        assert parameter.requires_grad is requires_grad_before[name]
        base_parameter = dict(base_model.named_parameters())[name]
        assert torch.allclose(parameter.grad, base_parameter.grad)


def test_monitor_does_not_retain_output_tensor() -> None:
    model = NewOutputModel()
    inputs = torch.ones(2)

    with watch_forward(model) as monitor:
        output = model(inputs)
        output_ref = weakref.ref(output)

    assert monitor.diagnostics == ()
    del output
    gc.collect()
    assert output_ref() is None


def test_integer_bool_empty_and_sparse_outputs_do_not_create_false_diagnostics() -> None:
    model = MixedUnsupportedModel()

    with watch_forward(model) as monitor:
        model(torch.ones(1))

    assert monitor.diagnostics == ()
    assert monitor.unsupported_tensor_count == 1


def test_invalid_module_scope_raises_value_error() -> None:
    with pytest.raises(ValueError, match="module_scope must be one of"):
        watch_forward(FiniteModel(), module_scope="invalid")


def test_repeated_start_fails_and_close_is_idempotent() -> None:
    model = FiniteModel()
    hooks_before = _hook_count(model)
    monitor = watch_forward(model)

    monitor.start()
    assert _hook_count(model) > hooks_before
    with pytest.raises(RuntimeError, match="already active"):
        monitor.start()

    monitor.close()
    monitor.close()
    assert _hook_count(model) == hooks_before
