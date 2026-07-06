import copy
import gc
import json
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import trainmedic.monitors.gradients as gradient_module
from trainmedic import Severity, inspect_optimizer, watch_gradients
from trainmedic.reports.json_report import diagnostics_to_json


class HealthyGradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor, *, use_unused: bool = False) -> torch.Tensor:
        if use_unused:
            return self.unused(inputs)
        return self.used(inputs)


class SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.ones(2))
        self.left = shared
        self.right = shared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.left * inputs).sum()


class ScaleModel(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.scale = scale

    def forward(self) -> torch.Tensor:
        return (self.weight * self.scale).sum()


class NoParameterModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        return torch.full((2,), float("nan"))


class InfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        return torch.full((2,), float("inf"))


class BothGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        return torch.tensor([float("nan"), float("inf")])


class BadGradientModel(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.mode = mode

    def forward(self) -> torch.Tensor:
        if self.mode == "nan":
            return NaNGradient.apply(self.weight).sum()
        if self.mode == "inf":
            return InfGradient.apply(self.weight).sum()
        return BothGradient.apply(self.weight).sum()


class TwoBadParameters(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.ones(2))
        self.second = nn.Parameter(torch.ones(2))

    def forward(self) -> torch.Tensor:
        return NaNGradient.apply(self.first).sum() + NaNGradient.apply(self.second).sum()


def _codes(diagnostics: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in diagnostics)


def _evidence(diagnostic: Any) -> dict[str, Any]:
    return {
        item["name"]: item["value"]
        for item in diagnostic.to_dict()["evidence"]
    }


def _parameter_hook_count(model: nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        hooks = getattr(parameter, "_post_accumulate_grad_hooks", None)
        if hooks is not None:
            total += len(hooks)
    return total


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_tensors(item)
    elif hasattr(value, "__dict__"):
        for item in value.__dict__.values():
            yield from _iter_tensors(item)


def test_healthy_gradients_have_no_diagnostics_and_are_not_modified() -> None:
    model = HealthyGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.ones(2)

    with watch_gradients(model, optimizer) as monitor:
        loss = model(inputs).sum()
        loss.backward()
        gradients_before = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        diagnostics = monitor.check_gradients()

    assert diagnostics == ()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, gradients_before[name])


def test_grad_none_is_reported_as_one_aggregate_diagnostic() -> None:
    model = BranchModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_gradients(model, optimizer) as monitor:
        model(torch.ones(1, 2)).sum().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2001",)
    evidence = _evidence(diagnostics[0])
    assert evidence["none_gradient_count"] == 1
    assert evidence["non_none_gradient_count"] == 1
    assert evidence["none_parameter_names_preview"] == ["unused.weight"]
    assert "used.weight" not in evidence["none_parameter_names_preview"]


def test_check_before_backward_reports_grad_none_without_backward_observed() -> None:
    model = HealthyGradientModel()

    with watch_gradients(model) as monitor:
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2001",)
    assert _evidence(diagnostics[0])["any_backward_observed"] is False


def test_zero_grad_set_to_none_before_check_reports_grad_none() -> None:
    model = HealthyGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_gradients(model, optimizer) as monitor:
        model(torch.ones(2)).sum().backward()
        optimizer.zero_grad(set_to_none=True)
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2001",)


def test_gradient_nan_is_reported_once_with_parameter_evidence() -> None:
    model = BadGradientModel("nan")

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002",)
    evidence = _evidence(diagnostics[0])
    assert diagnostics[0].severity is Severity.ERROR
    assert diagnostics[0].object_name == "weight"
    assert evidence["parameter_name"] == "weight"
    assert evidence["parameter_aliases"] == ["weight"]
    assert evidence["parameter_shape"] == [2]
    assert evidence["gradient_shape"] == [2]
    assert evidence["nan_count"] == 2
    assert "root cause" not in diagnostics[0].message


def test_gradient_inf_is_reported_once() -> None:
    model = BadGradientModel("inf")

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2003",)
    assert _evidence(diagnostics[0])["inf_count"] == 2


def test_same_gradient_with_nan_and_inf_reports_nan_then_inf() -> None:
    model = BadGradientModel("both")

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002", "TM2003")


def test_multiple_abnormal_parameters_report_each_issue_once() -> None:
    model = TwoBadParameters()

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002",)


def test_shared_parameter_registers_one_hook_and_reports_aliases() -> None:
    model = SharedParameterModel()
    hooks_before = _parameter_hook_count(model)

    with watch_gradients(model) as monitor:
        assert _parameter_hook_count(model) == hooks_before + 1
        model(torch.ones(2)).backward()
        diagnostics = monitor.check_gradients()

    assert diagnostics == ()
    assert _parameter_hook_count(model) == hooks_before

    bad_model = SharedParameterModel()
    with watch_gradients(bad_model) as bad_monitor:
        NaNGradient.apply(bad_model.left).sum().backward()
        bad_diagnostics = bad_monitor.check_gradients()

    evidence = _evidence(bad_diagnostics[0])
    assert evidence["parameter_aliases"] == ["left", "right"]


def test_multiple_backward_accumulates_gradients_and_hook_call_index_increments() -> None:
    model = BadGradientModel("nan")

    with watch_gradients(model) as monitor:
        model().backward(retain_graph=True)
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002",)
    assert _evidence(diagnostics[0])["hook_call_index"] == 1
    assert monitor._hook_call_counts[id(model.weight)] == 2
    assert torch.isnan(model.weight.grad).all()


def test_optimizer_selection_ignores_trainable_parameters_not_in_optimizer() -> None:
    model = BranchModel()
    optimizer = torch.optim.SGD(model.used.parameters(), lr=0.1)

    with watch_gradients(model, optimizer) as monitor:
        model(torch.ones(1, 2)).sum().backward()
        diagnostics = monitor.check_gradients()

    assert diagnostics == ()
    assert tuple(
        diagnostic.code
        for diagnostic in inspect_optimizer(model, optimizer)
    ) == ("TM1001",)


def test_without_optimizer_monitors_all_trainable_parameters() -> None:
    model = BranchModel()

    with watch_gradients(model) as monitor:
        model(torch.ones(1, 2)).sum().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2001",)


def test_no_parameter_model_reports_tm2000() -> None:
    model = NoParameterModel()

    with watch_gradients(model) as monitor:
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2000",)


def test_all_frozen_model_reports_tm2000() -> None:
    model = HealthyGradientModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with watch_gradients(model) as monitor:
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2000",)


def test_optimizer_with_no_matching_model_parameters_reports_tm2000() -> None:
    model = HealthyGradientModel()
    external = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([external], lr=0.1)

    with watch_gradients(model, optimizer) as monitor:
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2000",)


def test_sparse_coo_gradient_is_checked_without_false_diagnostics() -> None:
    embedding = nn.Embedding(5, 3, sparse=True)
    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)

    with watch_gradients(embedding, optimizer) as monitor:
        embedding(torch.tensor([0, 2, 2])).sum().backward()
        diagnostics = monitor.check_gradients()

    assert diagnostics == ()
    assert monitor.unsupported_gradient_count == 0
    assert embedding.weight.grad.layout is torch.sparse_coo


def test_sparse_coo_gradient_values_with_nan_are_reported() -> None:
    embedding = nn.Embedding(5, 3, sparse=True)

    with watch_gradients(embedding) as monitor:
        loss = embedding(torch.tensor([0, 2])).sum() * torch.tensor(float("nan"))
        loss.backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002",)
    evidence = _evidence(diagnostics[0])
    assert evidence["gradient_layout"] == "torch.sparse_coo"
    assert evidence["gradient_nnz"] == 2
    assert evidence["nan_count"] == 6


def test_unsupported_gradient_count_increases_without_false_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HealthyGradientModel()

    monkeypatch.setattr(gradient_module, "_summarize_gradient", lambda gradient: None)
    with watch_gradients(model) as monitor:
        model(torch.ones(2)).sum().backward()
        diagnostics = monitor.check_gradients()

    assert diagnostics == ()
    assert monitor.unsupported_gradient_count >= 1


def test_global_norm_exceeds_user_threshold() -> None:
    model = ScaleModel(scale=3.0)

    with watch_gradients(model, max_global_norm=4.0) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2004",)
    evidence = _evidence(diagnostics[0])
    assert evidence["actual_global_norm"] == pytest.approx(4.242640687)
    assert evidence["configured_threshold"] == 4.0


def test_global_norm_below_user_threshold() -> None:
    model = ScaleModel(scale=0.1)

    with watch_gradients(model, min_global_norm=1.0) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2005",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_global_norm": 0}, "max_global_norm"),
        ({"max_global_norm": -1}, "max_global_norm"),
        ({"min_global_norm": -1}, "min_global_norm"),
        ({"min_global_norm": 2, "max_global_norm": 1}, "min_global_norm"),
        ({"max_global_norm": float("nan")}, "max_global_norm"),
        ({"max_global_norm": float("inf")}, "max_global_norm"),
        ({"max_global_norm": True}, "max_global_norm"),
        ({"min_global_norm": False}, "min_global_norm"),
        ({"max_global_norm": "1"}, "max_global_norm"),
    ],
)
def test_invalid_thresholds_raise_value_error(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        watch_gradients(HealthyGradientModel(), **kwargs)


def test_hooks_are_removed_after_context_and_later_backward_is_not_recorded() -> None:
    model = BadGradientModel("nan")
    hooks_before = _parameter_hook_count(model)

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002",)
    assert _parameter_hook_count(model) == hooks_before
    before = monitor.diagnostics
    model.weight.grad = None
    model().backward()
    assert monitor.diagnostics == before


def test_hooks_are_removed_after_backward_exception() -> None:
    class FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
            del ctx
            return inputs.clone()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            del ctx, grad_output
            raise ValueError("backward failed")

    model = ScaleModel(scale=1.0)
    hooks_before = _parameter_hook_count(model)

    with pytest.raises(ValueError, match="backward failed"), watch_gradients(model):
        FailingBackward.apply(model.weight).sum().backward()

    assert _parameter_hook_count(model) == hooks_before


def test_start_cleans_registered_gradient_hooks_if_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BranchModel()
    monitor = watch_gradients(model)
    hooks_before = _parameter_hook_count(model)
    original_error = RuntimeError("register failed")

    def fail_register_post_accumulate_grad_hook(hook: object) -> object:
        del hook
        raise original_error

    monkeypatch.setattr(
        model.unused.weight,
        "register_post_accumulate_grad_hook",
        fail_register_post_accumulate_grad_hook,
    )

    with pytest.raises(RuntimeError) as exc_info:
        monitor.start()

    assert exc_info.value is original_error
    assert _parameter_hook_count(model) == hooks_before
    assert monitor._active is False

    monkeypatch.undo()
    monitor.start()
    monitor.close()
    assert _parameter_hook_count(model) == hooks_before


def test_repeated_start_close_and_check_rules() -> None:
    model = HealthyGradientModel()
    monitor = watch_gradients(model)

    with pytest.raises(RuntimeError, match="must be started"):
        monitor.check_gradients()

    monitor.start()
    with pytest.raises(RuntimeError, match="already active"):
        monitor.start()
    model(torch.ones(2)).sum().backward()
    monitor.check_gradients()
    with pytest.raises(RuntimeError, match="only be called once"):
        monitor.check_gradients()

    monitor.close()
    monitor.close()


def test_gradient_monitor_has_no_training_side_effects_and_step_matches() -> None:
    torch.manual_seed(0)
    base_model = HealthyGradientModel()
    monitored_model = copy.deepcopy(base_model)
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=0.1, momentum=0.9)
    monitored_optimizer = torch.optim.SGD(monitored_model.parameters(), lr=0.1, momentum=0.9)
    inputs = torch.randn(4, 2)

    base_output = base_model(inputs)
    base_loss = base_output.sum()
    base_loss.backward()

    with watch_gradients(monitored_model, monitored_optimizer) as monitor:
        monitored_output = monitored_model(inputs)
        monitored_loss = monitored_output.sum()
        monitored_loss.backward()
        diagnostics = monitor.check_gradients()

    assert diagnostics == ()
    torch.testing.assert_close(monitored_output, base_output)
    torch.testing.assert_close(monitored_loss, base_loss)
    for (_, base_parameter), (_, monitored_parameter) in zip(
        base_model.named_parameters(),
        monitored_model.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(monitored_parameter.grad, base_parameter.grad)
        assert monitored_parameter.requires_grad is base_parameter.requires_grad

    base_optimizer.step()
    monitored_optimizer.step()
    for (_, base_parameter), (_, monitored_parameter) in zip(
        base_model.named_parameters(),
        monitored_model.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(monitored_parameter, base_parameter)
    assert monitored_model.training is base_model.training
    assert len(monitored_optimizer.param_groups) == len(base_optimizer.param_groups)
    assert len(monitored_optimizer.state) == len(base_optimizer.state)
    for (_, base_parameter), (_, monitored_parameter) in zip(
        base_model.named_parameters(),
        monitored_model.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            monitored_optimizer.state[monitored_parameter]["momentum_buffer"],
            base_optimizer.state[base_parameter]["momentum_buffer"],
        )


def test_monitor_does_not_store_gradient_tensors_or_masks() -> None:
    model = BadGradientModel("nan")

    with watch_gradients(model) as monitor:
        model().backward()
        monitor.check_gradients()

    model.weight.grad = None
    gc.collect()

    assert monitor._handles == []
    assert all(isinstance(value, int) for value in monitor._hook_call_counts.values())
    for diagnostic in monitor.diagnostics:
        for evidence in diagnostic.evidence:
            assert list(_iter_tensors(evidence.value)) == []


def test_gradient_diagnostics_format_as_strict_json() -> None:
    model = BadGradientModel("both")

    with watch_gradients(model) as monitor:
        model().backward()
        diagnostics = monitor.check_gradients()

    payload = diagnostics_to_json(diagnostics)
    parsed = json.loads(payload)
    assert [item["code"] for item in parsed] == ["TM2002", "TM2003"]
    assert parsed[0]["evidence"][-2]["value"] == 1
