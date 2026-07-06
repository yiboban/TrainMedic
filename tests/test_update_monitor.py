import copy
import gc
import json
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import trainmedic.monitors.updates as update_module
from trainmedic import inspect_optimizer, watch_gradients, watch_updates
from trainmedic.reports.console import format_diagnostics
from trainmedic.reports.json_report import diagnostics_to_json


class TwoParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.ones(2))
        self.second = nn.Parameter(torch.ones(2))

    def forward(self) -> torch.Tensor:
        return self.first.sum() + self.second.sum()


class BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.used(inputs)


class SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.ones(4))
        self.left = shared
        self.right = shared

    def forward(self) -> torch.Tensor:
        return self.left.sum()


class NoParameterModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Any, *, lr: float = 0.1) -> None:
        super().__init__(params, {"lr": lr})

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        return loss


class PartialOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Any, *, lr: float = 0.1) -> None:
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        updated = False
        for group in self.param_groups:
            lr = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not updated:
                    parameter.add_(parameter.grad, alpha=-lr)
                    updated = True
        return loss


class FailingOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Any) -> None:
        super().__init__(params, {"lr": 0.1})

    def step(self, closure: Any = None) -> Any:
        if closure is not None:
            closure()
        raise RuntimeError("optimizer step failed")


class NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        return torch.full((2,), float("nan"))


def _codes(diagnostics: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in diagnostics)


def _evidence(diagnostic: Any) -> dict[str, Any]:
    return {
        item["name"]: item["value"]
        for item in diagnostic.to_dict()["evidence"]
    }


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_tensors(item)


def _run_sgd_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> Any:
    optimizer.zero_grad()
    loss = model().sum()
    loss.backward()
    return optimizer.step()


def test_healthy_sgd_update_has_no_diagnostics() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    assert monitor.diagnostics == ()
    assert monitor.step_count == 1
    assert monitor._pending_snapshot is None


def test_missing_optimizer_step_reports_tm4001() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        model().backward()

    assert _codes(monitor.diagnostics) == ("TM4001",)
    assert "this monitor session" in monitor.diagnostics[0].message


def test_user_exception_without_step_does_not_report_tm4001() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with (
        pytest.raises(ValueError, match="user failure"),
        watch_updates(model, optimizer) as monitor,
    ):
        raise ValueError("user failure")

    assert monitor.diagnostics == ()
    assert monitor._pending_snapshot is None


def test_empty_parameter_selection_reports_tm4000_not_tm4001() -> None:
    model = NoParameterModel()
    external = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([external], lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        pass

    assert _codes(monitor.diagnostics) == ("TM4000",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["selected_parameter_count"] == 0
    assert evidence["optimizer_model_parameter_count"] == 0


def test_zero_learning_rate_reports_tm4002_not_tm4003_and_parameter_stays_fixed() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    assert _codes(monitor.diagnostics) == ("TM4002",)
    assert "TM4003" not in _codes(monitor.diagnostics)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_partial_zero_lr_group_only_reports_zero_lr_candidates() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.first], "lr": 0.0},
            {"params": [model.second], "lr": 0.1},
        ]
    )

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    assert _codes(monitor.diagnostics) == ("TM4002",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["affected_parameter_names_preview"] == ["first"]
    assert evidence["optimizer_group_indices"] == [0]
    assert torch.equal(model.first, torch.ones(2))
    assert not torch.equal(model.second, torch.ones(2))


def test_no_op_optimizer_reports_tm4003() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    assert _codes(monitor.diagnostics) == ("TM4003",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["candidate_parameter_count"] == 2
    assert evidence["changed_candidate_count"] == 0
    assert evidence["unchanged_candidate_count"] == 2


def test_partial_parameter_update_reports_only_unchanged_preview() -> None:
    model = TwoParameterModel()
    optimizer = PartialOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    assert _codes(monitor.diagnostics) == ("TM4003",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["changed_candidate_count"] == 1
    assert evidence["unchanged_candidate_count"] == 1
    assert evidence["unchanged_parameter_names_preview"] == ["second"]


def test_grad_none_does_not_generate_tm4003() -> None:
    model = BranchModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

    assert monitor.diagnostics == ()


def test_zero_gradient_does_not_generate_tm4003() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()

    assert monitor.diagnostics == ()


def test_nan_gradient_is_left_to_gradient_monitor() -> None:
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(2))
    optimizer = NoOpOptimizer([model.weight], lr=0.1)

    with watch_updates(model, optimizer) as update_monitor:
        NaNGradient.apply(model.weight).sum().backward()
        optimizer.step()

    assert update_monitor.diagnostics == ()

    model.weight.grad = None
    with watch_gradients(model) as gradient_monitor:
        NaNGradient.apply(model.weight).sum().backward()
        diagnostics = gradient_monitor.check_gradients()

    assert _codes(diagnostics) == ("TM2002",)


def test_exact_snapshot_evidence_for_small_parameter() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer, sample_size=64) as monitor:
        _run_sgd_step(model, optimizer)

    preview = _evidence(monitor.diagnostics[0])["per_parameter_preview"]
    assert preview[0]["coverage"] == "exact"


def test_sampled_snapshot_evidence_for_large_parameter() -> None:
    model = nn.Linear(128, 1, bias=False)
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer, sample_size=3) as monitor:
        optimizer.zero_grad()
        model(torch.ones(1, 128)).sum().backward()
        optimizer.step()

    diagnostic = monitor.diagnostics[0]
    evidence = _evidence(diagnostic)
    preview = evidence["per_parameter_preview"]
    assert evidence["sampled_unchanged_count"] == 1
    assert preview[0]["coverage"] == "sampled"
    assert preview[0]["sampled_element_count"] == 3
    assert "sampled elements" in diagnostic.message
    assert "entire parameter" not in diagnostic.message


def test_global_snapshot_budget_is_enforced() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer, sample_size=64, max_snapshot_elements=2) as monitor:
        _run_sgd_step(model, optimizer)

    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["unsupported_or_skipped_count"] == 1
    assert monitor.unsupported_parameter_count >= 1
    assert monitor._pending_snapshot is None


def test_non_contiguous_parameter_is_sampled_without_crashing() -> None:
    class NonContiguousModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 2).t())

        def forward(self) -> torch.Tensor:
            return self.weight.sum()

    model = NonContiguousModel()
    assert not model.weight.is_contiguous()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer, sample_size=3) as monitor:
        _run_sgd_step(model, optimizer)

    assert _codes(monitor.diagnostics) == ("TM4003",)
    preview = _evidence(monitor.diagnostics[0])["per_parameter_preview"]
    assert preview[0]["coverage"] in {"exact", "sampled"}


def test_shared_parameter_is_sampled_once_and_keeps_aliases() -> None:
    model = SharedParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["candidate_parameter_count"] == 1
    assert evidence["per_parameter_preview"][0]["aliases"] == ["left", "right"]


def test_multiple_steps_keep_count_and_do_not_repeat_tm4003() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)
        _run_sgd_step(model, optimizer)

    assert monitor.step_count == 2
    assert _codes(monitor.diagnostics) == ("TM4003",)
    assert _evidence(monitor.diagnostics[0])["step_index"] == 1


def test_step_exception_cleans_snapshot_and_does_not_finalize_no_step() -> None:
    model = TwoParameterModel()
    optimizer = FailingOptimizer(model.parameters())

    with (
        pytest.raises(RuntimeError, match="optimizer step failed"),
        watch_updates(model, optimizer) as monitor,
    ):
        model().backward()
        optimizer.step()

    assert monitor.step_count == 0
    assert monitor.diagnostics == ()
    assert monitor._pending_snapshot is None
    assert monitor._handles == []


def test_start_cleans_pre_hook_if_post_hook_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monitor = watch_updates(model, optimizer)
    original_error = RuntimeError("post hook failed")

    def fail_register_post_hook(hook: object) -> object:
        del hook
        raise original_error

    monkeypatch.setattr(optimizer, "register_step_post_hook", fail_register_post_hook)

    with pytest.raises(RuntimeError) as exc_info:
        monitor.start()

    assert exc_info.value is original_error
    assert monitor._active is False

    monkeypatch.undo()
    _run_sgd_step(model, optimizer)
    assert monitor.diagnostics == ()

    monitor.start()
    monitor.close()


def test_repeated_start_close_and_restart_are_session_scoped() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)
    monitor = watch_updates(model, optimizer)

    monitor.start()
    with pytest.raises(RuntimeError, match="already active"):
        monitor.start()
    _run_sgd_step(model, optimizer)
    monitor.close()
    first = monitor.diagnostics
    monitor.close()
    assert monitor.diagnostics == first

    monitor.start()
    assert monitor.diagnostics == ()
    assert monitor.step_count == 0
    _run_sgd_step(model, optimizer)
    monitor.close()
    assert _evidence(monitor.diagnostics[0])["step_index"] == 1


def test_external_optimizer_hooks_still_run_and_are_not_removed() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls: list[str] = []

    def user_pre_hook(
        optimizer_arg: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        assert optimizer_arg is optimizer
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        calls.append("pre")

    def user_post_hook(
        optimizer_arg: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        assert optimizer_arg is optimizer
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        calls.append("post")

    pre_handle = optimizer.register_step_pre_hook(user_pre_hook)
    post_handle = optimizer.register_step_post_hook(user_post_hook)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)

    assert monitor.diagnostics == ()
    assert calls == ["pre", "post"]

    _run_sgd_step(model, optimizer)
    assert calls == ["pre", "post", "pre", "post"]
    pre_handle.remove()
    post_handle.remove()


def test_closure_is_called_and_return_value_is_preserved() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        return loss

    with watch_updates(model, optimizer) as monitor:
        result = optimizer.step(closure)

    assert closure_calls == 1
    assert isinstance(result, torch.Tensor)
    assert monitor.step_count == 1


@pytest.mark.parametrize(
    "optimizer_factory",
    [
        lambda params: torch.optim.SGD(params, lr=0.1),
        lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        lambda params: torch.optim.Adam(params, lr=0.01),
    ],
)
def test_update_monitor_has_no_training_side_effects(optimizer_factory: Any) -> None:
    torch.manual_seed(0)
    base_model = nn.Linear(3, 2)
    monitored_model = copy.deepcopy(base_model)
    base_optimizer = optimizer_factory(base_model.parameters())
    monitored_optimizer = optimizer_factory(monitored_model.parameters())
    inputs = torch.randn(4, 3)

    base_optimizer.zero_grad()
    base_output = base_model(inputs)
    base_loss = base_output.sum()
    base_loss.backward()
    base_step_result = base_optimizer.step()

    with watch_updates(monitored_model, monitored_optimizer) as monitor:
        monitored_optimizer.zero_grad()
        monitored_output = monitored_model(inputs)
        monitored_loss = monitored_output.sum()
        monitored_loss.backward()
        monitored_step_result = monitored_optimizer.step()

    assert monitor.diagnostics == ()
    assert monitored_step_result == base_step_result
    torch.testing.assert_close(monitored_output, base_output)
    torch.testing.assert_close(monitored_loss, base_loss)
    assert monitored_model.training is base_model.training

    for (_, base_parameter), (_, monitored_parameter) in zip(
        base_model.named_parameters(),
        monitored_model.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(monitored_parameter, base_parameter)
        torch.testing.assert_close(monitored_parameter.grad, base_parameter.grad)
        assert monitored_parameter.requires_grad is base_parameter.requires_grad

    assert len(monitored_optimizer.param_groups) == len(base_optimizer.param_groups)
    assert monitored_optimizer.param_groups[0]["lr"] == base_optimizer.param_groups[0]["lr"]
    assert len(monitored_optimizer.state) == len(base_optimizer.state)
    for (_, base_parameter), (_, monitored_parameter) in zip(
        base_model.named_parameters(),
        monitored_model.named_parameters(),
        strict=True,
    ):
        base_state = base_optimizer.state.get(base_parameter, {})
        monitored_state = monitored_optimizer.state.get(monitored_parameter, {})
        assert base_state.keys() == monitored_state.keys()
        for key in base_state:
            if isinstance(base_state[key], torch.Tensor):
                torch.testing.assert_close(monitored_state[key], base_state[key])
            else:
                assert monitored_state[key] == base_state[key]


def test_scheduler_behavior_is_not_changed_by_update_monitor() -> None:
    base_model = TwoParameterModel()
    monitored_model = copy.deepcopy(base_model)
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=0.1)
    monitored_optimizer = torch.optim.SGD(monitored_model.parameters(), lr=0.1)
    base_scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=1, gamma=0.5)
    monitored_scheduler = torch.optim.lr_scheduler.StepLR(
        monitored_optimizer,
        step_size=1,
        gamma=0.5,
    )

    _run_sgd_step(base_model, base_optimizer)
    base_scheduler.step()

    with watch_updates(monitored_model, monitored_optimizer) as monitor:
        _run_sgd_step(monitored_model, monitored_optimizer)
    monitored_scheduler.step()

    assert monitor.diagnostics == ()
    assert monitored_scheduler.get_last_lr() == base_scheduler.get_last_lr()


def test_snapshot_tensors_are_released_after_post_and_close() -> None:
    model = TwoParameterModel()
    optimizer = NoOpOptimizer(model.parameters(), lr=0.1)

    with watch_updates(model, optimizer) as monitor:
        _run_sgd_step(model, optimizer)
        assert monitor._pending_snapshot is None

    assert monitor._pending_snapshot is None
    gc.collect()
    for diagnostic in monitor.diagnostics:
        for evidence in diagnostic.evidence:
            assert list(_iter_tensors(evidence.value)) == []


@pytest.mark.parametrize("name", ["sample_size", "max_snapshot_elements"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1", None])
def test_invalid_snapshot_configuration_raises_value_error(name: str, value: object) -> None:
    kwargs = {"sample_size": 64, "max_snapshot_elements": 100_000}
    kwargs[name] = value

    with pytest.raises(ValueError, match=name):
        watch_updates(
            TwoParameterModel(),
            NoOpOptimizer(TwoParameterModel().parameters()),
            **kwargs,
        )


def test_tm4000_to_tm4003_format_as_console_and_strict_json() -> None:
    empty_model = NoParameterModel()
    external = nn.Parameter(torch.ones(1))
    empty_optimizer = torch.optim.SGD([external], lr=0.1)
    with watch_updates(empty_model, empty_optimizer) as empty_monitor:
        pass

    no_step_model = TwoParameterModel()
    no_step_optimizer = torch.optim.SGD(no_step_model.parameters(), lr=0.1)
    with watch_updates(no_step_model, no_step_optimizer) as no_step_monitor:
        no_step_model().backward()

    zero_lr_model = TwoParameterModel()
    zero_lr_optimizer = torch.optim.SGD(zero_lr_model.parameters(), lr=0.0)
    with watch_updates(zero_lr_model, zero_lr_optimizer) as zero_lr_monitor:
        _run_sgd_step(zero_lr_model, zero_lr_optimizer)

    noop_model = TwoParameterModel()
    noop_optimizer = NoOpOptimizer(noop_model.parameters(), lr=0.1)
    with watch_updates(noop_model, noop_optimizer) as noop_monitor:
        _run_sgd_step(noop_model, noop_optimizer)

    diagnostics = (
        empty_monitor.diagnostics
        + no_step_monitor.diagnostics
        + zero_lr_monitor.diagnostics
        + noop_monitor.diagnostics
    )
    assert _codes(diagnostics) == ("TM4000", "TM4001", "TM4002", "TM4003")
    text = format_diagnostics(diagnostics)
    assert "TM4000" in text
    parsed = json.loads(diagnostics_to_json(diagnostics))
    assert [item["code"] for item in parsed] == ["TM4000", "TM4001", "TM4002", "TM4003"]
    assert "NaN" not in diagnostics_to_json(diagnostics)


def test_optimizer_duplicate_model_parameter_count_remains_unique_for_tm4000() -> None:
    model = TwoParameterModel()
    optimizer = torch.optim.SGD([model.first], lr=0.1)
    optimizer.param_groups[0]["params"].append(model.first)
    model.first.requires_grad_(False)
    model.second.requires_grad_(False)

    with watch_updates(model, optimizer) as monitor:
        pass

    assert _codes(monitor.diagnostics) == ("TM4000",)
    assert _evidence(monitor.diagnostics[0])["optimizer_model_parameter_count"] == 1
    assert "TM1007" in _codes(inspect_optimizer(model, optimizer))


def test_internal_sampling_indices_are_deterministic_and_include_edges() -> None:
    first = update_module._deterministic_sample_indices(10, 4)
    second = update_module._deterministic_sample_indices(10, 4)

    assert first == second
    assert first[0] == 0
    assert first[-1] == 9
