import copy
import gc
import json
import weakref
from collections.abc import Iterator
from typing import Any, cast

import pytest
import torch
from torch import nn

from trainmedic import Severity, watch_modes
from trainmedic.reports.console import format_diagnostics
from trainmedic.reports.json_report import diagnostics_to_json


class LinearWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.child = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.child(inputs)


class BranchModeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 2, bias=False)
        self.unused = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor, *, use_unused: bool = False) -> torch.Tensor:
        if use_unused:
            return self.unused(inputs)
        return self.used(inputs)


class DropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.25, inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs)


class BatchNormModel(nn.Module):
    def __init__(self, *, track_running_stats: bool = True) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(2, track_running_stats=track_running_stats)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(inputs)


class SharedModuleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(2, 2, bias=False)
        self.left = shared
        self.right = shared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.left(inputs) + self.right(inputs)


class FailingForward(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        del inputs
        raise RuntimeError("forward failed")


def _codes(diagnostics: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in diagnostics)


def _evidence(diagnostic: Any) -> dict[str, Any]:
    return {
        item["name"]: item["value"]
        for item in diagnostic.to_dict()["evidence"]
    }


def _forward_pre_hook_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(module._forward_pre_hooks)
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


def test_healthy_training_has_no_diagnostics() -> None:
    model = LinearWrapper()
    model.train()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2))

    assert monitor.diagnostics == ()
    assert monitor.forward_call_count == 2


def test_healthy_evaluation_with_no_grad_has_no_diagnostics() -> None:
    model = LinearWrapper()
    model.eval()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(1, 2))

    assert monitor.diagnostics == ()


def test_root_model_eval_during_training_reports_tm5001() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5001",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["module_name"] == "<root>"
    assert evidence["expected_mode"] == "train"
    assert evidence["actual_mode"] == "eval"


def test_root_model_train_during_evaluation_reports_tm5002() -> None:
    model = nn.Linear(2, 1)
    model.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5002",)


def test_called_plain_submodule_mode_mismatch_is_info() -> None:
    model = LinearWrapper()
    model.train()
    model.child.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5003",)
    assert monitor.diagnostics[0].severity is Severity.INFO
    assert monitor.diagnostics[0].object_name == "child"


def test_uncalled_submodule_mode_mismatch_is_not_reported() -> None:
    model = BranchModeModel()
    model.train()
    model.unused.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2), use_unused=False)

    assert monitor.diagnostics == ()


def test_dropout_disabled_during_training_reports_tm5004() -> None:
    model = DropoutModel()
    model.train()
    model.dropout.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(2, 2))

    assert _codes(monitor.diagnostics) == ("TM5004",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["dropout_probability"] == 0.25
    assert evidence["inplace"] is False
    assert evidence["expected_stochastic_behavior"] is True
    assert evidence["actual_stochastic_behavior"] is False


def test_dropout_active_during_evaluation_reports_tm5004() -> None:
    model = DropoutModel()
    model.eval()
    model.dropout.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(2, 2))

    assert _codes(monitor.diagnostics) == ("TM5004",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["expected_stochastic_behavior"] is False
    assert evidence["actual_stochastic_behavior"] is True


def test_batchnorm_eval_during_training_reports_tm5005() -> None:
    model = BatchNormModel()
    model.train()
    model.batch_norm.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(2, 2))

    assert _codes(monitor.diagnostics) == ("TM5005",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["num_features"] == 2
    assert evidence["affine"] is True
    assert evidence["track_running_stats"] is True


def test_batchnorm_train_during_evaluation_reports_tm5005() -> None:
    model = BatchNormModel()
    model.eval()
    model.batch_norm.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(2, 2))

    assert _codes(monitor.diagnostics) == ("TM5005",)


def test_batchnorm_track_running_stats_false_message_is_bounded() -> None:
    model = BatchNormModel(track_running_stats=False)
    model.eval()
    model.batch_norm.train()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(2, 2))

    assert _codes(monitor.diagnostics) == ("TM5005",)
    assert "track_running_stats=False" in monitor.diagnostics[0].message


def test_eval_with_grad_enabled_reports_tm5006() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="eval") as monitor:
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5006",)
    assert monitor.diagnostics[0].severity is Severity.INFO


def test_eval_with_no_grad_does_not_report_tm5006() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="eval") as monitor, torch.no_grad():
        model(torch.ones(1, 2))

    assert monitor.diagnostics == ()


def test_eval_with_inference_mode_does_not_report_tm5006() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="eval") as monitor, torch.inference_mode():
        model(torch.ones(1, 2))

    assert monitor.diagnostics == ()


def test_train_with_no_grad_reports_tm5007() -> None:
    model = nn.Linear(2, 1)
    model.train()

    with watch_modes(model, expected_mode="train") as monitor, torch.no_grad():
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5007",)


def test_check_grad_context_false_disables_grad_context_rules_only() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with (
        watch_modes(model, expected_mode="train", check_grad_context=False) as monitor,
        torch.no_grad(),
    ):
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5001",)


def test_mode_changes_during_session_are_observed() -> None:
    model = nn.Linear(2, 1)
    model.train()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2))
        model.eval()
        model(torch.ones(1, 2))

    assert "TM5001" in _codes(monitor.diagnostics)


def test_repeated_error_forwards_do_not_duplicate_diagnostics() -> None:
    model = nn.Linear(2, 1)
    model.eval()

    with watch_modes(model, expected_mode="train") as monitor:
        model(torch.ones(1, 2))
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5001",)


def test_shared_module_has_one_hook_aliases_and_call_index() -> None:
    model = SharedModuleModel()
    model.train()
    model.left.eval()
    hooks_before = _forward_pre_hook_count(model)

    with watch_modes(model, expected_mode="train") as monitor:
        assert _forward_pre_hook_count(model) == hooks_before + 2
        model(torch.ones(1, 2))

    assert _forward_pre_hook_count(model) == hooks_before
    assert _codes(monitor.diagnostics) == ("TM5003",)
    evidence = _evidence(monitor.diagnostics[0])
    assert evidence["module_aliases"] == ["left", "right"]
    assert evidence["module_call_index"] == 1
    assert monitor.forward_call_count == 3


def test_no_forward_reports_tm5000_on_normal_exit() -> None:
    model = nn.Linear(2, 1)

    with watch_modes(model, expected_mode="train") as monitor:
        pass

    assert _codes(monitor.diagnostics) == ("TM5000",)


def test_user_exception_without_forward_does_not_report_tm5000() -> None:
    model = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="user failure"), watch_modes(
        model,
        expected_mode="train",
    ) as monitor:
        raise ValueError("user failure")

    assert monitor.diagnostics == ()
    assert monitor._handles == []


def test_forward_exception_propagates_and_keeps_real_pre_hook_diagnostic() -> None:
    model = FailingForward()
    model.eval()

    with pytest.raises(RuntimeError, match="forward failed"), watch_modes(
        model,
        expected_mode="train",
    ) as monitor:
        model(torch.ones(1, 2))

    assert _codes(monitor.diagnostics) == ("TM5001",)
    assert monitor._handles == []


def test_start_cleans_registered_hooks_if_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LinearWrapper()
    monitor = watch_modes(model, expected_mode="train")
    hooks_before = _forward_pre_hook_count(model)
    original_error = RuntimeError("register failed")

    def fail_register_forward_pre_hook(hook: object) -> object:
        del hook
        raise original_error

    monkeypatch.setattr(model.child, "register_forward_pre_hook", fail_register_forward_pre_hook)

    with pytest.raises(RuntimeError) as exc_info:
        monitor.start()

    assert exc_info.value is original_error
    assert _forward_pre_hook_count(model) == hooks_before
    assert monitor._active is False

    monkeypatch.undo()
    monitor.start()
    monitor.close()
    assert _forward_pre_hook_count(model) == hooks_before


def test_monitor_uses_lightweight_module_records_and_releases_them() -> None:
    model = LinearWrapper()
    monitor = watch_modes(model, expected_mode="train")

    monitor.start()
    assert monitor._module_records
    assert all(not hasattr(record, "module") for record in monitor._module_records)

    model(torch.ones(1, 2))
    monitor.close()

    assert monitor._module_records == ()
    assert monitor._module_call_counts == {}


def test_repeated_start_close_and_restart() -> None:
    model = nn.Linear(2, 1)
    monitor = watch_modes(model, expected_mode="train")

    monitor.start()
    with pytest.raises(RuntimeError, match="already active"):
        monitor.start()
    model(torch.ones(1, 2))
    monitor.close()
    monitor.close()

    monitor.start()
    assert monitor.diagnostics == ()
    assert monitor.forward_call_count == 0
    monitor.close()
    assert _codes(monitor.diagnostics) == ("TM5000",)


def test_dropout_monitor_has_no_rng_side_effect() -> None:
    torch.manual_seed(0)
    control_model = DropoutModel()
    monitored_model = copy.deepcopy(control_model)
    inputs = torch.ones(4, 2)

    torch.manual_seed(123)
    control_output = control_model(inputs)

    torch.manual_seed(123)
    with watch_modes(monitored_model, expected_mode="train") as monitor:
        monitored_output = monitored_model(inputs)

    assert monitor.diagnostics == ()
    torch.testing.assert_close(monitored_output, control_output)


def test_batchnorm_monitor_has_no_running_stats_side_effect() -> None:
    control_model = BatchNormModel()
    monitored_model = copy.deepcopy(control_model)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    control_output = control_model(inputs)
    with watch_modes(monitored_model, expected_mode="train") as monitor:
        monitored_output = monitored_model(inputs)

    assert monitor.diagnostics == ()
    torch.testing.assert_close(monitored_output, control_output)
    torch.testing.assert_close(
        monitored_model.batch_norm.running_mean,
        control_model.batch_norm.running_mean,
    )
    torch.testing.assert_close(
        monitored_model.batch_norm.running_var,
        control_model.batch_norm.running_var,
    )
    torch.testing.assert_close(
        monitored_model.batch_norm.num_batches_tracked,
        control_model.batch_norm.num_batches_tracked,
    )


def test_input_tensor_is_not_retained() -> None:
    model = nn.Linear(2, 1)
    model.train()
    inputs = torch.ones(1, 2)
    input_ref = weakref.ref(inputs)

    with watch_modes(model, expected_mode="train") as monitor:
        output = model(inputs)

    del inputs
    del output
    gc.collect()

    assert input_ref() is None
    for diagnostic in monitor.diagnostics:
        for evidence in diagnostic.evidence:
            assert list(_iter_tensors(evidence.value)) == []


@pytest.mark.parametrize("value", ["training", "test", "", "TRAIN", 1])
def test_invalid_expected_mode_raises_value_error(value: object) -> None:
    with pytest.raises(ValueError, match="train.*eval"):
        watch_modes(nn.Linear(2, 1), expected_mode=cast(Any, value))


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_invalid_check_grad_context_raises_value_error(value: object) -> None:
    with pytest.raises(ValueError, match="check_grad_context"):
        watch_modes(nn.Linear(2, 1), expected_mode="train", check_grad_context=cast(Any, value))


def test_tm5000_to_tm5007_format_as_console_and_strict_json() -> None:
    diagnostics = []

    with watch_modes(nn.Linear(2, 1), expected_mode="train") as no_forward:
        pass
    diagnostics.extend(no_forward.diagnostics)

    root_eval = nn.Linear(2, 1)
    root_eval.eval()
    with watch_modes(root_eval, expected_mode="train") as root_eval_monitor:
        root_eval(torch.ones(1, 2))
    diagnostics.extend(root_eval_monitor.diagnostics)

    root_train = nn.Linear(2, 1)
    root_train.train()
    with watch_modes(root_train, expected_mode="eval") as root_train_monitor, torch.no_grad():
        root_train(torch.ones(1, 2))
    diagnostics.extend(root_train_monitor.diagnostics)

    plain = LinearWrapper()
    plain.train()
    plain.child.eval()
    with watch_modes(plain, expected_mode="train") as plain_monitor:
        plain(torch.ones(1, 2))
    diagnostics.extend(plain_monitor.diagnostics)

    dropout = DropoutModel()
    dropout.eval()
    dropout.dropout.train()
    with watch_modes(dropout, expected_mode="eval") as dropout_monitor, torch.no_grad():
        dropout(torch.ones(2, 2))
    diagnostics.extend(dropout_monitor.diagnostics)

    batch_norm = BatchNormModel()
    batch_norm.eval()
    batch_norm.batch_norm.train()
    with watch_modes(batch_norm, expected_mode="eval") as batch_norm_monitor, torch.no_grad():
        batch_norm(torch.ones(2, 2))
    diagnostics.extend(batch_norm_monitor.diagnostics)

    eval_grad = nn.Linear(2, 1)
    eval_grad.eval()
    with watch_modes(eval_grad, expected_mode="eval") as eval_grad_monitor:
        eval_grad(torch.ones(1, 2))
    diagnostics.extend(eval_grad_monitor.diagnostics)

    train_no_grad = nn.Linear(2, 1)
    train_no_grad.train()
    with (
        watch_modes(train_no_grad, expected_mode="train") as train_no_grad_monitor,
        torch.no_grad(),
    ):
        train_no_grad(torch.ones(1, 2))
    diagnostics.extend(train_no_grad_monitor.diagnostics)

    assert _codes(tuple(diagnostics)) == (
        "TM5000",
        "TM5001",
        "TM5002",
        "TM5003",
        "TM5004",
        "TM5005",
        "TM5006",
        "TM5007",
    )
    text = format_diagnostics(diagnostics)
    assert "TM5007" in text
    parsed = json.loads(diagnostics_to_json(diagnostics))
    assert [item["code"] for item in parsed] == list(_codes(tuple(diagnostics)))
    for diagnostic in diagnostics:
        for evidence in diagnostic.evidence:
            assert list(_iter_tensors(evidence.value)) == []
