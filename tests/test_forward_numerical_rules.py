from trainmedic.monitors.forward import ForwardTensorObservation
from trainmedic.rules.numerical_rules import diagnose_forward_observation


def _observation(*, nan_count: int, inf_count: int) -> ForwardTensorObservation:
    return ForwardTensorObservation(
        sequence_index=1,
        module_name="bad",
        module_aliases=("bad",),
        module_type="tests.Bad",
        module_call_index=1,
        tensor_path="output",
        shape=(2,),
        dtype="torch.float32",
        device="cpu",
        numel=2,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def test_nan_and_inf_from_same_observation_are_reported_in_fixed_order() -> None:
    diagnostics = diagnose_forward_observation(
        _observation(nan_count=1, inf_count=1),
        nan_already_reported=False,
        inf_already_reported=False,
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM3001", "TM3002")


def test_already_reported_numerical_issue_is_not_reported_again() -> None:
    diagnostics = diagnose_forward_observation(
        _observation(nan_count=1, inf_count=1),
        nan_already_reported=True,
        inf_already_reported=False,
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM3002",)
