from trainmedic.monitors.gradients import GradientObservation
from trainmedic.rules.gradient_rules import diagnose_gradient_observation


def _observation(*, nan_count: int, inf_count: int) -> GradientObservation:
    return GradientObservation(
        sequence_index=1,
        parameter_name="weight",
        parameter_aliases=("weight",),
        parameter_shape=(2,),
        parameter_dtype="torch.float32",
        parameter_device="cpu",
        parameter_numel=2,
        hook_call_index=1,
        gradient_shape=(2,),
        gradient_dtype="torch.float32",
        gradient_device="cpu",
        gradient_layout="torch.strided",
        gradient_numel=2,
        gradient_nnz=None,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def test_nan_and_inf_from_same_gradient_are_reported_in_fixed_order() -> None:
    diagnostics = diagnose_gradient_observation(
        _observation(nan_count=1, inf_count=1),
        nan_already_reported=False,
        inf_already_reported=False,
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM2002", "TM2003")


def test_already_reported_gradient_issue_is_not_reported_again() -> None:
    diagnostics = diagnose_gradient_observation(
        _observation(nan_count=1, inf_count=1),
        nan_already_reported=True,
        inf_already_reported=False,
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("TM2003",)
