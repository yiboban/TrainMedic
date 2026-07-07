from trainmedic import Severity
from trainmedic.monitors.modes import ModeObservation
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


def _observation(*, expected: str = "train", actual: str = "eval") -> ModeObservation:
    return ModeObservation(
        sequence_index=1,
        module_name="module",
        module_aliases=("module",),
        module_type="torch.nn.modules.linear.Linear",
        module_call_index=1,
        is_root=False,
        expected_mode=expected,
        actual_mode=actual,
        grad_enabled=True,
        inference_mode_enabled=False,
    )


def test_mode_rule_constants_are_stable() -> None:
    assert FORWARD_NOT_OBSERVED_DURING_MODE_MONITORING == "TM5000"
    assert MODEL_IN_EVAL_DURING_TRAINING == "TM5001"
    assert MODEL_IN_TRAIN_DURING_EVALUATION == "TM5002"
    assert CALLED_SUBMODULE_MODE_MISMATCH == "TM5003"
    assert DROPOUT_MODE_MISMATCH == "TM5004"
    assert BATCHNORM_MODE_MISMATCH == "TM5005"
    assert GRADIENT_TRACKING_ENABLED_DURING_EVALUATION == "TM5006"
    assert GRADIENT_TRACKING_DISABLED_DURING_TRAINING == "TM5007"


def test_mode_rule_severities() -> None:
    diagnostics = (
        forward_not_observed_diagnostic(expected_mode="train"),
        model_in_eval_during_training_diagnostic(_observation()),
        model_in_train_during_evaluation_diagnostic(_observation(expected="eval", actual="train")),
        called_submodule_mode_mismatch_diagnostic(_observation()),
        dropout_mode_mismatch_diagnostic(_observation(), p=0.5, inplace=False),
        batchnorm_mode_mismatch_diagnostic(
            _observation(),
            num_features=2,
            affine=True,
            track_running_stats=True,
            momentum=0.1,
        ),
        eval_grad_enabled_diagnostic(_observation(expected="eval", actual="eval")),
        train_grad_disabled_diagnostic(_observation()),
    )

    assert tuple(diagnostic.severity for diagnostic in diagnostics) == (
        Severity.INFO,
        Severity.WARNING,
        Severity.WARNING,
        Severity.INFO,
        Severity.WARNING,
        Severity.WARNING,
        Severity.INFO,
        Severity.WARNING,
    )
