from trainmedic import Severity
from trainmedic.rules.update_rules import (
    NO_PARAMETERS_SELECTED_FOR_UPDATE_MONITORING,
    OPTIMIZER_STEP_DID_NOT_COMPLETE,
    OPTIMIZER_STEP_NOT_OBSERVED,
    PARAMETER_UPDATE_NOT_DETECTED,
    UPDATE_CHECK_SKIPPED_FOR_UNKNOWN_LEARNING_RATE,
    ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES,
    no_parameters_selected_diagnostic,
    parameter_update_not_detected_diagnostic,
    step_did_not_complete_diagnostic,
    step_not_observed_diagnostic,
    unknown_learning_rate_diagnostic,
    zero_learning_rate_diagnostic,
)


def test_update_rule_constants_are_stable() -> None:
    assert NO_PARAMETERS_SELECTED_FOR_UPDATE_MONITORING == "TM4000"
    assert OPTIMIZER_STEP_NOT_OBSERVED == "TM4001"
    assert ZERO_LEARNING_RATE_FOR_UPDATE_CANDIDATES == "TM4002"
    assert PARAMETER_UPDATE_NOT_DETECTED == "TM4003"
    assert OPTIMIZER_STEP_DID_NOT_COMPLETE == "TM4004"
    assert UPDATE_CHECK_SKIPPED_FOR_UNKNOWN_LEARNING_RATE == "TM4005"


def test_update_rule_severities() -> None:
    diagnostics = (
        no_parameters_selected_diagnostic(
            model_parameter_count=0,
            trainable_parameter_count=0,
            optimizer_model_parameter_count=0,
            selected_parameter_count=0,
            optimizer_group_count=1,
        ),
        step_not_observed_diagnostic(
            selected_parameter_count=1,
            optimizer_group_count=1,
        ),
        zero_learning_rate_diagnostic(
            step_index=1,
            affected_parameter_count=1,
            affected_parameter_names_preview=("weight",),
            omitted_parameter_count=0,
            optimizer_group_indices=(0,),
            learning_rate_values=(0.0,),
            selection_count=1,
        ),
        parameter_update_not_detected_diagnostic(
            step_index=1,
            candidate_parameter_count=1,
            changed_candidate_count=0,
            unchanged_candidate_count=1,
            exact_unchanged_count=1,
            sampled_unchanged_count=0,
            unsupported_or_skipped_count=0,
            unchanged_parameter_names_preview=("weight",),
            omitted_parameter_count=0,
            per_parameter_preview=(
                {
                    "name": "weight",
                    "aliases": ["weight"],
                    "group_index": 0,
                    "learning_rate": 0.1,
                    "coverage": "exact",
                    "parameter_numel": 2,
                    "sampled_element_count": 2,
                    "gradient_norm": 1.0,
                },
            ),
            configured_sample_size=64,
            configured_max_snapshot_elements=100_000,
        ),
        step_did_not_complete_diagnostic(
            attempted_step_count=1,
            successful_step_count=0,
            incomplete_step_count=1,
            last_attempted_step_index=1,
            pending_snapshot_present=True,
        ),
        unknown_learning_rate_diagnostic(
            step_index=1,
            affected_parameter_count=1,
            affected_parameter_names_preview=("weight",),
            omitted_parameter_count=0,
            optimizer_group_indices=(0,),
            learning_rate_representations=("bool:True",),
            selected_parameter_count=1,
        ),
    )

    assert tuple(diagnostic.severity for diagnostic in diagnostics) == (
        Severity.INFO,
        Severity.WARNING,
        Severity.WARNING,
        Severity.WARNING,
        Severity.WARNING,
        Severity.INFO,
    )
