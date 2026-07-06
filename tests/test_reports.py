import json

from trainmedic import Diagnostic, Evidence, Severity
from trainmedic.reports.console import format_diagnostics
from trainmedic.reports.json_report import diagnostics_to_json


def test_empty_console_report() -> None:
    assert format_diagnostics(()) == "TrainMedic found no diagnostics."


def test_console_report_preserves_diagnostic_order_and_details() -> None:
    diagnostics = (
        Diagnostic(
            code="TM1001",
            severity=Severity.ERROR,
            title="Missing optimizer parameter",
            message="Parameter second.weight is trainable but missing.",
            object_name="second.weight",
            evidence=(
                Evidence("parameter_name", "second.weight"),
                Evidence("requires_grad", True),
            ),
            possible_causes=("Optimizer received only part of the model.",),
            suggestions=("Pass all trainable parameters to the optimizer.",),
        ),
        Diagnostic(
            code="TM1002",
            severity=Severity.WARNING,
            title="External optimizer parameter",
            message="The optimizer contains an external parameter.",
            object_name="optimizer.param_groups[1].params[0]",
        ),
    )

    output = format_diagnostics(diagnostics)

    assert output.index("TM1001") < output.index("TM1002")
    assert "[1] TM1001 ERROR - Missing optimizer parameter" in output
    assert "Object: second.weight" in output
    assert "Evidence:\n  - parameter_name: second.weight\n  - requires_grad: true" in output
    assert "Possible causes:\n  - Optimizer received only part of the model." in output
    assert "Suggestions:\n  - Pass all trainable parameters to the optimizer." in output


def test_json_report_is_strict_json_and_preserves_order() -> None:
    diagnostics = (
        Diagnostic(
            code="TM1001",
            severity=Severity.ERROR,
            title="First",
            message="First diagnostic.",
            evidence=(Evidence("bad_float", [float("nan"), float("inf"), float("-inf")]),),
        ),
        Diagnostic(
            code="TM1002",
            severity=Severity.WARNING,
            title="Second",
            message="Second diagnostic.",
        ),
    )

    output = diagnostics_to_json(diagnostics)
    parsed = json.loads(output)

    assert [item["code"] for item in parsed] == ["TM1001", "TM1002"]
    assert parsed[0]["severity"] == "error"
    assert parsed[0]["evidence"][0]["value"] == ["nan", "inf", "-inf"]
    assert "NaN" not in output
    assert "Infinity" not in output
