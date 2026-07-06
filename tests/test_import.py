import json
from dataclasses import FrozenInstanceError
from enum import Enum

import pytest

import trainmedic
from trainmedic import Diagnostic, Evidence, Severity


class NotJsonSerializable:
    def __str__(self) -> str:
        return "not-json-serializable"


class NumericEnum(Enum):
    VALUE = 7


def test_import_trainmedic() -> None:
    assert trainmedic.Diagnostic is Diagnostic


def test_version() -> None:
    assert trainmedic.__version__ == "0.1.0.dev0"


def test_severity_values() -> None:
    assert Severity.INFO.value == "info"
    assert Severity.WARNING.value == "warning"
    assert Severity.ERROR.value == "error"
    assert Severity.CRITICAL.value == "critical"


def test_evidence_serializes_to_json_compatible_dict() -> None:
    evidence = Evidence(name="shape", value=(2, 3), description="Tensor shape")

    assert evidence.to_dict() == {
        "name": "shape",
        "value": [2, 3],
        "description": "Tensor shape",
    }
    json.dumps(evidence.to_dict())


def test_diagnostic_serializes_to_stable_json_compatible_dict() -> None:
    diagnostic = Diagnostic(
        code="TM0001",
        severity=Severity.INFO,
        title="TrainMedic initialized",
        message="The diagnostic system is available.",
        object_name="model",
        evidence=(Evidence(name="enabled", value=True),),
        possible_causes=("No issue; this is an initialization check.",),
        suggestions=("Continue to Phase 1.",),
    )

    assert diagnostic.to_dict() == {
        "code": "TM0001",
        "severity": "info",
        "title": "TrainMedic initialized",
        "message": "The diagnostic system is available.",
        "object_name": "model",
        "evidence": [
            {
                "name": "enabled",
                "value": True,
                "description": None,
            }
        ],
        "possible_causes": ["No issue; this is an initialization check."],
        "suggestions": ["Continue to Phase 1."],
    }
    json.dumps(diagnostic.to_dict())


def test_enum_values_inside_evidence_are_serialized_as_strings() -> None:
    evidence = Evidence(name="severity", value=Severity.ERROR)

    assert evidence.to_dict()["value"] == "error"


def test_mapping_and_non_string_enum_values_are_serialized_safely() -> None:
    evidence = Evidence(
        name="nested",
        value={
            "shape": (2, 3),
            "enum": NumericEnum.VALUE,
        },
    )

    assert evidence.to_dict()["value"] == {
        "shape": [2, 3],
        "enum": "7",
    }


def test_float_values_are_serialized_safely() -> None:
    evidence = Evidence(name="stats", value=[1.5, float("inf")])

    assert evidence.to_dict()["value"] == [1.5, "inf"]
    json.dumps(evidence.to_dict())


def test_non_json_serializable_evidence_value_is_converted_to_string() -> None:
    evidence = Evidence(name="object", value=NotJsonSerializable())

    assert evidence.to_dict()["value"] == "not-json-serializable"
    json.dumps(evidence.to_dict())


def test_frozen_evidence_cannot_be_modified() -> None:
    evidence = Evidence(name="requires_grad", value=True)

    with pytest.raises(FrozenInstanceError):
        evidence.name = "changed"


def test_frozen_diagnostic_cannot_be_modified() -> None:
    diagnostic = Diagnostic(
        code="TM0001",
        severity=Severity.INFO,
        title="TrainMedic initialized",
        message="The diagnostic system is available.",
    )

    with pytest.raises(FrozenInstanceError):
        diagnostic.code = "TM9999"
