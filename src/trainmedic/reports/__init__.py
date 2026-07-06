"""Diagnostic report formatters."""

from trainmedic.reports.console import format_diagnostics
from trainmedic.reports.json_report import diagnostics_to_json

__all__ = ["diagnostics_to_json", "format_diagnostics"]
