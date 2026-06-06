"""
Smoke tests for diagnostic outputs after running:
    python diagnostics/run_component_value_audit.py
"""

from pathlib import Path
import csv


def test_component_engagement_report_exists_if_diagnostics_ran():
    path = Path("diagnostic_reports/component_engagement.csv")
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows, "component_engagement.csv should not be empty"


def test_component_value_report_has_action_columns_if_present():
    path = Path("diagnostic_reports/component_value_report.csv")
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert "value_status" in rows[0]
    assert "recommended_action" in rows[0]
