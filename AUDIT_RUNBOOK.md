# Audit Runbook

This repo contains lightweight static-audit helpers intended to be run locally and kept out of the main pipeline.

## 1) Deep Logic Audit

- Script: `audit_logic.py`
- Purpose: AST-based scan for higher-risk correctness issues (leakage patterns, pandas footguns, math hazards, async hazards, resource leaks).
- Examples:
  - `python audit_logic.py --root src`
  - `python audit_logic.py --root src --json --output audit_logic_report.json`
  - `python audit_logic.py --root src --category LEAK,PANDAS,MATH,ASYNC,RES,FEAT`
  - `python audit_logic.py --root src --severity HIGH --max-issues 200`

## 2) Engagement & Coverage Audit

- Script: `audit_engagement.py`
- Purpose: scan for monitoring/explainability/engagement/test/doc presence signals.
- Examples:
  - `python audit_engagement.py --root src`
  - `python audit_engagement.py --root src --json --output audit_engagement_report.json`
  - `python audit_engagement.py --root src --category MON,TEST,DOC --max-issues 200`

## 3) Pipeline Operational Coverage Audit

- Script: `audit_tool_coverage.py`
- Purpose: instantiate the pipeline/analytics engine and print what "tool-like" components appear attached to stages and which analyzers are active.
- Run:
  - `python audit_tool_coverage.py`

Notes:
- This script imports pipeline modules and may take longer to start than the pure AST scanners.
- It is best treated as an "integration visibility" report, not a correctness proof.

