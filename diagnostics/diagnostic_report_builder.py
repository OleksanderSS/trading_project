"""
Build a single markdown report from diagnostic outputs.

Run:
    python diagnostics/diagnostic_report_builder.py --reports diagnostic_reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path, default=""):
    return path.read_text(encoding="utf-8") if path.exists() else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="diagnostic_reports")
    args = ap.parse_args()
    r = Path(args.reports)
    r.mkdir(parents=True, exist_ok=True)

    parts = ["# Full Diagnostic Report", ""]

    for name in [
        "diagnostic_summary.md",
        "domain_rule_findings.md",
        "registry_consistency_report.md",
        "config_reachability_summary.md",
        "dead_code_classification.md",
    ]:
        p = r / name
        if p.exists():
            parts.append(f"\n---\n\n")
            parts.append(read(p))

    parts.append("\n---\n\n## Files to review\n")
    for filename in [
        "risk_findings.csv",
        "domain_rule_findings.csv",
        "module_inventory.csv",
        "orphan_modules.txt",
        "registry_consistency_report.json",
        "config_reachability.csv",
        "dead_code_classification.csv",
        "runtime_usage_report.json",
    ]:
        if (r / filename).exists():
            parts.append(f"- `{filename}`")

    (r / "FULL_DIAGNOSTIC_REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print(r / "FULL_DIAGNOSTIC_REPORT.md")


if __name__ == "__main__":
    main()
