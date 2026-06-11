"""
Dead/useful code classifier.

Inputs:
- module_inventory.csv from module_diagnostic.py
- risk_findings.csv from module_diagnostic.py

Output:
- dead_code_classification.csv
- dead_code_classification.md

Run:
    python diagnostics/dead_code_classifier.py --reports diagnostic_reports
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classify(row):
    imported_by = int(row.get("imported_by_count") or 0)
    risk_count = int(row.get("risk_count") or 0)
    category = row.get("category_guess", "")
    path = row.get("path", "")

    if imported_by > 0 and risk_count > 0:
        return "ACTIVE_BUGGY_OR_RISKY", "Used by source and has risk findings. Review/fix first."
    if imported_by > 0 and risk_count == 0:
        return "ACTIVE_UNTESTED_OR_OK", "Used by source. Check tests/runtime before declaring OK."
    if imported_by == 0 and category in {"factory", "validator", "analyzer", "detector", "selector", "risk", "target_calculator"}:
        return "UNUSED_USEFUL_CANDIDATE", "Orphan statically, but category may contain reusable production logic."
    if imported_by == 0 and ("legacy" in path.lower() or "old" in path.lower() or "backup" in path.lower()):
        return "UNUSED_LEGACY", "Likely legacy. Quarantine before deletion."
    if imported_by == 0 and risk_count > 0:
        return "UNUSED_RISKY_QUARANTINE", "Not imported and risky. Do not integrate without tests."
    if imported_by == 0:
        return "UNUSED_DELETE_CANDIDATE", "No static imports. Confirm config/runtime usage before deletion."
    return "UNKNOWN_NEEDS_REVIEW", "Manual review needed."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="diagnostic_reports")
    args = ap.parse_args()

    reports = Path(args.reports)
    inventory = read_csv(reports / "module_inventory.csv")
    out_rows = []

    for row in inventory:
        status, rec = classify(row)
        out_rows.append({
            "path": row.get("path", ""),
            "category_guess": row.get("category_guess", ""),
            "imported_by_count": row.get("imported_by_count", ""),
            "risk_count": row.get("risk_count", ""),
            "status": status,
            "recommendation": rec,
        })

    out_csv = reports / "dead_code_classification.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(out_rows[0].keys()) if out_rows else ["path", "category_guess", "imported_by_count", "risk_count", "status", "recommendation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = {}
    for row in out_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    md = ["# Dead Code Classification", ""]
    for status, count in sorted(counts.items(), key=lambda x: x[0]):
        md.append(f"- **{status}**: {count}")
    md.append("")
    md.append("Do not delete modules based only on static classification. Check config and runtime reports first.")
    (reports / "dead_code_classification.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
