"""
Build component value report.

Merges available outputs:
- component_engagement.csv
- component_harness_results.csv
- feature_lineage_report.json
- component_ablation_results.csv

Output:
- component_value_report.csv
- component_value_report.md

Run:
    python diagnostics/component_value_report.py --reports diagnostic_reports
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index_by(rows: list[dict], key: str) -> dict[str, dict]:
    return {r.get(key, ""): r for r in rows if r.get(key)}


def value_status(eng: dict, harness: dict | None, lineage_component: dict | None, ablation: dict | None) -> tuple[str, str]:
    eng_status = eng.get("status", "")
    risk_count = int(eng.get("risk_count") or 0)

    if risk_count > 0:
        return "NEEDS_FIX_BEFORE_VALUE_TEST", "Risk findings exist; fix correctness/leakage before judging value."

    if harness and harness.get("status") == "EXECUTED":
        if "TARGET_COLUMN_ADDED" in harness.get("warnings", ""):
            return "EXECUTED_BUT_LEAKAGE_RISK", "Harness executed and detected target column output."
        if harness.get("added_columns"):
            if lineage_component:
                reached = lineage_component.get("reached_model_input", [])
                if reached:
                    return "OUTPUT_REACHES_MODEL", "Component output reaches model input; run ablation to estimate value."
                return "OUTPUT_DROPPED_OR_NOT_MARKED", "Component output exists but did not reach model input or lineage not complete."
            return "EXECUTED_OUTPUT_UNKNOWN_LINEAGE", "Component adds output; integrate feature lineage tracker."
        return "EXECUTED_NO_OUTPUT_CHANGE", "Component executed but did not change dataframe output."

    if ablation and ablation.get("status") == "DONE":
        try:
            delta = float(ablation.get("delta_with_vs_baseline"))
            if delta > 0:
                return "ACTIVE_VALUE_POSITIVE", "Ablation suggests positive impact."
            if delta < 0:
                return "ACTIVE_VALUE_NEGATIVE", "Ablation suggests negative impact."
            return "ACTIVE_VALUE_NEUTRAL", "Ablation suggests neutral impact."
        except Exception:
            pass

    if eng_status == "UNUSED_POTENTIALLY_VALUABLE":
        return "UNUSED_POTENTIALLY_VALUABLE", "Not engaged statically but category suggests possible useful logic."

    if eng_status.startswith("ACTIVE"):
        return "ACTIVE_VALUE_UNKNOWN", "Engaged, but runtime lineage/ablation not yet available."

    return "LOW_PRIORITY_ORPHAN", "No evidence of engagement/value yet."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="diagnostic_reports")
    args = ap.parse_args()

    reports = Path(args.reports)
    engagement = read_csv(reports / "component_engagement.csv")
    harness = index_by(read_csv(reports / "component_harness_results.csv"), "component")
    ablations = index_by(read_csv(reports / "component_ablation_results.csv"), "component")

    lineage_by_component = {}
    lineage_path = reports / "feature_lineage_report.json"
    if lineage_path.exists():
        data = json.loads(lineage_path.read_text(encoding="utf-8"))
        for comp in data.get("components", []):
            lineage_by_component[comp.get("component", "")] = comp

    out_rows = []
    for eng in engagement:
        comp = eng.get("component", "")
        # Harness component naming uses full module.class, so exact match often works.
        h = harness.get(comp)
        l = lineage_by_component.get(comp) or lineage_by_component.get(eng.get("class_or_function", ""))
        a = ablations.get(comp)
        status, action = value_status(eng, h, l, a)

        out_rows.append({
            "component": comp,
            "class_or_function": eng.get("class_or_function", ""),
            "category": eng.get("category", ""),
            "file": eng.get("file", ""),
            "engagement_status": eng.get("status", ""),
            "value_status": status,
            "risk_rules": eng.get("risk_rules", ""),
            "added_columns_static": eng.get("added_columns_static", ""),
            "imported_by_count": eng.get("imported_by_count", ""),
            "referenced_by_count": eng.get("referenced_by_count", ""),
            "has_test_reference": eng.get("has_test_reference", ""),
            "harness_status": h.get("status", "") if h else "",
            "harness_added_columns": h.get("added_columns", "") if h else "",
            "lineage_reached_model": ";".join(l.get("reached_model_input", [])) if l else "",
            "ablation_delta": a.get("delta_with_vs_baseline", "") if a else "",
            "recommended_action": action,
        })

    out_csv = reports / "component_value_report.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(out_rows[0].keys()) if out_rows else ["component", "category", "file", "value_status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = {}
    for r in out_rows:
        counts[r["value_status"]] = counts.get(r["value_status"], 0) + 1

    md = ["# Component Value Report", "", "## Value status counts"]
    for k, v in sorted(counts.items(), key=lambda x: x[0]):
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("Review `component_value_report.csv` for component-level actions.")
    (reports / "component_value_report.md").write_text("\n".join(md), encoding="utf-8")

    print(out_csv)


if __name__ == "__main__":
    main()
