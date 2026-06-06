"""
Pipeline stage checker.

Purpose:
- find likely pipeline/stage/orchestrator files
- extract stage-like classes/functions
- detect stage files that are not referenced elsewhere
- detect entrypoints that may import heavy modules too early

Run:
    python diagnostics/pipeline_stage_checker.py --root src --out diagnostic_reports
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StageRecord:
    file: str
    stage_symbols: str
    imported_by_count: int
    has_execute_or_run: bool
    status: str
    recommendation: str


def iter_py(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def extract_symbols(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], False, text
    names = []
    has_execute = False
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and ("Stage" in n.name or "Pipeline" in n.name or "Orchestrator" in n.name):
            names.append(n.name)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in {"run", "execute", "process", "fit", "transform"}:
            has_execute = True
    return names, has_execute, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src")
    ap.add_argument("--out", default="diagnostic_reports")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = iter_py(root)
    all_text = {}
    module_names = {}
    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        all_text[rel] = p.read_text(encoding="utf-8", errors="ignore")
        module_names[rel] = ".".join(("src", *Path(rel).with_suffix("").parts))

    records = []
    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        if not any(token in rel.lower() for token in ["pipeline", "stage", "orchestrator", "workflow"]):
            continue

        symbols, has_execute, text = extract_symbols(p)
        mod = module_names[rel]
        import_refs = 0
        for other_rel, other_text in all_text.items():
            if other_rel == rel:
                continue
            if mod in other_text or Path(rel).stem in other_text:
                import_refs += 1

        if import_refs > 0 and has_execute:
            status = "LIKELY_REACHABLE"
            rec = "Add smoke/runtime test for this stage."
        elif import_refs == 0:
            status = "POSSIBLY_ORPHAN_STAGE"
            rec = "Check if this stage is loaded by config/dynamic import before deleting."
        else:
            status = "NEEDS_REVIEW"
            rec = "Stage-like file found, but run/execute/process contract is unclear."

        records.append(StageRecord(rel, ";".join(symbols), import_refs, has_execute, status, rec))

    with (out / "pipeline_stage_report.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(records[0]).keys()) if records else ["file", "stage_symbols", "imported_by_count", "has_execute_or_run", "status", "recommendation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    summary = {
        "stage_like_files": len(records),
        "possibly_orphan_stage_files": sum(1 for r in records if r.status == "POSSIBLY_ORPHAN_STAGE"),
        "likely_reachable_stage_files": sum(1 for r in records if r.status == "LIKELY_REACHABLE"),
    }
    (out / "pipeline_stage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
