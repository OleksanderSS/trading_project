#!/usr/bin/env python3
"""Run the full offline audit workflow in one command.

This is intentionally dependency-light and wraps:
  1. deep_static_audit.py -> findings.json/csv/md/summary.json
  2. triage_findings.py -> compact P0/P1 report
  3. audit_gate.py -> baseline comparison / CI gate

Examples:
  python audit/engine/full_audit_workflow.py --root src --mode scan
  python audit/engine/full_audit_workflow.py --root src --mode baseline
  python audit/engine/full_audit_workflow.py --root src --mode check --fail-on P0,P1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src", help="Source root to scan, usually src or .")
    ap.add_argument("--out", default="audit_reports", help="Audit report directory")
    ap.add_argument("--tools", default="audit/engine", help="Audit tools directory")
    ap.add_argument("--mode", choices=["scan", "baseline", "check", "all"], default="check")
    ap.add_argument("--baseline", default="audit/engine/audit_baseline.json")
    ap.add_argument("--suppressions", default="audit/engine/audit_suppressions.yaml")
    ap.add_argument("--fail-on", default="P0,P1")
    ap.add_argument("--triage-severity", default="P0,P1")
    ap.add_argument("--max-per-rule", default="8")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    tools = Path(args.tools)
    out.mkdir(parents=True, exist_ok=True)

    findings = out / "findings.json"
    triage = out / "triage_P0_P1.md"
    gate_report = out / "gate_report.md"

    python = sys.executable

    if args.mode in {"scan", "baseline", "check", "all"}:
        run([python, str(tools / "deep_static_audit.py"), "--root", str(root), "--out", str(out), "--format", "all"])
        run([
            python, str(tools / "triage_findings.py"),
            "--findings", str(findings),
            "--out", str(triage),
            "--severity", args.triage_severity,
            "--max-per-rule", args.max_per_rule,
        ])

    if args.mode in {"baseline", "all"}:
        run([python, str(tools / "audit_gate.py"), "baseline", "--findings", str(findings), "--baseline", args.baseline])

    if args.mode in {"check", "all"}:
        run([
            python, str(tools / "audit_gate.py"), "check",
            "--findings", str(findings),
            "--baseline", args.baseline,
            "--suppressions", args.suppressions,
            "--fail-on", args.fail_on,
            "--out", str(gate_report),
        ])

    print(f"\nReports written under: {out}")
    print(f"Triage: {triage}")
    if args.mode in {"check", "all"}:
        print(f"Gate report: {gate_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
