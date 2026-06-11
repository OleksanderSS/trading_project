"""
Run all available diagnostics in sequence.

Run:
    python diagnostics/run_all_diagnostics.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


COMMANDS = [
    [sys.executable, "diagnostics/module_diagnostic.py", "--root", "src", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/domain_rule_scanner.py", "--root", "src", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/registry_consistency_checker.py", "--root", "src", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/config_reachability_checker.py", "--root", "src", "--configs", "configs", ".", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/pipeline_stage_checker.py", "--root", "src", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/dead_code_classifier.py", "--reports", "diagnostic_reports"],
    [sys.executable, "diagnostics/diagnostic_report_builder.py", "--reports", "diagnostic_reports"],
]


def main():
    for cmd in COMMANDS:
        script = Path(cmd[1])
        if not script.exists():
            print(f"SKIP missing {script}")
            continue
        print("\n>", " ".join(cmd))
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"WARNING: command failed with exit code {result.returncode}: {' '.join(cmd)}")
    print("\nDiagnostics finished. Check diagnostic_reports/.")


if __name__ == "__main__":
    main()
