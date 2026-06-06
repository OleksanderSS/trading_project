"""
Run component engagement/value audit.

Run:
    python diagnostics/run_component_value_audit.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


COMMANDS = [
    [sys.executable, "diagnostics/component_engagement_audit.py", "--root", "src", "--configs", "configs", ".", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/component_harness_runner.py", "--root", "src", "--out", "diagnostic_reports"],
    [sys.executable, "diagnostics/component_value_report.py", "--reports", "diagnostic_reports"],
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
            print(f"WARNING: command failed: {' '.join(cmd)}")
    print("\nDone. Review diagnostic_reports/component_value_report.md and CSV files.")


if __name__ == "__main__":
    main()
