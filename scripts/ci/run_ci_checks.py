"""CI validation: config + YAML tests.

Usage:
    python scripts/ci/run_ci_checks.py

Returns exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
SCAFFOLD = PROJECT / "dean_domain_scaffold.py"


def _run_pytest(*args: str) -> bool:
    import os
    cmd = [
        sys.executable, "-m", "pytest",
        *args,
        "-v", "-p", "no:cacheprovider",
    ]
    env = {**dict(os.environ.items()), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main() -> int:
    all_ok = True

    print("=" * 60)
    print("  Step 1: validate-config")
    print("=" * 60)
    r1 = subprocess.run(
        [sys.executable, str(SCAFFOLD), "validate-config"],
        capture_output=True, text=True, timeout=120,
    )
    print(r1.stdout)
    if r1.returncode != 0:
        print(r1.stderr)
        all_ok = False

    print("=" * 60)
    print("  Step 2: YAML config tests")
    print("=" * 60)
    if not _run_pytest(f"{PROJECT}/tests/dean_os/test_config_yamls.py"):
        all_ok = False

    print("=" * 60)
    print("  Step 3: Agent unit tests")
    print("=" * 60)
    if not _run_pytest(
        f"{PROJECT}/tests/dean_os/test_freshness_audit_agent.py",
        f"{PROJECT}/tests/dean_os/test_coherence_scan_agent.py",
        f"{PROJECT}/tests/dean_os/test_pipeline_readiness_agent.py",
    ):
        all_ok = False

    print("=" * 60)
    print("  Step 4: CLI smoke tests")
    print("=" * 60)
    if not _run_pytest(f"{PROJECT}/tests/dean_os/test_cli_smoke.py"):
        all_ok = False

    if all_ok:
        print("\n  All CI checks passed.")
        return 0
    else:
        print("\n  Some CI checks FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
