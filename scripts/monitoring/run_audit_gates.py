#!/usr/bin/env python3
"""
Local audit gates for repository health.

This script is intentionally lightweight: it catches blockers before deeper ML
or trading validation runs. It does not require network access.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_TARGETS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "run_hybrid_pipeline.py",
]
ARTIFACT_PATTERNS = [
    "*.pyc",
    "*/__pycache__/*",
    "catboost_info/*",
    "trained_models/*",
    "full_pipeline_trading/*",
    "reports/evaluation/*",
    "*.joblib",
    "*.pkl",
    "*.pt",
    "*.parquet",
    "*.duckdb",
]


def run_command(command: Sequence[str], label: str, required: bool = True) -> bool:
    """Run a command and report a compact pass/fail result."""
    print(f"\n== {label} ==")
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        if required:
            print(f"FAIL: command not found: {command[0]}")
            return False
        print(f"SKIP: command not found: {command[0]}")
        return True

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode == 0:
        print("PASS")
        return True

    print(f"FAIL: exit code {result.returncode}")
    return not required


def syntax_gate() -> bool:
    """Compile all project Python files without writing bytecode."""
    print("\n== Syntax compile ==")
    ok = True
    files: List[Path] = []
    for target in PYTHON_TARGETS:
        if not target.exists():
            continue
        if target.is_dir():
            files.extend(path for path in target.rglob("*.py") if "__pycache__" not in path.parts)
        else:
            files.append(target)

    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except Exception as exc:
            rel = path.relative_to(PROJECT_ROOT)
            logger.error(f"Error compiling {rel}: {exc}", exc_info=True)
            print(f"{rel}: {type(exc).__name__}: {exc}")
            ok = False

    print("PASS" if ok else "FAIL")
    return ok


def git_ls_files() -> List[str]:
    """Return tracked files when git is available."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def artifact_gate(fix: bool = False) -> bool:
    """Fail if generated artifacts are tracked by git."""
    print("\n== Tracked artifact check ==")
    tracked = git_ls_files()
    offenders = [
        path
        for path in tracked
        if any(fnmatch.fnmatch(path, pattern) for pattern in ARTIFACT_PATTERNS)
    ]
    if not offenders:
        print("PASS")
        return True

    if fix:
        result = subprocess.run(
            ["git", "rm", "--cached", "--pathspec-from-file=-"],
            cwd=PROJECT_ROOT,
            input="\n".join(offenders) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode == 0:
            print(f"PASS: removed {len(offenders)} artifact(s) from the git index")
            return True
        print(f"FAIL: could not update git index, exit code {result.returncode}")
        return False

    print("FAIL: tracked generated artifacts detected")
    for path in offenders[:50]:
        print(f"  {path}")
    if len(offenders) > 50:
        print(f"  ... and {len(offenders) - 50} more")
    return False


def optional_tool_gate(command: Sequence[str], label: str) -> bool:
    """Run a tool if it is installed; otherwise report a non-blocking skip."""
    executable = command[0]
    if executable == sys.executable and len(command) > 2 and command[1] == "-m":
        module = command[2]
        check = subprocess.run(
            [sys.executable, "-c", f"import importlib.util; raise SystemExit(importlib.util.find_spec('{module}') is None)"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode != 0:
            print(f"\n== {label} ==\nSKIP: Python module '{module}' is not installed")
            return True
    elif shutil.which(executable) is None:
        print(f"\n== {label} ==\nSKIP: command not found: {executable}")
        return True

    return run_command(command, label, required=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local project audit gates.")
    parser.add_argument("--full", action="store_true", help="Run optional static-analysis tools too.")
    parser.add_argument("--fix-artifacts", action="store_true", help="Run git rm --cached for tracked artifacts.")
    args = parser.parse_args()

    checks = [
        syntax_gate(),
        artifact_gate(fix=args.fix_artifacts),
        run_command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], "pytest", required=True),
    ]

    if args.full:
        checks.extend(
            [
                optional_tool_gate([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"], "ruff"),
                optional_tool_gate([sys.executable, "-m", "mypy", "src"], "mypy"),
                optional_tool_gate(
                    [sys.executable, "-m", "bandit", "-q", "-r", "src", "-x", "*/__pycache__/*"],
                    "bandit",
                ),
            ]
        )

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
