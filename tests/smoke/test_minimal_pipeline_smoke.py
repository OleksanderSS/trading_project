"""
Minimal pipeline smoke tests.

This file is intentionally conservative because project entrypoints differ.
Extend PROJECT_ENTRYPOINTS after confirming your CLI/API.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ENTRYPOINTS = [
    "src.cli.pipeline_executor",
    "src.main.system_orchestrator",
]


@pytest.mark.parametrize("module_name", PROJECT_ENTRYPOINTS)
def test_entrypoint_module_imports(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Entrypoint not present in this project version: {exc}")


def test_run_hybrid_pipeline_help_if_available():
    script = Path("run_hybrid_pipeline.py")
    if not script.exists():
        pytest.skip("run_hybrid_pipeline.py not available in provided context")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr[-1000:]
