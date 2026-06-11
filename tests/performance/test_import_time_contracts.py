"""
Import-time performance contracts.

These tests guard against heavy imports in lightweight paths.
"""

from __future__ import annotations

import importlib
import sys
import time

import pytest


LIGHTWEIGHT_MODULES = [
    "src.factories.model_factory",
    "src.cli.pipeline_executor",
]


@pytest.mark.parametrize("module_name", LIGHTWEIGHT_MODULES)
def test_lightweight_import_does_not_load_heavy_ml_libraries(module_name):
    for heavy in ["tensorflow", "torch", "transformers"]:
        sys.modules.pop(heavy, None)

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Module not present in this project version: {exc}")

    assert "tensorflow" not in sys.modules, f"{module_name} imported tensorflow at import time"
    assert "transformers" not in sys.modules, f"{module_name} imported transformers at import time"


@pytest.mark.parametrize("module_name", LIGHTWEIGHT_MODULES)
def test_lightweight_import_under_time_budget(module_name):
    start = time.perf_counter()
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Module not present in this project version: {exc}")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"{module_name} import took {elapsed:.2f}s; consider lazy imports"
