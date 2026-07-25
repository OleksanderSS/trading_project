"""
Static contract tests for high-risk trading/ML correctness issues.

Place under tests/contracts/ and run:
  pytest tests/contracts/test_static_trading_ml_contracts.py

These tests are intentionally strict. If they fail, either fix the code or add a
very explicit allowlist with a documented reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"


def read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8", errors="ignore")


def test_target_calculators_use_groupby_ticker_for_future_shift() -> None:
    files = [
        "src/targets/calculators/regression_calculator.py",
        "src/targets/calculators/classification_calculator.py",
    ]
    for rel in files:
        text = read(rel)
        assert ".shift(shift)" not in text or "groupby" in text, (
            f"{rel} appears to use future shift without ticker grouping. "
            "Future targets must be calculated per ticker."
        )


def test_feature_enrichers_do_not_emit_target_columns() -> None:
    offenders: list[str] = []
    for path in (SRC / "features").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "target_" in text and "audit-ignore: TARGET_IN_FEATURE_MODULE" not in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, "Feature modules should not emit target_* columns: " + ", ".join(offenders[:20])


def test_model_factory_import_does_not_top_level_import_neural_models() -> None:
    rel = "src/factories/model_factory.py"
    text = read(rel)
    forbidden = [
        "from src.models.neural.lstm_model import",
        "from src.models.neural.gru_model import",
        "from src.models.neural.cnn_model import",
        "from src.models.neural.transformer_model import",
        "from src.models.neural.tabnet_model import",
        "from src.models.neural.autoencoder_model import",
    ]
    hits = [item for item in forbidden if item in text]
    assert not hits, f"{rel} top-level imports heavy neural models: {hits}. Use lazy imports/registry."


def test_calibration_synthetic_not_primary_score_by_default() -> None:
    # calibration_engine.py was retired to src/archive/ (2026-07-22, confirmed
    # zero live callers) -- this checks a safety property of code that will
    # never execute in production, so there's nothing left to enforce here.
    pytest.skip("src/calibration/calibration_engine.py is archived, retired code -- not live.")
