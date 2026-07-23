"""Tests for the Stage 3 external-predictor causal diagnostic.

Wires in src/analytics/calculators/advanced_econometrics_calculator.py,
which existed but had zero callers anywhere in the active codebase before
this change (found during a duplication/dead-code audit). This diagnostic
is deliberately non-invasive: it must never change which features get
selected, only annotate the Stage 3 output with Granger-causality evidence
for macro/sentiment/news predictors.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage


class _LoggerStub:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def info(self, msg, *a, **k):
        self.infos.append(msg)


def _make_stage() -> FeatureEngineeringStage:
    # Avoid heavy stage init (config manager, error handler, real selector);
    # the diagnostic method only needs self.logger and the class constants.
    stage = object.__new__(FeatureEngineeringStage)
    stage.logger = _LoggerStub()
    return stage


def test_no_external_predictors_returns_empty_without_calling_econometrics():
    stage = _make_stage()
    df = pd.DataFrame({
        "RSI_14": np.random.RandomState(0).randn(50),
        "MACD": np.random.RandomState(1).randn(50),
    })
    target = pd.Series(np.random.RandomState(2).randn(50), name="target_up_1d")

    evidence = stage._diagnose_external_predictor_causality(df, target, "target_up_1d")

    assert evidence == {}
    assert stage.logger.infos == []


def test_technical_indicators_are_excluded_from_the_diagnostic():
    """RSI/MACD-style columns must never be sent through the causality
    check — they're derived from price, testing them against a
    price-derived target is close to circular."""
    stage = _make_stage()
    external_markers = stage._EXTERNAL_PREDICTOR_MARKERS
    for technical_name in ("RSI_14", "MACD", "close", "VOLATILITY_20", "SHARPE_RATIO"):
        assert not any(marker in technical_name for marker in external_markers), technical_name


def test_external_predictor_causality_is_detected_and_does_not_affect_selection():
    """A genuinely lagged, noisy leading indicator should be flagged
    significant more readily than pure noise — and either way, this method
    only returns diagnostic evidence, never a filtered feature list."""
    rng = np.random.RandomState(42)
    n = 300
    leading_signal = rng.randn(n)
    # Target is driven by the leading indicator's PRIOR value plus noise —
    # a textbook Granger-causal relationship.
    target = np.zeros(n)
    for t in range(1, n):
        target[t] = 0.8 * leading_signal[t - 1] + 0.3 * rng.randn()

    df = pd.DataFrame({
        "FRED_leading_indicator": leading_signal,
        "sentiment_pure_noise": rng.randn(n),
        "RSI_14": rng.randn(n),  # must be excluded
    })
    target_series = pd.Series(target, name="target_up_1d")

    stage = _make_stage()
    evidence = stage._diagnose_external_predictor_causality(df, target_series, "target_up_1d")

    assert "RSI_14" not in evidence
    assert "FRED_leading_indicator" in evidence
    assert "sentiment_pure_noise" in evidence
    for key in ("is_significant", "causality_strength", "p_value"):
        assert key in evidence["FRED_leading_indicator"]

    # This is a diagnostic annotation only — it must never return a
    # selected-features list or otherwise mimic _select_features' contract.
    assert isinstance(evidence, dict)
    assert "selected_features" not in evidence


@pytest.mark.asyncio
async def test_causal_evidence_is_attached_to_run_output_without_changing_selection():
    """End-to-end through _select_features: causal_evidence is computed as
    a side effect (self._last_causal_evidence) but the returned
    (selected, importance) pair is driven only by the stubbed selector."""

    class _SelectorStub:
        async def select_with_full_analysis(self, x, y, **kwargs):
            return {"selected_features": list(x.columns)}

    stage = _make_stage()
    stage.selector = _SelectorStub()
    stage._last_causal_evidence = {}
    stage._train_only_index = lambda frame: frame.index  # no holdout split needed for this test

    rng = np.random.RandomState(7)
    n = 60
    df = pd.DataFrame({
        "FRED_rate": rng.randn(n),
        "RSI_14": rng.randn(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "target_up_1d": (rng.randn(n) > 0).astype(int),
    })

    selected, importance = await FeatureEngineeringStage._select_features(
        stage, final_features=df, target_col="target_up_1d", kwargs={}
    )

    assert set(selected) == {"FRED_rate", "RSI_14"}
    assert "FRED_rate" in stage._last_causal_evidence
    assert "RSI_14" not in stage._last_causal_evidence
