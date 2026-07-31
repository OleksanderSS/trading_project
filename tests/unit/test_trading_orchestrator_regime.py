"""The regime detector must actually be called, and reach the kill switch.

`TradingOrchestrator.execute_trading_cycle` used to read:

    regime = 'ranging'
    if self.regime_detector:
        self.logger.info('Detecting market regime for decision optimization...')

so the detector was injected, the log line was printed, and the detector was
never invoked -- every cycle ran as 'ranging' while the logs suggested
detection was happening.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trading.trading_orchestrator import TradingOrchestrator


class _Detector:
    def __init__(self, regime: str):
        self.regime = regime
        self.calls = 0

    def detect_regime(self, returns, data_bundle=None):
        self.calls += 1
        return {"regime": self.regime, "confidence": 0.9}


class _PortfolioManager:
    def __init__(self):
        self.current_regime = None


def _orchestrator(detector=None, portfolio_manager=None):
    orch = object.__new__(TradingOrchestrator)
    orch.regime_detector = detector
    orch.portfolio_manager = portfolio_manager
    orch.logger = __import__("logging").getLogger("test")
    return orch


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, n))})


def test_detector_is_actually_invoked():
    detector = _Detector("TRENDING_UP")
    orch = _orchestrator(detector)

    orch._detect_regime(_frame())

    assert detector.calls == 1, "the detector was injected but never called"


def test_detected_regime_reaches_the_portfolio_manager():
    pm = _PortfolioManager()
    orch = _orchestrator(_Detector("CRISIS"), pm)

    regime = orch._detect_regime(_frame())

    assert regime == "volatile"
    assert pm.current_regime == "volatile"


@pytest.mark.parametrize(
    "detected,expected",
    [
        ("TRENDING_UP", "trending_up"),
        ("trend_down", "trending_down"),
        ("CRISIS", "volatile"),
        ("HIGH_VOLATILITY", "volatile"),
        ("DEAD", "dead"),
        ("NORMAL", "ranging"),
        ("something_unheard_of", "ranging"),
    ],
)
def test_detector_vocabulary_maps_onto_market_regime_members(detected, expected):
    """AdaptiveParameterManager knows trending_up/trending_down/ranging/
    volatile/dead. Unknown names must land on 'ranging', never on something
    that would widen a limit."""
    assert TradingOrchestrator._map_detected_regime(detected.lower()) == expected


def test_no_detector_means_the_previous_default():
    pm = _PortfolioManager()
    orch = _orchestrator(None, pm)

    assert orch._detect_regime(_frame()) == "ranging"
    assert pm.current_regime == "ranging"


def test_too_few_returns_falls_back_without_calling_the_detector():
    detector = _Detector("CRISIS")
    orch = _orchestrator(detector)

    assert orch._detect_regime(_frame(n=10)) == "ranging"
    assert detector.calls == 0


def test_a_failing_detector_does_not_break_the_cycle():
    class Exploding:
        def detect_regime(self, *_args, **_kwargs):
            raise ValueError("boom")

    pm = _PortfolioManager()
    orch = _orchestrator(Exploding(), pm)

    assert orch._detect_regime(_frame()) == "ranging"
    assert pm.current_regime == "ranging"


def test_missing_close_column_is_handled():
    orch = _orchestrator(_Detector("CRISIS"), _PortfolioManager())
    assert orch._detect_regime(pd.DataFrame({"open": [1.0, 2.0]})) == "ranging"
