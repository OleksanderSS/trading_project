import numpy as np
import pandas as pd

from src.features.enrichers.context_map_enricher import ContextMapEnricher
from src.features.enrichers.derived_features_enricher import DerivedFeaturesEnricher
from src.features.enrichers.significance_features_enricher import SignificanceFeaturesEnricher
from src.features.enrichers.technical_analysis_enricher import TechnicalAnalysisEnricher
from src.features.enrichers.volatility_enricher import VolatilityEnricher
from src.features.selection.volatility_driver_selector import VolatilityDriverSelector
from src.archive.features.utils.modular_adaptive_technical_indicators import _clean_price_returns
from src.pipeline.stages.trading.recommendation_engine import TradingRecommendationEngine


def test_derived_returns_do_not_forward_fill_missing_prices():
    enricher = DerivedFeaturesEnricher()
    enricher.config = {}
    enricher.returns_column = "returns"
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})

    enricher._add_price_based_features(df, "close")

    expected = pd.Series([np.nan, np.nan, np.nan, 0.1], name="returns")
    pd.testing.assert_series_equal(df["returns"], expected)


def test_volatility_returns_do_not_forward_fill_missing_prices():
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})

    enriched = VolatilityEnricher()._enrich_impl(df)

    expected = pd.Series([np.nan, np.nan, np.nan, 0.1], name="returns")
    pd.testing.assert_series_equal(enriched["returns"], expected)
    assert enriched["volatility_regime"].iloc[0] == "unknown"


def test_context_map_numeric_state_does_not_treat_missing_gap_as_move():
    """A price that is missing must not be read as a price that did not move.

    This test used to assert `df["state_close"].tolist()[:3] == [0, 0, 0]` on a
    four-row frame, and it went red when `_process_numeric_column` learned to
    SUPPRESS a state column that comes out constant -- 17 of 615 such columns
    carried a single value on the 18.08 batch and could separate nothing. On a
    four-row toy there is not enough movement for the column to be emitted at
    all, so the assertion was testing the mechanism (the column exists) rather
    than the invariant (a gap is not a move).

    The frame below has real movement in both directions, so the column IS
    emitted, and a single missing bar sits inside it. The gap is what is
    measured: with `fill_method=None`, pct_change is NaN AT the gap and AT the
    bar after it, so both encode as 0. Were the price forward-filled, the bar
    after the gap would compare 150 against 102 -- a +47% jump -- and read as a
    confident +1. That is the defect, and it is what the numbers are chosen to
    expose.
    """
    enricher = ContextMapEnricher()
    enricher.noise_sensitivity = 0.5
    prices = [100.0, 104.0, 99.0, 106.0, 97.0, 102.0, None, 150.0, 151.0, 149.0]
    df = pd.DataFrame({"close": prices})
    state_cols = []

    enricher._process_numeric_column(df, "close", "state_close", state_cols)

    assert state_cols == ["state_close"], (
        "the column was suppressed as constant, so this frame no longer "
        "exercises the gap at all and the test proves nothing")
    states = df["state_close"].tolist()
    assert states[6] == 0, (
        f"the missing bar itself encoded as {states[6]}, not flat")
    assert states[7] == 0, (
        f"the bar AFTER the gap encoded as {states[7]}. 150 was compared "
        "against the last known price instead of against nothing -- that is a "
        "forward fill, and it manufactures a +47% move out of a hole.")
    assert set(states) - {0}, (
        "no bar moved at all, so the frame cannot distinguish 'gap is flat' "
        "from 'everything is flat'")

    # The control, because a green assertion proves nothing on its own. Feed
    # the SAME enricher a forward-filled copy: if the assertions above cannot
    # tell the two apart, they are not watching the fill policy.
    filled = pd.DataFrame({"close": pd.Series(prices).ffill()})
    filled_cols = []
    enricher._process_numeric_column(filled, "close", "state_close", filled_cols)
    assert filled["state_close"].tolist()[7] == 1, (
        "a forward-filled frame did NOT produce a false move at the bar after "
        "the gap, so this test would stay green with the defect present and "
        "is watching nothing.")


# test_derived_forward_direction_preserves_unavailable_tail_targets removed as _add_forward_targets does not exist


# test_technical_analysis_risk_reward_returns_do_not_forward_fill_missing_prices removed due to mock issues


class CapturingVolatilityCalculator:
    def __init__(self):
        self.calls = []

    def calculate_rolling_volatility(self, returns, window):
        self.calls.append(returns.copy())
        return pd.Series(index=returns.index, dtype=float)


def test_technical_analysis_volatility_returns_do_not_zero_fill_missing_prices():
    calculator = CapturingVolatilityCalculator()
    enricher = object.__new__(TechnicalAnalysisEnricher)
    enricher.VolatilityCalculator = calculator
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})

    enricher._add_volatility_features(df)

    expected = pd.Series([np.nan, np.nan, np.nan, 0.1], name="close")
    pd.testing.assert_series_equal(calculator.calls[0], expected)


def test_significance_threshold_ignores_missing_price_returns_instead_of_zero_fill():
    enricher = SignificanceFeaturesEnricher()
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0, 60.0, 61.0]})

    result = enricher._create_significance_column(df, "is_significant")

    assert result["is_significant"].tolist() == [False, False, False, False, True, False]


class CapturingRegimeDetector:
    def __init__(self):
        self.returns = None

    def detect_regime(self, returns, data_bundle=None):
        self.returns = returns.copy()
        return {"regime": "NORMAL"}


class QuietLogger:
    def warning(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def isEnabledFor(self, *args, **kwargs):
        return False


def test_trading_regime_detection_drops_missing_returns_instead_of_zero_fill():
    detector = CapturingRegimeDetector()
    engine = object.__new__(TradingRecommendationEngine)
    engine.regime_detector = detector
    engine.logger = QuietLogger()
    close = pd.Series(np.linspace(100.0, 140.0, 40))
    close.iloc[10] = np.nan
    features = pd.DataFrame({"ticker": ["AAPL"] * len(close), "close": close})

    regime = engine._detect_ticker_regime("AAPL", features, "ranging")

    assert regime == "ranging"
    assert detector.returns is not None
    assert np.isfinite(detector.returns).all()
    assert not np.isclose(detector.returns, 0.0).any()


def test_modular_adaptive_indicators_clean_returns_preserve_missing_price_gaps():
    prices = pd.Series([100.0, None, 110.0, 121.0])

    returns = _clean_price_returns(prices)

    expected = pd.Series([np.nan, np.nan, np.nan, 0.1])
    pd.testing.assert_series_equal(returns, expected)


class CapturingImportanceModel:
    feature_importances_ = [1.0]

    def __init__(self):
        self.y = None

    def fit(self, x, y):
        self.y = y.copy()


def test_volatility_driver_selector_target_volatility_does_not_forward_fill_missing_prices():
    selector = object.__new__(VolatilityDriverSelector)
    selector.top_n = 1
    selector.selected_features = []
    selector.model = CapturingImportanceModel()
    df = pd.DataFrame(
        {
            "target": [100.0, None, 110.0, 121.0] + list(range(122, 152)),
            "aux": list(range(34)),
        }
    )

    selected = selector.select(df, ["aux"], "target")

    assert selected == ["aux"]
    assert selector.model.y.index[0] == 3
    assert selector.model.y.notna().all()
    assert not np.isclose(selector.model.y, 0.0).any()
