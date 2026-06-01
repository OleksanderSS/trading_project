import numpy as np
import pandas as pd

from src.features.enrichers.context_map_enricher import ContextMapEnricher
from src.features.enrichers.derived_features_enricher import DerivedFeaturesEnricher
from src.features.enrichers.technical_analysis_enricher import TechnicalAnalysisEnricher
from src.features.enrichers.volatility_enricher import VolatilityEnricher
from src.features.selection.volatility_driver_selector import VolatilityDriverSelector


def test_derived_returns_do_not_forward_fill_missing_prices():
    enricher = object.__new__(DerivedFeaturesEnricher)
    enricher.config = {}
    enricher.returns_column = "returns"
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})

    enricher._add_price_based_features(df, "close")

    assert np.allclose(df["returns"], [0.0, 0.0, 0.0, 0.1])


def test_volatility_returns_do_not_forward_fill_missing_prices():
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})

    enriched = VolatilityEnricher()._enrich_impl(df)

    assert np.allclose(enriched["returns"], [0.0, 0.0, 0.0, 0.1])


def test_context_map_numeric_state_does_not_treat_missing_gap_as_move():
    enricher = object.__new__(ContextMapEnricher)
    enricher.noise_sensitivity = 0.5
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})
    state_cols = []

    enricher._process_numeric_column(df, "close", "state_close", state_cols)

    assert df["state_close"].tolist()[:3] == [0, 0, 0]
    assert state_cols == ["state_close"]


class CapturingRiskRewardCalculator:
    def __init__(self):
        self.sharpe_returns = None

    def calculate_sharpe_ratio(self, returns):
        self.sharpe_returns = returns.copy()
        return 0.0

    def calculate_sortino_ratio(self, returns):
        return 0.0


def test_technical_analysis_risk_reward_returns_do_not_forward_fill_missing_prices():
    calculator = CapturingRiskRewardCalculator()
    enricher = object.__new__(TechnicalAnalysisEnricher)
    enricher.RiskRewardCalculator = calculator
    df = pd.DataFrame({"close": [100.0, None, 110.0, 121.0]})

    enricher._add_risk_reward_features(df)

    expected = pd.Series([np.nan, np.nan, np.nan, 0.1], name="close")
    pd.testing.assert_series_equal(calculator.sharpe_returns, expected)


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
    assert selector.model.y.iloc[:3].tolist() == [0.0, 0.0, 0.0]
