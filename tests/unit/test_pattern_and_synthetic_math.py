import numpy as np
import pandas as pd

from src.data.synthetic.data_generator import DataGenerator
from src.features.enrichers.volume_enricher import VolumeEnricher
from src.scripts.optimization.portfolio.optimizer import PortfolioOptimizer
from src.archive.patterns.pattern_analyzer import PatternAnalyzer


def test_fractal_similarity_ignores_initial_pct_change_nan():
    analyzer = object.__new__(PatternAnalyzer)
    prices = pd.DataFrame(
        {"close": [100, 101, 102, 101, 103, 104, 105, 107, 106, 108, 109, 111, 112, 113, 115]},
        index=pd.date_range("2024-01-01", periods=15, freq="h"),
    )

    result = analyzer._find_fractal_similarity(prices)

    assert result
    assert np.isfinite(result["similarity_score"])
    assert np.isfinite(result["historical_outcome"])


def test_volume_enricher_short_series_marks_unavailable_history_as_missing():
    df = pd.DataFrame(
        {
            "close": [10.0, 11.0, 10.5],
            "volume": [100.0, 0.0, 150.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    enriched = VolumeEnricher()._enrich_impl(df)

    stable_cols = ["volume_sma_5", "volume_sma_10", "obv", "volume_rs"]
    assert enriched.index.equals(df.index)
    assert enriched["volume_roc"].isna().all()
    assert pd.isna(enriched["price_volume_trend"].iloc[0])
    assert enriched[stable_cols].notna().all().all()
    assert np.isfinite(enriched[stable_cols].to_numpy()).all()


def test_synthetic_targets_use_distinct_future_horizons():
    generator = DataGenerator(config_manager=None)
    index = pd.date_range("2024-01-01", periods=40, freq="h")
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)
    features = pd.DataFrame(
        {
            "close": close,
            "sma_20": close.rolling(2, min_periods=1).mean(),
            "sma_50": close.rolling(3, min_periods=1).mean(),
            "macd": np.linspace(-1.0, 1.0, len(index)),
            "macd_signal": 0.0,
            "rsi": 55.0,
            "volatility": 0.01,
        },
        index=index,
    )
    generator.generate_synthetic_features = lambda: features

    targets = generator.generate_synthetic_targets()
    first_idx = targets.index[0]

    assert np.isclose(targets.loc[first_idx, "return_1h"], close.shift(-1).loc[first_idx] / close.loc[first_idx] - 1)
    assert np.isclose(targets.loc[first_idx, "return_4h"], close.shift(-4).loc[first_idx] / close.loc[first_idx] - 1)
    assert targets["return_1h"].ne(targets["return_4h"]).any()
    assert targets[["volatility_1h", "volatility_4h"]].notna().all().all()


def test_synthetic_generator_does_not_mix_future_targets_into_features():
    """
    shift(-N) is expected for target construction, but targets must never leak into features.
    """
    generator = DataGenerator(config_manager=None)
    index = pd.date_range("2024-01-01", periods=40, freq="h")
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)
    features = pd.DataFrame(
        {
            "close": close,
            "sma_20": close.rolling(2, min_periods=1).mean(),
            "sma_50": close.rolling(3, min_periods=1).mean(),
            "macd": np.linspace(-1.0, 1.0, len(index)),
            "macd_signal": 0.0,
            "rsi": 55.0,
            "volatility": 0.01,
        },
        index=index,
    )
    generator.generate_synthetic_features = lambda: features

    targets = generator.generate_synthetic_targets()

    assert targets.columns.str.startswith("target_").sum() == 0
    assert not any(c in features.columns for c in targets.columns)


def test_portfolio_returns_do_not_forward_fill_missing_prices():
    optimizer = object.__new__(PortfolioOptimizer)
    optimizer.logger = type("Logger", (), {"info": lambda *args, **kwargs: None, "error": lambda *args, **kwargs: None})()
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0],
            "BBB": [50.0, None, 55.0],
        }
    )

    returns = optimizer.calculate_returns(prices)

    assert returns.empty
