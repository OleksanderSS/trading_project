import numpy as np
import pandas as pd

from src.pipeline.stages.stage_0_data_generation import DataGenerator


def _features(periods=40):
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)
    return pd.DataFrame(
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


def test_stage0_synthetic_targets_drop_rows_without_future_labels():
    generator = DataGenerator(config_manager=None)
    features = _features()

    targets = generator.generate_synthetic_targets(features)

    assert targets.index.max() < features.index[-24]
    assert targets[
        ["return_1h", "return_4h", "return_24h", "volatility_1h", "volatility_4h"]
    ].notna().all().all()
    assert not targets[["return_1h", "return_4h", "return_24h"]].tail().eq(0).all().any()


def test_stage0_generated_features_and_targets_are_aligned():
    generator = DataGenerator(config_manager=None)
    features = _features()
    generator.generate_synthetic_features = lambda: features

    result = generator.generate_synthetic_data()

    assert result["features_df"].index.equals(result["targets_df"].index)
    assert result["data_points"] == len(result["targets_df"])
