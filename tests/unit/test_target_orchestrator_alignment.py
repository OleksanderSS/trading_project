import pandas as pd

from src.pipeline.stages.feature_engineering.targets import TargetGenerator
from src.targets.target_orchestrator import TargetOrchestrator


TARGETS_CONFIG = {
    "target_return_1d": {
        "type": "regression",
        "params": {
            "base_col": "close",
            "shift": -1,
        },
    },
}


def test_target_orchestrator_preserves_original_ticker_row_alignment():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "ticker": ["B", "A", "B", "A"],
            "interval": ["1d"] * 4,
            "close": [100.0, 200.0, 110.0, 300.0],
        }
    )

    targets = TargetOrchestrator(TARGETS_CONFIG).generate_targets(df)

    assert targets.loc[0, "ticker"] == "B"
    assert targets.loc[1, "ticker"] == "A"
    assert targets.loc[0, "target_return_1d"] == 0.1
    assert targets.loc[1, "target_return_1d"] == 0.5
    assert pd.isna(targets.loc[2, "target_return_1d"])
    assert pd.isna(targets.loc[3, "target_return_1d"])


def test_target_generator_appends_targets_without_dropping_features():
    class StubConfig:
        def get(self, key, default=None):
            if key == "targets":
                return TARGETS_CONFIG
            return default

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=2),
            "ticker": ["A", "A"],
            "interval": ["1d", "1d"],
            "close": [100.0, 110.0],
            "feature_x": [1.0, 2.0],
        }
    )

    enriched = TargetGenerator(StubConfig()).append_targets(df)

    assert "feature_x" in enriched.columns
    assert "target_return_1d" in enriched.columns
    assert enriched.loc[0, "target_return_1d"] == 0.1


def test_hourly_target_resolves_to_four_15m_bars_and_one_60m_bar():
    config = {
        "target_hourly_up_1h": {
            "type": "classification_binary",
            "params": {
                "base_col": "close",
                "horizon": "1h",
                "shift": -4,
                "threshold": 0.0,
            },
        },
    }
    fifteen_minute = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-02 14:30",
                periods=6,
                freq="15min",
                tz="UTC",
            ),
            "ticker": ["A"] * 6,
            "interval": ["15m"] * 6,
            "close": [100.0, 99.0, 98.0, 97.0, 101.0, 102.0],
        }
    )
    hourly = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-02 14:30",
                periods=3,
                freq="60min",
                tz="UTC",
            ),
            "ticker": ["A"] * 3,
            "interval": ["60m"] * 3,
            "close": [100.0, 101.0, 99.0],
        }
    )

    target_15m = TargetOrchestrator(config).generate_targets(fifteen_minute)
    target_60m = TargetOrchestrator(config).generate_targets(hourly)

    assert target_15m.loc[0, "target_hourly_up_1h"] == 1.0
    assert pd.isna(target_15m.loc[2, "target_hourly_up_1h"])
    assert target_60m.loc[0, "target_hourly_up_1h"] == 1.0
    assert target_60m.loc[1, "target_hourly_up_1h"] == 0.0


def test_targets_are_blank_across_gaps_and_partition_boundaries():
    config = {
        "target_intraday_up_15m": {
            "type": "classification_binary",
            "params": {
                "base_col": "close",
                "horizon": "15m",
                "shift": -1,
                "threshold": 0.0,
            },
        },
    }
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 14:30Z",
                    "2025-01-02 14:45Z",
                    "2025-01-02 16:00Z",
                    "2025-01-02 16:15Z",
                ]
            ),
            "ticker": ["A"] * 4,
            "interval": ["15m"] * 4,
            "partition_id": ["development", "development", "evaluation", "evaluation"],
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    targets = TargetOrchestrator(config).generate_targets(frame)

    assert targets.loc[0, "target_intraday_up_15m"] == 1.0
    assert pd.isna(targets.loc[1, "target_intraday_up_15m"])
    assert targets.loc[2, "target_intraday_up_15m"] == 1.0
    assert pd.isna(targets.loc[3, "target_intraday_up_15m"])


def test_same_ticker_targets_never_cross_interval_groups():
    config = {
        "target_hourly_up_1h": {
            "type": "classification_binary",
            "params": {
                "base_col": "close",
                "horizon": "1h",
                "shift": -4,
                "threshold": 0.0,
            },
        },
    }
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 14:30Z",
                    "2025-01-02 15:30Z",
                    "2025-01-02 14:30Z",
                    "2025-01-02 14:45Z",
                ]
            ),
            "ticker": ["A"] * 4,
            "interval": ["60m", "60m", "15m", "15m"],
            "close": [100.0, 110.0, 1000.0, 900.0],
        }
    )

    targets = TargetOrchestrator(config).generate_targets(frame)

    assert targets.loc[0, "target_hourly_up_1h"] == 1.0
    assert pd.isna(targets.loc[1, "target_hourly_up_1h"])
    assert pd.isna(targets.loc[2, "target_hourly_up_1h"])
    assert pd.isna(targets.loc[3, "target_hourly_up_1h"])
