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
