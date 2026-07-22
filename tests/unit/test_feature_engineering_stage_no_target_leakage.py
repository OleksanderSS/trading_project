import pytest
import pandas as pd

from src.pipeline.target_column_utils import is_target_like_column


class _SelectorStub:
    def __init__(self):
        self.last_x = None
        self.last_y = None

    async def select_with_full_analysis(self, x, y, **kwargs):
        self.last_x = x
        self.last_y = y
        return {"selected_features": []}


@pytest.mark.asyncio
async def test_feature_selection_never_sees_target_columns():
    # Avoid heavy stage init; we only care about the selection pre-processing.
    from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage

    stage = object.__new__(FeatureEngineeringStage)
    stage.selector = _SelectorStub()
    stage.logger = type("L", (), {"warning": lambda *args, **kwargs: None})()

    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "target_up_1d": [0, 1, 0, 1, 0],
            "target_down_1d": [1, 0, 1, 0, 1],
            "TARGET_RETURN_1P": [0.01, 0.02, -0.01, 0.03, -0.02],
            "state_TARGET_RETURN_1P": [0.01, 0.02, -0.01, 0.03, -0.02],
        }
    )

    selected, importance = await FeatureEngineeringStage._select_features(
        stage,
        final_features=df,
        target_col="target_up_1d",
        kwargs={},
    )

    # We don't assert on selection outcome; we assert on data-contract to the selector.
    x = stage.selector.last_x
    y = stage.selector.last_y
    assert x is not None and y is not None
    assert "target_up_1d" not in x.columns
    assert not any(is_target_like_column(col) for col in x.columns)
    assert "timestamp" not in x.columns
    assert set(selected).issubset(set(df.columns))
    assert isinstance(importance, dict)

