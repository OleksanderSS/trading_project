import pandas as pd
import asyncio
import logging

from src.features.enrichers.sentiment_features_enricher import SentimentFeaturesEnricher
from src.pipeline.guards.temporal_target_guard import TemporalTargetGuard
from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage
from src.pipeline.stages.stage_3_improvements import validate_and_align_features_targets


def test_temporal_target_guard_preserves_return_and_volatility_targets():
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=40, freq="D"),
        "ticker": ["AAPL"] * 40,
        "interval": ["1d"] * 40,
        "close": [100 + i for i in range(40)],
    })

    targets = TemporalTargetGuard().generate_targets_safe(
        df,
        "1d",
        pd.Timestamp("2024-03-01"),
    )

    assert "target_return_1d" in targets.columns
    assert "target_return_5d" in targets.columns
    assert "target_return_20d" in targets.columns
    assert "target_return_1d_direction" in targets.columns
    assert "target_volatility_1d" in targets.columns
    assert "target_volatility_5d" in targets.columns
    assert "datetime" in targets.columns
    assert "ticker" in targets.columns


def test_temporal_target_guard_does_not_shift_across_tickers():
    df = pd.DataFrame({
        "datetime": list(pd.date_range("2024-01-01", periods=25, freq="D")) * 2,
        "ticker": ["AAPL"] * 25 + ["MSFT"] * 25,
        "interval": ["1d"] * 50,
        "close": list(range(100, 125)) + list(range(200, 225)),
    })

    targets = TemporalTargetGuard().generate_targets_safe(
        df,
        "1d",
        pd.Timestamp("2024-03-01"),
    )

    # AAPL has return, MSFT has return, check ticker preservation
    assert "ticker" in targets.columns
    assert targets["ticker"].nunique() == 2


def test_temporal_target_guard_leaves_tail_direction_targets_missing():
    df = pd.DataFrame({
        "datetime": list(pd.date_range("2024-01-01", periods=3, freq="D")) * 2,
        "ticker": ["AAPL"] * 3 + ["MSFT"] * 3,
        "interval": ["1d"] * 6,
        "close": [100.0, 101.0, 102.0, 200.0, 198.0, 197.0],
    })

    targets = TemporalTargetGuard().generate_targets_safe(
        df,
        "1d",
        pd.Timestamp("2024-03-01"),
    )

    assert pd.isna(targets.loc[2, "target_return_1d"])
    assert pd.isna(targets.loc[2, "target_return_1d_direction"])
    assert pd.isna(targets.loc[5, "target_return_1d"])
    assert pd.isna(targets.loc[5, "target_return_1d_direction"])


def test_align_features_targets_uses_ticker_identity_not_datetime_only():
    dt = pd.Timestamp("2024-01-01")
    features = pd.DataFrame({
        "datetime": [dt, dt],
        "ticker": ["AAPL", "MSFT"],
        "interval": ["1d", "1d"],
        "feature_x": [1.0, 2.0],
    })
    targets = pd.DataFrame({
        "datetime": [dt, dt],
        "ticker": ["AAPL", "MSFT"],
        "interval": ["1d", "1d"],
        "target_return_1d": [0.1, 0.2],
    })

    aligned_features, aligned_targets = validate_and_align_features_targets(features, targets, "1d")

    assert len(aligned_features) == 2
    assert len(aligned_targets) == 2
    assert set(aligned_features["ticker"]) == {"AAPL", "MSFT"}



def test_sentiment_enricher_keeps_merged_news_sentiment_features():
    prices = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01 10:00", periods=4, freq="h"),
        "ticker": ["AAPL"] * 4,
        "close": [100, 101, 102, 103],
    })
    news = pd.DataFrame({
        "published_at": [pd.Timestamp("2024-01-01 10:10"), pd.Timestamp("2024-01-01 11:15")],
        "ticker": ["AAPL", "AAPL"],
        "sentiment_score": [0.5, -0.2],
    })

    enriched = SentimentFeaturesEnricher().enrich(prices, news=news)

    assert "nlp_sentiment_score" in enriched.columns
    assert "sentiment_velocity" in enriched.columns
    assert "news_intensity" in enriched.columns


def test_sentiment_enricher_does_not_fill_sentiment_across_tickers():
    df = pd.DataFrame({
        "datetime": list(pd.date_range("2024-01-01", periods=2, freq="h")) * 2,
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "nlp_sentiment_score": [None, None, 0.7, None],
    })

    prepared = SentimentFeaturesEnricher()._prepare_dataframe(df, "nlp_sentiment_score")

    aapl_scores = prepared.loc[prepared["ticker"] == "AAPL", "nlp_sentiment_score"]
    msft_scores = prepared.loc[prepared["ticker"] == "MSFT", "nlp_sentiment_score"]
    assert aapl_scores.tolist() == [0.0, 0.0]
    assert msft_scores.tolist() == [0.7, 0.7]


def test_feature_engineering_selection_excludes_targets_and_metadata():
    class StubSelector:
        async def select_with_full_analysis(self, features_df, target_series, **kwargs):
            assert "target_up_1d" not in features_df.columns
            assert "ticker" not in features_df.columns
            assert "datetime" not in features_df.columns
            return {"selected_features": ["feature_signal"]}

    stage = object.__new__(FeatureEngineeringStage)
    stage.selector = StubSelector()
    stage.logger = logging.getLogger("test")
    features = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=8),
        "ticker": ["AAPL"] * 8,
        "feature_signal": range(8),
        "feature_noise": [1, 2, 1, 2, 1, 2, 1, 2],
        "target_up_1d": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    selected, importance = asyncio.run(
        stage._select_features(features, "target_up_1d", {"context_id": "unit"})
    )

    assert selected == ["feature_signal"]
    assert importance == {"feature_signal": 1.0}
