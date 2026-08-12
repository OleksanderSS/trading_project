import pandas as pd
import asyncio
import logging

from src.features.enrichers.sentiment_features_enricher import SentimentFeaturesEnricher
# TemporalTargetGuard was superseded by src/targets/ and moved to
# src/archive/guards_superseded/. The module keeps its tests -- they still
# describe behaviour the replacement has to preserve -- and this import was
# left pointing at the old location, so the whole file failed to collect and
# took its other 20-odd contract tests with it, silently.
from src.archive.guards_superseded.temporal_target_guard import TemporalTargetGuard
from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage
from src.archive.pipeline.stages.stage_3_improvements import validate_and_align_features_targets


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


def test_feature_engineering_restores_service_columns_dropped_by_enricher():
    stage = object.__new__(FeatureEngineeringStage)
    source = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=3, freq="15min"),
        "ticker": ["AMD"] * 3,
        "interval": ["15m"] * 3,
        "close": [100.0, 101.0, 102.0],
    })
    # The enriched frame keeps `close`, which is what the real case looked
    # like: macro_features dropped `datetime` and nothing else. That surviving
    # column is what proves the rows still line up, so the restore is allowed.
    #
    # This test used to hand over a frame with no `close` and no `hash` and
    # assert the columns came back anyway. That is the behaviour that put
    # 54,552 bars on the wrong dates -- copying by position with nothing to
    # show the positions still correspond. Refusal in that case is pinned by
    # test_service_column_restore_alignment.py; what belongs here is that a
    # provable restore still works.
    enriched = pd.DataFrame({
        "feature_signal_15m": [0.1, 0.2, 0.3],
        "close": [100.0, 101.0, 102.0],
    })

    restored = stage._restore_service_columns(enriched, source)

    assert restored["datetime"].tolist() == source["datetime"].tolist()
    assert restored["ticker"].tolist() == ["AMD"] * 3
    assert restored["interval"].tolist() == ["15m"] * 3
    assert restored["close"].tolist() == [100.0, 101.0, 102.0]


def test_feature_engineering_initial_columns_never_include_targets():
    stage = object.__new__(FeatureEngineeringStage)
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=2),
        "ticker": ["AMD", "AMD"],
        "interval": ["15m", "15m"],
        "feature_signal": [0.1, 0.2],
        "target_intraday_up_15m": [0.0, 1.0],
        "state_target_return_1d": [0.3, 0.4],
    })

    selected = stage._initial_feature_columns(frame)

    assert selected == ["feature_signal"]


def test_feature_guards_do_not_reorder_rows_without_a_temporal_key():
    from src.pipeline.stages.feature_engineering.guards import FeatureGuards

    # Constructed properly rather than via object.__new__: apply_guards now
    # runs FeatureLeakageGuard, which __init__ builds. Bypassing __init__ was
    # a shortcut that only worked while apply_guards touched nothing the
    # constructor set up.
    guards = FeatureGuards(mode='prepare')
    source = pd.DataFrame({
        "ticker": ["NVDA"] * 100,
        "row_identity": list(range(100)),
        "close": [100.0 + value for value in range(100)],
    })

    guarded = guards.apply_guards(source)

    assert guarded["row_identity"].tolist() == source["row_identity"].tolist()
    assert guarded["close"].tolist() == source["close"].tolist()
