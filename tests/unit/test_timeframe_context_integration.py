import pandas as pd

from src.pipeline.stages.feature_engineering.timeframe_context import (
    BackwardTimeframeContextAssembler,
)


def test_hourly_context_is_unavailable_until_the_bar_is_complete():
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 10:30Z",
                    "2025-01-02 10:45Z",
                    "2025-01-02 11:00Z",
                ]
            ),
            "ticker": ["A"] * 3,
            "interval": ["15m"] * 3,
            "feature_15m": [1.0, 2.0, 3.0],
            "target_intraday_up_15m": [0.0, 1.0, 1.0],
        }
    )
    base["datetime"] = base["datetime"].astype("datetime64[us, UTC]")
    hourly = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2025-01-02 10:00Z", "2025-01-02 11:00Z"]
            ),
            "ticker": ["A", "A"],
            "interval": ["60m", "60m"],
            "hourly_signal": [10.0, 20.0],
            "target_hourly_up_1h": [1.0, 0.0],
        }
    )

    combined, report = BackwardTimeframeContextAssembler().assemble(
        {"15m": base, "60m": hourly}
    )
    intraday = combined.loc[combined["interval"].eq("15m")].reset_index(drop=True)

    assert pd.isna(intraday.loc[0, "ctx_60m_hourly_signal"])
    assert intraday.loc[1, "ctx_60m_hourly_signal"] == 10.0
    assert intraday.loc[2, "ctx_60m_hourly_signal"] == 10.0
    assert "ctx_60m_target_hourly_up_1h" not in combined.columns
    assert report["summary"]["future_context_violations"] == 0
    assert report["summary"]["row_identity_preserved"] is True
    assert report["summary"]["output_rows"] == len(base) + len(hourly)


def test_daily_context_is_not_visible_to_same_day_intraday_rows():
    intraday = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2025-01-02 15:00Z", "2025-01-03 15:00Z"]
            ),
            "ticker": ["A", "A"],
            "interval": ["15m", "15m"],
            "signal": [1.0, 2.0],
        }
    )
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 00:00Z"]),
            "ticker": ["A"],
            "interval": ["1d"],
            "daily_regime": [7.0],
        }
    )

    combined, _ = BackwardTimeframeContextAssembler().assemble(
        {"15m": intraday, "1d": daily}
    )
    intraday_result = combined.loc[
        combined["interval"].eq("15m")
    ].reset_index(drop=True)

    assert pd.isna(intraday_result.loc[0, "ctx_1d_daily_regime"])
    assert intraday_result.loc[1, "ctx_1d_daily_regime"] == 7.0


def test_context_join_does_not_cross_partition_boundaries():
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2025-01-02 11:00Z", "2025-01-02 11:00Z"]
            ),
            "ticker": ["A", "A"],
            "interval": ["15m", "15m"],
            "partition_id": ["development", "evaluation"],
            "signal": [1.0, 2.0],
        }
    )
    hourly = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 10:00Z"]),
            "ticker": ["A"],
            "interval": ["60m"],
            "partition_id": ["development"],
            "hourly_signal": [9.0],
        }
    )

    combined, _ = BackwardTimeframeContextAssembler().assemble(
        {"15m": base, "60m": hourly}
    )
    result = combined.loc[combined["interval"].eq("15m")].reset_index(drop=True)

    assert result.loc[0, "ctx_60m_hourly_signal"] == 9.0
    assert pd.isna(result.loc[1, "ctx_60m_hourly_signal"])


def test_context_join_rejects_mismatched_partition_metadata():
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 11:00Z"]),
            "ticker": ["A"],
            "interval": ["15m"],
            "partition_id": ["development"],
            "signal": [1.0],
        }
    )
    hourly = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 10:00Z"]),
            "ticker": ["A"],
            "interval": ["60m"],
            "hourly_signal": [9.0],
        }
    )

    try:
        BackwardTimeframeContextAssembler().assemble(
            {"15m": base, "60m": hourly}
        )
    except ValueError as exc:
        assert "same partition metadata" in str(exc)
    else:
        raise AssertionError("Mismatched partition metadata must block context join.")


def test_stage4_isolates_ticker_and_timeframe_contexts():
    from src.pipeline.modeling_context import iter_model_contexts

    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B"],
            "interval": ["15m", "15m", "60m", "15m"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    contexts = list(iter_model_contexts(frame))
    identities = {(ticker, timeframe): len(group) for ticker, timeframe, group in contexts}

    assert identities == {("A", "15m"): 2, ("A", "60m"): 1, ("B", "15m"): 1}


def test_stage3_schema_preserves_context_lineage_for_stage4():
    from src.validation.pipeline_schemas import EnrichedDataSchema

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 11:00Z"]),
            "ticker": ["A"],
            "interval": ["15m"],
            "feature_signal": [1.0],
            "target_intraday_up_15m": [1.0],
        }
    )
    report = {
        "status": "causal_timeframe_context_ready",
        "join_direction": "backward",
        "summary": {"future_context_violations": 0},
    }

    schema = EnrichedDataSchema(
        enriched_prices={"15m": frame},
        selected_features=["feature_signal"],
        feature_importance={},
        all_targets={"15m": frame[["target_intraday_up_15m"]]},
        combined_features=frame,
        enriched_data=frame,
        timeframe_context_report=report,
    )
    schema.validate()

    assert schema.model_dump()["timeframe_context_report"] == report


def test_model_preparation_excludes_every_target_like_column_and_missing_labels():
    from src.models.adapters.data_preparation import prepare_data_for_models

    rows = 50
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="15min")[::-1],
            "ticker": ["A"] * rows,
            "interval": ["15m"] * rows,
            "feature_signal": list(range(rows)),
            "all_missing_feature": [float("nan")] * rows,
            "target_intraday_up_15m": [0.0, 1.0] * (rows // 2),
            "target_hourly_up_1h": [1.0, 0.0] * (rows // 2),
        }
    )
    frame.loc[[0, 1], "target_intraday_up_15m"] = float("nan")

    prepared = prepare_data_for_models(
        frame,
        ticker="A",
        timeframe="15m",
        target_cols=["target_intraday_up_15m"],
        gap_size=2,
    )

    assert prepared is not None
    feature_names = prepared["light_models"]["feature_names"]
    assert feature_names == ["feature_signal"]
    assert prepared["metadata"]["samples"] == rows - 2


def test_target_filter_respects_semantic_timeframe_contract():
    from src.targets.timeframe_contract import target_applies_to_timeframe

    config = {
        "target_intraday_up_15m": {
            "type": "classification_binary",
            "params": {"base_col": "close", "horizon": "15m", "shift": -1},
        },
        "target_hourly_up_1h": {
            "type": "classification_binary",
            "params": {"base_col": "close", "horizon": "1h", "shift": -4},
        },
        "target_return_1d": {
            "type": "regression",
            "params": {"base_col": "close", "shift": -1},
        },
        "target_rsi_14_f1": {
            "type": "indicator_prediction",
            "params": {"indicator_col": "RSI_14", "shift": -1},
        },
    }

    names = {
        name
        for name, target in config.items()
        if target_applies_to_timeframe({"name": name, **target}, "60m")
    }

    assert names == {"target_hourly_up_1h", "target_rsi_14_f1"}


def test_indicator_target_with_daily_source_is_not_generated_on_intraday():
    from src.targets.timeframe_contract import target_applies_to_timeframe

    target = {
        "name": "target_rsi_14_f1",
        "type": "indicator_prediction",
        "params": {
            "indicator_col": "RSI_14_1d",
            "source_timeframe": "1d",
            "shift": -1,
        },
    }

    assert target_applies_to_timeframe(target, "15m") is False
    assert target_applies_to_timeframe(target, "60m") is False
    assert target_applies_to_timeframe(target, "1d") is True


def test_indicator_source_timeframe_is_inferred_from_column_suffix():
    from src.targets.timeframe_contract import target_applies_to_timeframe

    target = {
        "name": "target_rsi_14_f1",
        "type": "indicator_prediction",
        "params": {
            "indicator_col": "RSI_14_1d",
            "shift": -1,
        },
    }

    assert target_applies_to_timeframe(target, "15m") is False
    assert target_applies_to_timeframe(target, "1d") is True
