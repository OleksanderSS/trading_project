import json

import pandas as pd
import pytest

from src.features.validation.feature_leakage_guard import FeatureLeakageGuard
from src.features.utils.datetime_utils import ensure_datetime_column
from src.pipeline.hybrid.colab_manager import (
    BatchPreparationConfig,
    ColabManager,
)
from src.pipeline.hybrid.feature_processor import FeatureProcessor


def test_load_colab_results_merges_wrapped_models_metadata_instead_of_overwriting(tmp_path):
    """trained_models_metadata.json and colab_results.json both map to the
    'models_metadata' result key and both may wrap their payload in a
    top-level models_metadata key. _load_single_file must merge these,
    not let the second file silently discard the first file's entries."""
    manager = ColabManager(output_dir=tmp_path, batch_name="batch")
    (tmp_path / "trained_models_metadata.json").write_text(
        json.dumps({"models_metadata": {"model_a": {"score": 1.0}}}),
        encoding="utf-8",
    )
    (tmp_path / "colab_results.json").write_text(
        json.dumps({"models_metadata": {"model_b": {"score": 2.0}}}),
        encoding="utf-8",
    )

    results = manager.load_colab_results("batch")

    assert results["models_metadata"] == {
        "model_a": {"score": 1.0},
        "model_b": {"score": 2.0},
    }


def test_feature_processor_drops_target_derived_columns_from_features():
    df = pd.DataFrame(
        {
            "ticker": ["AMD"],
            "datetime": ["2026-05-08"],
            "feature_a": [1.0],
            "TARGET_RETURN_1P": [0.1],
            "state_TARGET_RETURN_1P": [0.2],
        }
    )

    features_df, targets_df = FeatureProcessor().split_features_and_targets(df)

    assert list(features_df.columns) == ["ticker", "datetime", "feature_a"]
    assert list(targets_df.columns) == ["TARGET_RETURN_1P", "ticker", "datetime"]


def test_colab_manager_removes_target_like_columns_from_features(tmp_path):
    manager = ColabManager(output_dir=tmp_path, batch_name="batch")
    features_df = pd.DataFrame(
        {
            "feature_a": [1.0],
            "TARGET_RETURN_1P": [0.1],
            "state_TARGET_RETURN_1P": [0.2],
        }
    )
    targets_df = pd.DataFrame({"TARGET_RETURN_1P": [0.1]})

    cleaned_features = manager._check_feature_leakage(features_df, targets_df)

    assert list(cleaned_features.columns) == ["feature_a"]


def test_feature_leakage_guard_flags_target_derived_feature_columns():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "target_up_1d": [0, 1, 0],
            "state_TARGET_RETURN_1P": [0.1, 0.2, 0.3],
        }
    )
    guard = FeatureLeakageGuard(block_on_forbidden=False, report_dir=None)

    report = guard.check(df, ticker="AMD")

    assert report.status == "blocked"
    assert "state_TARGET_RETURN_1P" in report.forbidden_cols


def test_feature_processor_preserves_timezone_as_utc_lineage():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-06-01T09:30:00-04:00"]
            )
        }
    )

    normalized = FeatureProcessor().normalize_timezone(frame)

    assert str(normalized["datetime"].dt.tz) == "UTC"
    assert normalized["datetime"].iloc[0].isoformat() == (
        "2026-06-01T13:30:00+00:00"
    )
    assert normalized.attrs["datetime_timezone"] == "UTC"


def test_naive_datetime_is_not_silently_declared_utc():
    frame = pd.DataFrame(
        {"datetime": [pd.Timestamp("2026-06-01T13:30:00")]}
    )

    normalized = ensure_datetime_column(frame)

    assert normalized["datetime"].dt.tz is None
    assert normalized.attrs["datetime_timezone_status"] == (
        "timezone_naive_unresolved"
    )


def test_missing_datetime_is_not_fabricated():
    normalized = ensure_datetime_column(
        pd.DataFrame({"feature": [1.0]}),
        raise_on_missing=False,
    )

    assert "datetime" not in normalized.columns


def test_colab_dedup_preserves_separate_timeframes():
    manager = ColabManager(output_dir="unused", batch_name="batch")
    timestamp = pd.Timestamp("2026-06-01T13:30:00Z")
    frame = pd.DataFrame(
        {
            "ticker": ["AMD", "AMD"],
            "datetime": [timestamp, timestamp],
            "interval": ["15m", "60m"],
            "feature": [1.0, 2.0],
        }
    )

    result = manager._deduplicate_df(frame)

    assert len(result) == 2
    assert set(result["interval"]) == {"15m", "60m"}


def test_colab_batch_rejects_declared_daily_intraday_cadence():
    manager = ColabManager(output_dir="unused", batch_name="batch")
    frame = pd.DataFrame(
        {
            "ticker": ["AMD"] * 8,
            "datetime": pd.date_range(
                "2026-06-01T13:30:00Z",
                periods=8,
                freq="15min",
            ),
            "interval": ["1d"] * 8,
            "feature": range(8),
        }
    )

    with pytest.raises(ValueError, match="observed 15m cadence"):
        manager._validate_batch_frame(
            frame,
            frame_name="features",
            requested_timeframes=["1d"],
        )


def test_colab_batch_persists_exact_identity_and_file_hashes(tmp_path):
    manager = ColabManager(output_dir=tmp_path, batch_name="batch")
    timestamps = pd.date_range(
        "2026-06-01T13:30:00Z",
        periods=8,
        freq="15min",
    )
    identity = {
        "ticker": ["AMD"] * 8,
        "datetime": timestamps,
        "interval": ["15m"] * 8,
    }
    features = pd.DataFrame(
        {
            **identity,
            "feature": [float(value) for value in range(8)],
        }
    )
    targets = pd.DataFrame(
        {
            **identity,
            "target_up_15m": [0, 1] * 4,
        }
    )

    manager.prepare_colab_batch(
        features,
        targets,
        BatchPreparationConfig(
            tickers=["AMD"],
            timeframes=["15m"],
            accumulate=False,
            check_feature_selection=False,
        ),
    )

    metadata = json.loads(
        (tmp_path / "batch_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    saved = pd.read_parquet(tmp_path / "features.parquet")
    assert len(saved) == 8
    assert set(saved["interval"]) == {"15m"}
    assert str(saved["datetime"].dt.tz) == "UTC"
    assert metadata["lineage"]["identity_columns"] == [
        "ticker",
        "datetime",
        "interval",
    ]
    assert metadata["lineage"]["feature_interval_counts"] == {
        "15m": 8
    }
    assert metadata["lineage"]["features_sha256"] == manager._sha256(
        tmp_path / "features.parquet"
    )
    assert metadata["lineage"]["targets_sha256"] == manager._sha256(
        tmp_path / "targets.parquet"
    )
