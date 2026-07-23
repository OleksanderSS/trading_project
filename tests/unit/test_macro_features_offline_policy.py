from __future__ import annotations

import pandas as pd

from src.features.enrichers.macro_features_enricher import MacroFeaturesEnricher


def test_macro_enricher_offline_policy_uses_only_provided_frame(monkeypatch):
    enricher = object.__new__(MacroFeaturesEnricher)

    def fail_cache():
        raise AssertionError("offline provided-only mode must not read the shared cache")

    monkeypatch.setattr(enricher, "_load_full_macro_from_cache", fail_cache)
    macro = pd.DataFrame(
        {
            "datetime": ["2025-01-01", "2025-01-02"],
            "series_id": ["DGS10", "DGS10"],
            "value": [4.1, 4.2],
        }
    )

    result = enricher._prepare_macro_data(
        pd.DataFrame(index=pd.date_range("2025-01-02", periods=2, freq="D")),
        macro_data=macro,
        offline_only=True,
    )

    assert list(result.columns) == ["FRED_DGS10"]
    assert result["FRED_DGS10"].tolist() == [4.1, 4.2]


def test_macro_enricher_offline_policy_never_falls_back_to_api(monkeypatch):
    enricher = object.__new__(MacroFeaturesEnricher)

    def fail_api(*args, **kwargs):
        raise AssertionError("offline mode must not call FRED")

    monkeypatch.setattr(enricher, "_load_macro_data", fail_api)
    result = enricher._prepare_macro_data(
        pd.DataFrame(index=pd.date_range("2025-01-02", periods=2, freq="D")),
        offline_only=True,
    )

    assert result.empty


def test_stale_legacy_cache_columns_never_pollute_the_merged_frame(monkeypatch, tmp_path):
    """Reproduces a bug found while reviewing a real pipeline run's
    feature_lineage_report.json: fed_funds_rate/cpi/gdp/vix/
    consumer_sentiment/etc. all showed nan_ratio=1.0. Root cause:
    ./cache/macro_data.parquet is shared with the older
    _load_macro_data()/_load_fred_series() fallback path, which writes
    semantically-named columns (from self.config's series-name mapping)
    instead of this (Stage-1-driven) path's FRED_-prefixed ones.
    Concatenating an unfiltered legacy cache pulled those semantic columns
    into the merged frame, permanently NaN since this path never populates
    them. Fix: only FRED_*-prefixed cache columns are kept before merging.
    """
    enricher = object.__new__(MacroFeaturesEnricher)
    enricher.cache_path = tmp_path / "macro_data.parquet"

    # Legacy cache: has both a real FRED_ column (worth keeping — this is
    # the actual caching benefit) AND stale semantic-named columns from the
    # old direct-fetch path (must be dropped, not merged in).
    legacy_cache = pd.DataFrame(
        {
            "FRED_DGS10": [4.0, 4.1],
            "fed_funds_rate": [5.25, 5.25],
            "cpi": [3.1, 3.1],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    legacy_cache.to_parquet(enricher.cache_path)

    macro = pd.DataFrame(
        {
            "datetime": ["2025-01-01", "2025-01-02"],
            "series_id": ["DGS10", "DGS10"],
            "value": [4.5, 4.6],
        }
    )

    result = enricher._prepare_macro_data(
        pd.DataFrame(index=pd.date_range("2025-01-02", periods=2, freq="D")),
        macro_data=macro,
    )

    assert "fed_funds_rate" not in result.columns
    assert "cpi" not in result.columns
    # The genuine FRED_ history from the cache must still be preserved —
    # this fix must not throw away the actual caching benefit.
    assert "FRED_DGS10" in result.columns
    assert pd.Timestamp("2024-01-01") in result.index
