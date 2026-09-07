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
    assert "FRED_DGS10" in result.columns

    # This test used to end with `assert pd.Timestamp("2024-01-01") in
    # result.index` -- the cached 2024 rows had to survive into the merged
    # frame, because the cache's stated purpose was "accumulating more
    # historical FRED_ rows". That union was itself a defect and was removed
    # on 2026-08-29, so the assertion was pinning the bug.
    #
    # What it cost: 695 of 8,447 cached rows carried an availability stamp
    # computed by code that has since been fixed -- midnight instead of
    # 23:59:59, values for only eighteen of forty-five series. Daily bars are
    # stamped at midnight too, so merge_asof matched those rows exactly and
    # handed the bar a NaN for nearly every series; the per-ticker forward
    # fill then filled the hole from each NAME's own history. META missed the
    # 2024-07-05 claims print and carried a two-year-old value for 514
    # consecutive sessions. That is how 44 of 45 macro columns came to
    # disagree between tickers on the same date -- a macro series labelling
    # which name it belongs to.
    #
    # So the contract is the opposite one now: this run's rows, and only this
    # run's.
    assert pd.Timestamp("2024-01-01") not in result.index, (
        "a cached row from a previous run reached the merged frame. The union "
        "was removed because those rows carry superseded availability stamps.")
    assert list(result.index) == list(pd.to_datetime(["2025-01-01", "2025-01-02"])), (
        f"the frame should hold exactly this run's two dates, got "
        f"{list(result.index)}")


def test_the_macro_cache_is_rebuilt_from_this_run_not_appended_to(monkeypatch, tmp_path):
    """The other half of the same change, which nothing was watching.

    Replacing the union with "this run's rows only" is safe only if the cache
    FILE is also rewritten. If the old file survived, the next run would read
    the same superseded rows again and the fix would last exactly one run.
    """
    enricher = object.__new__(MacroFeaturesEnricher)
    enricher.cache_path = tmp_path / "macro_data.parquet"
    pd.DataFrame(
        {"FRED_DGS10": [4.0, 4.1], "fed_funds_rate": [5.25, 5.25]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    ).to_parquet(enricher.cache_path)

    macro = pd.DataFrame({
        "datetime": ["2025-01-01", "2025-01-02"],
        "series_id": ["DGS10", "DGS10"],
        "value": [4.5, 4.6],
    })
    enricher._prepare_macro_data(
        pd.DataFrame(index=pd.date_range("2025-01-02", periods=2, freq="D")),
        macro_data=macro,
    )

    rewritten = pd.read_parquet(enricher.cache_path)
    assert pd.Timestamp("2024-01-01") not in rewritten.index, (
        "the previous run's rows are still in the cache file, so the next run "
        "will read the superseded availability stamps back in and the fix "
        "survives exactly one run")
    assert "fed_funds_rate" not in rewritten.columns
