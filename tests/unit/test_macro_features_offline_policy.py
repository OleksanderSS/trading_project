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
