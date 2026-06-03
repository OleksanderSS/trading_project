import numpy as np
import pandas as pd

from src.analytics.analyzers.performance_attribution_analyzer import PerformanceAttributionAnalyzer


def test_risk_adjusted_attribution_drops_unaligned_nan_rows():
    analyzer = PerformanceAttributionAnalyzer()
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    portfolio = pd.DataFrame(
        {
            "asset_a": [0.01, 0.02, np.nan, 0.03, 0.01],
            "asset_b": [0.02, 0.01, np.nan, 0.02, 0.02],
        },
        index=index,
    )
    benchmark = pd.DataFrame({"spy": [0.01, 0.015, 0.02, 0.025, 0.012]}, index=index)

    result = analyzer._risk_adjusted_attribution(portfolio, benchmark)

    assert np.isfinite(result["jensen_alpha"])
    assert np.isfinite(result["m2_measure"])
    assert np.isfinite(result["realized_beta"])

