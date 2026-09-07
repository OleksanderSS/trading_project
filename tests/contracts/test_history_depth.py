"""
Declared history depth must cover the windows the feature layer computes.

Every enabled source was capped at 60 days while the enrichers compute 252-day
rolling statistics (SHARPE_RATIO, SORTINO_RATIO, AUTOCORR, SKEWNESS, KURTOSIS
in technical_analysis_enricher) and a 200-period moving average. Over 60 rows
each of those either comes out NaN or silently falls back to min_periods=30 —
a one-month estimate carrying an annual label. That is not a short history; it
is a feature computed on data that is not there.

Two of the sources did not even honour the depth they declared: vix_collector
hardcoded period="60d" and alternative_me hardcoded limit=100 into the URL.
"""
from __future__ import annotations

import ast
from pathlib import Path

import yaml

CONFIG = Path("src/config/collectors.yaml")
COLLECTORS = Path("src/data/collectors")

# The longest rolling window any enricher applies to a daily series.
LONGEST_DAILY_WINDOW = 252
# Yahoo caps intraday history at 60 days server-side; that is not our choice.
INTRADAY_TIMEFRAMES = {"15m", "60m", "1m", "5m", "30m", "90m"}
TRADING_DAYS_PER_YEAR = 252


def _config() -> dict:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return data.get("collectors", data)


def _period_to_trading_days(period: str) -> int:
    """Convert a yfinance-style period ('5y', '60d') to approximate trading days."""
    period = period.strip().lower()
    if period.endswith("y"):
        return int(float(period[:-1]) * TRADING_DAYS_PER_YEAR)
    if period.endswith("mo"):
        return int(float(period[:-2]) * TRADING_DAYS_PER_YEAR / 12)
    if period.endswith("d"):
        # Calendar days -> trading days, roughly 5 in 7.
        return int(int(period[:-1]) * 5 / 7)
    raise ValueError(f"Unsupported period format: {period!r}")


def test_daily_market_history_covers_the_longest_feature_window():
    timeframes = _config()["yahoo_finance"]["timeframes"]
    daily = {tf: cfg for tf, cfg in timeframes.items() if tf not in INTRADAY_TIMEFRAMES}
    assert daily, "no daily timeframe configured for yahoo_finance"

    for tf, cfg in daily.items():
        days = _period_to_trading_days(cfg["period"])
        assert days >= LONGEST_DAILY_WINDOW, (
            f"yahoo_finance[{tf}] declares {cfg['period']} ~ {days} trading days, "
            f"below the {LONGEST_DAILY_WINDOW}-period windows the enrichers "
            "compute. Those features would be NaN or silently fall back to "
            "min_periods=30."
        )


def test_daily_context_sources_cover_the_longest_feature_window():
    """VIX feeds the same 252-day volatility windows as prices do."""
    days = _period_to_trading_days(_config()["vix"]["params"]["period"])
    assert days >= LONGEST_DAILY_WINDOW


def test_macro_history_is_declared_in_years_not_days():
    """
    FRED series are monthly or quarterly. A depth in days is almost certainly a
    mistake: 60d of GDP is zero to one observation.
    """
    period = _config()["fred"]["params"]["period"].strip().lower()
    assert period.endswith("y"), (
        f"fred declares period={period!r}. Macro series need years of history; "
        "a window measured in days yields almost no observations."
    )
    assert _period_to_trading_days(period) >= LONGEST_DAILY_WINDOW


def test_intraday_depth_is_left_at_the_provider_cap():
    """Guard the other way: don't 'fix' a limit the provider imposes."""
    timeframes = _config()["yahoo_finance"]["timeframes"]
    for tf, cfg in timeframes.items():
        if tf in INTRADAY_TIMEFRAMES:
            assert cfg["period"] == "60d", (
                f"yahoo_finance[{tf}] is intraday; Yahoo caps that at 60d "
                "server-side, so a larger value would silently return less."
            )


def _assigned_string_constants(path: Path) -> set[str]:
    """Every string literal passed as a keyword argument in the module."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    out.add(f"{kw.arg}={kw.value.value}")
    return out


def test_vix_collector_does_not_hardcode_its_period():
    """It used to call history(period="60d"), ignoring the configured value."""
    literals = _assigned_string_constants(COLLECTORS / "vix_collector.py")
    hardcoded = {lit for lit in literals if lit.startswith("period=")}
    assert not hardcoded, (
        f"vix_collector passes a literal period {hardcoded}; it must use the "
        "configured self.period or the declared depth means nothing."
    )


def test_alternative_me_does_not_hardcode_its_limit():
    """The limit used to be baked into the request URL."""
    text = (COLLECTORS / "alternative_me_collector.py").read_text(encoding="utf-8")
    assert "?limit=100" not in text and "?limit=1000" not in text, (
        "alternative_me hardcodes its limit into the URL, so params.limit in "
        "collectors.yaml has no effect."
    )
    assert "limit={self.limit}" in text
