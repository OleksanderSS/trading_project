"""Volatility is annualised on the frame's own bar count, not always on 252.

`volatility_5/10/20` were all computed as `returns.rolling(...).std() *
np.sqrt(252)`. 252 is the number of trading DAYS in a year, so that factor is
right for a daily bar and wrong by the square root of the intraday bar count
for anything finer.

Measured on the export of 2026-08-31: 25 bars per trading day on the 15m
frame and 7 on the 60m one, so those columns came out 5.00x and 2.65x too
small. Within one timeframe a constant factor is harmless -- StandardScaler
removes it before any model sees it -- so this stayed invisible for as long
as nothing compared the number to an absolute threshold.

One thing does: `volatility_regime` bins it at 0.15 / 0.25 / 0.35. Those cuts
put 95.3% of 15m rows and 82.3% of 60m rows in "low", against a well-spread
33/25/24/18 on the daily frame. Rescaled by the measured cadence the three
frames agree -- 37/29/21/14 and 33/29/22/16 against 33/25/24/18 -- and three
timeframes agreeing once one factor is corrected is what says the bins were
never the problem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.enrichers.volatility_enricher import VolatilityEnricher


#: Bars per trading day and their spacing, as the three frames really are.
CADENCE = {"15m": (26, 15), "60m": (7, 60), "1d": (1, 0)}


def _frame(timeframe: str, days: int = 40, tickers=("AAA", "BBB"),
           sigma: float = 0.001):
    """A frame whose SPACING matches the timeframe, not only its bar count.

    The first version spaced every frame 30 minutes apart and varied only how
    many bars a day held. `infer_periods_per_year` keys off the gap between
    bars, so a "26 bars a day" frame at 30-minute spacing is an hourly frame
    with a long day -- the fixture was describing a market that does not
    exist, and the test that failed on it was right to.
    """
    bars_per_day, minutes = CADENCE[timeframe]
    rng = np.random.default_rng(4)
    rows = []
    for ticker in tickers:
        stamps = []
        for day in range(days):
            base = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
            stamps.extend(
                base + pd.Timedelta(minutes=minutes * i)
                for i in range(bars_per_day)
            )
        price = 100 * np.exp(np.cumsum(rng.normal(0, sigma, len(stamps))))
        rows.append(pd.DataFrame({
            "datetime": stamps,
            "ticker": ticker,
            "close": price,
            "high": price * 1.001,
            "low": price * 0.999,
        }))
    return pd.concat(rows, ignore_index=True)


def test_a_daily_frame_is_unchanged():
    """One bar per day is the case sqrt(252) was always right for."""
    assert VolatilityEnricher._bars_per_year(_frame("1d")) == 252


def test_an_intraday_frame_is_annualised_on_its_own_cadence():
    """And on the CANONICAL table, not on a second copy of it.

    `infer_periods_per_year` already answers this question for Sharpe and for
    the backtest engine. A private bar count here would be a third copy of a
    number this project has already been bitten by having two of.
    """
    from src.metrics.financial.financial_metrics_library import (
        infer_periods_per_year,
    )

    for timeframe in ("15m", "60m"):
        frame = _frame(timeframe)
        stamps = frame[frame["ticker"] == "AAA"]["datetime"]
        expected = infer_periods_per_year(
            pd.Series(0.0, index=pd.DatetimeIndex(stamps))
        )
        assert VolatilityEnricher._bars_per_year(frame) == expected
        assert expected > 252


def test_a_pooled_frame_is_not_read_as_sub_minute_bars():
    """Interleaved tickers share timestamps, so raw gaps there are zero."""
    frame = _frame("60m", tickers=("AAA", "BBB", "CCC")).sort_values("datetime")
    assert VolatilityEnricher._bars_per_year(frame) == 7 * 252


def test_the_intraday_scale_is_corrected_by_the_measured_factor():
    """The same returns must annualise higher on a finer frame, not lower."""
    enricher = VolatilityEnricher()
    daily = enricher.enrich(_frame("1d", days=200))
    intraday = enricher.enrich(_frame("15m", days=200))

    # Same generator, same per-bar volatility: the annualised figure has to
    # come out roughly sqrt(26) = 5.1x larger on the frame with 26x the bars.
    ratio = (
        pd.to_numeric(intraday["volatility_10"], errors="coerce").median()
        / pd.to_numeric(daily["volatility_10"], errors="coerce").median()
    )
    assert 3.5 < ratio < 7.0, ratio


def test_a_frame_without_timestamps_falls_back_loudly_not_silently(caplog):
    """An unusable frame must say the scale may be wrong, not assume it is right."""
    frame = _frame("60m").drop(columns=["datetime"])
    with caplog.at_level("WARNING"):
        assert VolatilityEnricher._bars_per_year(frame) == 252
    assert any("annualised as if the bars were daily" in r.message
               for r in caplog.records)


def test_the_regime_bins_stop_collapsing_onto_one_label():
    """The consequence the correction exists for.

    On the real export the 15m frame binned 95.3% of rows as "low". A frame
    with real intraday variation must not produce a near-constant column,
    because a column that never varies is not a regime and cannot be a
    feature either.
    """
    # Per-bar sigma taken from the real 15m frame: its median volatility_10
    # was 0.0374 at the wrong sqrt(252), i.e. 0.0374 / 15.87 = 0.0024 per bar.
    # The first version of this test used 0.001 and every row still binned as
    # "low" -- the fixture was calmer than any real market, not the code wrong.
    enriched = VolatilityEnricher().enrich(
        _frame("15m", days=200, sigma=0.0024))
    share = enriched["volatility_regime"].value_counts(normalize=True)
    assert share.max() < 0.95, dict(share)

    # And the same returns under the old blanket sqrt(252) would have been
    # nearly all "low" -- the state this corrects.
    old_scale = pd.cut(
        pd.to_numeric(enriched["volatility_10"], errors="coerce")
        * np.sqrt(252) / np.sqrt(26 * 252),
        bins=[0, 0.15, 0.25, 0.35, float("inf")],
        labels=["low", "normal", "high", "extreme"],
    ).value_counts(normalize=True)
    assert old_scale["low"] > 0.95, dict(old_scale)
