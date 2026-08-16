"""`cftc_available` and `fear_greed_available` must mean what their siblings mean.

Three flags in this pipeline -- hype_available, news_impact_available,
sentiment_available -- were deliberately settled on "is there a reading at THIS
row", after sentiment_available spent three timeframes as the constant 1.0 by
being read off a forward-filled series.

These two were left computing `value.notna()` on exactly such a series. On the
v14 batch `cftc_available` is the constant 1.0 on 15m, 60m AND 1d: CFTC covers
the whole bar history and an unbounded backward as-of join carries the last
report forever, so every bar answered yes. A column with one distinct value
cannot inform a model.

The first repair swapped one useless constant for another and is worth keeping
in mind: "fresh" defined as "within one bar-width of the release" measured as
the constant ZERO, because a daily figure with a one-day lag becomes readable
at midnight and no bar trades within an hour of midnight. Freshness is anchored
on the release -- the first bar to see it -- not on the clock.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.market_wide_enricher import MarketWideEnricher

TICKERS = ['AAPL', 'MSFT', 'XOM']
DAYS = pd.date_range('2026-03-02', periods=40, freq='B')
BARS = [day + pd.Timedelta(hours=14 + hour) for day in DAYS for hour in range(7)]


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _bars():
    return pd.DataFrame({
        'datetime': BARS * len(TICKERS),
        'ticker': sum([[t] * len(BARS) for t in TICKERS], []),
        'close': 1.0,
    })


def _cftc():
    """Weekly, as the real Commitments of Traders report is."""
    return pd.DataFrame({
        'date': pd.date_range('2026-02-24', periods=30, freq='W-TUE'),
        'instrument': 'S&P',
        'net_position_pct': np.linspace(-5, 5, 30),
        'long_short_ratio': np.linspace(0.8, 1.2, 30),
    })


def _fear_greed():
    return pd.DataFrame({
        'date': pd.date_range('2026-02-24', periods=200, freq='D'),
        'value': np.linspace(20, 80, 200),
    })


@pytest.fixture
def enriched():
    return MarketWideEnricher().enrich(
        _bars(), cftc_data=_cftc(), fear_greed_data=_fear_greed()
    )


@pytest.mark.parametrize('flag', ['cftc_available', 'fear_greed_available'])
def test_the_flag_is_not_a_constant(enriched, flag):
    assert enriched[flag].nunique() == 2, (
        f'{flag} has one distinct value, so it cannot inform a model'
    )


@pytest.mark.parametrize('flag', ['cftc_available', 'fear_greed_available'])
def test_the_flag_is_neither_always_on_nor_always_off(enriched, flag):
    share = float(enriched[flag].mean())
    assert 0.0 < share < 1.0, f'{flag} reads {share:.3f} everywhere'


def test_a_daily_source_is_fresh_once_a_day_per_ticker(enriched):
    """The case the first attempt got wrong: the release lands at midnight and
    the bars trade from 14:00, so a bar-width window matched nothing at all."""
    assert int(enriched['fear_greed_available'].sum()) == len(DAYS) * len(TICKERS)


def test_a_weekly_source_is_fresh_far_less_often_than_a_daily_one(enriched):
    assert enriched['cftc_available'].sum() < enriched['fear_greed_available'].sum() / 3


def test_every_ticker_sees_the_release(enriched):
    per_ticker = enriched.groupby('ticker')['fear_greed_available'].sum()
    assert per_ticker.nunique() == 1, (
        'a market-wide release reaches every instrument; each has its own '
        'first sight of it'
    )
    assert (per_ticker > 0).all()


def test_the_value_still_carries_forward(enriched):
    """The flag changed meaning; the series did not. A model that wants the
    level still gets it on every bar."""
    assert float(enriched['cftc_sp500_net_pct'].notna().mean()) == 1.0
    assert enriched['cftc_sp500_net_pct'].nunique() > 1
    assert float(enriched['fear_greed_index'].notna().mean()) > 0.9


def test_bars_before_the_first_release_are_not_marked_fresh():
    bars = _bars()
    late = _fear_greed().assign(
        date=pd.date_range('2026-04-01', periods=200, freq='D')
    )
    enriched = MarketWideEnricher().enrich(bars, fear_greed_data=late)
    early = pd.to_datetime(enriched['datetime']) < pd.Timestamp('2026-04-01')
    assert early.any(), 'the fixture must contain bars predating the source'
    assert int(enriched.loc[early, 'fear_greed_available'].sum()) == 0
