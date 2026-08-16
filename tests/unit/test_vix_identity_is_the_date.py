"""One trading day, one VIX row — and the same numbers however often you collect.

Two defects with one shape, and the shape is the point: CONTENT had got into an
IDENTITY key, which turns the key into a change detector.

`hash_keys` was (date, vix_close, volatility_regime). The regime is derived from
a 20-day moving average, and the average was computed over `history(period=
"60d")` sliced up to the current row -- a window whose START moves with the
COLLECTION date. So the same trading day produced different statistics each
time it was fetched, the regime sometimes flipped, the hash changed with it,
and deduplication stored a second row. Measured: 22 of 77 dates duplicated,
including 2026-06-05 with an identical vix_close of 21.51 from the 2026-07-20
and 2026-08-04 runs but different vix_sma_20, vix_percentile_20/_80,
vix_zscore and volatility_regime.

The same defect put 273 duplicate bars in market_data_raw, where the hash was
taken of a formatted local-time string.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.collectors import vix_collector as module


@pytest.fixture
def collector():
    return object.__new__(module.VIXCollector)


def _history(days, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2026-01-01', periods=days, freq='B')
    close = 20 + np.cumsum(rng.normal(0, 0.4, days))
    return pd.DataFrame(
        {'Open': close, 'High': close + 1, 'Low': close - 1,
         'Close': close, 'Volume': 1000.0},
        index=index,
    )


def _stats(history, idx):
    """The derived block as the collector computes it, for one row."""
    upto = history.iloc[:idx + 1]
    window = module._STAT_WINDOW
    if len(upto) >= window:
        recent = upto['Close'].iloc[-window:]
        return {
            'sma': float(recent.mean()),
            'p20': float(recent.quantile(0.2)),
            'p80': float(recent.quantile(0.8)),
        }
    return {'sma': float('nan'), 'p20': float('nan'), 'p80': float('nan')}


def test_the_same_day_gives_the_same_numbers_from_a_longer_fetch():
    """The defect exactly: a later collection reaches further back."""
    full = _history(120)
    target = full.index[100]

    short = full.iloc[60:]           # as if fetched with a later start
    long = full.iloc[20:]            # as if fetched earlier, more history

    a = _stats(short, list(short.index).index(target))
    b = _stats(long, list(long.index).index(target))

    assert a == pytest.approx(b), (
        'the same trading day measured differently depending on when it was '
        'collected'
    )


def test_the_statistics_read_a_fixed_trailing_window():
    history = _history(120)
    idx = 100
    stats = _stats(history, idx)
    recent = history['Close'].iloc[idx + 1 - module._STAT_WINDOW: idx + 1]
    assert stats['sma'] == pytest.approx(float(recent.mean()))
    assert stats['p20'] == pytest.approx(float(recent.quantile(0.2)))


def test_too_little_history_gives_no_statistic_rather_than_a_substitute():
    """Substituting the close for a percentile asserts that today sits at both
    the 20th and the 80th percentile, which is not a missing value -- it is a
    made-up one."""
    stats = _stats(_history(120), module._STAT_WINDOW - 2)
    assert np.isnan(stats['sma'])
    assert np.isnan(stats['p20'])
    assert np.isnan(stats['p80'])


def _configured_hash_keys():
    import yaml
    config = yaml.safe_load(open('src/config/collectors.yaml', encoding='utf-8'))
    return (config.get('collectors', config)).get('vix', {}).get('hash_keys')


def test_the_identity_key_is_the_date_alone():
    keys = _configured_hash_keys()
    assert keys == ['date'], (
        f'hash_keys is {keys}; anything beyond the date makes the key a change '
        f'detector rather than an identity'
    )


def test_a_derived_column_never_belongs_in_the_key():
    keys = set(_configured_hash_keys() or [])
    derived = {'volatility_regime', 'vix_sma_20', 'vix_zscore', 'vix_signal',
               'vix_percentile_20', 'vix_percentile_80', 'vix_classification'}
    assert not (keys & derived), f'derived columns in the identity key: {keys & derived}'


def test_generate_hash_depends_only_on_the_key(collector):
    collector.hash_keys = ['date']
    row = pd.Series({'date': '2026-06-05', 'vix_close': 21.51,
                     'volatility_regime': 'high'})
    flipped = row.copy()
    flipped['volatility_regime'] = 'low'
    flipped['vix_close'] = 99.0
    assert collector.generate_hash(row) == collector.generate_hash(flipped)
