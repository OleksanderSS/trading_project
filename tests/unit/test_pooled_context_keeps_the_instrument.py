"""Pooling must hand the model every ticker AND tell it which one a row is.

Measured on target_hourly_breakout_1h: one pooled model beats 22 per-ticker
models at every cost ratio from 0.5 to 3.0. This covers the data layer that
makes that possible, and the two ways it could quietly fail to:

- the instrument column being dropped by the identity rule, so a pooled model
  cannot tell AAPL from XOM and nobody reports anything;
- the purge gap being applied in rows while the frame holds 22 rows per bar,
  which shrinks the gap by 22x and reopens the leak it exists to close.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.adapters.data_preparation import (
    filter_data_by_ticker_timeframe,
    prepare_data_for_models,
)
from src.pipeline.modeling_context import (
    INSTRUMENT_CODE_COLUMN,
    POOLED_TICKER,
    instrument_code,
    iter_model_contexts,
)
from src.pipeline.target_column_utils import is_identity_column

TICKERS = ['AAPL', 'MSFT', 'XOM']


def _frame(bars=120):
    stamps = pd.date_range('2026-01-01', periods=bars, freq='h')
    rows = []
    rng = np.random.default_rng(0)
    for ticker in TICKERS:
        rows.append(pd.DataFrame({
            'ticker': ticker,
            'interval': '60m',
            'datetime': stamps,
            'hash': [f'{ticker}-{i}' for i in range(bars)],
            'close_60m': rng.normal(100, 5, bars),
            'rsi_14_60m': rng.uniform(10, 90, bars),
            'target_up_1h': rng.integers(0, 2, bars),
        }))
    return pd.concat(rows, ignore_index=True)


def test_pooling_yields_one_frame_per_timeframe_with_every_ticker():
    contexts = list(iter_model_contexts(_frame(), pool_tickers=True))
    assert len(contexts) == 1
    ticker, timeframe, frame = contexts[0]
    assert ticker == POOLED_TICKER
    assert timeframe == '60m'
    assert set(frame['ticker']) == set(TICKERS)


def test_without_pooling_nothing_changes():
    contexts = list(iter_model_contexts(_frame()))
    assert sorted(t for t, _, _ in contexts) == sorted(TICKERS)
    for _, _, frame in contexts:
        assert INSTRUMENT_CODE_COLUMN not in frame.columns


def test_a_dict_of_frames_is_concatenated_not_recursed():
    # Recursing per dict entry would yield one "pooled" frame per ticker --
    # the per-ticker split wearing a different name.
    per_ticker = {t: g.drop(columns=['ticker'])
                  for t, g in _frame().groupby('ticker')}
    contexts = list(iter_model_contexts(per_ticker, pool_tickers=True))
    assert len(contexts) == 1
    assert set(contexts[0][2]['ticker']) == set(TICKERS)


def test_the_instrument_column_survives_feature_selection():
    assert not is_identity_column(INSTRUMENT_CODE_COLUMN)
    # It must also dodge the filter-column search, which matches 'ticker' or
    # 'symbol' ANYWHERE in a column name and then uses the first hit.
    assert 'ticker' not in INSTRUMENT_CODE_COLUMN
    assert 'symbol' not in INSTRUMENT_CODE_COLUMN


def test_the_instrument_code_needs_no_state_to_reproduce():
    assert instrument_code('AAPL') == instrument_code('aapl') == instrument_code(' AAPL ')
    assert instrument_code('AAPL') != instrument_code('MSFT')
    # The point of hashing: adding a ticker must not renumber the others.
    before = {t: instrument_code(t) for t in TICKERS}
    after = {t: instrument_code(t) for t in [*TICKERS, 'AAAA', 'ZZZZ']}
    assert all(after[t] == before[t] for t in TICKERS)


def test_the_pooled_filter_keeps_every_ticker_and_one_timeframe():
    frame = _frame()
    frame.loc[frame.index[:10], 'interval'] = '15m'
    kept = filter_data_by_ticker_timeframe(frame, POOLED_TICKER, '60m')
    assert set(kept['ticker']) == set(TICKERS)
    assert set(kept['interval']) == {'60m'}


def test_a_named_ticker_still_gets_only_its_own_rows():
    kept = filter_data_by_ticker_timeframe(_frame(), 'AAPL', '60m')
    assert set(kept['ticker']) == {'AAPL'}


def test_the_purge_gap_is_widened_for_a_pooled_frame(caplog):
    _, _, pooled = next(iter(iter_model_contexts(_frame(), pool_tickers=True)))
    with caplog.at_level('INFO'):
        prepared = prepare_data_for_models(
            df=pooled, ticker=POOLED_TICKER, timeframe='60m',
            target_cols=['target_up_1h'], gap_size=10, test_size=0.2,
        )
    assert prepared is not None
    # 3 tickers share every timestamp, so a 10-bar gap must become ~30 rows.
    scaled = [r for r in caplog.messages if 'Purge gap scaled' in r]
    assert scaled, 'the gap was applied in rows over a frame with 3 rows per bar'
    assert '10 -> 30 bars' in scaled[0]


def test_a_per_ticker_frame_keeps_its_gap_untouched(caplog):
    single = _frame()[lambda d: d['ticker'] == 'AAPL']
    with caplog.at_level('INFO'):
        prepared = prepare_data_for_models(
            df=single, ticker='AAPL', timeframe='60m',
            target_cols=['target_up_1h'], gap_size=10, test_size=0.2,
        )
    assert prepared is not None
    assert not [r for r in caplog.messages if 'Purge gap scaled' in r]


def test_prediction_derives_the_same_instrument_code_as_training():
    """The two ends must agree without anything being persisted between them.

    Stage 5 refuses a context outright when a selected feature is missing from
    its frame ("skipping prediction instead of filling zeros"), so a pooled
    champion whose feature set names `instrument_code` would drop EVERY
    context and the run would report no predictions at all. The column is
    derived at both ends from the same function rather than carried through the
    export, which also means a pooled champion works against feature sets built
    before pooling existed.
    """
    from src.pipeline.stages.prediction import data_preparation_service as svc

    assert svc.INSTRUMENT_CODE_COLUMN == INSTRUMENT_CODE_COLUMN
    _, _, pooled = next(iter(iter_model_contexts(_frame(), pool_tickers=True)))
    for ticker in TICKERS:
        trained = pooled.loc[pooled['ticker'] == ticker, INSTRUMENT_CODE_COLUMN].unique()
        assert list(trained) == [svc.instrument_code(ticker)], (
            f'{ticker} would be a different instrument at prediction time'
        )


def test_the_pooled_model_is_given_the_instrument_as_a_feature():
    _, _, pooled = next(iter(iter_model_contexts(_frame(), pool_tickers=True)))
    prepared = prepare_data_for_models(
        df=pooled, ticker=POOLED_TICKER, timeframe='60m',
        target_cols=['target_up_1h'], gap_size=10, test_size=0.2,
    )
    assert prepared is not None
    columns = list(prepared['light_models']['feature_names'])
    assert INSTRUMENT_CODE_COLUMN in columns, (
        'a pooled model that cannot tell one instrument from another is a '
        'model trained on a mixture it has no way to separate'
    )
    # And it must reach the matrix the model is actually fitted on, not just
    # the name list beside it.
    assert INSTRUMENT_CODE_COLUMN in prepared['light_models']['X_train'].columns
