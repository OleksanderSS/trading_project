"""A pooled model must be asked about each instrument, not about the pool.

Stage 5 asked a pooled champion once, for a ticker named `__POOLED__`, and
turned 5,500 rows (110 instruments x 50 bars) into one number with no
instrument attached -- unpriceable, unexecutable, unevaluable (REGISTER #216).

These tests pin the expansion itself: how many requests a context becomes,
what identity each carries, and what happens when the instruments cannot be
read. They do not re-test the per-ticker prediction path, which already
existed and already worked.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.modeling_context import (
    INSTRUMENT_META_KEY,
    POOLED_TICKER,
    artifact_ticker,
    instrument_ticker,
)


@pytest.fixture
def stage():
    from src.pipeline.stages.prediction.orchestrator import PredictionStage
    return PredictionStage.__new__(PredictionStage)


@pytest.fixture
def logger_on(stage):
    import logging
    stage.logger = logging.getLogger('test-expand')
    return stage


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'AAPL', 'NVDA'],
        'close': [1.0, 2.0, 3.0, 4.0],
    })


def test_a_per_ticker_context_stays_a_single_request(logger_on, frame):
    meta = {'ticker': 'AAPL', 'model_path': 'x.joblib'}
    assert logger_on._expand_context('ctx', meta, frame) == [('ctx', meta)]


def test_a_pooled_context_becomes_one_request_per_instrument(logger_on, frame):
    meta = {'ticker': POOLED_TICKER, 'model_path': 'x.joblib'}
    expanded = logger_on._expand_context('POOLED_ctx', meta, frame)

    assert [key for key, _ in expanded] == [
        'POOLED_ctx::AAPL', 'POOLED_ctx::MSFT', 'POOLED_ctx::NVDA'
    ]
    assert [instrument_ticker(m) for _, m in expanded] == ['AAPL', 'MSFT', 'NVDA']
    # The model is unchanged; only the instrument it is asked about differs.
    assert {m['model_path'] for _, m in expanded} == {'x.joblib'}
    # And the ARTIFACT identity stays pooled. Overriding `ticker` instead sent
    # the preprocessor lookup to PREP_BA_... , which does not exist, so the
    # model was served raw features instead of the z-scores it was trained on
    # -- a wrong answer with a warning, not a failure.
    assert {artifact_ticker(m) for _, m in expanded} == {POOLED_TICKER}
    # The original metadata must not be mutated -- the same dict is reused
    # across contexts by the caller.
    assert meta['ticker'] == POOLED_TICKER


def test_the_sentinel_never_becomes_one_of_its_own_instruments(logger_on):
    frame = pd.DataFrame({'ticker': ['AAPL', POOLED_TICKER], 'close': [1.0, 2.0]})
    expanded = logger_on._expand_context(
        'ctx', {'ticker': POOLED_TICKER}, frame
    )
    assert [instrument_ticker(m) for _, m in expanded] == ['AAPL']


def test_no_instruments_means_no_requests_rather_than_a_pooled_one(logger_on):
    """Falling back to the pooled request would restore the silent behaviour.

    Zero predictions is now a run-level failure, which is the point: it is
    reported. A pooled request would be reported as a success.
    """
    empty = pd.DataFrame({'close': [1.0]})
    assert logger_on._expand_context('ctx', {'ticker': POOLED_TICKER}, empty) == []


def test_a_per_ticker_context_reads_the_same_both_ways():
    """With no fan-out the two identities must stay identical."""
    meta = {'ticker': 'AAPL'}
    assert artifact_ticker(meta) == instrument_ticker(meta) == 'AAPL'
    assert INSTRUMENT_META_KEY not in meta
